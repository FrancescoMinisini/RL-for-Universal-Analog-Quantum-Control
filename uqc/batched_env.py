from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch

from .env import EnvConfig
from .physics import GmonSystem


Tensor = torch.Tensor


class BatchedQuantumControlEnv:
    """
    QuantumControlEnv for B episodes at once.

    Same arithmetic as env.QuantumControlEnv and physics.TSWTLeakageEstimator, in double
    precision, with a leading batch dimension; the operators and masks are taken from the
    NumPy GmonSystem. analysis/batched_env_checks.py compares the two step by step.

    The only difference is the random stream: the noise comes from a torch generator, so a
    batched rollout is not sample-by-sample the rollout of the scalar environment.
    """

    def __init__(self, system: GmonSystem, config: EnvConfig, batch_size: int, seed: int | None = None):
        if abs(system.dt_ns - config.dt_ns) > 1e-12:
            raise ValueError("EnvConfig.dt_ns must match GmonSystem.config.dt_ns")
        self.system = system
        self.config = config
        self.batch_size = int(batch_size)
        self.dt_ns = float(system.dt_ns)
        self.scale = float(system.mhz_to_rad_per_ns)
        self.eta_base_mhz = float(system.config.eta_base_mhz)
        self.max_steps = int(round(config.max_time_ns / config.dt_ns))
        self.observation_dim = 2 * (system.dim * system.dim) + 1
        self.action_dim = 7

        if seed is None:
            seed = config.seed
        self.generator = torch.Generator()
        self.generator.manual_seed(0 if seed is None else int(seed))

        def c(a: np.ndarray) -> Tensor:
            return torch.as_tensor(np.asarray(a), dtype=torch.complex128)

        # Operators with a real coefficient, in the order eta, delta1, delta2, g.
        self.real_ops = torch.stack([c(system.drift_shape), c(system.n1), c(system.n2), c(system.op_coupling)])
        self.drift_shape = c(system.drift_shape)
        self.a1, self.ad1, self.a2, self.ad2 = c(system.a1), c(system.ad1), c(system.a2), c(system.ad2)
        self.block_mask = torch.as_tensor(system.block_mask, dtype=torch.float64)
        self.offdiag_mask = torch.as_tensor(system.offdiag_mask, dtype=torch.float64)
        inv = np.zeros_like(system.coeff_diff_matrix)
        np.divide(1.0, system.coeff_diff_matrix, out=inv, where=system.denom_mask)
        self.inv_coeff_diff = torch.as_tensor(np.where(system.offdiag_mask, inv, 0.0), dtype=torch.float64)
        self.comp = torch.as_tensor(system.comp_indices, dtype=torch.long)
        self.target_gate = c(system.target_gate(config.target_alpha, config.target_gamma))
        self.eye = c(system.eye)

        f_sample_ghz = 1.0 / self.dt_ns
        alpha = math.exp(-math.pi * (config.filter_bandwidth_mhz * 1e-3) / f_sample_ghz)
        self.filter_a1 = (1.0 - alpha) ** 2
        self.filter_b1 = -2.0 * alpha
        self.filter_b2 = alpha ** 2
        self.linear = torch.tensor([0, 1, 2, 4, 6], dtype=torch.long)
        self.phase = torch.tensor([3, 5], dtype=torch.long)

        self.reset()

    def reset(self) -> Tensor:
        B = self.batch_size
        self.U = self.eye.expand(B, -1, -1).clone()
        self.steps = 0
        self.filter_prev1 = torch.zeros(B, 7, dtype=torch.float64)
        self.filter_prev2 = torch.zeros(B, 7, dtype=torch.float64)
        self.prev_s1: Tensor | None = None
        self.prev_s2: Tensor | None = None
        self.prev_hod_1: Tensor | None = None
        self.prev_hod_2: Tensor | None = None
        self.leakage_first = torch.zeros(B, dtype=torch.float64)
        self.leakage_integral = torch.zeros(B, dtype=torch.float64)
        self.first_controls: Tensor | None = None
        self.done = torch.zeros(B, dtype=torch.bool)
        return self._obs()

    def _obs(self) -> Tensor:
        flat = self.U.reshape(self.batch_size, -1)
        frac = torch.full((self.batch_size, 1), self.steps / max(self.max_steps, 1), dtype=torch.float64)
        return torch.cat([flat.real, flat.imag, frac], dim=1).to(torch.float32)

    def hamiltonian(self, controls: Tensor, eta_mhz: Tensor) -> Tensor:
        coeffs = torch.stack([eta_mhz, controls[:, 0], controls[:, 1], controls[:, 6]], dim=1) * self.scale
        H = torch.einsum("bk,kij->bij", coeffs.to(torch.complex128), self.real_ops)
        z1 = (1j * self.scale * controls[:, 2] * torch.exp(1j * controls[:, 3]))[:, None, None]
        z2 = (1j * self.scale * controls[:, 4] * torch.exp(1j * controls[:, 5]))[:, None, None]
        return H + z1 * self.a1 + z1.conj() * self.ad1 + z2 * self.a2 + z2.conj() * self.ad2

    @staticmethod
    def spectral_norm(M: Tensor) -> Tensor:
        # M is Hermitian wherever this is called, so the largest singular value is max |eigenvalue|.
        return torch.linalg.eigvalsh(M).abs().amax(dim=1)

    def _leakage_step(self, H: Tensor, eta_mhz: Tensor) -> Tensor:
        eta_rad = (eta_mhz * self.scale)[:, None, None]
        h_rest = H - eta_rad * self.drift_shape
        h1 = h_rest * self.block_mask
        h2 = h_rest - h1

        def solve_offdiag(numerator: Tensor) -> Tensor:
            return numerator * self.inv_coeff_diff / eta_rad

        def comm(A: Tensor, B: Tensor) -> Tensor:
            return A @ B - B @ A

        s1 = solve_offdiag(h2)
        ds1_dt = torch.zeros_like(s1) if self.prev_s1 is None else (s1 - self.prev_s1) / self.dt_ns
        s2 = solve_offdiag((comm(h1, s1) - 1j * ds1_dt) * self.offdiag_mask)
        ds2_dt = torch.zeros_like(s2) if self.prev_s2 is None else (s2 - self.prev_s2) / self.dt_ns

        residual = comm(h1, s2) + (1.0 / 3.0) * comm(comm(h2, s1), s1) - 1j * ds2_dt
        hod_eff = (0.5 * (residual + residual.conj().transpose(1, 2))) * self.offdiag_mask
        delta_rad = eta_mhz.abs() * self.scale
        last = self.spectral_norm(hod_eff) / delta_rad.clamp_min(1e-12)

        if self.prev_hod_1 is None:
            self.leakage_first = last
        if self.prev_hod_2 is not None and self.prev_hod_1 is not None:
            d2 = (hod_eff - 2.0 * self.prev_hod_1 + self.prev_hod_2) / (self.dt_ns ** 2)
            self.leakage_integral = self.leakage_integral + self.spectral_norm(d2) * self.dt_ns / (delta_rad ** 2).clamp_min(1e-12)

        self.prev_s1 = s1
        self.prev_s2 = s2
        self.prev_hod_2 = self.prev_hod_1
        self.prev_hod_1 = hod_eff
        return self.leakage_first + last + self.leakage_integral

    def step(self, action: Tensor, noise_mhz: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """
        action: [B, 7] in [-1, 1]. noise_mhz: optional [B, 6] perturbations of
        (eta, delta1, delta2, f1, f2, g), used instead of drawing them.

        Returns (obs, reward, done, alive, info). alive marks the episodes that were still
        running when the step was taken; the rows of finished episodes keep evolving and must
        be discarded by the caller.
        """
        cfg = self.config
        weights = cfg.cost_weights
        action = action.to(torch.float64).clamp(-1.0, 1.0)
        proposed = torch.empty_like(action)
        proposed[:, self.linear] = action[:, self.linear] * cfg.max_control_mhz
        proposed[:, self.phase] = (action[:, self.phase] + 1.0) * math.pi

        out = self.filter_a1 * proposed - self.filter_b1 * self.filter_prev1 - self.filter_b2 * self.filter_prev2
        self.filter_prev2 = self.filter_prev1
        self.filter_prev1 = out
        filtered = out.clone()
        filtered[:, self.phase] = torch.remainder(filtered[:, self.phase], 2.0 * math.pi)

        eta_mhz = torch.full((self.batch_size,), self.eta_base_mhz, dtype=torch.float64)
        noisy = filtered
        if cfg.noise_optimized:
            if noise_mhz is None:
                noise_mhz = torch.randn(self.batch_size, 6, dtype=torch.float64, generator=self.generator) * cfg.train_noise_std_mhz
            eta_mhz = eta_mhz + noise_mhz[:, 0]
            noisy = filtered.clone()
            noisy[:, self.linear] = noisy[:, self.linear] + noise_mhz[:, 1:]

        H = self.hamiltonian(noisy, eta_mhz)
        Hh = 0.5 * (H + H.conj().transpose(1, 2))
        eigvals, eigvecs = torch.linalg.eigh(Hh)
        phases = torch.exp(-1j * eigvals * self.dt_ns)
        self.U = (eigvecs * phases[:, None, :]) @ eigvecs.conj().transpose(1, 2) @ self.U

        if cfg.leakage_from_nominal_controls and cfg.noise_optimized:
            eta_nominal = torch.full_like(eta_mhz, self.eta_base_mhz)
            leakage = self._leakage_step(self.hamiltonian(filtered, eta_nominal), eta_nominal)
        else:
            leakage = self._leakage_step(H, eta_mhz)

        if self.first_controls is None:
            self.first_controls = filtered
        self.steps += 1
        t_ns = self.steps * self.dt_ns

        K = self.U[:, self.comp][:, :, self.comp]
        tr = (K.conj() * self.target_gate).sum(dim=(1, 2))
        fidelity = tr.abs() ** 2 / (self.target_gate.shape[0] ** 2)
        c0, ct = self.first_controls, filtered
        boundary_raw = c0[:, 6] ** 2 + c0[:, 2] ** 2 + c0[:, 4] ** 2 + ct[:, 6] ** 2 + ct[:, 2] ** 2 + ct[:, 4] ** 2
        boundary_cost = weights.mu * boundary_raw
        time_cost = weights.kappa * (t_ns / cfg.runtime_norm_ns)
        cost = weights.chi * (1.0 - fidelity) + weights.beta * leakage + boundary_cost + time_cost

        alive = ~self.done
        if self.steps >= self.max_steps:
            ended = torch.ones_like(self.done)
        else:
            ended = cost <= cfg.termination_cost
            if cfg.min_fidelity is not None:
                ended = ended & (fidelity >= cfg.min_fidelity)
        self.done = self.done | ended

        if cfg.reward_mode == "dense_current_cost":
            reward = -cost
        else:
            reward = torch.where(ended, -cost, torch.zeros_like(cost))
        info = {
            "fidelity": fidelity,
            "leakage": leakage,
            "boundary_cost": boundary_cost,
            "time_cost": torch.full_like(cost, time_cost),
            "cost": cost,
            "time_ns": torch.full_like(cost, t_ns),
            "nominal_controls": filtered,
        }
        return self._obs(), reward, self.done, alive, info

    def rollout(self, policy: torch.nn.Module) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Run B stochastic episodes of `policy` (a trpo.PolicyNetwork on the CPU).

        Returns the transitions as flat arrays, episode after episode, in the layout that
        TRPOAgent.update_from_batch expects, and the per-episode final infos as arrays.
        """
        B, T = self.batch_size, self.max_steps
        obs = self.reset()
        obs_buf = torch.empty(T, B, self.observation_dim, dtype=torch.float32)
        raw_buf = torch.empty(T, B, self.action_dim, dtype=torch.float32)
        rew_buf = torch.zeros(T, B, dtype=torch.float32)
        alive_buf = torch.zeros(T, B, dtype=torch.bool)

        final_keys = ("cost", "fidelity", "leakage", "time_ns", "boundary_cost", "time_cost")
        finals = {k: torch.zeros(B, dtype=torch.float64) for k in final_keys}
        min_cost = torch.full((B,), float("inf"), dtype=torch.float64)
        min_cost_time_ns = torch.zeros(B, dtype=torch.float64)

        with torch.no_grad():
            for t in range(T):
                mean, std = policy(obs)
                raw = mean + std * torch.randn(mean.shape, dtype=mean.dtype, generator=self.generator)
                obs_buf[t] = obs
                raw_buf[t] = raw
                obs, reward, done, alive, info = self.step(torch.tanh(raw))
                rew_buf[t] = reward.to(torch.float32)
                alive_buf[t] = alive
                for k in final_keys:
                    finals[k] = torch.where(alive, info[k], finals[k])
                better = alive & (info["cost"] < min_cost)
                min_cost = torch.where(better, info["cost"], min_cost)
                min_cost_time_ns = torch.where(better, info["time_ns"], min_cost_time_ns)
                if bool(done.all()):
                    break

        # Episode-major order: all steps of episode 0, then episode 1, ...
        alive_bt = alive_buf.t()
        lengths = alive_bt.sum(dim=1)
        last_step = torch.zeros(B, T, dtype=torch.bool)
        last_step[torch.arange(B), lengths - 1] = True
        batch = {
            "obs": obs_buf.transpose(0, 1)[alive_bt].numpy(),
            "raw_actions": raw_buf.transpose(0, 1)[alive_bt].numpy(),
            "rewards": rew_buf.t()[alive_bt].numpy(),
            "masks": (~last_step)[alive_bt].to(torch.float32).numpy(),
        }
        final_arrays = {k: v.numpy() for k, v in finals.items()}
        final_arrays["min_cost"] = min_cost.numpy()
        final_arrays["min_cost_time_ns"] = min_cost_time_ns.numpy()
        return batch, final_arrays
