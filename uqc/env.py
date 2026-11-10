from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .physics import GmonSystem, TSWTLeakageEstimator, UFOCostWeights


@dataclass
class EnvConfig:
    target_alpha: float
    target_gamma: float = np.pi / 2.0
    max_time_ns: float = 600.0
    dt_ns: float = 2.0
    noise_optimized: bool = False
    train_noise_std_mhz: float = 1.0
    reward_mode: str = "dense_current_cost"  # or 'terminal_ufo'
    termination_cost: float = 0.15
    max_control_mhz: float = 20.0
    filter_bandwidth_mhz: float = 10.0
    runtime_norm_ns: float = 60.0
    min_fidelity: float | None = None
    # Accumulate the leakage bound from the noise-free controls instead of the noisy Hamiltonian.
    leakage_from_nominal_controls: bool = False
    # "white": a new draw at every step (the UFO training noise); "quasi_static": one draw of the
    # offsets (eta, delta1, delta2, f1, f2, g) per episode, at reset, held for the whole episode.
    train_noise_mode: str = "white"
    # Observe the propagator of the noise-free controls instead of the noisy one, so that the
    # policy has no information about the noise realization and can only learn an open-loop pulse.
    # The cost and the reward stay those of the noisy propagator.
    observe_nominal: bool = False
    seed: int | None = None
    cost_weights: UFOCostWeights = UFOCostWeights()


class ExactExponentialFilter:
    """
    Two-pole normalized double exponential smoothing filter from Appendix C.
    """

    def __init__(self, dim: int, dt_ns: float, bandwidth_mhz: float = 10.0):
        self.dim = dim
        self.dt_ns = float(dt_ns)
        self.bandwidth_mhz = float(bandwidth_mhz)
        f_sample_ghz = 1.0 / self.dt_ns
        bw_ghz = self.bandwidth_mhz * 1e-3
        alpha = np.exp(-np.pi * bw_ghz / f_sample_ghz)
        self.a1 = (1.0 - alpha) ** 2
        self.b1 = -2.0 * alpha
        self.b2 = alpha ** 2
        self.reset()

    def reset(self) -> None:
        self.prev1 = np.zeros(self.dim, dtype=np.float64)
        self.prev2 = np.zeros(self.dim, dtype=np.float64)

    def filter(self, u_rl: Sequence[float]) -> np.ndarray:
        u_rl = np.asarray(u_rl, dtype=np.float64)
        out = self.a1 * u_rl - self.b1 * self.prev1 - self.b2 * self.prev2
        self.prev2 = self.prev1.copy()
        self.prev1 = out.copy()
        return out


class QuantumControlEnv:
    def __init__(self, system: GmonSystem, config: EnvConfig):
        self.system = system
        self.config = config
        if abs(system.dt_ns - config.dt_ns) > 1e-12:
            raise ValueError("EnvConfig.dt_ns must match GmonSystem.config.dt_ns")

        self.max_steps = int(round(config.max_time_ns / config.dt_ns))
        self.target_gate = system.target_gate(config.target_alpha, config.target_gamma)
        self.filter = ExactExponentialFilter(7, dt_ns=config.dt_ns, bandwidth_mhz=config.filter_bandwidth_mhz)
        self.rng = np.random.default_rng(config.seed)
        self.leakage = TSWTLeakageEstimator(system)

        dim = system.dim
        self.observation_dim = 2 * (dim * dim) + 1
        self.action_dim = 7
        if config.train_noise_mode not in ("white", "quasi_static"):
            raise ValueError(f"Unknown train_noise_mode: {config.train_noise_mode!r}")

        self.reset()

    def reset(self) -> np.ndarray:
        self.U = self.system.initial_unitary()
        self.t_ns = 0.0
        self.steps = 0
        self.filter.reset()
        self.leakage.reset()
        self.done = False
        self.current_eta_mhz = self.system.config.eta_base_mhz
        # Noise-free propagator, kept only when it can differ from U and the policy observes it.
        self.U_nominal = self.U.copy() if self._tracks_nominal() else None
        self.quasi_static_offsets: np.ndarray | None = None
        if self.config.noise_optimized and self.config.train_noise_mode == "quasi_static":
            std = self.config.train_noise_std_mhz
            # Drawn in the order of a white step: eta first, then the five control offsets.
            eta_offset = self.rng.normal(0.0, std)
            self.quasi_static_offsets = np.concatenate([[eta_offset], self.rng.normal(0.0, std, size=5)])
        self.nominal_controls_history: List[np.ndarray] = []
        self.noisy_controls_history: List[np.ndarray] = []
        self.reward_history: List[float] = []
        self.cost_history: List[float] = []
        self.fidelity_history: List[float] = []
        return self._obs()

    def _tracks_nominal(self) -> bool:
        return self.config.observe_nominal and self.config.noise_optimized

    def _obs(self) -> np.ndarray:
        flat = (self.U if self.U_nominal is None else self.U_nominal).reshape(-1)
        frac = self.steps / max(self.max_steps, 1)
        obs = np.concatenate([flat.real, flat.imag, np.array([frac], dtype=np.float64)])
        return obs.astype(np.float32)

    def map_action_to_controls(self, action: Sequence[float]) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        controls = np.zeros(7, dtype=np.float64)
        linear = [0, 1, 2, 4, 6]
        phase = [3, 5]
        controls[linear] = action[linear] * self.config.max_control_mhz
        controls[phase] = (action[phase] + 1.0) * np.pi
        return controls

    def _apply_noise(self, nominal_controls: np.ndarray) -> Tuple[np.ndarray, float]:
        controls = nominal_controls.copy()
        eta_mhz = self.system.config.eta_base_mhz
        if self.config.noise_optimized and self.quasi_static_offsets is not None:
            eta_mhz = self.system.config.eta_base_mhz + self.quasi_static_offsets[0]
            noise = self.quasi_static_offsets[1:]
        elif self.config.noise_optimized:
            std = self.config.train_noise_std_mhz
            eta_mhz = self.system.config.eta_base_mhz + self.rng.normal(0.0, std)
            noise = self.rng.normal(0.0, std, size=5)
        if self.config.noise_optimized:
            controls[0] += noise[0]
            controls[1] += noise[1]
            controls[2] += noise[2]
            controls[4] += noise[3]
            controls[6] += noise[4]
        controls[[3, 5]] = np.mod(controls[[3, 5]], 2.0 * np.pi)
        return controls, eta_mhz

    def current_cost(self) -> Tuple[float, Dict[str, float]]:
        fidelity = self.system.gate_fidelity(self.U, self.target_gate)
        leakage_total, leakage_parts = self.leakage.current_leakage_bound()
        weights = self.config.cost_weights

        if self.nominal_controls_history:
            c0 = self.nominal_controls_history[0]
            ct = self.nominal_controls_history[-1]
            boundary_raw = c0[6] ** 2 + c0[2] ** 2 + c0[4] ** 2 + ct[6] ** 2 + ct[2] ** 2 + ct[4] ** 2
        else:
            boundary_raw = 0.0
        boundary_cost = weights.mu * boundary_raw

        time_cost = weights.kappa * (self.t_ns / self.config.runtime_norm_ns)
        ufo = weights.chi * (1.0 - fidelity) + weights.beta * leakage_total + boundary_cost + time_cost
        info = {
            "fidelity": float(fidelity),
            "leakage": float(leakage_total),
            "leakage_boundary": leakage_parts["boundary"],
            "leakage_integral": leakage_parts["integral"],
            "boundary_cost": float(boundary_cost),
            "boundary_raw": float(boundary_raw),
            "time_cost": float(time_cost),
            "cost": float(ufo),
        }
        return float(ufo), info

    def step(self, action: Sequence[float]) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        if self.done:
            raise RuntimeError("Episode already finished; call reset().")

        proposed_controls = self.map_action_to_controls(action)
        filtered = self.filter.filter(proposed_controls)
        filtered[[3, 5]] = np.mod(filtered[[3, 5]], 2.0 * np.pi)
        noisy_controls, eta_mhz = self._apply_noise(filtered)

        self.current_eta_mhz = eta_mhz
        H = self.system.hamiltonian(noisy_controls, eta_mhz)
        self.U = scipy_expm_step(self.U, H, self.system.dt_ns)
        H_nominal = None
        if self.U_nominal is not None or (self.config.leakage_from_nominal_controls and self.config.noise_optimized):
            H_nominal = self.system.hamiltonian(filtered, self.system.config.eta_base_mhz)
        if self.U_nominal is not None:
            self.U_nominal = scipy_expm_step(self.U_nominal, H_nominal, self.system.dt_ns)
        if self.config.leakage_from_nominal_controls and self.config.noise_optimized:
            self.leakage.step(H_nominal, self.system.config.eta_base_mhz)
        else:
            self.leakage.step(H, eta_mhz)
        self.nominal_controls_history.append(filtered.copy())
        self.noisy_controls_history.append(noisy_controls.copy())

        self.steps += 1
        self.t_ns = self.steps * self.system.dt_ns
        cost, info = self.current_cost()
        self.cost_history.append(cost)
        self.fidelity_history.append(info["fidelity"])

        done = False
        if self.steps >= self.max_steps:
            done = True
            info["done_reason"] = "time_limit"
        elif cost <= self.config.termination_cost:
            if self.config.min_fidelity is None or info["fidelity"] >= self.config.min_fidelity:
                done = True
                info["done_reason"] = "cost_threshold"

        reward = -cost if self.config.reward_mode == "dense_current_cost" else (0.0 if not done else -cost)
        self.reward_history.append(float(reward))
        self.done = done
        info["time_ns"] = float(self.t_ns)
        return self._obs(), float(reward), done, info

    def rollout(self, policy, deterministic: bool = False) -> Dict[str, object]:
        obs = self.reset()
        transitions: List[Dict[str, object]] = []
        info = {
            "cost": np.inf,
            "fidelity": 0.0,
            "leakage": np.inf,
            "time_ns": 0.0,
        }
        while True:
            action, raw_action = policy(obs, deterministic=deterministic)
            next_obs, reward, done, info = self.step(action)
            transitions.append(
                {
                    "obs": obs,
                    "action": np.asarray(action, dtype=np.float32),
                    "raw_action": np.asarray(raw_action, dtype=np.float32),
                    "reward": float(reward),
                    "mask": 0.0 if done else 1.0,
                    "info": dict(info),
                }
            )
            obs = next_obs
            if done:
                break
        
        if self.cost_history:
            min_cost_idx = int(np.argmin(self.cost_history))
            info["min_cost"] = float(self.cost_history[min_cost_idx])
            info["min_cost_time_ns"] = float((min_cost_idx + 1) * self.system.dt_ns)
        else:
            info["min_cost"] = np.inf
            info["min_cost_time_ns"] = 0.0

        return {
            "transitions": transitions,
            "nominal_controls": np.asarray(self.nominal_controls_history, dtype=np.float64),
            "noisy_controls": np.asarray(self.noisy_controls_history, dtype=np.float64),
            "rewards": np.asarray(self.reward_history, dtype=np.float64),
            "costs": np.asarray(self.cost_history, dtype=np.float64),
            "fidelities": np.asarray(self.fidelity_history, dtype=np.float64),
            "final_info": info,
        }


# H is Hermitian; eigh is much faster than a generic matrix exponential here.
def scipy_expm_step(U: np.ndarray, H: np.ndarray, dt_ns: float) -> np.ndarray:
    Hh = 0.5 * (H + H.conj().T)
    eigvals, eigvecs = np.linalg.eigh(Hh)
    phases = np.exp(-1j * eigvals * dt_ns)
    return (eigvecs * phases) @ eigvecs.conj().T @ U
