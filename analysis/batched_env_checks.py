"""Checks that the batched TRPO engine computes what the scalar one does.

uqc.batched_env.BatchedQuantumControlEnv is a fourth copy of the physics (after uqc.env,
uqc.eval and uqc.baseline_adam), used by train_trpo_single_target.py --engine batched. This
script compares it with uqc.env.QuantumControlEnv and prints the largest differences:

  step by step     same actions and same noise in both environments: cost, fidelity, leakage
                   bound and observation at every step, without noise, with noise, and with
                   the leakage bound taken from the noise-free controls
  rollouts         batched stochastic rollouts of a policy, with a stopping cost that ends
                   about half of the episodes early, replayed action by action in the scalar
                   environment: observations, rewards, episode boundaries and final infos
  advantages       TRPOAgent.compute_returns_advantages against the per-transition recurrence
                   it replaced

It writes no files and exits with an error if a difference exceeds the tolerance.

Run from anywhere: python analysis/batched_env_checks.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
from dataclasses import replace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from uqc.batched_env import BatchedQuantumControlEnv  # noqa: E402
from uqc.env import EnvConfig, QuantumControlEnv  # noqa: E402
from uqc.physics import GmonSystem, GmonSystemConfig, UFOCostWeights  # noqa: E402
from uqc.trpo import PolicyNetwork, TRPOAgent, TRPOConfig  # noqa: E402

torch.set_num_threads(1)

TOLERANCE = 1e-10          # double-precision quantities
OBS_TOLERANCE = 1e-6       # observations and rewards are stored in single precision
FAILURES: list[str] = []


def report(label: str, value: float, tolerance: float) -> None:
    ok = value <= tolerance
    print(f"  {label:<58} {value:9.2e}  {'ok' if ok else 'FAIL'}")
    if not ok:
        FAILURES.append(label)


def base_config(**kwargs) -> EnvConfig:
    return EnvConfig(
        target_alpha=2.2,
        max_time_ns=180.0,
        dt_ns=2.0,
        train_noise_std_mhz=1.0,
        termination_cost=-1.0,
        runtime_norm_ns=60.0,
        cost_weights=UFOCostWeights(),
        **kwargs,
    )


def check_steps(system: GmonSystem) -> None:
    print("step by step, same actions and noise (max |batched - scalar| over 6 episodes x 90 steps)")
    B, T = 6, 90
    rng = np.random.default_rng(123)
    actions = np.tanh(np.cumsum(rng.standard_normal((T, B, 7)) * 0.25, axis=0))
    cases = {
        "no noise": base_config(noise_optimized=False),
        "noise": base_config(noise_optimized=True),
        "noise, nominal leakage": base_config(noise_optimized=True, leakage_from_nominal_controls=True),
    }
    for name, cfg in cases.items():
        keys = ("cost", "fidelity", "leakage", "boundary_cost")
        ref = {k: np.zeros((T, B)) for k in keys}
        ref_obs = np.zeros((T, B, 163), dtype=np.float32)
        noise = np.zeros((T, B, 6))
        for b in range(B):
            env = QuantumControlEnv(system, replace(cfg, seed=1000 + b))
            twin = np.random.default_rng(1000 + b)  # the stream the scalar environment draws from
            for t in range(T):
                if cfg.noise_optimized:
                    noise[t, b, 0] = twin.normal(0.0, cfg.train_noise_std_mhz)
                    noise[t, b, 1:] = twin.normal(0.0, cfg.train_noise_std_mhz, size=5)
                obs, _, _, info = env.step(actions[t, b])
                ref_obs[t, b] = obs
                for k in keys:
                    ref[k][t, b] = info[k]
        benv = BatchedQuantumControlEnv(system, cfg, B, seed=0)
        err = {k: 0.0 for k in keys}
        err_obs = 0.0
        for t in range(T):
            obs, _, _, _, info = benv.step(torch.as_tensor(actions[t]), noise_mhz=torch.as_tensor(noise[t]))
            for k in keys:
                err[k] = max(err[k], float(np.abs(info[k].numpy() - ref[k][t]).max()))
            err_obs = max(err_obs, float(np.abs(obs.numpy() - ref_obs[t]).max()))
        for k in keys:
            report(f"{name}: {k}", err[k], TOLERANCE)
        report(f"{name}: observation", err_obs, OBS_TOLERANCE)
        print(f"  {'':<58} (leakage bound at the last step: {ref['leakage'][-1].mean():.4f})")


def check_rollouts(system: GmonSystem) -> None:
    print("batched rollouts replayed in the scalar environment (no noise)")
    B = 48
    torch.manual_seed(7)
    policy = PolicyNetwork(163, 7, (64, 32, 32), -0.5).eval()
    cfg = base_config(noise_optimized=False, seed=11)
    _, finals = BatchedQuantumControlEnv(system, cfg, B).rollout(policy)
    cfg = replace(cfg, termination_cost=float(np.median(finals["min_cost"])))
    batch, finals = BatchedQuantumControlEnv(system, cfg, B).rollout(policy)

    ends = np.flatnonzero(batch["masks"] == 0.0)
    starts = np.concatenate([[0], ends[:-1] + 1])
    lengths = ends - starts + 1
    print(f"  {B} episodes, stopping cost {cfg.termination_cost:.3f}: {int((lengths < 90).sum())} end early, "
          f"lengths {int(lengths.min())}-{int(lengths.max())} steps, {batch['obs'].shape[0]} transitions")
    report("episodes found in the mask", abs(len(ends) - B), 0)

    err_obs = err_rew = 0.0
    err_final = {k: 0.0 for k in finals}
    wrong_length = 0
    for b, (i0, i1) in enumerate(zip(starts, ends)):
        env = QuantumControlEnv(system, cfg)
        obs = env.reset()
        steps = 0
        info: dict = {}
        for i in range(i0, i1 + 1):
            err_obs = max(err_obs, float(np.abs(obs - batch["obs"][i]).max()))
            obs, reward, done, info = env.step(np.tanh(batch["raw_actions"][i].astype(np.float64)))
            err_rew = max(err_rew, abs(reward - float(batch["rewards"][i])) / max(1.0, abs(reward)))
            steps += 1
            if done:
                break
        wrong_length += int(steps != i1 - i0 + 1 or not done)
        info["min_cost"] = float(np.min(env.cost_history))
        info["min_cost_time_ns"] = float((int(np.argmin(env.cost_history)) + 1) * system.dt_ns)
        for k in finals:
            err_final[k] = max(err_final[k], abs(float(finals[k][b]) - float(info[k])))
    report("episodes with a different length", wrong_length, 0)
    report("observations", err_obs, OBS_TOLERANCE)
    report("rewards (relative)", err_rew, OBS_TOLERANCE)
    for k, v in err_final.items():
        # The actions are stored in single precision, so the replay is not exact to 1e-10.
        report(f"final info: {k}", v, 1e-5)


def reference_gae(rewards, values, masks, gamma, lam):
    n = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    adv = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype)
    running_return = torch.zeros((), dtype=rewards.dtype)
    next_value = torch.zeros((), dtype=rewards.dtype)
    for t in reversed(range(n)):
        running_return = rewards[t] + gamma * running_return * masks[t]
        returns[t] = running_return
        delta = rewards[t] + gamma * next_value * masks[t] - values[t]
        gae = delta + gamma * lam * masks[t] * gae
        adv[t] = gae
        next_value = values[t]
    return returns, (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)


def check_advantages() -> None:
    print("returns and advantages against the per-transition recurrence")
    agent = TRPOAgent(163, 7, config=TRPOConfig())
    rng = np.random.default_rng(3)
    for name, last_mask in (("every episode finished", 0.0), ("last episode cut", 1.0)):
        lengths = rng.integers(1, 91, size=200)
        lengths[3] = 1
        n = int(lengths.sum())
        rewards = torch.as_tensor(-rng.random(n) * 10.0, dtype=torch.float32)
        values = torch.as_tensor(rng.standard_normal(n), dtype=torch.float32)
        masks = torch.ones(n)
        masks[np.cumsum(lengths) - 1] = 0.0
        masks[-1] = last_mask
        ret, adv = agent.compute_returns_advantages(rewards, values, masks)
        ret0, adv0 = reference_gae(rewards, values, masks, agent.config.gamma, agent.config.lam)
        report(f"{name}: returns", float((ret - ret0).abs().max()), 0.0)
        report(f"{name}: advantages", float((adv - adv0).abs().max()), 0.0)


def main() -> None:
    system = GmonSystem(GmonSystemConfig(dt_ns=2.0, runtime_norm_ns=60.0, bandwidth_mhz=10.0))
    check_steps(system)
    check_rollouts(system)
    check_advantages()
    if FAILURES:
        raise SystemExit(f"FAILED: {', '.join(FAILURES)}")
    print("all checks passed")


if __name__ == "__main__":
    main()
