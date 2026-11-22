"""How much of the closed-loop gain under quasi-static noise survives with calibrated offsets only?

In closed loop on the noisy propagator, a TRPO policy recovers almost all the fidelity lost to quasi-static
offsets (analysis/quasi_static_trpo.py), but it does so by observing the full 9x9 unitary during the pulse,
which no device provides. Quasi-static offsets are constant from shot to shot, so the information can come
from a calibration made before the gate instead: estimate the offsets, run the policy in closed loop on a
model with the estimated offsets, and send the resulting pulse to the device in open loop. No measurement
during the pulse is needed, and the policy becomes a generator of pulses conditioned on the calibration.

For every checkpoint this script draws CLOSED_SAMPLES true offsets eps (the same as quasi_static_trpo.py) and
a calibration measurement m = eps + sigma_m z for each measurement error sigma_m in SIGMA_M (z shared by all
checkpoints and levels). The estimate is the posterior mean, eps_hat = m sigma^2 / (sigma^2 + sigma_m^2), so
that sigma_m = 0 gives the true offsets (the closed loop of quasi_static_trpo.py, which the script checks)
and sigma_m = inf gives eps_hat = 0 (the policy's own noise-free pulse, the open loop). The pulse is the
deterministic policy rolled out on the model with eps_hat; it is scored by replaying it on eps.

Runs (read from final_results/<name> if it exists, else runs/<name>), seeds 1-3:
  qs40_nominal, qs40_feedback        40 MHz filter, no training noise / quasi-static training noise
  matched_nominal, matched_noise     10 MHz filter (the preprint's matched pairs), no training noise /
                                     white training noise (matched_noise_nomleak_seed<N>)

Outputs (--out, default runs/calibrated_feedback_results/):
  calibrated_feedback.csv  one row per (run, seed, iteration, sigma_m): average fidelity at SIGMA_MHZ
  summary.csv              per run, sigma_m and seed: medians over the iterations, and the fraction of the
                           closed-loop gain recovered, (F(sigma_m) - F(inf)) / (F(0) - F(inf))

Run from anywhere: python analysis/calibrated_feedback.py [--workers N] [--iterations 25:101:5]
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

import torch  # noqa: E402

torch.set_num_threads(1)

from noise_with_memory import write_csv  # noqa: E402
from quasi_static_trpo import PHASES, PLAN_TOLERANCE_MHZ, make_env  # noqa: E402
from uqc.eval import ControlPlan  # noqa: E402
from uqc.physics import GmonSystem, GmonSystemConfig, evaluate_average_fidelity  # noqa: E402
from uqc.trpo import TRPOAgent, TRPOConfig  # noqa: E402

# label -> (run directory pattern, filter bandwidth in MHz, training noise)
RUNS = {
    "qs40_nominal": ("qs40_nominal_seed{seed}", 40.0, "none"),
    "qs40_feedback": ("qs40_feedback_seed{seed}", 40.0, "quasi_static"),
    "matched_nominal": ("matched_nominal_seed{seed}", 10.0, "none"),
    "matched_noise": ("matched_noise_nomleak_seed{seed}", 10.0, "white"),
}
SEEDS = (1, 2, 3)
SIGMA_MHZ = 0.3                                   # true offset strength, as in quasi_static_trpo.py
SIGMA_M = (0.0, 0.03, 0.1, 0.3, 1.0, np.inf)      # calibration measurement error, MHz
CLOSED_SAMPLES = 200
CLOSED_SEED = 21                                  # same true offsets as quasi_static_trpo.py
CALIBRATION_SEED = 22


def run_dir(name: str) -> str:
    for base in ("final_results", "runs", os.path.join("final_results", "matched_retraining")):
        path = os.path.join(base, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(name)


def true_offsets() -> np.ndarray:
    return SIGMA_MHZ * np.random.default_rng(CLOSED_SEED).standard_normal((CLOSED_SAMPLES, 6))


def calibration_noise() -> np.ndarray:
    return np.random.default_rng(CALIBRATION_SEED).standard_normal((CLOSED_SAMPLES, 6))


def estimate(eps: np.ndarray, z: np.ndarray, sigma_m: float) -> np.ndarray:
    """Posterior mean of the offsets after a Gaussian measurement with error sigma_m."""
    if np.isinf(sigma_m):
        return np.zeros_like(eps)
    return (eps + sigma_m * z) * SIGMA_MHZ ** 2 / (SIGMA_MHZ ** 2 + sigma_m ** 2)


def residual_mhz(sigma_m: float) -> float:
    """Standard deviation of eps - eps_hat for each channel."""
    if np.isinf(sigma_m):
        return SIGMA_MHZ
    return SIGMA_MHZ * sigma_m / np.sqrt(SIGMA_MHZ ** 2 + sigma_m ** 2)


def rollout(env, agent, steps, offsets):
    """Deterministic closed loop of the policy on env with the given quasi-static offsets: its actions."""
    obs = env.reset()
    env.quasi_static_offsets = offsets.copy()
    actions = []
    for _ in range(steps):
        action = agent.get_action(obs, deterministic=True)[0]
        actions.append(action)
        obs, _, _, _ = env.step(action)
    return actions, env.U.copy()


def replay(env, system, actions, offsets):
    env.reset()
    env.quasi_static_offsets = offsets.copy()
    for action in actions:
        env.step(action)
    return system.projected_unitary(env.U)


def evaluate_checkpoint(task):
    label, seed, iteration = task
    directory = run_dir(RUNS[label][0].format(seed=seed))
    plan = ControlPlan.load(os.path.join(directory, "plans", f"iter_{iteration:06d}_control_plan.npz"))
    system = GmonSystem(GmonSystemConfig(dt_ns=plan.dt_ns, runtime_norm_ns=plan.runtime_norm_ns))
    target = system.target_gate(plan.target_alpha, plan.target_gamma)
    steps = plan.controls_mhz_and_phase.shape[0]

    # the policy acts on the noisy propagator (it was trained that way, or without noise): model and
    # device are the same environment, with different offsets
    env = make_env(system, plan, noisy=True, observe_nominal=False)
    if abs(env.config.filter_bandwidth_mhz - RUNS[label][1]) > 1e-9:
        raise RuntimeError(f"{label} seed {seed}: filter {env.config.filter_bandwidth_mhz} MHz, expected {RUNS[label][1]}")
    agent = TRPOAgent(env.observation_dim, 7, config=TRPOConfig(hidden_sizes=(64, 32, 32)))
    agent.load_state_dict(torch.load(os.path.join(directory, "checkpoints", f"iter_{iteration:06d}.pt"),
                                     map_location="cpu")["agent_state"])

    # zero offsets: must reproduce the stored plan (see quasi_static_trpo.evaluate_policy)
    rollout(env, agent, steps, np.zeros(6))
    replayed, stored = np.asarray(env.nominal_controls_history), plan.controls_mhz_and_phase
    if replayed.shape != stored.shape:
        raise RuntimeError(f"{label} seed {seed} iteration {iteration}: plan length differs")
    diff = np.abs(replayed - stored)
    diff[:, PHASES] = np.minimum(diff[:, PHASES], 2.0 * np.pi - diff[:, PHASES])
    if diff.max() > PLAN_TOLERANCE_MHZ:
        raise RuntimeError(f"{label} seed {seed} iteration {iteration}: checkpoint does not reproduce its plan")

    eps, z = true_offsets(), calibration_noise()
    rows = []
    for sigma_m in SIGMA_M:
        unitaries = []
        for k in range(CLOSED_SAMPLES):
            actions, u_model = rollout(env, agent, steps, estimate(eps[k], z[k], sigma_m))
            u_device = replay(env, system, actions, eps[k])
            if sigma_m == 0.0 and not np.allclose(u_model, env.U, atol=1e-12, rtol=0):
                raise RuntimeError("exact calibration does not reproduce the closed loop")
            unitaries.append(u_device)
        rows.append({
            "run": label, "filter_bandwidth_mhz": RUNS[label][1], "training_noise": RUNS[label][2],
            "seed": seed, "iteration": iteration, "time_ns": steps * plan.dt_ns,
            "sigma_mhz": SIGMA_MHZ, "sigma_m_mhz": float(sigma_m), "residual_mhz": residual_mhz(sigma_m),
            "num_samples": CLOSED_SAMPLES,
            "average_fidelity": float(evaluate_average_fidelity(system, target, unitaries)),
        })
    return rows


def summarize(rows):
    out = []
    for label in RUNS:
        for seed in SEEDS:
            sel = [r for r in rows if r["run"] == label and r["seed"] == seed]
            if not sel:
                continue
            by = {}
            for r in sel:
                by.setdefault(r["iteration"], {})[r["sigma_m_mhz"]] = r["average_fidelity"]
            exact = np.array([v[0.0] for v in by.values()])
            blind = np.array([v[np.inf] for v in by.values()])
            for sigma_m in SIGMA_M:
                f = np.array([v[float(sigma_m)] for v in by.values()])
                out.append({
                    "run": label, "filter_bandwidth_mhz": RUNS[label][1], "training_noise": RUNS[label][2],
                    "seed": seed, "sigma_m_mhz": float(sigma_m), "residual_mhz": residual_mhz(sigma_m),
                    "n_iterations": len(f),
                    "average_fidelity_median": float(np.median(f)),
                    "gain_over_open_loop_median": float(np.median(f - blind)),
                    "fraction_of_closed_loop_gain_median": float(np.median((f - blind) / (exact - blind))),
                    "fraction_iterations_beating_open_loop": float(np.mean(f > blind)),
                })
    return out


def parse_range(text: str) -> range:
    start, stop, step = (int(x) for x in text.split(":"))
    return range(start, stop, step)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--iterations", type=str, default="25:101:5")
    ap.add_argument("--runs", type=str, default=",".join(RUNS))
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--out", type=str, default="runs/calibrated_feedback_results")
    args = ap.parse_args()
    labels = [s for s in args.runs.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    tasks = [(label, seed, it) for label in labels for seed in seeds for it in parse_range(args.iterations)]

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for done, result in enumerate(ex.map(evaluate_checkpoint, tasks), start=1):
            rows += result
            if done % 12 == 0:
                print(f"{done}/{len(tasks)} checkpoints evaluated", flush=True)
    summary = summarize(rows)
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "calibrated_feedback.csv"), rows)
    write_csv(os.path.join(args.out, "summary.csv"), summary)

    print(f"\naverage fidelity at {SIGMA_MHZ} MHz quasi-static noise, median over iterations (per seed)")
    for label in labels:
        for sigma_m in SIGMA_M:
            sel = [r for r in summary if r["run"] == label and r["sigma_m_mhz"] == float(sigma_m)]
            fs = " ".join(f"{r['average_fidelity_median']:.4f}" for r in sel)
            fr = " ".join(f"{r['fraction_of_closed_loop_gain_median']:.2f}" for r in sel)
            print(f"  {label:16s} sigma_m={sigma_m:5.2f} (residual {residual_mhz(sigma_m):.3f})  F={fs}  fraction={fr}")
    print("written to", args.out)


if __name__ == "__main__":
    main()
