"""Baseline for calibrated_feedback.py: subtract the calibrated offsets from the noise-free pulse.

Five of the six quasi-static offsets (delta_1, delta_2, f_1, f_2, g) add to control channels, so once they
are calibrated the obvious correction needs no policy: subtract the estimate from the commanded pulse.
The anharmonicity offset eta is not a control channel and stays uncorrected. Two versions:

  subtract_ideal     the estimate is removed from the filtered controls at every step (an upper bound:
                     it ignores the filter)
  subtract_filtered  the estimate is removed from the commanded controls, so the correction reaches the
                     Hamiltonian through the two-pole filter, which starts from rest and needs a few
                     1/(pi bandwidth) to pass it on
  subtract_preemphasis  the commanded controls (recovered by inverting the filter on the stored pulse)
                     are chosen step by step so that the filtered controls equal the stored ones minus
                     the estimate, clipped to the command bound of the environment (+-max_control_mhz):
                     the correction arrives as fast as the bound allows (6 ns at 10 MHz for 0.3 MHz)

The pulse is the stored plan of each checkpoint that calibrated_feedback.py evaluates (the policy's
noise-free pulse), with the same true offsets, the same calibration draws and the same posterior-mean
estimate. The replay is uqc.noise_response.replay_with_noise, exact for additive offsets; at
sigma_m = inf it is the open loop, which the script compares with calibrated_feedback.csv when present.

Outputs (--out, default runs/calibrated_feedback_results/):
  calibrated_feedforward.csv  one row per (run, seed, iteration, sigma_m, method)
  comparison.csv              per run, seed, sigma_m and method (policy included, from calibrated_feedback.csv):
                              median average fidelity over the iterations

Run from anywhere: python analysis/calibrated_feedforward.py [--workers N] [--iterations 25:101:5]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

from calibrated_feedback import (  # noqa: E402
    CLOSED_SAMPLES, RUNS, SIGMA_M, SIGMA_MHZ, calibration_noise, estimate, parse_range, residual_mhz,
    run_dir, true_offsets)
from noise_with_memory import write_csv  # noqa: E402
from uqc.env import EnvConfig, ExactExponentialFilter  # noqa: E402
from uqc.eval import ControlPlan  # noqa: E402
from uqc.noise_response import fidelity_statistics, replay_with_noise  # noqa: E402
from uqc.physics import GmonSystem, GmonSystemConfig  # noqa: E402

METHODS = ("subtract_ideal", "subtract_filtered", "subtract_preemphasis")
# offsets are stored as (eta, delta_1, delta_2, f_1, f_2, g); replay_with_noise wants
# (delta_1, delta_2, f_1, f_2, g, eta)
TO_REPLAY = [1, 2, 3, 4, 5, 0]
LINEAR_CONTROLS = [0, 1, 2, 4, 6]         # control indices of delta_1, delta_2, f_1, f_2, g
MAX_CONTROL_MHZ = EnvConfig.__dataclass_fields__["max_control_mhz"].default


def step_response(steps: int, dt_ns: float, bandwidth_mhz: float) -> np.ndarray:
    filt = ExactExponentialFilter(1, dt_ns=dt_ns, bandwidth_mhz=bandwidth_mhz)
    return np.array([filt.filter([1.0])[0] for _ in range(steps)])


def preemphasis_correction(plan: ControlPlan, eps_hat: np.ndarray) -> np.ndarray:
    """Correction [samples, 5, steps] reaching the Hamiltonian when the commands aim at y_nominal - eps_hat."""
    filt = ExactExponentialFilter(1, dt_ns=plan.dt_ns, bandwidth_mhz=plan.filter_bandwidth_mhz)
    a1, b1, b2 = filt.a1, filt.b1, filt.b2
    y_nom = plan.controls_mhz_and_phase[:, LINEAR_CONTROLS]            # [steps, 5]
    steps = y_nom.shape[0]
    # commands of the stored pulse: the filter inverted (it starts from rest)
    padded = np.vstack([np.zeros((2, 5)), y_nom])
    u_nom = (padded[2:] + b1 * padded[1:-1] + b2 * padded[:-2]) / a1
    if np.abs(u_nom).max() > MAX_CONTROL_MHZ * (1 + 1e-6):
        raise RuntimeError(f"recovered commands exceed the bound: {np.abs(u_nom).max():.3f} MHz")
    samples = eps_hat.shape[0]
    y1 = np.zeros((samples, 5))
    y2 = np.zeros((samples, 5))
    out = np.empty((samples, 5, steps))
    for k in range(steps):
        u = np.clip((y_nom[k] - eps_hat[:, :5] + b1 * y1 + b2 * y2) / a1, -MAX_CONTROL_MHZ, MAX_CONTROL_MHZ)
        y = a1 * u - b1 * y1 - b2 * y2
        out[:, :, k] = y - y_nom[k]
        y2, y1 = y1, y
    return out


def evaluate_plan(task):
    label, seed, iteration = task
    directory = run_dir(RUNS[label][0].format(seed=seed))
    plan = ControlPlan.load(os.path.join(directory, "plans", f"iter_{iteration:06d}_control_plan.npz"))
    system = GmonSystem(GmonSystemConfig(dt_ns=plan.dt_ns, runtime_norm_ns=plan.runtime_norm_ns))
    steps = plan.controls_mhz_and_phase.shape[0]
    response = {"subtract_ideal": np.ones(steps),
                "subtract_filtered": step_response(steps, plan.dt_ns, plan.filter_bandwidth_mhz)}
    eps = true_offsets()[:, TO_REPLAY]
    z = calibration_noise()[:, TO_REPLAY]
    rows = []
    for sigma_m in SIGMA_M:
        eps_hat = estimate(eps, z, sigma_m)
        eps_hat[:, 5] = 0.0                        # eta cannot be corrected through a control
        for method in METHODS:
            noise = np.repeat(eps[:, :, None], steps, axis=2)
            if method == "subtract_preemphasis":
                noise[:, :5, :] += preemphasis_correction(plan, eps_hat)
            else:
                noise -= eps_hat[:, :, None] * response[method][None, None, :]
            fbar, _, _ = fidelity_statistics(system, plan, replay_with_noise(system, plan, noise))
            rows.append({
                "run": label, "filter_bandwidth_mhz": RUNS[label][1], "training_noise": RUNS[label][2],
                "seed": seed, "iteration": iteration, "time_ns": steps * plan.dt_ns, "method": method,
                "sigma_mhz": SIGMA_MHZ, "sigma_m_mhz": float(sigma_m), "residual_mhz": residual_mhz(sigma_m),
                "num_samples": CLOSED_SAMPLES, "average_fidelity": float(fbar),
            })
    return rows


def read_policy(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{**r, "method": "policy", "seed": int(r["seed"]), "iteration": int(r["iteration"]),
                 "sigma_m_mhz": float(r["sigma_m_mhz"]), "average_fidelity": float(r["average_fidelity"])}
                for r in csv.DictReader(f)]


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
        for result in ex.map(evaluate_plan, tasks):
            rows += result
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "calibrated_feedforward.csv"), rows)

    policy = [r for r in read_policy(os.path.join(args.out, "calibrated_feedback.csv"))
              if r["run"] in labels and r["seed"] in seeds]
    if policy:
        # at sigma_m = inf both baselines and the policy replay the noise-free pulse
        ref = {(r["run"], r["seed"], r["iteration"]): r["average_fidelity"] for r in policy if np.isinf(r["sigma_m_mhz"])}
        worst = max(abs(r["average_fidelity"] - ref[(r["run"], r["seed"], r["iteration"])])
                    for r in rows if np.isinf(r["sigma_m_mhz"]) and (r["run"], r["seed"], r["iteration"]) in ref)
        print(f"open loop: largest difference from calibrated_feedback.csv {worst:.1e}")
    comparison = []
    for label in labels:
        for seed in seeds:
            for sigma_m in SIGMA_M:
                for method in ("policy",) + METHODS:
                    f = [r["average_fidelity"] for r in (policy if method == "policy" else rows)
                         if r["run"] == label and r["seed"] == seed and r["sigma_m_mhz"] == float(sigma_m)
                         and r["method"] == method]
                    if f:
                        comparison.append({"run": label, "filter_bandwidth_mhz": RUNS[label][1], "seed": seed,
                                           "sigma_m_mhz": float(sigma_m), "residual_mhz": residual_mhz(sigma_m),
                                           "method": method, "n_iterations": len(f),
                                           "average_fidelity_median": float(np.median(f))})
    write_csv(os.path.join(args.out, "comparison.csv"), comparison)

    print(f"\naverage fidelity at {SIGMA_MHZ} MHz quasi-static noise, median over iterations, seeds 1/2/3")
    for label in labels:
        for sigma_m in SIGMA_M:
            for method in ("policy",) + METHODS:
                sel = [r for r in comparison if r["run"] == label and r["sigma_m_mhz"] == float(sigma_m)
                       and r["method"] == method]
                if sel:
                    fs = " ".join(f"{r['average_fidelity_median']:.4f}" for r in sel)
                    print(f"  {label:16s} sigma_m={sigma_m:5.2f}  {method:18s} {fs}")
    print("written to", args.out)


if __name__ == "__main__":
    main()
