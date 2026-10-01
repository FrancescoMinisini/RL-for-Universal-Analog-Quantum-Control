"""Robustness of every per-iteration control plan of the two matched TRPO runs.

The nominal agent (final_results/nominal) and the noise-trained agent
(final_results/trpo_noise_alpha_2.2) share seed, architecture, UFO weights, batch
size (10 000 episodes) and stopping cost (0.4) for their first 100 iterations; the
only difference is the 1 MHz control noise in the training environment. Comparing
the deterministic plans they produce at every iteration therefore isolates the
effect of stochastic training far better than comparing one selected plan each.

For every plan and every noise strength on SIGMAS this script runs the same Monte
Carlo evaluation as benchmark_robustness.py (uqc.eval.robustness_metrics), with the
seed tied to the noise-strength index so that all plans see the same random
stream. Output (deterministic, independent of the number of workers):

  final_results/trajectory_robustness_results/trajectory_robustness.csv
      one row per (agent, iteration, sigma)
  final_results/trajectory_robustness_results/trajectory_robustness_summary.csv
      per sigma: medians and quartiles over iterations 21-100, the fraction of
      iterations at which the noise-trained plan beats the nominal one, and the
      mean paired difference (noise - nominal, same iteration) of the average
      fidelity with a 95% block-bootstrap interval (8 blocks of 10 iterations,
      fixed seed, to respect the autocorrelation along training)

Run from anywhere:  python analysis/robustness_training_trajectories.py [--workers N]
                    python analysis/robustness_training_trajectories.py --summary-only
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

from uqc.eval import ControlPlan, robustness_metrics, simulate_nominal_plan  # noqa: E402
from uqc.physics import GmonSystem, GmonSystemConfig  # noqa: E402

AGENTS = {
    "nominal": "final_results/nominal/plans",
    "noise": "final_results/trpo_noise_alpha_2.2/plans",
}
MATCHED_ITERATIONS = range(1, 101)          # identical settings apart from training noise
SUMMARY_ITERATIONS = range(21, 101)         # skip the initial transient
SIGMAS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5]
SAMPLES = 200
SEED = 1
TARGET_ALPHA = 2.2
OUT_DIR = "final_results/trajectory_robustness_results"
BOOT_BLOCK = 10
BOOT_SAMPLES = 4000
BOOT_SEED = 0


def evaluate(task):
    agent, iteration, path = task
    plan = ControlPlan.load(path)
    if abs(plan.target_alpha - TARGET_ALPHA) > 1e-9 or abs(plan.target_gamma - np.pi / 2) > 1e-9:
        raise ValueError(f"{path} targets alpha={plan.target_alpha}, gamma={plan.target_gamma}, not N(2.2, 2.2, pi/2)")
    system = GmonSystem(GmonSystemConfig(dt_ns=plan.dt_ns, runtime_norm_ns=plan.runtime_norm_ns))
    nominal = simulate_nominal_plan(system, plan)
    rows = []
    for idx, sigma in enumerate(SIGMAS):
        m = robustness_metrics(system, plan, sigma_mhz=sigma, num_samples=SAMPLES, seed=SEED + idx)
        rows.append({
            "agent": agent,
            "iteration": iteration,
            "sigma_mhz": sigma,
            "num_samples": SAMPLES,
            "average_fidelity": m["average_fidelity"],
            "average_gate_fidelity": m["average_gate_fidelity"],
            "fidelity_variance": m["fidelity_variance"],
            "nominal_fidelity": nominal["fidelity"],
            "nominal_cost": nominal["cost"],
            "nominal_leakage": nominal["leakage"],
            "time_ns": nominal["time_ns"],
        })
    return rows


def summarize(rows):
    out = []
    for sigma in SIGMAS:
        rec = {"sigma_mhz": sigma}
        per_agent = {}
        for agent in AGENTS:
            sel = {r["iteration"]: r for r in rows
                   if r["agent"] == agent and r["sigma_mhz"] == sigma and r["iteration"] in SUMMARY_ITERATIONS}
            per_agent[agent] = sel
            f = np.array([sel[i]["average_fidelity"] for i in SUMMARY_ITERATIONS])
            v = np.array([sel[i]["fidelity_variance"] for i in SUMMARY_ITERATIONS])
            q1, med, q3 = np.percentile(f, [25, 50, 75])
            rec[f"{agent}_fidelity_q1"] = q1
            rec[f"{agent}_fidelity_median"] = med
            rec[f"{agent}_fidelity_q3"] = q3
            rec[f"{agent}_fidelity_mean"] = f.mean()
            vq1, vmed, vq3 = np.percentile(v, [25, 50, 75])
            rec[f"{agent}_variance_q1"] = vq1
            rec[f"{agent}_variance_median"] = vmed
            rec[f"{agent}_variance_q3"] = vq3
        fn = np.array([per_agent["noise"][i]["average_fidelity"] for i in SUMMARY_ITERATIONS])
        fm = np.array([per_agent["nominal"][i]["average_fidelity"] for i in SUMMARY_ITERATIONS])
        vn = np.array([per_agent["noise"][i]["fidelity_variance"] for i in SUMMARY_ITERATIONS])
        vm = np.array([per_agent["nominal"][i]["fidelity_variance"] for i in SUMMARY_ITERATIONS])
        rec["frac_noise_higher_fidelity"] = float(np.mean(fn > fm))
        rec["frac_noise_lower_variance"] = float(np.mean(vn < vm))
        rec["median_infidelity_ratio_nominal_over_noise"] = float(np.median(1 - fm) / np.median(1 - fn))
        diff = fn - fm
        blocks = diff.reshape(-1, BOOT_BLOCK)
        rng = np.random.default_rng(BOOT_SEED)
        boot = [blocks[rng.integers(0, len(blocks), len(blocks))].mean() for _ in range(BOOT_SAMPLES)]
        rec["mean_fidelity_difference_noise_minus_nominal"] = float(diff.mean())
        rec["difference_ci95_low"] = float(np.percentile(boot, 2.5))
        rec["difference_ci95_high"] = float(np.percentile(boot, 97.5))
        rec["n_iterations"] = len(SUMMARY_ITERATIONS)
        out.append(rec)
    return out


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(float(v)) if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--summary-only", action="store_true",
                    help="recompute the summary from an existing trajectory_robustness.csv")
    args = ap.parse_args()

    if args.summary_only:
        with open(os.path.join(OUT_DIR, "trajectory_robustness.csv"), newline="", encoding="utf-8") as f:
            rows = []
            for r in csv.DictReader(f):
                r["iteration"] = int(r["iteration"])
                for k in r:
                    if k not in ("agent", "iteration"):
                        r[k] = int(r[k]) if k == "num_samples" else float(r[k])
                rows.append(r)
        write_csv(os.path.join(OUT_DIR, "trajectory_robustness_summary.csv"), summarize(rows))
        print("summary rewritten in", OUT_DIR)
        return

    tasks = []
    for agent, plan_dir in AGENTS.items():
        for it in MATCHED_ITERATIONS:
            tasks.append((agent, it, os.path.join(plan_dir, f"iter_{it:06d}_control_plan.npz")))

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for done, part in enumerate(ex.map(evaluate, tasks), start=1):
            rows.extend(part)
            if done % 20 == 0:
                print(f"{done}/{len(tasks)} plans evaluated", flush=True)
    rows.sort(key=lambda r: (r["agent"], r["iteration"], r["sigma_mhz"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    write_csv(os.path.join(OUT_DIR, "trajectory_robustness.csv"), rows)
    write_csv(os.path.join(OUT_DIR, "trajectory_robustness_summary.csv"), summarize(rows))
    print("written to", OUT_DIR)


if __name__ == "__main__":
    main()
