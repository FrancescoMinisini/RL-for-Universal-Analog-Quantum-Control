"""Matched nominal / noise-trained TRPO histories for several seeds, with the nominal leakage bound.

robustness_training_trajectories.py compares the per-iteration plans of one nominal and one
noise-trained agent (seed 1). There the noise-trained agent had its leakage bound accumulated
from the noisy Hamiltonian. This script repeats that comparison, with the same evaluation
(same noise strengths, samples and random streams), on the runs of run_matched_retraining.py:
for every seed a nominal agent and an agent trained under 1 MHz noise with the leakage bound
taken from the noise-free controls, both with the batched engine.

Input:  <runs-root>/matched_nominal_seed<N>/plans, <runs-root>/matched_noise_nomleak_seed<N>/plans
Output (final_results/matched_retraining_results/, deterministic, independent of the workers):

  trajectory_robustness.csv        one row per (seed, agent, iteration, sigma)
  per_seed_summary.csv             the summary of robustness_training_trajectories.py for each
                                   seed: iterations 21-100, fraction of iterations at which the
                                   noise-trained plan is the more robust one, mean paired
                                   difference with its 95% block-bootstrap interval
  seed_summary.csv                 per sigma, across seeds: the same-seed differences, the
                                   differences for all pairings of a noise-trained with a nominal
                                   seed, and the spread between seeds within each arm
  plan_statistics.csv              per run: median duration, nominal fidelity and cost of the
                                   plans of iterations 21-100, and the lowest-cost plan

Run from anywhere:  python analysis/matched_retraining_robustness.py [--workers N]
                    python analysis/matched_retraining_robustness.py --summary-only
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robustness_training_trajectories import (  # noqa: E402  (also changes into the repo root)
    MATCHED_ITERATIONS,
    SIGMAS,
    SUMMARY_ITERATIONS,
    evaluate,
    summarize,
    write_csv,
)

RUN_NAMES = {"nominal": "matched_nominal_seed{seed}", "noise": "matched_noise_nomleak_seed{seed}"}
OUT_DIR = "final_results/matched_retraining_results"


def read_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in r:
                if k in ("seed", "iteration", "num_samples"):
                    r[k] = int(r[k])
                elif k != "agent":
                    r[k] = float(r[k])
            rows.append(r)
    return rows


def seed_summary(rows, seeds):
    """Per sigma: paired differences (noise - nominal, same iteration) for every pairing of seeds."""
    fid = {}
    for r in rows:
        if r["iteration"] in SUMMARY_ITERATIONS:
            fid[(r["seed"], r["agent"], r["sigma_mhz"], r["iteration"])] = r["average_fidelity"]

    def history(seed, agent, sigma):
        return np.array([fid[(seed, agent, sigma, i)] for i in SUMMARY_ITERATIONS])

    out = []
    for sigma in SIGMAS:
        rec = {"sigma_mhz": sigma}
        same, cross, frac = [], [], []
        for sn in seeds:
            for sm in seeds:
                d = history(sn, "noise", sigma) - history(sm, "nominal", sigma)
                (same if sn == sm else cross).append(float(d.mean()))
                if sn == sm:
                    frac.append(float(np.mean(d > 0)))
        for seed, d, fr in zip(seeds, same, frac):
            rec[f"difference_seed{seed}"] = d
            rec[f"frac_noise_higher_seed{seed}"] = fr
        rec["difference_same_seed_mean"] = float(np.mean(same))
        rec["difference_same_seed_min"] = float(np.min(same))
        rec["difference_same_seed_max"] = float(np.max(same))
        rec["difference_all_pairings_mean"] = float(np.mean(same + cross))
        rec["difference_all_pairings_min"] = float(np.min(same + cross))
        rec["difference_all_pairings_max"] = float(np.max(same + cross))
        rec["frac_noise_higher_pooled"] = float(np.mean(frac))
        for agent in ("nominal", "noise"):
            means = np.array([history(s, agent, sigma).mean() for s in seeds])
            rec[f"{agent}_fidelity_mean_over_seeds"] = float(means.mean())
            rec[f"{agent}_fidelity_min_over_seeds"] = float(means.min())
            rec[f"{agent}_fidelity_max_over_seeds"] = float(means.max())
        rec["n_seeds"] = len(seeds)
        rec["n_iterations"] = len(SUMMARY_ITERATIONS)
        out.append(rec)
    return out


def plan_statistics(rows, seeds):
    out = []
    ref_sigma = 1.0
    for seed in seeds:
        for agent in ("nominal", "noise"):
            sel = [r for r in rows if r["seed"] == seed and r["agent"] == agent and r["sigma_mhz"] == ref_sigma]
            late = [r for r in sel if r["iteration"] in SUMMARY_ITERATIONS]
            best = min(sel, key=lambda r: r["nominal_cost"])
            out.append({
                "seed": seed,
                "agent": agent,
                "median_time_ns": float(np.median([r["time_ns"] for r in late])),
                "median_nominal_fidelity": float(np.median([r["nominal_fidelity"] for r in late])),
                "median_nominal_cost": float(np.median([r["nominal_cost"] for r in late])),
                "median_average_fidelity_1mhz": float(np.median([r["average_fidelity"] for r in late])),
                "best_iteration": best["iteration"],
                "best_nominal_cost": best["nominal_cost"],
                "best_nominal_fidelity": best["nominal_fidelity"],
                "best_nominal_leakage": best["nominal_leakage"],
                "best_time_ns": best["time_ns"],
                "best_average_fidelity_1mhz": best["average_fidelity"],
            })
    return out


def write_summaries(rows, seeds):
    per_seed = []
    for seed in seeds:
        for rec in summarize([r for r in rows if r["seed"] == seed]):
            per_seed.append({"seed": seed, **rec})
    write_csv(os.path.join(OUT_DIR, "per_seed_summary.csv"), per_seed)
    write_csv(os.path.join(OUT_DIR, "seed_summary.csv"), seed_summary(rows, seeds))
    write_csv(os.path.join(OUT_DIR, "plan_statistics.csv"), plan_statistics(rows, seeds))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-root", type=str, default="final_results/matched_retraining")
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--summary-only", action="store_true",
                    help="recompute the summaries from an existing trajectory_robustness.csv")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.summary_only:
        write_summaries(read_rows(os.path.join(OUT_DIR, "trajectory_robustness.csv")), seeds)
        print("summaries rewritten in", OUT_DIR)
        return

    tasks, task_seeds = [], []
    for seed in seeds:
        for agent, name in RUN_NAMES.items():
            plan_dir = os.path.join(args.runs_root, name.format(seed=seed), "plans")
            for it in MATCHED_ITERATIONS:
                tasks.append((agent, it, os.path.join(plan_dir, f"iter_{it:06d}_control_plan.npz")))
                task_seeds.append(seed)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for done, (seed, part) in enumerate(zip(task_seeds, ex.map(evaluate, tasks)), start=1):
            rows.extend({"seed": seed, **r} for r in part)
            if done % 50 == 0:
                print(f"{done}/{len(tasks)} plans evaluated", flush=True)
    rows.sort(key=lambda r: (r["seed"], r["agent"], r["iteration"], r["sigma_mhz"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    write_csv(os.path.join(OUT_DIR, "trajectory_robustness.csv"), rows)
    write_summaries(rows, seeds)
    print("written to", OUT_DIR)


if __name__ == "__main__":
    main()
