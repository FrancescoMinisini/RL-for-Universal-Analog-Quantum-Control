"""Room for pulse-shape robustness against the bandwidth of the control filter.

The paper finds that Adam trained under quasi-static noise lowers the quasi-static sensitivity of
its pulses by 18-24% at the 10 MHz filter of the UFO model, and attributes the small size of the
gain to the filter, which leaves a 70-130 ns pulse little room for refocusing structure. This script
tests that reading: it evaluates the fixed-horizon Adam experiment of train_adam_noise_models.py
(leakage bound from the noise-free controls) repeated at wider filters, with exactly the evaluation
of noise_with_memory.py (same offsets, same sample counts, same first-order kernel).

Runs read, for each bandwidth, from final_results/<name> if it exists and from runs/<name>
otherwise:
  10 MHz  adam_noise_models_nominal_leakage          (the run of the paper)
  20 MHz  adam_noise_models_bw20_nominal_leakage
  40 MHz  adam_noise_models_bw40_nominal_leakage
  40 MHz  adam_noise_models_bw40_nominal_leakage_converged   convergence check: no-noise and
          quasi-static models only, seeds 1-3, 4 noise samples per step and 800 steps instead of
          1 and 400 (source prefix bw40c)

Outputs (--out, default runs/filter_bandwidth_results/):
  plan_sensitivity.csv   one row per plan, with its filter bandwidth
  adam_noise_models.csv  per bandwidth, training-noise model and horizon: statistics over the seeds
                         (the columns of noise_memory_results/adam_noise_models_nominal_leakage.csv)

Run from anywhere: python analysis/filter_bandwidth.py [--workers N]
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

from noise_with_memory import evaluate, summarize_adam, write_csv  # noqa: E402

# source prefix -> (filter bandwidth in MHz, run directory name)
RUNS = {
    "bw10": (10.0, "adam_noise_models_nominal_leakage"),
    "bw20": (20.0, "adam_noise_models_bw20_nominal_leakage"),
    "bw40": (40.0, "adam_noise_models_bw40_nominal_leakage"),
    "bw40c": (40.0, "adam_noise_models_bw40_nominal_leakage_converged"),
}


def run_dir(name: str) -> str:
    final = os.path.join("final_results", name)
    return final if os.path.isdir(final) else os.path.join("runs", name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=str, default="runs/filter_bandwidth_results")
    args = ap.parse_args()

    tasks, bandwidth_of, prefixes = [], {}, []
    for prefix, (bandwidth, name) in RUNS.items():
        directory = run_dir(name)
        summary = os.path.join(directory, "summary.csv")
        if not os.path.exists(summary):
            print(f"skipping {prefix}: {summary} not found", flush=True)
            continue
        prefixes.append(prefix)
        with open(summary, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                source = f"{prefix}_{r['model']}"
                bandwidth_of[source] = bandwidth
                tasks.append((source, int(r["seed"]), os.path.join(directory, r["plan"]), True))

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = []
        for done, row in enumerate(ex.map(evaluate, tasks), start=1):
            rows.append({"filter_bandwidth_mhz": bandwidth_of[row["source"]], **row})
            if done % 25 == 0:
                print(f"{done}/{len(tasks)} plans evaluated", flush=True)

    summary_rows = []
    for prefix in prefixes:
        for rec in summarize_adam(rows, prefix):
            summary_rows.append({"run": prefix, "filter_bandwidth_mhz": RUNS[prefix][0], **rec})

    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "plan_sensitivity.csv"), rows)
    write_csv(os.path.join(args.out, "adam_noise_models.csv"), summary_rows)
    print("written to", args.out)


if __name__ == "__main__":
    main()
