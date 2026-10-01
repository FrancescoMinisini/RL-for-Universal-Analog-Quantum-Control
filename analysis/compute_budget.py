"""Simulator steps spent by the single-target controllers.

One simulator step is one propagation of the 9-dimensional propagator over dt with the update of
the leakage bound. A TRPO iteration spends (episodes per batch) x (mean episode length / dt) of
them; the mean episode length of every iteration is in training_log.jsonl, the batch size is the
one documented in final_results/README.md (it is not logged per iteration). An Adam iteration at
horizon T spends T / dt forward steps and as many backward ones; only the forward steps are
counted here.

Output: final_results/compute_budget_results/compute_budget.csv
Run from anywhere: python analysis/compute_budget.py
"""
from __future__ import annotations

import csv
import json
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = "final_results/compute_budget_results"
DT_NS = 2.0
# run -> (iteration of the reported plan, episodes per batch as a function of the iteration)
TRPO_RUNS = {
    "trpo_nominal": ("final_results/nominal", 111, lambda it: 10000 if it <= 100 else 5000),
    "trpo_noise": ("final_results/trpo_noise_alpha_2.2", 11, lambda it: 10000),
}
ADAM_RUN = "final_results/adam_noise"


def main() -> None:
    rows = []
    for name, (run, reported, batch) in TRPO_RUNS.items():
        with open(os.path.join(run, "training_log.jsonl"), encoding="utf-8") as f:
            log = [json.loads(line) for line in f if line.strip()]
        steps = {r["iteration"]: batch(r["iteration"]) * r["avg_time_ns"] / DT_NS for r in log}
        rows.append({
            "controller": name, "iterations": len(log), "reported_iteration": reported,
            "episodes": sum(batch(r["iteration"]) for r in log),
            "simulator_steps_total": sum(steps.values()),
            "simulator_steps_to_reported_plan": sum(v for it, v in steps.items() if it <= reported),
            "wall_clock_seconds": "",
        })
    with open(os.path.join(ADAM_RUN, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    total = summary["adam_iters"] * sum(h / DT_NS for h in summary["horizons_ns"])
    rows.append({
        "controller": "adam_horizon_search", "iterations": summary["adam_iters"] * len(summary["horizons_ns"]),
        "reported_iteration": "", "episodes": summary["adam_iters"] * len(summary["horizons_ns"]),
        "simulator_steps_total": total, "simulator_steps_to_reported_plan": total,
        "wall_clock_seconds": summary["total_elapsed_seconds"],
    })
    adam = rows[-1]["simulator_steps_total"]
    for r in rows:
        r["total_over_adam"] = r["simulator_steps_total"] / adam
        r["to_reported_plan_over_adam"] = r["simulator_steps_to_reported_plan"] / adam
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "compute_budget.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
            print(r)


if __name__ == "__main__":
    main()
