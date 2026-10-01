"""Numbers quoted in the preprint's robustness section, from robustness_analysis_3/.

Reads final_results/robustness_analysis_3/combined_ewma_data.csv (written by
export_ewma_data.py) and writes, next to it:
  representative_values.csv  EWMA average fidelity and variance of every controller at the
                             noise strengths of the paper's table
  crossovers.csv             for every pair of controllers, the noise strengths at which their
                             EWMA mean-fidelity (and variance) curves cross, and the share of
                             the grid on which each controller is best
Run from anywhere: python analysis/summarize_robustness_3.py
"""
from __future__ import annotations

import csv
import itertools
import os

import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = "final_results/robustness_analysis_3"
CONTROLLERS = ["adam70", "adam60", "nominal", "noise", "noise91"]
SIGMAS = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]


def fmt(v):
    return repr(float(v)) if isinstance(v, (float, np.floating)) else v


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v) for k, v in r.items()})


def crossings(x, d):
    """Noise strengths where the sign of d changes (linear interpolation between grid points)."""
    out = []
    s = np.sign(d)
    for i in np.nonzero(s[:-1] * s[1:] < 0)[0]:
        out.append(x[i] - d[i] * (x[i + 1] - x[i]) / (d[i + 1] - d[i]))
    return out


def main() -> None:
    df = pd.read_csv(os.path.join(BASE, "combined_ewma_data.csv"))
    x = df["sigma_mhz"].to_numpy()

    rows = []
    for sigma in SIGMAS:
        i = int(np.argmin(np.abs(x - sigma)))
        rec = {"sigma_mhz": float(x[i])}
        for c in CONTROLLERS:
            rec[f"{c}_fidelity_ewma"] = df[f"{c}_fidelity_ewma"].iloc[i]
            rec[f"{c}_variance_ewma"] = df[f"{c}_variance_ewma"].iloc[i]
        rows.append(rec)
    last, first = df.iloc[-1], df.iloc[0]
    rec = {"sigma_mhz": "loss_0.1_to_3.5"}
    for c in CONTROLLERS:
        rec[f"{c}_fidelity_ewma"] = first[f"{c}_fidelity_ewma"] - last[f"{c}_fidelity_ewma"]
        rec[f"{c}_variance_ewma"] = ""
    rows.append(rec)
    write(os.path.join(BASE, "representative_values.csv"), rows)

    cross = []
    for a, b in itertools.combinations(CONTROLLERS, 2):
        dm = (df[f"{a}_fidelity_ewma"] - df[f"{b}_fidelity_ewma"]).to_numpy()
        dv = (df[f"{a}_variance_ewma"] - df[f"{b}_variance_ewma"]).to_numpy()
        cm, cv = crossings(x, dm), crossings(x, dv)
        cross.append({
            "pair": f"{a}_vs_{b}",
            "fidelity_crossings_mhz": " ".join(f"{v:.3f}" for v in cm),
            "variance_crossings_mhz": " ".join(f"{v:.3f}" for v in cv),
            "share_a_higher_fidelity": float(np.mean(dm > 0)),
            "share_a_lower_variance": float(np.mean(dv < 0)),
        })
    main4 = ["adam70", "adam60", "nominal", "noise"]
    fid = np.vstack([df[f"{c}_fidelity_ewma"] for c in main4])
    var = np.vstack([df[f"{c}_variance_ewma"] for c in main4])
    best_f, best_v = np.argmax(fid, axis=0), np.argmin(var, axis=0)
    for k, c in enumerate(main4):
        cross.append({
            "pair": f"best_of_four:{c}",
            "fidelity_crossings_mhz": "", "variance_crossings_mhz": "",
            "share_a_higher_fidelity": float(np.mean(best_f == k)),
            "share_a_lower_variance": float(np.mean(best_v == k)),
        })
    write(os.path.join(BASE, "crossovers.csv"), cross)
    print("written representative_values.csv and crossovers.csv in", BASE)


if __name__ == "__main__":
    main()
