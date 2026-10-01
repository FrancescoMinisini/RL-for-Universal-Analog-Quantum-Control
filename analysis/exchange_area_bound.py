"""How fast the gmon model can deliver the exchange area of N(alpha, alpha, pi/2).

Every member of the family is an exchange rotation by eps = alpha - pi/2 (analyze_gate_structure.py).
In the qubit projection the coupling alone generates it once |int g dt| = 2 |eps|. Each control is
bounded by 20 MHz and reaches the Hamiltonian through the two-pole filter of Appendix C, and the
boundary term of the UFO cost asks the coupling to vanish at the end of the pulse. The largest
exchange area that a pulse of N steps can deliver is therefore a linear program in the raw controls,
    maximize  s dt sum_k g~_k   subject to  |u_k| <= u_max  and  g~_{N-1} = 0,   g~ = filter(u),
solved here for every N. Dropping the end condition (the boundary term is a soft penalty) gives a
second, smaller time; the two bracket what a pulse that uses the coupling alone can do. This is an
estimate for that route, not a speed limit of the model: near the local ends of the family the
optimizer also uses the detunings.

The same capacity gives the time of a single-qubit rotation under the model's own constraints: a
rotation by theta needs a drive area theta / 2, a Z rotation by theta a detuning area theta.

Outputs (final_results/gate_structure_results/):
  exchange_area_capacity.csv     per duration: largest deliverable area, with and without the end condition
  exchange_area_band.csv         on a fine grid of alpha: the two times, interpolated between durations
  runtime_vs_exchange_area.csv   per target of the curriculum and of the Adam sweep: required area,
                                 the two times, the measured runtime and the areas its pulse delivers
  reference_rotation_times.csv   single-qubit rotations under the same bound and filter
Run from anywhere: python analysis/exchange_area_bound.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from uqc.eval import ControlPlan  # noqa: E402

OUT_DIR = "final_results/gate_structure_results"
DT_NS = 2.0
BANDWIDTH_MHZ = 10.0
U_MAX_MHZ = 20.0
MAX_TIME_NS = 180.0
SCALE = 2 * np.pi * 1e-3            # MHz -> rad/ns
TRPO_CONVERGED_COST = 0.4


def filter_response(n: int) -> np.ndarray:
    """L[k, j]: filtered sample k per unit raw sample j, from rest."""
    a = np.exp(-np.pi * BANDWIDTH_MHZ * 1e-3 * DT_NS)
    k = np.arange(n)
    impulse = (1.0 - a) ** 2 * (k + 1) * a ** k
    L = np.zeros((n, n))
    for j in range(n):
        L[j:, j] = impulse[: n - j]
    return L


def max_area(n: int, end_at_zero: bool) -> float:
    L = filter_response(n)
    weights = SCALE * DT_NS * L.sum(axis=0)
    if not end_at_zero:
        return float(U_MAX_MHZ * weights.sum())      # the impulse response is positive: u = u_max throughout
    res = linprog(-weights, A_eq=L[-1][None, :], b_eq=[0.0], bounds=[(-U_MAX_MHZ, U_MAX_MHZ)] * n, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return float(-res.fun)


def time_for_area(area: float, times: np.ndarray, capacity: np.ndarray, on_grid: bool) -> float:
    if on_grid:
        return float(times[np.argmax(capacity >= area - 1e-12)])
    return float(np.interp(area, np.concatenate([[0.0], capacity]), np.concatenate([[0.0], times])))


def local_runtime_plan(windows_path: str) -> str:
    p = windows_path.replace("\\", "/")
    return "final_results/runtime/" + p[p.index("gamma_1.570796/"):]


def delivered_areas(path: str) -> tuple[float, float, float]:
    c = ControlPlan.load(path).controls_mhz_and_phase
    return tuple(float(SCALE * DT_NS * c[:, i].sum()) for i in (6, 0, 1))


def write_csv(path: str, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main() -> None:
    steps = np.arange(1, int(round(MAX_TIME_NS / DT_NS)) + 1)
    times = steps * DT_NS
    free = np.array([max_area(n, False) for n in steps])
    zero = np.array([max_area(n, True) for n in steps])
    assert np.all(np.diff(free) > 0) and np.all(np.diff(zero) >= 0) and np.all(zero <= free + 1e-12)
    os.makedirs(OUT_DIR, exist_ok=True)
    write_csv(os.path.join(OUT_DIR, "exchange_area_capacity.csv"),
              [{"time_ns": float(t), "area_free_end_rad": float(a), "area_zero_end_rad": float(b)}
               for t, a, b in zip(times, free, zero)])

    band = []
    for alpha in np.linspace(0.0, np.pi, 315):
        area = 2 * abs(alpha - np.pi / 2)
        band.append({"alpha": float(alpha), "abs_exchange_angle": float(area / 2),
                     "t_free_end_ns": time_for_area(area, times, free, False),
                     "t_zero_end_ns": time_for_area(area, times, zero, False)})
    write_csv(os.path.join(OUT_DIR, "exchange_area_band.csv"), band)

    trpo = pd.read_csv("final_results/runtime_results/phase_final_results_log.csv")
    adam = pd.read_csv("final_results/adam_results/adam_family_sweep.csv")
    targets = []
    for _, r in trpo.iterrows():
        targets.append(("trpo_curriculum", float(r["alpha"]), float(r["eval_time_ns"]), float(r["eval_fidelity"]),
                        bool(r["eval_cost"] <= TRPO_CONVERGED_COST), local_runtime_plan(r["eval_plan_path"])))
    for _, r in adam.iterrows():
        plan = f"final_results/adam_runtime_sweep/gamma_1.570796/alpha_{float(r['alpha']):.6f}/best_control_plan.npz"
        targets.append(("adam", float(r["alpha"]), float(r["runtime_ns"]), float(r["best_fidelity"]), True, plan))
    rows = []
    for method, alpha, runtime, fidelity, converged, plan in targets:
        area = 2 * abs(alpha - np.pi / 2)
        t_free = time_for_area(area, times, free, True)
        t_zero = time_for_area(area, times, zero, True)
        g_area, d1_area, d2_area = delivered_areas(plan)
        rows.append({
            "method": method, "alpha": alpha, "abs_exchange_angle": area / 2, "required_area_rad": area,
            "t_free_end_ns": t_free, "t_zero_end_ns": t_zero, "runtime_ns": runtime, "converged": converged,
            "nominal_fidelity": fidelity,
            "position": "below" if runtime < t_free else ("inside" if runtime <= t_zero else "above"),
            "runtime_over_t_zero_end": runtime / t_zero,
            "delivered_exchange_area_rad": abs(g_area),
            "delivered_over_required": abs(g_area) / area,
            "detuning_1_area_rad": d1_area, "detuning_2_area_rad": d2_area,
        })
    write_csv(os.path.join(OUT_DIR, "runtime_vs_exchange_area.csv"), rows)

    rotations = []
    for gate, area in (("pi rotation by the drive", np.pi / 2), ("pi/2 rotation by the drive", np.pi / 4),
                       ("Z rotation by pi by the detuning", np.pi)):
        rotations.append({"gate": gate, "required_area_rad": float(area),
                          "t_free_end_ns": time_for_area(area, times, free, True),
                          "t_zero_end_ns": time_for_area(area, times, zero, True)})
    write_csv(os.path.join(OUT_DIR, "reference_rotation_times.csv"), rotations)

    for method in ("adam", "trpo_curriculum"):
        sel = [r for r in rows if r["method"] == method and r["converged"]]
        counts = {p: sum(r["position"] == p for r in sel) for p in ("below", "inside", "above")}
        ratio = np.array([r["runtime_over_t_zero_end"] for r in sel if r["abs_exchange_angle"] > 0.05])
        print(f"{method:16s} {counts}  runtime / zero-end time: {ratio.min():.2f} - {ratio.max():.2f} "
              f"(median {np.median(ratio):.2f})")
    for r in rotations:
        print(f"{r['gate']:34s} {r['t_free_end_ns']:.0f} ns (free end), {r['t_zero_end_ns']:.0f} ns (zero end)")
    print("written to", OUT_DIR)


if __name__ == "__main__":
    main()
