"""Conditional phase from the second excited levels, and the fidelity it costs.

In the two-transmon model the coupling g connects |11> to |02> and |20>, which lie |eta| away. To
second order this shifts |11> by 4 g^2 / |eta|, so an exchange pulse carries a conditional phase
    phi = (4 / |eta|) int g(t)^2 dt .
It is the ZZ term g^2/|eta| of gmon processors and the phi of their fSim gates. The family
N(alpha, alpha, pi/2) contains no such phase and local rotations cannot remove it, so a pulse that
is otherwise perfect is left with the infidelity
    1 - F = sin^2(phi / 4) .
This script evaluates phi and that infidelity for the stored pulses and sets them against the
nominal infidelity from the full simulation. For the Adam pulses optimized without noise the two
agree, which identifies the conditional phase as what limits them. The last column gives the
smallest infidelity the same exchange area allows in the same duration, a constant coupling.

A reference row gives phi for a square pulse at the 20 MHz bound that delivers the iSWAP area.

Output: final_results/gate_structure_results/conditional_phase.csv
Run from anywhere: python analysis/conditional_phase.py
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from uqc.eval import ControlPlan, simulate_nominal_plan  # noqa: E402
from uqc.physics import GmonSystem, GmonSystemConfig  # noqa: E402

PLANS = {
    **{"adam_no_noise_" + os.path.basename(p)[6:9].lstrip("0") + "ns": p
       for p in sorted(glob.glob("final_results/adam_noise_models/plans/none_h*_control_plan.npz"))},
    "adam_70ns": "final_results/adam_noise_70ns/best_control_plan.npz",
    "adam_60ns": "final_results/adam_noise/best_control_plan.npz",
    "trpo_nominal": "final_results/nominal/best_control_plan.npz",
    "trpo_noise": "final_results/trpo_noise_alpha_2.2/plans/iter_000011_control_plan.npz",
}
OUT = "final_results/gate_structure_results/conditional_phase.csv"


def main() -> None:
    rows = []
    for name, path in PLANS.items():
        plan = ControlPlan.load(path)
        system = GmonSystem(GmonSystemConfig(dt_ns=plan.dt_ns, runtime_norm_ns=plan.runtime_norm_ns))
        eta = abs(system.config.eta_base_mhz) * system.mhz_to_rad_per_ns
        g = plan.controls_mhz_and_phase[:, 6] * system.mhz_to_rad_per_ns
        time_ns = g.size * plan.dt_ns
        area = abs(g.sum() * plan.dt_ns)
        phase = 4.0 * np.sum(g ** 2) * plan.dt_ns / eta
        flat_phase = 4.0 * area ** 2 / (eta * time_ns)
        infidelity = 1.0 - simulate_nominal_plan(system, plan)["fidelity"]
        rows.append({
            "plan": name, "time_ns": time_ns, "nominal_infidelity": infidelity,
            "exchange_area_rad": area, "conditional_phase_rad": phase,
            "conditional_phase_infidelity": np.sin(phase / 4.0) ** 2,
            "nominal_over_conditional": infidelity / np.sin(phase / 4.0) ** 2,
            "constant_coupling_infidelity": np.sin(flat_phase / 4.0) ** 2,
        })
    system = GmonSystem(GmonSystemConfig())
    g_max = system.config.max_control_mhz * system.mhz_to_rad_per_ns
    eta = abs(system.config.eta_base_mhz) * system.mhz_to_rad_per_ns
    time_ns = (np.pi / 2) / g_max
    phase = 4.0 * g_max ** 2 * time_ns / eta
    rows.append({
        "plan": "square_pulse_at_20mhz_iswap_area", "time_ns": time_ns, "nominal_infidelity": "",
        "exchange_area_rad": np.pi / 2, "conditional_phase_rad": phase,
        "conditional_phase_infidelity": np.sin(phase / 4.0) ** 2, "nominal_over_conditional": "",
        "constant_coupling_infidelity": np.sin(phase / 4.0) ** 2,
    })
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
            print({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
