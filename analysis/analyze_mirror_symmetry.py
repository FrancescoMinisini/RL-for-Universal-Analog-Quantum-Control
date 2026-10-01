"""Mirror symmetry of the family N(alpha, alpha, pi/2) about alpha = pi/2.

Conjugation by the parity of the first transmon, P1 = (-1)^{n_1}, maps a1 -> -a1.
It leaves the anharmonicity, the detunings, the leakage structure and the boundary
penalty unchanged, and maps the Hamiltonian with controls (g, phi_1) to the one
with (-g, phi_1 + pi). On the qubits it maps N(pi/2 + eps) to N(pi/2 - eps). The
control problems at alpha and pi - alpha are therefore exactly equivalent: a plan
for one, with g -> -g and phi_1 -> phi_1 + pi, is a plan for the other with the
same fidelity, leakage bound, boundary penalty and duration.

This script mirrors the final-rollout plan of every curriculum target (the plan
behind Table/Fig. runtime) onto pi - alpha, re-simulates it, and records its
metrics next to those of the curriculum's own plan for the mirrored target.

Output: final_results/gate_structure_results/curriculum_mirror_check.csv
Run from anywhere: python analysis/analyze_mirror_symmetry.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from uqc.eval import ControlPlan, simulate_nominal_plan  # noqa: E402
from uqc.physics import GmonSystem, GmonSystemConfig  # noqa: E402

OUT = "final_results/gate_structure_results/curriculum_mirror_check.csv"
CONVERGED_COST = 0.4


def local_plan_path(windows_path: str) -> str:
    p = windows_path.replace("\\", "/")
    i = p.index("gamma_1.570796/")
    return "final_results/runtime/" + p[i:]


def mirrored(plan: ControlPlan) -> ControlPlan:
    c = plan.controls_mhz_and_phase.copy()
    c[:, 6] = -c[:, 6]
    c[:, 3] = np.mod(c[:, 3] + np.pi, 2 * np.pi)
    return ControlPlan(c, np.pi - plan.target_alpha, plan.target_gamma, plan.dt_ns, plan.runtime_norm_ns,
                       plan.cost_weights, plan.filter_bandwidth_mhz, "mirror of " + plan.note)


def main() -> None:
    table = pd.read_csv("final_results/runtime_results/phase_final_results_log.csv")
    system = GmonSystem(GmonSystemConfig(dt_ns=2.0, runtime_norm_ns=60.0))
    own = {}
    for _, r in table.iterrows():
        plan = ControlPlan.load(local_plan_path(r["eval_plan_path"]))
        assert abs(plan.target_alpha - r["alpha"]) < 1e-9
        own[round(float(r["alpha"]), 3)] = (plan, simulate_nominal_plan(system, plan))
    rows = []
    for alpha, (plan, nom) in own.items():
        m = mirrored(plan)
        mnom = simulate_nominal_plan(system, m)
        partner = min(own, key=lambda a: abs(a - m.target_alpha))
        pnom = own[partner][1]
        rows.append({
            "alpha": alpha,
            "time_ns": nom["time_ns"], "fidelity": nom["fidelity"], "cost": nom["cost"],
            "converged": nom["cost"] <= CONVERGED_COST,
            "mirror_alpha": m.target_alpha,
            "mirror_fidelity": mnom["fidelity"], "mirror_cost": mnom["cost"],
            "mirror_minus_own_cost": mnom["cost"] - nom["cost"],
            "partner_alpha": partner,
            "partner_time_ns": pnom["time_ns"], "partner_fidelity": pnom["fidelity"], "partner_cost": pnom["cost"],
            "partner_converged": pnom["cost"] <= CONVERGED_COST,
        })
    worst = max(abs(r.pop("mirror_minus_own_cost")) for r in rows)
    assert worst < 1e-9, worst
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    for r in rows:
        print(f"alpha={r['alpha']:.3f} T={r['time_ns']:5.0f} C={r['cost']:.3f} conv={r['converged']!s:5} | "
              f"mirror->{r['mirror_alpha']:.4f}: F={r['mirror_fidelity']:.5f} C={r['mirror_cost']:.3f} | "
              f"curriculum at {r['partner_alpha']:.3f}: T={r['partner_time_ns']:.0f} C={r['partner_cost']:.3f} conv={r['partner_converged']}")
    print(f"largest |mirror cost - own cost| = {worst:.1e}; written {OUT}")


if __name__ == "__main__":
    main()
