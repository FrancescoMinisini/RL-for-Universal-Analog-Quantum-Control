"""Two-qubit structure of the family N(alpha, alpha, pi/2) and its runtime data.

Every member of the family is, up to a global phase, a pure exchange rotation:
    N(alpha, alpha, pi/2) = i * exp[i (alpha - pi/2) (XX + YY)],
because exp(i pi/2 ZZ) = i ZZ and exp(i pi/2 (XX + YY)) = ZZ. The target at
alpha = pi/2 is therefore the identity, and alpha = 0, pi give the local gate ZZ.

For every alpha of the TRPO curriculum sweep and of the Adam family sweep this
script checks that identity numerically, the local-equivalence class through the
Makhlin invariants, and the minimal CNOT count through the Shende-Markov-Bullock
criteria, and derives:
  * the Weyl-chamber coordinate a (the class is (a, a, 0)),
  * the exchange angle eps = alpha - pi/2 and the bang-bang coupling time
    2|eps| / g_max needed to generate it with g alone (g_max = 20 MHz),
  * a target-specific synthesis time with the gate times of Niu et al.
    (20 ns per single-qubit layer, 45 ns per CNOT): 0 ns for the identity, one
    single-qubit layer (20 ns) for ZZ, two CNOTs and three layers (150 ns) for
    every other member; the generic three-CNOT reference is 215 ns.

Output: final_results/gate_structure_results/runtime_vs_gate_structure.csv
Run from anywhere: python analysis/analyze_gate_structure.py
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pandas as pd
import scipy.linalg as la

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = "final_results/gate_structure_results"
T_1Q_NS, T_CNOT_NS = 20.0, 45.0
T_SYN_GENERIC_NS = 4 * T_1Q_NS + 3 * T_CNOT_NS          # 215 ns, Ref. [Niu2019]
G_MAX_RAD_PER_NS = 2 * np.pi * 20e-3                      # 20 MHz coupling bound
TRPO_CONVERGED_COST = 0.4                                 # curriculum advancement threshold

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
XX, YY, ZZ = np.kron(X, X), np.kron(Y, Y), np.kron(Z, Z)
MAGIC = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]], dtype=complex) / np.sqrt(2)


def target(alpha: float, gamma: float = np.pi / 2) -> np.ndarray:
    return la.expm(1j * (alpha * XX + alpha * YY + gamma * ZZ))


def canonical(a: float, b: float, c: float) -> np.ndarray:
    return la.expm(1j * (a * XX + b * YY + c * ZZ))


def makhlin(U: np.ndarray) -> tuple[complex, float]:
    Ub = MAGIC.conj().T @ U @ MAGIC
    m = Ub.T @ Ub
    det = np.linalg.det(U)
    tr = np.trace(m)
    g1 = tr ** 2 / (16 * det)
    g2 = (tr ** 2 - np.trace(m @ m)) / (4 * det)
    return complex(g1), float(np.real(g2))


def min_cnots(U: np.ndarray, tol: float = 1e-9) -> int:
    U = U / np.linalg.det(U) ** 0.25
    g = U @ np.kron(Y, Y) @ U.T @ np.kron(Y, Y)
    tr = np.trace(g)
    if np.allclose(g, np.eye(4), atol=tol) or np.allclose(g, -np.eye(4), atol=tol):
        return 0
    if abs(tr) < tol and np.allclose(g @ g, -np.eye(4), atol=tol):
        return 1
    if abs(np.imag(tr)) < tol:
        return 2
    return 3


def local_layers(U: np.ndarray, tol: float = 1e-9) -> int:
    """Single-qubit layers needed for a local gate: 0 for the identity up to phase, else 1."""
    phase = U[0, 0] / abs(U[0, 0]) if abs(U[0, 0]) > tol else 1.0
    return 0 if np.allclose(U / phase, np.eye(4), atol=tol) else 1


def structure(alpha: float) -> dict:
    U = target(alpha)
    eps = alpha - np.pi / 2
    exchange_form = 1j * la.expm(1j * eps * (XX + YY))
    r = abs(eps) % (np.pi / 2)
    a = min(r, np.pi / 2 - r)
    g_u, g_c = makhlin(U), makhlin(canonical(a, a, 0.0))
    n_cnot = min_cnots(U)
    if n_cnot == 0:
        t_syn = local_layers(U) * T_1Q_NS
    else:
        t_syn = n_cnot * T_CNOT_NS + (n_cnot + 1) * T_1Q_NS
    return {
        "alpha": alpha,
        "exchange_angle": eps,
        "abs_exchange_angle": abs(eps),
        "exchange_form_error": float(np.max(np.abs(U - exchange_form))),
        "weyl_a": a,
        "makhlin_mismatch": float(max(abs(g_u[0] - g_c[0]), abs(g_u[1] - g_c[1]))),
        "min_cnots": n_cnot,
        "t_syn_generic_ns": T_SYN_GENERIC_NS,
        "t_syn_target_ns": t_syn,
        "t_exchange_bangbang_ns": 2 * abs(eps) / G_MAX_RAD_PER_NS,
    }


def main() -> None:
    trpo = pd.read_csv("final_results/runtime_results/phase_final_results_log.csv")
    adam = pd.read_csv("final_results/adam_results/adam_family_sweep.csv")
    rows = []
    for _, r in trpo.iterrows():
        s = structure(float(r["alpha"]))
        s.update({
            "method": "trpo_curriculum",
            "runtime_ns": float(r["eval_time_ns"]),
            "min_cost_time_ns": float(r["eval_min_cost_time_ns"]),
            "nominal_fidelity": float(r["eval_fidelity"]),
            "nominal_cost": float(r["eval_cost"]),
            "nominal_leakage": float(r["eval_leakage"]),
            "avg_fidelity_1mhz": float(r["robustness_sigma_1p0_average_fidelity"]),
            "converged": bool(r["eval_cost"] <= TRPO_CONVERGED_COST),
        })
        rows.append(s)
    for _, r in adam.iterrows():
        s = structure(float(r["alpha"]))
        s.update({
            "method": "adam",
            "runtime_ns": float(r["runtime_ns"]),
            "min_cost_time_ns": float(r["runtime_ns"]),
            "nominal_fidelity": float(r["best_fidelity"]),
            "nominal_cost": float(r["best_cost"]),
            "nominal_leakage": float(r["best_leakage"]),
            "avg_fidelity_1mhz": float(r["robustness_sigma_1p0_average_fidelity"]),
            "converged": True,
        })
        rows.append(s)
    for s in rows:
        s["speedup_generic"] = s["t_syn_generic_ns"] / s["runtime_ns"]
        s["speedup_target"] = s["t_syn_target_ns"] / s["runtime_ns"]

    worst = max(max(s["exchange_form_error"], s["makhlin_mismatch"]) for s in rows)
    assert worst < 1e-10, f"structure check failed: {worst}"

    cols = ["method", "alpha", "exchange_angle", "abs_exchange_angle", "weyl_a", "min_cnots",
            "t_syn_generic_ns", "t_syn_target_ns", "t_exchange_bangbang_ns", "runtime_ns",
            "min_cost_time_ns", "nominal_fidelity", "nominal_cost", "nominal_leakage", "avg_fidelity_1mhz", "converged",
            "speedup_generic", "speedup_target"]
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "runtime_vs_gate_structure.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for s in rows:
            w.writerow({k: (f"{s[k]:.6g}" if isinstance(s[k], float) else s[k]) for k in cols})
    print(f"largest check error {worst:.2e}; written {path}")


if __name__ == "__main__":
    main()
