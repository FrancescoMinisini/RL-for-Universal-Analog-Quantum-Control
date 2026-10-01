"""How the UFO cost behaves under the training noise model.

Noise-aware training (TRPO environment with noise_optimized=True, Adam objective
with train_noise_std > 0) accumulates the TSWT leakage bound from the *noisy*
Hamiltonian. This script replays four control plans for N(2.2, 2.2, pi/2) under
the same noise model (Gaussian, per 2 ns step, on g, delta_1,2, f_1,2 and eta,
after the bandwidth filter) and records, per noise strength, the gate fidelity,
the leakage bound with its boundary/integral split, and the leakage population
actually present at the end of the pulse, 1 - Tr(K^dag K)/4 with K the
computational block of the propagator.

Output: final_results/noisy_leakage_results/noisy_leakage_bound.csv
Run from anywhere: python analysis/analyze_noisy_leakage.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from uqc.eval import ControlPlan  # noqa: E402
from uqc.physics import GmonSystem, GmonSystemConfig, TSWTLeakageEstimator  # noqa: E402

PLANS = {
    "trpo_nominal": "final_results/nominal/best_control_plan.npz",
    "trpo_noise": "final_results/trpo_noise_alpha_2.2/plans/iter_000011_control_plan.npz",
    "adam_60ns": "final_results/adam_noise/best_control_plan.npz",
    "adam_70ns": "final_results/adam_noise_70ns/best_control_plan.npz",
}
SIGMAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
SAMPLES = 100
SEED = 7
OUT_DIR = "final_results/noisy_leakage_results"


def rollout(system, plan, sigma, rng):
    U = system.initial_unitary()
    est = TSWTLeakageEstimator(system)
    for c in plan.controls_mhz_and_phase:
        x = np.asarray(c, dtype=np.float64).copy()
        eta = system.config.eta_base_mhz
        if sigma > 0:
            eta = eta + rng.normal(0.0, sigma)
            n = rng.normal(0.0, sigma, size=5)
            x[0] += n[0]; x[1] += n[1]; x[2] += n[2]; x[4] += n[3]; x[6] += n[4]
        x[[3, 5]] = np.mod(x[[3, 5]], 2.0 * np.pi)
        H = system.hamiltonian(x, eta)
        U = system.evolve_step(U, x, eta)
        est.step(H, eta)
    K = system.projected_unitary(U)
    target = system.target_gate(plan.target_alpha, plan.target_gamma)
    fid = system.gate_fidelity_from_projected(K, target)
    bound, parts = est.current_leakage_bound()
    pop = 1.0 - float(np.real(np.trace(K.conj().T @ K))) / 4.0
    return fid, bound, parts["boundary"], parts["integral"], pop


def main():
    rows = []
    for name, path in PLANS.items():
        plan = ControlPlan.load(path)
        assert abs(plan.target_alpha - 2.2) < 1e-9, path
        system = GmonSystem(GmonSystemConfig(dt_ns=plan.dt_ns, runtime_norm_ns=plan.runtime_norm_ns))
        w = plan.cost_weights
        c0, ct = plan.controls_mhz_and_phase[0], plan.controls_mhz_and_phase[-1]
        boundary_cost = w.mu * (c0[6] ** 2 + c0[2] ** 2 + c0[4] ** 2 + ct[6] ** 2 + ct[2] ** 2 + ct[4] ** 2)
        time_ns = plan.controls_mhz_and_phase.shape[0] * plan.dt_ns
        time_cost = w.kappa * time_ns / plan.runtime_norm_ns
        for k, sigma in enumerate(SIGMAS):
            rng = np.random.default_rng(SEED + k)
            n = 1 if sigma == 0 else SAMPLES
            a = np.array([rollout(system, plan, sigma, rng) for _ in range(n)])
            f, lb, lbb, lbi, pop = a.mean(axis=0)
            rows.append({
                "plan": name, "time_ns": time_ns, "sigma_mhz": sigma, "num_samples": n,
                "gate_fidelity": f, "leakage_bound": lb, "leakage_bound_boundary": lbb,
                "leakage_bound_integral": lbi, "leakage_population": pop,
                "infidelity_term": w.chi * (1 - f), "leakage_term": w.beta * lb,
                "boundary_term": boundary_cost, "time_term": time_cost,
                "ufo_cost": w.chi * (1 - f) + w.beta * lb + boundary_cost + time_cost,
            })
            print(f"{name:13s} sigma={sigma:4.2f}  F={f:.4f}  bound={lb:.3e}  population={pop:.2e}  "
                  f"chi(1-F)={w.chi*(1-f):.3f}  beta*bound={w.beta*lb:.3f}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "noisy_leakage_bound.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
