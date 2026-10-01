"""How the leakage bound under white noise depends on the time step.

analyze_noisy_leakage.py shows that at dt = 2 ns and 1 MHz the TSWT bound, accumulated from the
noisy Hamiltonian, grows by about 3.8e-3 per ns whatever the pulse. The bound contains the second
time derivative of a residual that itself contains the second derivative of the controls, so white
noise enters through its fourth finite difference and the growth scales as dt^-4. This script
checks that scaling: each plan is replayed on a grid 1, 2, 4 and 8 times finer (the controls are
held within a 2 ns step, the noise is redrawn at every fine step) and the bound of the noise-free
replay on the same grid is subtracted.

It matters for the comparison with Niu et al., whose pulses have about a thousand steps: on such
a grid a bound evaluated on the noisy Hamiltonian would be dominated by the noise far more
strongly than in this implementation.

Output: final_results/noisy_leakage_results/noisy_leakage_time_step.csv
Run from anywhere: python analysis/noisy_leakage_time_step.py
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
REFINEMENTS = (1, 2, 4, 8)
SIGMA = 1.0
SAMPLES = 40
SEED = 7
OUT = "final_results/noisy_leakage_results/noisy_leakage_time_step.csv"


def bound(system: GmonSystem, controls: np.ndarray, sigma: float, rng) -> float:
    est = TSWTLeakageEstimator(system)
    for c in controls:
        x = c.copy()
        eta = system.config.eta_base_mhz
        if sigma > 0:
            eta = eta + rng.normal(0.0, sigma)
            n = rng.normal(0.0, sigma, size=5)
            x[0] += n[0]; x[1] += n[1]; x[2] += n[2]; x[4] += n[3]; x[6] += n[4]
        est.step(system.hamiltonian(x, eta), eta)
    return est.current_leakage_bound()[0]


def main() -> None:
    rows = []
    for name, path in PLANS.items():
        plan = ControlPlan.load(path)
        time_ns = plan.controls_mhz_and_phase.shape[0] * plan.dt_ns
        reference = None
        for refinement in REFINEMENTS:
            dt = plan.dt_ns / refinement
            system = GmonSystem(GmonSystemConfig(dt_ns=dt, runtime_norm_ns=plan.runtime_norm_ns))
            controls = np.repeat(plan.controls_mhz_and_phase, refinement, axis=0)
            rng = np.random.default_rng([SEED, refinement])
            noisy = float(np.mean([bound(system, controls, SIGMA, rng) for _ in range(SAMPLES)]))
            rate = (noisy - bound(system, controls, 0.0, rng)) / time_ns
            reference = rate if reference is None else reference
            rows.append({"plan": name, "time_ns": time_ns, "dt_ns": dt, "sigma_mhz": SIGMA, "num_samples": SAMPLES,
                         "noise_bound_per_ns": rate, "ratio_to_2ns": rate / reference,
                         "dt_power_law": (2.0 / dt) ** 4})
            print(f"{name:13s} dt={dt:5.2f} ns  noise part of the bound per ns = {rate:.4g}  "
                  f"({rate / reference:.1f} x; dt^-4 gives {(2.0 / dt) ** 4:.0f})", flush=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
