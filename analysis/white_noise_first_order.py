"""First-order infidelity rate of the white control-noise model, compared with the data.

The noise model adds, at every step of length dt, an independent Gaussian kick of standard
deviation sigma (MHz) to each of delta_1, delta_2, f_1, f_2, g and eta. Moving each kick to the
end of the pulse conjugates its generator G by the evolution so far. As long as that evolution
keeps the computational subspace P invariant, the first-order infidelity of a kick depends only
on the variance of G over P,
    V(G) = Tr(P G^2 P)/d - |Tr(P G P)/d|^2 ,
which conjugation leaves unchanged. The noise-induced average infidelity is then independent of
the pulse shape and proportional to the pulse duration T:
    1 - Fbar  ~=  (1 - Fbar_0) + K sigma^2 T,     K = d/(d+1) (2 pi 1e-3)^2 dt sum_c V_c .
The part of G that leads out of P (to states detuned by the anharmonicity) is averaged by the
drift during the step; its weight is multiplied by sinc^2(omega dt / 2), omega being the
transition frequency, which is the zero-order-hold response of the step at that frequency.

This script computes K from the operators of uqc.physics.GmonSystem and compares it with the
measured excess infidelity per ns of every plan in the matched-trajectory evaluation and of the
selected controllers on the dense grid.

Outputs (final_results/white_noise_results/):
  first_order_rate.csv     channel variances, predicted rate K, measured rates
  excess_vs_duration.csv   per plan: duration and excess infidelity at 1 MHz,
                           Fbar(0.1 MHz) - Fbar(1 MHz); the 160 plans of the matched-trajectory
                           evaluation (200 samples each) and the five controllers of the dense
                           grid (raw values averaged over the 11 grid points of 0.100-0.110 MHz and
                           of 0.995-1.005 MHz, 660 samples each), with the first-order prediction
                           K (<sigma^2 near 1 MHz> - <sigma^2 near 0.1 MHz>) T
Run from anywhere: python analysis/white_noise_first_order.py
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

from uqc.physics import GmonSystem, GmonSystemConfig  # noqa: E402

OUT_DIR = "final_results/white_noise_results"
D = 4


def channel_variance(system: GmonSystem, G: np.ndarray, eta_rad: float, dt: float, zoh: bool) -> float:
    P = system.comp_indices
    L = system.leak_indices
    energies = eta_rad * np.diag(system.drift_shape).real
    Gpp = G[np.ix_(P, P)]
    second = np.real(np.trace(Gpp.conj().T @ Gpp))
    for li in L:
        for pi in P:
            w = 1.0
            if zoh:
                x = (energies[li] - energies[pi]) * dt / 2.0
                w = (np.sin(x) / x) ** 2 if abs(x) > 1e-12 else 1.0
            second += w * abs(G[li, pi]) ** 2
    first = np.trace(Gpp) / D
    return float(second / D - abs(first) ** 2)


def main() -> None:
    system = GmonSystem(GmonSystemConfig(dt_ns=2.0, runtime_norm_ns=60.0))
    dt = system.dt_ns
    eta_rad = system.config.eta_base_mhz * system.mhz_to_rad_per_ns
    channels = {
        "delta_1": system.n1,
        "delta_2": system.n2,
        "f_1": 1j * (system.a1 - system.ad1),
        "f_2": 1j * (system.a2 - system.ad2),
        "g": system.op_coupling,
        "eta": system.drift_shape,
    }
    prefactor = D / (D + 1) * (2 * np.pi * 1e-3) ** 2 * dt
    rows = []
    totals = {}
    for zoh in (False, True):
        total = 0.0
        for name, G in channels.items():
            v = channel_variance(system, G, eta_rad, dt, zoh)
            total += v
            rows.append({"quantity": f"variance_{name}", "zero_order_hold": zoh, "value": v})
        totals[zoh] = prefactor * total
        rows.append({"quantity": "predicted_rate_per_ns_per_mhz2", "zero_order_hold": zoh, "value": totals[zoh]})

    traj = pd.read_csv("final_results/trajectory_robustness_results/trajectory_robustness.csv")
    for agent in ("nominal", "noise"):
        a = traj[(traj.agent == agent) & (traj.iteration >= 21)]
        base = a[a.sigma_mhz == 0.1].set_index("iteration").average_fidelity
        for sigma in (0.5, 1.0, 2.0):
            s = a[a.sigma_mhz == sigma].set_index("iteration")
            rate = (base - s.average_fidelity) / (s.time_ns * (sigma ** 2 - 0.1 ** 2))
            rows.append({"quantity": f"measured_rate_{agent}_iterations21to100_sigma{sigma}_median",
                         "zero_order_hold": "", "value": float(rate.median())})
            rows.append({"quantity": f"measured_rate_{agent}_iterations21to100_sigma{sigma}_iqr_low",
                         "zero_order_hold": "", "value": float(rate.quantile(0.25))})
            rows.append({"quantity": f"measured_rate_{agent}_iterations21to100_sigma{sigma}_iqr_high",
                         "zero_order_hold": "", "value": float(rate.quantile(0.75))})
        s1 = a[a.sigma_mhz == 1.0].set_index("iteration")
        corr = np.corrcoef(base - s1.average_fidelity, s1.time_ns)[0, 1]
        rows.append({"quantity": f"corr_excess_infidelity_vs_duration_{agent}_sigma1.0",
                     "zero_order_hold": "", "value": float(corr)})

    os.makedirs(OUT_DIR, exist_ok=True)
    k = totals[True]
    excess = []
    for agent in ("nominal", "noise"):
        a = traj[(traj.agent == agent) & (traj.iteration >= 21)]
        base = a[a.sigma_mhz == 0.1].set_index("iteration")
        s1 = a[a.sigma_mhz == 1.0].set_index("iteration")
        for it in s1.index:
            t = float(s1.loc[it, "time_ns"])
            excess.append({"source": f"trajectory_{agent}", "iteration": int(it), "time_ns": t,
                           "nominal_fidelity": float(s1.loc[it, "nominal_fidelity"]),
                           "excess_infidelity_1mhz": float(base.loc[it, "average_fidelity"] - s1.loc[it, "average_fidelity"]),
                           "predicted_excess_1mhz": k * (1.0 - 0.1 ** 2) * t})
    dense = pd.read_csv("final_results/robustness_analysis_3/combined_ewma_data.csv")
    low = dense[(dense.sigma_mhz >= 0.0995) & (dense.sigma_mhz <= 0.1105)]
    high = dense[(dense.sigma_mhz >= 0.9945) & (dense.sigma_mhz <= 1.0055)]
    assert len(low) == 11 and len(high) == 11
    dsig2 = float((high.sigma_mhz ** 2).mean() - (low.sigma_mhz ** 2).mean())
    durations = {"adam70": 70.0, "adam60": 60.0, "nominal": 116.0, "noise": 98.0, "noise91": 104.0}
    for c, t in durations.items():
        f01 = float(low[f"{c}_fidelity_raw"].mean())
        f1 = float(high[f"{c}_fidelity_raw"].mean())
        excess.append({"source": f"selected_{c}", "iteration": "", "time_ns": t, "nominal_fidelity": "",
                       "excess_infidelity_1mhz": f01 - f1, "predicted_excess_1mhz": k * dsig2 * t})
    with open(os.path.join(OUT_DIR, "excess_vs_duration.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(excess[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in excess:
            w.writerow({kk: (f"{v:.6g}" if isinstance(v, float) else v) for kk, v in r.items()})
    with open(os.path.join(OUT_DIR, "first_order_rate.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "zero_order_hold", "value"], lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "value": f"{r['value']:.6g}"})
    for r in rows:
        print(f"{r['quantity']:62s} {str(r['zero_order_hold']):6s} {r['value']:.4g}")


if __name__ == "__main__":
    main()
