"""Checks of the first-order description of the white control-noise model.

white_noise_first_order.py derives the rate K of the white-noise infidelity from the channel
operators alone. This script evaluates the same first-order expression exactly on each stored
pulse (uqc.noise_response: generators in the toggling frame, with the exact response of every
step instead of the sinc^2 weight) and tests three of its consequences by Monte Carlo replay of
the five single-target controllers:

  exact_rate.csv         per plan (the five controllers and every plan of the matched TRPO
                         histories): first-order rate per channel and in total, in ns^-1 MHz^-2;
                         the rows "summary_*" give its range over the plans
  channel_resolved.csv   noise on one channel at a time (1 MHz): Monte Carlo excess infidelity
                         against the first-order value
  noise_interval.csv     noise redrawn every 0.1, 0.5, 1, 2, 4 or 8 ns at the same sigma (1 MHz; the
                         model uses 2 ns, Niu et al. about a thousand steps per pulse): the rate
                         scales with the interval, so a given sigma describes a different noise
                         level at a different time step
  prefilter_noise.csv    the same white noise added to the raw controls, before the bandwidth
                         filter, instead of after it

Before anything else the script checks that the replay of uqc.noise_response reproduces
uqc.eval.noisy_projected_unitary_samples sample by sample. Every Monte Carlo row has its own random
stream, so adding a row does not change the others.

Run from anywhere: python analysis/white_noise_checks.py
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

from uqc.eval import ControlPlan, noisy_projected_unitary_samples, simulate_nominal_plan  # noqa: E402
from uqc.noise_response import (  # noqa: E402
    NOISE_CHANNELS,
    fidelity_statistics,
    first_order_excess_infidelity,
    first_order_kernel,
    held_covariance,
    prefilter_covariance,
    replay_with_noise,
    sample_noise,
    toggling_frame_generators,
    white_covariance,
)
from uqc.physics import GmonSystem, GmonSystemConfig  # noqa: E402

SELECTED = {
    "adam_70ns": "final_results/adam_noise_70ns/best_control_plan.npz",
    "adam_60ns": "final_results/adam_noise/best_control_plan.npz",
    "trpo_nominal": "final_results/nominal/best_control_plan.npz",
    "trpo_noise": "final_results/trpo_noise_alpha_2.2/plans/iter_000011_control_plan.npz",
    "trpo_noise_it091": "final_results/trpo_noise_alpha_2.2/best_control_plan.npz",
}
HISTORIES = {
    "trajectory_nominal": "final_results/nominal/plans",
    "trajectory_noise": "final_results/trpo_noise_alpha_2.2/plans",
}
MATCHED_ITERATIONS = range(1, 101)
SUMMARY_ITERATIONS = range(21, 101)
SIGMA = 1.0
SAMPLES = 4000
SEED = 5
# noise interval in ns -> (parts of a 2 ns step, steps over which the noise is held)
INTERVALS = {0.1: (20, 1), 0.5: (4, 1), 1.0: (2, 1), 2.0: (1, 1), 4.0: (1, 2), 8.0: (1, 4)}
OUT_DIR = "final_results/white_noise_results"
D = 4


def load(path: str):
    plan = ControlPlan.load(path)
    if abs(plan.target_alpha - 2.2) > 1e-9 or abs(plan.target_gamma - np.pi / 2) > 1e-9:
        raise ValueError(f"{path} does not target N(2.2, 2.2, pi/2)")
    system = GmonSystem(GmonSystemConfig(dt_ns=plan.dt_ns, runtime_norm_ns=plan.runtime_norm_ns))
    return system, plan


def check_replay(system, plan) -> float:
    """Largest difference between the two replays when both are fed the same random numbers."""
    samples, n = 20, plan.controls_mhz_and_phase.shape[0]
    reference = noisy_projected_unitary_samples(system, plan, SIGMA, samples, seed=SEED)
    rng = np.random.default_rng(SEED)
    noise = np.zeros((samples, len(NOISE_CHANNELS), n))
    for s in range(samples):
        for k in range(n):
            noise[s, 5, k] = rng.normal(0.0, SIGMA)          # eta is drawn first in uqc.eval
            noise[s, :5, k] = rng.normal(0.0, SIGMA, size=5)  # delta_1, delta_2, f_1, f_2, g
    return float(np.max(np.abs(replay_with_noise(system, plan, noise) - np.asarray(reference))))


def write_csv(path: str, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- exact first-order rate of every plan ------------------------------------------------
    sources = [(name, "", path) for name, path in SELECTED.items()]
    for name, plan_dir in HISTORIES.items():
        sources += [(name, it, os.path.join(plan_dir, f"iter_{it:06d}_control_plan.npz")) for it in MATCHED_ITERATIONS]
    rate_rows, cache = [], {}
    for source, iteration, path in sources:
        system, plan = load(path)
        kernel = first_order_kernel(system, toggling_frame_generators(system, plan))
        n = kernel.shape[1]
        time_ns = n * plan.dt_ns
        rates = first_order_excess_infidelity(system, kernel, white_covariance(n)) / time_ns
        row = {"source": source, "iteration": iteration, "time_ns": time_ns}
        row.update({f"rate_{c}": float(r) for c, r in zip(NOISE_CHANNELS, rates)})
        row["rate_total"] = float(rates.sum())
        rate_rows.append(row)
        if source in SELECTED:
            cache[source] = (system, plan, kernel)
    matched = np.array([r["rate_total"] for r in rate_rows
                        if r["source"] in HISTORIES and r["iteration"] in SUMMARY_ITERATIONS])
    selected = np.array([r["rate_total"] for r in rate_rows if r["source"] in SELECTED])
    for label, values in (("summary_matched_iterations_21_100", matched), ("summary_selected", selected)):
        for stat, value in (("min", values.min()), ("median", np.median(values)), ("max", values.max())):
            rate_rows.append({"source": f"{label}_{stat}", "iteration": "", "time_ns": "",
                              **{f"rate_{c}": "" for c in NOISE_CHANNELS}, "rate_total": float(value)})
    write_csv(os.path.join(OUT_DIR, "exact_rate.csv"), rate_rows)
    print(f"exact first-order rate: {matched.min():.6g} - {matched.max():.6g} over the matched plans, "
          f"{selected.min():.6g} - {selected.max():.6g} over the selected ones", flush=True)

    # ---- Monte Carlo checks on the selected controllers --------------------------------------
    channel_rows, interval_rows, prefilter_rows = [], [], []
    for index, (name, (system, plan, kernel)) in enumerate(cache.items()):
        worst = check_replay(system, plan)
        assert worst < 1e-9, f"{name}: replay mismatch {worst}"
        n = kernel.shape[1]
        time_ns = n * plan.dt_ns
        nominal = simulate_nominal_plan(system, plan)
        fbar0 = (D * nominal["fidelity"] + 1.0) / (D + 1.0)

        def measured(noise, substeps=1):
            fbar, _, variance = fidelity_statistics(system, plan, replay_with_noise(system, plan, noise, substeps))
            return fbar0 - fbar, np.sqrt(variance / noise.shape[0]) * D / (D + 1)

        white = first_order_excess_infidelity(system, kernel, SIGMA ** 2 * white_covariance(n))
        for ci, channel in enumerate(NOISE_CHANNELS):
            rng = np.random.default_rng([SEED, index, 0, ci])
            noise = np.zeros((SAMPLES, len(NOISE_CHANNELS), n))
            noise[:, ci, :] = rng.normal(0.0, SIGMA, size=(SAMPLES, n))
            excess, stderr = measured(noise)
            channel_rows.append({"plan": name, "time_ns": time_ns, "channel": channel, "sigma_mhz": SIGMA,
                                 "num_samples": SAMPLES, "excess_infidelity": float(excess),
                                 "standard_error": float(stderr), "first_order": float(white[ci])})

        for interval, (substeps, hold) in INTERVALS.items():
            rng = np.random.default_rng([SEED, index, 1, int(round(10 * interval))])
            sub_kernel = kernel if substeps == 1 else first_order_kernel(
                system, toggling_frame_generators(system, plan, substeps=substeps))
            cov = held_covariance(n * substeps, hold)
            predicted = first_order_excess_infidelity(system, sub_kernel, SIGMA ** 2 * cov).sum()
            excess, stderr = measured(sample_noise(rng, cov, SAMPLES, SIGMA), substeps)
            interval_rows.append({"plan": name, "time_ns": time_ns, "noise_interval_ns": interval,
                                  "sigma_mhz": SIGMA, "num_samples": SAMPLES,
                                  "excess_infidelity": float(excess), "standard_error": float(stderr),
                                  "first_order": float(predicted),
                                  "first_order_rate_per_ns": float(predicted / time_ns),
                                  "first_order_ratio_to_2ns": float(predicted / white.sum())})

        cov = prefilter_covariance(n, plan.dt_ns, plan.filter_bandwidth_mhz)
        predicted = first_order_excess_infidelity(system, kernel, SIGMA ** 2 * cov).sum()
        excess, stderr = measured(sample_noise(np.random.default_rng([SEED, index, 2]), cov, SAMPLES, SIGMA))
        prefilter_rows.append({"plan": name, "time_ns": time_ns, "sigma_mhz": SIGMA, "num_samples": SAMPLES,
                               "excess_infidelity": float(excess), "standard_error": float(stderr),
                               "first_order": float(predicted),
                               "first_order_after_filter": float(white.sum()),
                               "ratio_before_over_after": float(predicted / white.sum())})
        print(f"{name:17s} replay check {worst:.1e}; prefilter/postfilter = {predicted / white.sum():.3f}", flush=True)

    write_csv(os.path.join(OUT_DIR, "channel_resolved.csv"), channel_rows)
    write_csv(os.path.join(OUT_DIR, "noise_interval.csv"), interval_rows)
    write_csv(os.path.join(OUT_DIR, "prefilter_noise.csv"), prefilter_rows)
    print("written to", OUT_DIR)


if __name__ == "__main__":
    main()
