"""Response of the stored pulses to control noise with memory.

Under the white noise of the UFO model the first-order infidelity of a pulse depends only on its
duration (white_noise_first_order.py, white_noise_checks.py). This script evaluates the same
pulses against noise that is correlated in time, where the pulse shape enters:

  quasi-static noise   one Gaussian offset per channel (delta_1,2, f_1,2, g, eta), constant over
                       the pulse and redrawn for every shot;
  exponential noise    stationary Ornstein-Uhlenbeck noise of correlation time tau, sampled at
                       every step; it tends to the white model for tau -> 0 and to the
                       quasi-static one for tau -> infinity.

For every plan it computes the first-order prediction of uqc.noise_response and a Monte Carlo
replay of the full dynamics. All plans see the same quasi-static offsets, so differences between
plans are not blurred by sampling. The plans are the five single-target controllers, every
per-iteration plan of the two matched TRPO runs, and the Adam pulses trained without noise,
under white noise and under quasi-static noise (final_results/adam_noise_models, written by
train_adam_noise_models.py), together with the repetition of that experiment in which the leakage
bound of the objective is accumulated from the noise-free controls
(final_results/adam_noise_models_nominal_leakage).

Outputs (final_results/noise_memory_results/):
  plan_sensitivity.csv        one row per plan
  matched_histories.csv       noise-trained against nominal TRPO plan of the same iteration under
                              quasi-static noise: paired differences with 95% block-bootstrap
                              intervals (8 blocks of 10 iterations, fixed seed), iterations 21-100
  adam_noise_models.csv       per training-noise model and horizon: statistics over the seeds
  adam_noise_models_nominal_leakage.csv   the same for the runs with the leakage bound on the
                              noise-free controls
  correlation_time.csv        excess infidelity against the correlation time for the three 70 ns
                              Adam pulses of seed 1 (first order on a fine grid, Monte Carlo at a
                              few values)
  filter_functions.csv        their filter functions, summed over the channels and normalized so
                              that the average over the Nyquist band is 1 for every pulse

Run from anywhere: python analysis/noise_with_memory.py [--workers N]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from uqc.eval import ControlPlan, simulate_nominal_plan  # noqa: E402
from uqc.noise_response import (  # noqa: E402
    NOISE_CHANNELS,
    exponential_covariance,
    fidelity_statistics,
    filter_function,
    first_order_excess_infidelity,
    first_order_kernel,
    quasi_static_covariance,
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
ADAM_DIR = "final_results/adam_noise_models"
# source prefix in plan_sensitivity.csv -> run directory
ADAM_RUNS = {"adam": ADAM_DIR, "adam_nominal_leakage": "final_results/adam_noise_models_nominal_leakage"}
ADAM_MODELS = ("none", "white", "quasi_static")
MATCHED_ITERATIONS = range(1, 101)
SUMMARY_ITERATIONS = range(21, 101)
SIGMA_WHITE = 1.0              # MHz, the UFO noise strength
SIGMAS_QUASI_STATIC = (0.1, 0.3)   # MHz
SAMPLES = 2000
SEED = 9
FIGURE_HORIZON, FIGURE_SEED = 70.0, 1
SIGMA_CORRELATED = 0.2         # MHz, exponential noise of the correlation-time scan
TAUS_FIRST_ORDER = np.geomspace(0.25, 4096.0, 57)
TAUS_MONTE_CARLO = (1.0, 4.0, 16.0, 64.0, 256.0, 1024.0)
SAMPLES_CORRELATED = 8000
FREQUENCIES_MHZ = np.geomspace(0.1, 250.0, 240)
OUT_DIR = "final_results/noise_memory_results"
BOOT_BLOCK, BOOT_SAMPLES, BOOT_SEED = 10, 4000, 0
D = 4


def load(path: str):
    plan = ControlPlan.load(path)
    if abs(plan.target_alpha - 2.2) > 1e-9 or abs(plan.target_gamma - np.pi / 2) > 1e-9:
        raise ValueError(f"{path} does not target N(2.2, 2.2, pi/2)")
    system = GmonSystem(GmonSystemConfig(dt_ns=plan.dt_ns, runtime_norm_ns=plan.runtime_norm_ns))
    return system, plan


def evaluate(task):
    source, label, path, with_white = task
    system, plan = load(path)
    nominal = simulate_nominal_plan(system, plan)
    fbar0 = (D * nominal["fidelity"] + 1.0) / (D + 1.0)
    kernel = first_order_kernel(system, toggling_frame_generators(system, plan))
    n = kernel.shape[1]
    time_ns = n * plan.dt_ns
    white = first_order_excess_infidelity(system, kernel, SIGMA_WHITE ** 2 * white_covariance(n)).sum()
    unit = first_order_excess_infidelity(system, kernel, quasi_static_covariance(n))
    row = {
        "source": source, "label": label, "time_ns": time_ns,
        "nominal_fidelity": nominal["fidelity"], "nominal_cost": nominal["cost"],
        "white_first_order_1mhz": float(white),
        # ratio of the quasi-static to the white first-order infidelity at equal sigma
        "memory_gain": float(unit.sum() * SIGMA_WHITE ** 2 / white),
    }
    # the same offsets for every plan, scaled to each sigma
    offsets = np.random.default_rng(SEED).standard_normal((SAMPLES, len(NOISE_CHANNELS), 1))
    for sigma in SIGMAS_QUASI_STATIC:
        tag = f"{sigma:g}".replace(".", "p")
        row[f"quasi_static_first_order_{tag}mhz"] = float(unit.sum() * sigma ** 2)
        noise = np.repeat(sigma * offsets, n, axis=2)
        fbar, _, variance = fidelity_statistics(system, plan, replay_with_noise(system, plan, noise))
        row[f"quasi_static_average_fidelity_{tag}mhz"] = fbar
        row[f"quasi_static_excess_{tag}mhz"] = fbar0 - fbar
        row[f"quasi_static_fidelity_variance_{tag}mhz"] = variance
    for channel, value in zip(NOISE_CHANNELS, unit):
        row[f"quasi_static_first_order_0p1mhz_{channel}"] = float(value * 0.1 ** 2)
    if with_white:
        noise = SIGMA_WHITE * np.random.default_rng(SEED + 1).standard_normal((SAMPLES, len(NOISE_CHANNELS), n))
        fbar, _, variance = fidelity_statistics(system, plan, replay_with_noise(system, plan, noise))
        row["white_average_fidelity_1mhz"] = fbar
        row["white_excess_1mhz"] = fbar0 - fbar
        row["white_fidelity_variance_1mhz"] = variance
    else:
        row["white_average_fidelity_1mhz"] = row["white_excess_1mhz"] = row["white_fidelity_variance_1mhz"] = ""
    return row


def block_ci(diff: np.ndarray) -> tuple[float, float]:
    blocks = diff.reshape(-1, BOOT_BLOCK)
    rng = np.random.default_rng(BOOT_SEED)
    boot = [blocks[rng.integers(0, len(blocks), len(blocks))].mean() for _ in range(BOOT_SAMPLES)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def summarize_histories(rows):
    series = {}
    for agent in HISTORIES:
        sel = {r["label"]: r for r in rows if r["source"] == agent}
        series[agent] = [sel[i] for i in SUMMARY_ITERATIONS]
    out = []
    quantities = {
        "quasi_static_average_fidelity_0p1mhz": lambda r: r["quasi_static_average_fidelity_0p1mhz"],
        "quasi_static_average_fidelity_0p3mhz": lambda r: r["quasi_static_average_fidelity_0p3mhz"],
        # sensitivity with the duration divided out: the part that depends on the pulse shape
        "quasi_static_first_order_0p1mhz_per_ns2": lambda r: r["quasi_static_first_order_0p1mhz"] / r["time_ns"] ** 2,
        "memory_gain": lambda r: r["memory_gain"],
        "time_ns": lambda r: r["time_ns"],
    }
    for name, get in quantities.items():
        nominal = np.array([get(r) for r in series["trajectory_nominal"]])
        noise = np.array([get(r) for r in series["trajectory_noise"]])
        diff = noise - nominal
        lo, hi = block_ci(diff)
        out.append({
            "quantity": name,
            "nominal_min": float(nominal.min()), "nominal_median": float(np.median(nominal)),
            "nominal_max": float(nominal.max()),
            "noise_min": float(noise.min()), "noise_median": float(np.median(noise)), "noise_max": float(noise.max()),
            "mean_difference_noise_minus_nominal": float(diff.mean()), "difference_ci95_low": lo,
            "difference_ci95_high": hi, "fraction_noise_larger": float(np.mean(diff > 0)),
            "n_iterations": len(diff),
        })
    return out


def summarize_adam(rows, prefix="adam"):
    out = []
    horizons = sorted({r["time_ns"] for r in rows if r["source"] == f"{prefix}_none"})
    for horizon in horizons:
        reference = next(r for r in rows if r["source"] == f"{prefix}_none" and r["time_ns"] == horizon)
        for model in ADAM_MODELS:
            sel = [r for r in rows if r["source"] == f"{prefix}_{model}" and r["time_ns"] == horizon]
            rec = {"model": model, "horizon_ns": horizon, "n_runs": len(sel)}
            for key in ("nominal_fidelity", "nominal_cost", "white_first_order_1mhz", "white_excess_1mhz",
                        "white_average_fidelity_1mhz", "quasi_static_first_order_0p1mhz",
                        "quasi_static_excess_0p1mhz", "quasi_static_average_fidelity_0p1mhz",
                        "quasi_static_excess_0p3mhz", "quasi_static_average_fidelity_0p3mhz", "memory_gain"):
                x = np.array([r[key] for r in sel])
                rec[f"{key}_mean"] = float(x.mean())
                rec[f"{key}_min"] = float(x.min())
                rec[f"{key}_max"] = float(x.max())
            x = np.array([r["quasi_static_first_order_0p1mhz"] for r in sel])
            rec["quasi_static_first_order_ratio_to_none_mean"] = float(x.mean() / reference["quasi_static_first_order_0p1mhz"])
            rec["quasi_static_first_order_ratio_to_none_min"] = float(x.min() / reference["quasi_static_first_order_0p1mhz"])
            rec["quasi_static_first_order_ratio_to_none_max"] = float(x.max() / reference["quasi_static_first_order_0p1mhz"])
            for key in ("nominal_fidelity", "white_average_fidelity_1mhz", "quasi_static_average_fidelity_0p1mhz",
                        "quasi_static_average_fidelity_0p3mhz"):
                rec[f"runs_above_none_{key}"] = int(sum(r[key] > reference[key] for r in sel)) if model != "none" else ""
            out.append(rec)
    return out


def figure_pulses():
    """The three 70 ns Adam pulses of the figure: no noise, white noise and quasi-static noise, seed 1."""
    tag = f"h{int(FIGURE_HORIZON):03d}"
    out = {}
    for model in ADAM_MODELS:
        seed = FIGURE_SEED if model != "none" else None
        names = [f for f in sorted(os.listdir(os.path.join(ADAM_DIR, "plans")))
                 if f.startswith(f"{model}_{tag}_") and (seed is None or f"_seed{seed}_" in f)]
        assert len(names) == 1, names
        out[model] = load(os.path.join(ADAM_DIR, "plans", names[0]))
    return out


def correlation_time_scan(pulses):
    rows = []
    for index, (model, (system, plan)) in enumerate(pulses.items()):
        kernel = first_order_kernel(system, toggling_frame_generators(system, plan))
        n = kernel.shape[1]
        nominal = simulate_nominal_plan(system, plan)
        fbar0 = (D * nominal["fidelity"] + 1.0) / (D + 1.0)
        for tau in TAUS_FIRST_ORDER:
            cov = SIGMA_CORRELATED ** 2 * exponential_covariance(n, plan.dt_ns, tau)
            rows.append({"model": model, "kind": "first_order", "correlation_time_ns": float(tau),
                         "sigma_mhz": SIGMA_CORRELATED, "num_samples": "",
                         "excess_infidelity": float(first_order_excess_infidelity(system, kernel, cov).sum()),
                         "standard_error": ""})
        for j, tau in enumerate(TAUS_MONTE_CARLO):
            rng = np.random.default_rng(SEED + 100 + j)   # the same draws for the three pulses
            noise = sample_noise(rng, exponential_covariance(n, plan.dt_ns, tau), SAMPLES_CORRELATED, SIGMA_CORRELATED)
            fbar, _, variance = fidelity_statistics(system, plan, replay_with_noise(system, plan, noise))
            rows.append({"model": model, "kind": "monte_carlo", "correlation_time_ns": float(tau),
                         "sigma_mhz": SIGMA_CORRELATED, "num_samples": SAMPLES_CORRELATED,
                         "excess_infidelity": float(fbar0 - fbar),
                         "standard_error": float(np.sqrt(variance / SAMPLES_CORRELATED) * D / (D + 1))})
        for limit, cov in (("white", white_covariance(n)), ("quasi_static", quasi_static_covariance(n))):
            rows.append({"model": model, "kind": f"first_order_{limit}_limit", "correlation_time_ns": "",
                         "sigma_mhz": SIGMA_CORRELATED, "num_samples": "",
                         "excess_infidelity": float(first_order_excess_infidelity(
                             system, kernel, SIGMA_CORRELATED ** 2 * cov).sum()),
                         "standard_error": ""})
    return rows


def filter_function_table(pulses):
    columns = {}
    for model, (system, plan) in pulses.items():
        kernel = first_order_kernel(system, toggling_frame_generators(system, plan)).sum(axis=0, keepdims=True)
        columns[model] = filter_function(kernel, FREQUENCIES_MHZ, plan.dt_ns)[0] / np.trace(kernel[0])
    return [{"frequency_mhz": float(f), **{f"normalized_filter_function_{m}": float(columns[m][i]) for m in columns}}
            for i, f in enumerate(FREQUENCIES_MHZ)]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(float(v)) if isinstance(v, float) else v) for k, v in r.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    tasks = [(f"selected_{name}", "", path, True) for name, path in SELECTED.items()]
    for agent, plan_dir in HISTORIES.items():
        tasks += [(agent, it, os.path.join(plan_dir, f"iter_{it:06d}_control_plan.npz"), False)
                  for it in MATCHED_ITERATIONS]
    for prefix, run_dir in ADAM_RUNS.items():
        with open(os.path.join(run_dir, "summary.csv"), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                tasks.append((f"{prefix}_{r['model']}", int(r["seed"]), os.path.join(run_dir, r["plan"]), True))

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = []
        for done, row in enumerate(ex.map(evaluate, tasks), start=1):
            rows.append(row)
            if done % 25 == 0:
                print(f"{done}/{len(tasks)} plans evaluated", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    write_csv(os.path.join(OUT_DIR, "plan_sensitivity.csv"), rows)
    write_csv(os.path.join(OUT_DIR, "matched_histories.csv"), summarize_histories(rows))
    write_csv(os.path.join(OUT_DIR, "adam_noise_models.csv"), summarize_adam(rows))
    write_csv(os.path.join(OUT_DIR, "adam_noise_models_nominal_leakage.csv"),
              summarize_adam(rows, "adam_nominal_leakage"))
    pulses = figure_pulses()
    write_csv(os.path.join(OUT_DIR, "correlation_time.csv"), correlation_time_scan(pulses))
    write_csv(os.path.join(OUT_DIR, "filter_functions.csv"), filter_function_table(pulses))
    print("written to", OUT_DIR)


if __name__ == "__main__":
    main()
