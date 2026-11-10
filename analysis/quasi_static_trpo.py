"""TRPO trained under quasi-static noise: robust pulse or feedback?

Evaluates the runs of run_matched_retraining.py --experiment quasi_static_40mhz (40 MHz filter,
seeds 1-3), three arms per seed that differ only in the training noise and in what the policy
observes:

  nominal    no training noise
  open       quasi-static noise of 0.3 MHz, the policy observes the noise-free propagator, so it
             can only learn an open-loop pulse
  feedback   the same noise, the policy observes the noisy propagator

Two evaluations, for every iteration of every run:

  pulses     the stored plan (the noise-free deterministic rollout, the pulse one would deploy)
             with exactly the evaluation of noise_with_memory.py: first-order quasi-static
             excess infidelity, Monte Carlo average fidelity under quasi-static noise of 0.1 and
             0.3 MHz (same offsets for every plan), duration and nominal fidelity
  policy     at 0.3 MHz, on CLOSED_SAMPLES fixed offsets shared by every checkpoint: the
             deterministic policy acting in the noisy environment with the observation it was
             trained with (closed loop), against its own plan replayed on the same offsets
             (open loop). For the open arm the two coincide, which the script checks.

The comparisons are paired by seed and iteration over iterations 21-100, with the block
bootstrap of noise_with_memory.py. The Adam pulses of the converged 40 MHz study
(adam_qs_converged_bw40) are the reference for what a quasi-static-aware optimizer achieves.

Runs are read from final_results/<name> if it exists and from runs/<name> otherwise.
Outputs (--out, default runs/quasi_static_trpo_results/):
  plan_sensitivity.csv   one row per (arm, seed, iteration): the pulse evaluation
  policy_closed_loop.csv one row per (arm, seed, iteration): open- and closed-loop average fidelity
  arm_summary.csv        per arm and seed, iterations 21-100: medians of the main quantities
  paired_summary.csv     per seed and comparison: mean paired difference, 95% block-bootstrap
                         interval and fraction of iterations with a positive difference

Run from anywhere: python analysis/quasi_static_trpo.py [--workers N] [--seeds 1,2,3]
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

import torch  # noqa: E402

torch.set_num_threads(1)

from noise_with_memory import block_ci, write_csv  # noqa: E402
from noise_with_memory import evaluate as evaluate_plan  # noqa: E402
from uqc.env import EnvConfig, QuantumControlEnv  # noqa: E402
from uqc.eval import ControlPlan  # noqa: E402
from uqc.physics import GmonSystem, GmonSystemConfig, evaluate_average_fidelity  # noqa: E402
from uqc.trpo import TRPOAgent, TRPOConfig  # noqa: E402

ARMS = ("nominal", "open", "feedback")
RUN_NAME = "qs40_{arm}_seed{seed}"
ITERATIONS = range(1, 101)
SUMMARY_ITERATIONS = range(21, 101)
SIGMA_CLOSED = 0.3          # MHz, the training strength
CLOSED_SAMPLES = 200
CLOSED_SEED = 21
MAX_TIME_NS = 180.0
ADAM_SENSITIVITY = "runs/filter_bandwidth_results/plan_sensitivity.csv"


def run_dir(name: str) -> str:
    final = os.path.join("final_results", name)
    return final if os.path.isdir(final) else os.path.join("runs", name)


def plan_path(arm: str, seed: int, iteration: int) -> str:
    return os.path.join(run_dir(RUN_NAME.format(arm=arm, seed=seed)), "plans", f"iter_{iteration:06d}_control_plan.npz")


def evaluate_pulse(task):
    arm, seed, iteration = task
    row = evaluate_plan((arm, iteration, plan_path(arm, seed, iteration), False))
    return {"arm": arm, "seed": seed, "iteration": iteration,
            **{k: v for k, v in row.items() if k not in ("source", "label") and not k.startswith("white_")
               or k == "white_first_order_1mhz"}}


def closed_loop_offsets() -> np.ndarray:
    """[CLOSED_SAMPLES, 6] offsets (eta, delta1, delta2, f1, f2, g) in MHz, the same for every checkpoint."""
    return SIGMA_CLOSED * np.random.default_rng(CLOSED_SEED).standard_normal((CLOSED_SAMPLES, 6))


def make_env(system: GmonSystem, plan: ControlPlan, noisy: bool, observe_nominal: bool) -> QuantumControlEnv:
    return QuantumControlEnv(system, EnvConfig(
        target_alpha=plan.target_alpha, target_gamma=plan.target_gamma, max_time_ns=MAX_TIME_NS,
        dt_ns=plan.dt_ns, noise_optimized=noisy, train_noise_std_mhz=SIGMA_CLOSED,
        train_noise_mode="quasi_static", observe_nominal=observe_nominal,
        filter_bandwidth_mhz=plan.filter_bandwidth_mhz, termination_cost=-1.0,
        runtime_norm_ns=plan.runtime_norm_ns, cost_weights=plan.cost_weights, seed=0))


def evaluate_policy(task):
    arm, seed, iteration = task
    directory = run_dir(RUN_NAME.format(arm=arm, seed=seed))
    plan = ControlPlan.load(plan_path(arm, seed, iteration))
    system = GmonSystem(GmonSystemConfig(dt_ns=plan.dt_ns, runtime_norm_ns=plan.runtime_norm_ns))
    target = system.target_gate(plan.target_alpha, plan.target_gamma)
    steps = plan.controls_mhz_and_phase.shape[0]
    observe_nominal = arm == "open"

    env = make_env(system, plan, noisy=False, observe_nominal=False)
    agent = TRPOAgent(env.observation_dim, 7, config=TRPOConfig(hidden_sizes=(64, 32, 32)))
    agent.load_state_dict(torch.load(os.path.join(directory, "checkpoints", f"iter_{iteration:06d}.pt"),
                                     map_location="cpu")["agent_state"])

    # noise-free deterministic rollout: must reproduce the stored plan; its actions are the open loop
    obs = env.reset()
    actions = []
    for _ in range(steps):
        action, _ = agent.get_action(obs, deterministic=True)
        actions.append(action)
        obs, _, _, _ = env.step(action)
    if not np.allclose(np.asarray(env.nominal_controls_history), plan.controls_mhz_and_phase, atol=1e-6, rtol=0):
        raise RuntimeError(f"{arm} seed {seed} iteration {iteration}: checkpoint does not reproduce its plan")

    env = make_env(system, plan, noisy=True, observe_nominal=observe_nominal)
    closed, opened = [], []
    for offsets in closed_loop_offsets():
        for mode, out in (("closed", closed), ("open", opened)):
            obs = env.reset()
            env.quasi_static_offsets = offsets.copy()
            for t in range(steps):
                action = agent.get_action(obs, deterministic=True)[0] if mode == "closed" else actions[t]
                obs, _, _, _ = env.step(action)
            out.append(system.projected_unitary(env.U))
    row = {
        "arm": arm, "seed": seed, "iteration": iteration, "time_ns": steps * plan.dt_ns,
        "sigma_mhz": SIGMA_CLOSED, "num_samples": CLOSED_SAMPLES,
        "open_loop_average_fidelity": float(evaluate_average_fidelity(system, target, opened)),
        "closed_loop_average_fidelity": float(evaluate_average_fidelity(system, target, closed)),
    }
    if observe_nominal and abs(row["open_loop_average_fidelity"] - row["closed_loop_average_fidelity"]) > 1e-9:
        raise RuntimeError(f"open arm seed {seed} iteration {iteration}: closed loop differs from open loop")
    return row


def summary_quantities():
    return {
        "time_ns": lambda r: r["time_ns"],
        "nominal_fidelity": lambda r: r["nominal_fidelity"],
        "quasi_static_first_order_0p1mhz": lambda r: r["quasi_static_first_order_0p1mhz"],
        # sensitivity with the duration divided out: the part that depends on the pulse shape
        "quasi_static_first_order_0p1mhz_per_ns2": lambda r: r["quasi_static_first_order_0p1mhz"] / r["time_ns"] ** 2,
        "quasi_static_average_fidelity_0p1mhz": lambda r: r["quasi_static_average_fidelity_0p1mhz"],
        "quasi_static_average_fidelity_0p3mhz": lambda r: r["quasi_static_average_fidelity_0p3mhz"],
    }


def series(rows, arm, seed, key):
    sel = {r["iteration"]: r for r in rows if r["arm"] == arm and r["seed"] == seed}
    return np.array([key(sel[i]) for i in SUMMARY_ITERATIONS])


def summarize(pulses, policy, seeds):
    arm_rows, paired_rows = [], []
    q_pulse = summary_quantities()
    q_policy = {
        "open_loop_average_fidelity_0p3mhz": lambda r: r["open_loop_average_fidelity"],
        "closed_loop_average_fidelity_0p3mhz": lambda r: r["closed_loop_average_fidelity"],
        "closed_minus_open_0p3mhz": lambda r: r["closed_loop_average_fidelity"] - r["open_loop_average_fidelity"],
    }
    for seed in seeds:
        for arm in ARMS:
            rec = {"arm": arm, "seed": seed, "n_iterations": len(SUMMARY_ITERATIONS)}
            for name, key in q_pulse.items():
                rec[f"{name}_median"] = float(np.median(series(pulses, arm, seed, key)))
            for name, key in q_policy.items():
                rec[f"{name}_median"] = float(np.median(series(policy, arm, seed, key)))
            arm_rows.append(rec)
        comparisons = [(f"{arm}_minus_nominal", arm, "nominal") for arm in ("open", "feedback")]
        comparisons.append(("feedback_minus_open", "feedback", "open"))
        for label, a, b in comparisons:
            for name, key in list(q_pulse.items()) + list(q_policy.items()):
                source = pulses if name in q_pulse else policy
                diff = series(source, a, seed, key) - series(source, b, seed, key)
                lo, hi = block_ci(diff)
                paired_rows.append({"seed": seed, "comparison": label, "quantity": name,
                                    "mean_difference": float(diff.mean()), "ci95_low": lo, "ci95_high": hi,
                                    "fraction_positive": float(np.mean(diff > 0))})
    return arm_rows, paired_rows


def adam_reference():
    """Shape factor of the converged 40 MHz Adam pulses (adam_qs_converged_bw40), per model and horizon."""
    if not os.path.exists(ADAM_SENSITIVITY):
        return []
    import csv
    out = {}
    with open(ADAM_SENSITIVITY, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source"] in ("c40_none", "c40_quasi_static"):
                key = (r["source"][4:], float(r["time_ns"]))
                out.setdefault(key, []).append(float(r["quasi_static_first_order_0p1mhz"]) / float(r["time_ns"]) ** 2)
    return [(model, t, float(np.mean(v))) for (model, t), v in sorted(out.items())]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--out", type=str, default="runs/quasi_static_trpo_results")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    tasks = [(arm, seed, it) for seed in seeds for arm in ARMS for it in ITERATIONS]
    missing = [t for t in tasks if not os.path.exists(plan_path(*t))]
    if missing:
        raise SystemExit(f"{len(missing)} plans missing, e.g. {plan_path(*missing[0])}")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        pulses = list(ex.map(evaluate_pulse, tasks, chunksize=4))
        print(f"{len(pulses)} pulses evaluated", flush=True)
        policy_tasks = [t for t in tasks if t[2] in SUMMARY_ITERATIONS]
        policy = list(ex.map(evaluate_policy, policy_tasks, chunksize=2))
        print(f"{len(policy)} checkpoints evaluated in closed loop", flush=True)

    arm_rows, paired_rows = summarize(pulses, policy, seeds)
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "plan_sensitivity.csv"), pulses)
    write_csv(os.path.join(args.out, "policy_closed_loop.csv"), policy)
    write_csv(os.path.join(args.out, "arm_summary.csv"), arm_rows)
    write_csv(os.path.join(args.out, "paired_summary.csv"), paired_rows)

    print("\nmedians over iterations 21-100 (shape factor = first-order QS excess at 0.1 MHz / T^2)")
    for r in arm_rows:
        print(f"  seed {r['seed']} {r['arm']:9s} T={r['time_ns_median']:5.0f} ns  1-F={1 - r['nominal_fidelity_median']:.2e}  "
              f"shape={r['quasi_static_first_order_0p1mhz_per_ns2_median']:.2e}  "
              f"Fbar(0.3, pulse)={r['quasi_static_average_fidelity_0p3mhz_median']:.4f}  "
              f"closed-open={r['closed_minus_open_0p3mhz_median']:+.4f}")
    for model, t, shape in adam_reference():
        print(f"  Adam 40 MHz {model:12s} T={t:5.0f} ns  shape={shape:.2e}")
    print("written to", args.out)


if __name__ == "__main__":
    main()
