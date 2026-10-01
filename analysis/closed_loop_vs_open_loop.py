"""Closed-loop versus open-loop robustness of the two matched TRPO agents.

During noise-aware training the agent observes the propagator produced by the
*noisy* dynamics, so it can learn to react to the realized noise. A deployed pulse
cannot: it is the open-loop plan of the mean action in the noise-free environment,
replayed under noise (this is what benchmark_robustness.py and the paper evaluate).

For every checkpoint of both agents in the matched phase (iterations 21-100), and
for the two controllers selected in the paper (nominal: iteration 111; noise-trained:
iteration 11), this script measures at sigma = 1 MHz
  * open loop:   the stored plan replayed under noise (uqc.eval.robustness_metrics);
  * closed loop: the deterministic policy acting in the noisy environment for the
                 same number of steps as its plan, observing the noisy propagator.
Both use the training noise model and SAMPLES realizations. The script first checks
that the noise-free deterministic rollout of each checkpoint reproduces its stored
plan exactly.

Outputs (final_results/closed_loop_results/):
  closed_loop_vs_open_loop.csv          one row per (agent, iteration)
  closed_loop_vs_open_loop_summary.csv  per agent and mode, over iterations 21-100: box-plot
      statistics of the average fidelity; per comparison: the mean paired difference with a
      95% block-bootstrap interval (8 blocks of 10 iterations, fixed seed)
Run from anywhere: python analysis/closed_loop_vs_open_loop.py [--workers N] [--summary-only]
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

import torch  # noqa: E402

torch.set_num_threads(1)

from uqc.env import EnvConfig, QuantumControlEnv  # noqa: E402
from uqc.eval import ControlPlan, robustness_metrics  # noqa: E402
from uqc.physics import GmonSystem, GmonSystemConfig, evaluate_average_fidelity  # noqa: E402
from uqc.trpo import TRPOAgent, TRPOConfig  # noqa: E402

RUNS = {"nominal": "final_results/nominal", "noise": "final_results/trpo_noise_alpha_2.2"}
ITERATIONS = {"nominal": list(range(21, 101)) + [111], "noise": [11] + list(range(21, 101))}
SIGMA = 1.0
SAMPLES = 100
SEED = 11
OUT_DIR = "final_results/closed_loop_results"
MATCHED = range(21, 101)
BOOT_BLOCK, BOOT_SAMPLES, BOOT_SEED = 10, 4000, 0


def load_agent(path: str, obs_dim: int) -> TRPOAgent:
    agent = TRPOAgent(obs_dim, 7, config=TRPOConfig(hidden_sizes=(64, 32, 32)))
    state = torch.load(path, map_location="cpu")
    agent.load_state_dict(state["agent_state"])
    return agent


def make_env(system, plan, noisy: bool, seed: int) -> QuantumControlEnv:
    return QuantumControlEnv(system, EnvConfig(
        target_alpha=plan.target_alpha, target_gamma=plan.target_gamma, max_time_ns=180.0,
        dt_ns=plan.dt_ns, noise_optimized=noisy, train_noise_std_mhz=SIGMA,
        termination_cost=-1.0, runtime_norm_ns=plan.runtime_norm_ns,
        cost_weights=plan.cost_weights, seed=seed))


def run(env: QuantumControlEnv, agent: TRPOAgent, steps: int) -> np.ndarray:
    obs = env.reset()
    for _ in range(steps):
        action, _ = agent.get_action(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
    return env.U


def evaluate(task):
    agent_name, iteration = task
    run_dir = RUNS[agent_name]
    plan = ControlPlan.load(os.path.join(run_dir, "plans", f"iter_{iteration:06d}_control_plan.npz"))
    assert abs(plan.target_alpha - 2.2) < 1e-9
    system = GmonSystem(GmonSystemConfig(dt_ns=plan.dt_ns, runtime_norm_ns=plan.runtime_norm_ns))
    steps = plan.controls_mhz_and_phase.shape[0]
    env0 = make_env(system, plan, noisy=False, seed=0)
    agent = load_agent(os.path.join(run_dir, "checkpoints", f"iter_{iteration:06d}.pt"), env0.observation_dim)

    run(env0, agent, steps)
    reproduced = np.asarray(env0.nominal_controls_history)
    if not np.allclose(reproduced, plan.controls_mhz_and_phase, atol=1e-6, rtol=0):
        raise RuntimeError(f"{agent_name} iteration {iteration}: checkpoint does not reproduce its plan")

    target = system.target_gate(plan.target_alpha, plan.target_gamma)
    env1 = make_env(system, plan, noisy=True, seed=SEED)
    projected = [system.projected_unitary(run(env1, agent, steps)) for _ in range(SAMPLES)]
    fids = np.array([system.gate_fidelity_from_projected(K, target) for K in projected])
    closed_avg = evaluate_average_fidelity(system, target, projected)
    open_m = robustness_metrics(system, plan, sigma_mhz=SIGMA, num_samples=SAMPLES, seed=SEED)
    return {
        "agent": agent_name, "iteration": iteration, "sigma_mhz": SIGMA, "num_samples": SAMPLES,
        "time_ns": steps * plan.dt_ns,
        "open_loop_average_fidelity": float(open_m["average_fidelity"]),
        "open_loop_fidelity_variance": float(open_m["fidelity_variance"]),
        "closed_loop_average_fidelity": float(closed_avg),
        "closed_loop_fidelity_variance": float(np.mean((fids - fids.mean()) ** 2)),
    }


def block_ci(diff: np.ndarray) -> tuple[float, float]:
    blocks = diff.reshape(-1, BOOT_BLOCK)
    rng = np.random.default_rng(BOOT_SEED)
    boot = [blocks[rng.integers(0, len(blocks), len(blocks))].mean() for _ in range(BOOT_SAMPLES)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def summarize(rows):
    series = {}
    for agent in RUNS:
        sel = sorted((r for r in rows if r["agent"] == agent and r["iteration"] in MATCHED),
                     key=lambda r: r["iteration"])
        for mode in ("open_loop", "closed_loop"):
            series[(agent, mode)] = np.array([r[f"{mode}_average_fidelity"] for r in sel])
    out = []
    for position, (agent, mode) in enumerate(series, start=1):
        x = series[(agent, mode)]
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        out.append({"kind": "distribution", "position": position, "label": f"{agent}_{mode}",
                    "min": float(x.min()), "q1": float(q1), "median": float(med), "q3": float(q3),
                    "max": float(x.max()), "mean_difference": "", "ci95_low": "", "ci95_high": "",
                    "fraction_positive": ""})
    comparisons = [
        ("noise_minus_nominal_open_loop", series[("noise", "open_loop")] - series[("nominal", "open_loop")]),
        ("noise_minus_nominal_closed_loop", series[("noise", "closed_loop")] - series[("nominal", "closed_loop")]),
        ("nominal_closed_minus_open", series[("nominal", "closed_loop")] - series[("nominal", "open_loop")]),
        ("noise_closed_minus_open", series[("noise", "closed_loop")] - series[("noise", "open_loop")]),
    ]
    for label, diff in comparisons:
        lo, hi = block_ci(diff)
        out.append({"kind": "paired_difference", "position": "", "label": label, "min": "", "q1": "",
                    "median": "", "q3": "", "max": "", "mean_difference": float(diff.mean()),
                    "ci95_low": lo, "ci95_high": hi, "fraction_positive": float(np.mean(diff > 0))})
    return out


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(float(v)) if isinstance(v, float) else v) for k, v in r.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--summary-only", action="store_true",
                    help="recompute the summary from an existing closed_loop_vs_open_loop.csv")
    args = ap.parse_args()
    path = os.path.join(OUT_DIR, "closed_loop_vs_open_loop.csv")
    if args.summary_only:
        with open(path, newline="", encoding="utf-8") as f:
            rows = []
            for r in csv.DictReader(f):
                for k in r:
                    if k in ("iteration", "num_samples"):
                        r[k] = int(r[k])
                    elif k != "agent":
                        r[k] = float(r[k])
                rows.append(r)
    else:
        tasks = [(a, it) for a in RUNS for it in ITERATIONS[a]]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(evaluate, tasks))
        rows.sort(key=lambda r: (r["agent"], r["iteration"]))
        os.makedirs(OUT_DIR, exist_ok=True)
        write_csv(path, rows)
    write_csv(os.path.join(OUT_DIR, "closed_loop_vs_open_loop_summary.csv"), summarize(rows))
    for agent in RUNS:
        sel = [r for r in rows if r["agent"] == agent and 21 <= r["iteration"] <= 100]
        o = np.median([r["open_loop_average_fidelity"] for r in sel])
        c = np.median([r["closed_loop_average_fidelity"] for r in sel])
        print(f"{agent:8s} iterations 21-100: median open-loop Fbar={o:.4f}  closed-loop Fbar={c:.4f}")
    print("written", path)


if __name__ == "__main__":
    main()
