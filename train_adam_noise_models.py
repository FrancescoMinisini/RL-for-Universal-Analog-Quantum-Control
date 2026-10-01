"""Adam at fixed horizons under three training-noise models.

For one target N(alpha, alpha, gamma) and every horizon of --horizons-ns, the pulse is optimized
with the Adam baseline (uqc.baseline_adam: same simulator, filter and UFO cost as everywhere else)
  none          on the deterministic objective (one run per horizon: with the single deterministic
                initialization of the baseline the result does not depend on the seed);
  white         under per-step Gaussian noise of standard deviation --white-noise-std, the UFO
                training noise;
  quasi_static  under one Gaussian offset per channel, held for the whole pulse, of standard
                deviation --quasi-static-noise-std,
with one run per seed for the two stochastic models. Each run is one call of
TorchGmonObjective.optimize_for_horizon and keeps, as the baseline does, the iterate with the
lowest sampled cost. Fixing the horizon removes the duration from the comparison, and unlike
train_adam_baseline.py every plan is stored.

By default every term of the objective is evaluated on the noisy trajectory, as in the TRPO
environment, including the leakage bound, which responds strongly to white noise. With
--leakage-bound nominal the bound is accumulated from the noise-free controls instead, so that the
noise enters the objective through the fidelity only.

Output (independent of the number of workers):
  args.json
  summary.csv          one row per run: the nominal re-evaluation of its plan
  training_log.jsonl   one record per Adam step and one per run
  plans/<model>_h<horizon>_seed<seed>_control_plan.npz
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Tuple

import torch

torch.set_num_threads(1)

from uqc.baseline_adam import AdamBaselineConfig, TorchGmonObjective
from uqc.eval import ControlPlan, simulate_nominal_plan
from uqc.physics import GmonSystem, GmonSystemConfig, UFOCostWeights
from uqc.utils import JsonlLogger, ensure_dir, parse_angle_expr

MODELS = ("none", "white", "quasi_static")


def plan_name(model: str, horizon_ns: float, seed: int) -> str:
    return f"{model}_h{int(round(horizon_ns)):03d}_seed{seed}_control_plan.npz"


def run_one(task: Tuple[str, float, int, Dict[str, Any]]) -> Dict[str, Any]:
    model, horizon_ns, seed, cfg = task
    torch.set_num_threads(1)
    weights = UFOCostWeights(chi=cfg["cost_chi"], beta=cfg["cost_beta"], mu=cfg["cost_mu"], kappa=cfg["cost_kappa"])
    sigma = {"none": 0.0, "white": cfg["white_noise_std"], "quasi_static": cfg["quasi_static_noise_std"]}[model]
    baseline_cfg = AdamBaselineConfig(
        dt_ns=cfg["dt_ns"],
        runtime_norm_ns=cfg["runtime_norm_ns"],
        lr=cfg["lr"],
        adam_iters=cfg["adam_iters"],
        horizons_ns=(horizon_ns,),
        train_noise_std_mhz=sigma,
        train_noise_mode="quasi_static" if model == "quasi_static" else "white",
        leakage_from_nominal_controls=cfg["leakage_bound"] == "nominal",
        seed=seed,
        cost_weights=weights,
    )
    t0 = time.time()
    result = TorchGmonObjective(baseline_cfg).optimize_for_horizon(cfg["alpha"], cfg["gamma"], horizon_ns)
    elapsed = time.time() - t0

    plan = ControlPlan(
        controls_mhz_and_phase=result["controls"],
        target_alpha=cfg["alpha"],
        target_gamma=cfg["gamma"],
        dt_ns=cfg["dt_ns"],
        runtime_norm_ns=cfg["runtime_norm_ns"],
        cost_weights=weights,
        note=f"Adam, training noise model {model}, horizon {horizon_ns} ns, seed {seed}",
    )
    name = plan_name(model, horizon_ns, seed)
    plan.save(os.path.join(cfg["out"], "plans", name))
    system = GmonSystem(GmonSystemConfig(dt_ns=cfg["dt_ns"], runtime_norm_ns=cfg["runtime_norm_ns"]))
    nominal = simulate_nominal_plan(system, plan)
    return {
        "model": model,
        "horizon_ns": float(horizon_ns),
        "seed": int(seed),
        "train_noise_std_mhz": float(sigma),
        "optimizer_cost": float(result["best_cost"]),
        "nominal_cost": float(nominal["cost"]),
        "nominal_fidelity": float(nominal["fidelity"]),
        "nominal_leakage": float(nominal["leakage"]),
        "nominal_leakage_boundary": float(nominal["leakage_boundary"]),
        "nominal_leakage_integral": float(nominal["leakage_integral"]),
        "nominal_boundary_cost": float(nominal["boundary_cost"]),
        "nominal_time_cost": float(nominal["time_cost"]),
        "nominal_time_ns": float(nominal["time_ns"]),
        "plan": f"plans/{name}",
        "history": result["history"],
        "elapsed_seconds": round(elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Adam at fixed horizons under three training-noise models.")
    parser.add_argument("--alpha", type=str, default="2.2")
    parser.add_argument("--gamma", type=str, default="pi/2")
    parser.add_argument("--horizons-ns", type=str, default="70,100,130")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5,6,7,8")
    parser.add_argument("--white-noise-std", type=float, default=1.0, help="MHz, redrawn at every step.")
    parser.add_argument("--quasi-static-noise-std", type=float, default=0.3, help="MHz, one draw per trajectory.")
    parser.add_argument(
        "--leakage-bound",
        choices=["noisy", "nominal"],
        default="noisy",
        help="noisy: the leakage bound of the objective is accumulated from the noisy Hamiltonian, as in the "
             "TRPO environment; nominal: from the noise-free controls.",
    )
    parser.add_argument("--dt-ns", type=float, default=2.0)
    parser.add_argument("--runtime-norm-ns", type=float, default=60.0)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--adam-iters", type=int, default=400)
    parser.add_argument("--cost-chi", type=float, default=10.0, help="Weight for fidelity cost")
    parser.add_argument("--cost-beta", type=float, default=10.0, help="Weight for leakage cost")
    parser.add_argument("--cost-mu", type=float, default=0.2, help="Weight for boundary cost")
    parser.add_argument("--cost-kappa", type=float, default=0.1, help="Weight for time cost")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    horizons = [float(x) for x in args.horizons_ns.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    ensure_dir(os.path.join(args.out, "plans"))
    with open(os.path.join(args.out, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    cfg = dict(vars(args))
    cfg["alpha"] = parse_angle_expr(args.alpha)
    cfg["gamma"] = parse_angle_expr(args.gamma)
    tasks: List[Tuple[str, float, int, Dict[str, Any]]] = []
    for horizon in horizons:
        tasks.append(("none", horizon, seeds[0], cfg))
        for model in ("white", "quasi_static"):
            tasks.extend((model, horizon, seed, cfg) for seed in seeds)

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        for done, row in enumerate(ex.map(run_one, tasks), start=1):
            rows.append(row)
            print(
                f"[{done}/{len(tasks)}] {row['model']:12s} {row['horizon_ns']:5.0f} ns seed {row['seed']}: "
                f"F={row['nominal_fidelity']:.5f} cost={row['nominal_cost']:.4f} ({row['elapsed_seconds']:.0f} s)",
                flush=True,
            )
    rows.sort(key=lambda r: (MODELS.index(r["model"]), r["horizon_ns"], r["seed"]))

    log_path = os.path.join(args.out, "training_log.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = JsonlLogger(log_path)
    for row in rows:
        best_cost_so_far = float("inf")
        for entry in row["history"]:
            best_cost_so_far = min(best_cost_so_far, float(entry["cost"]))
            logger.write({
                "type": "adam_step",
                "model": row["model"],
                "horizon_ns": row["horizon_ns"],
                "seed": row["seed"],
                "adam_iter": int(entry["iter"]),
                "cost": float(entry["cost"]),
                "best_cost_so_far": best_cost_so_far,
            })
        logger.write({"type": "run_summary", **{k: v for k, v in row.items() if k != "history"}})

    columns = [k for k in rows[0] if k not in ("history", "elapsed_seconds")]
    with open(os.path.join(args.out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (repr(row[k]) if isinstance(row[k], float) else row[k]) for k in columns})
    print(f"written {len(rows)} runs to {args.out}", flush=True)


if __name__ == "__main__":
    main()
