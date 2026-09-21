# final_results — data behind the thesis and the preprint

Licensed under CC BY 4.0 (see [`../LICENSE-DATA`](../LICENSE-DATA)).

This folder holds the training runs retained for the thesis and the processed data derived from them.
It contains two kinds of directories:

- **Run directories** (`nominal/`, `noise/`, `adam_noise/`, `runtime/`, the two sweeps, …) — raw output of
  the training and evaluation scripts at the repository root, copied here from the working folder `runs/`.
- **`*_results` directories** — CSVs and reference PNGs written by the scripts in [`../analysis/`](../analysis/).
  Every CSV the thesis plots is a byte-identical copy of a file here (it lives under `data/` in the
  [thesis repository](https://github.com/FrancescoMinisini/Document)). Re-running the analysis scripts
  regenerates each of these CSVs byte-for-byte.

## From figure to data

Thesis figure sources are the `plots/*.tex` files of the thesis repository.

| Thesis figure | Data file | Written by | Computed from |
| --- | --- | --- | --- |
| `nominal_trpo_training_curves` | `trpo_nominal_results/nominal_trpo_training_curves.csv` | `analysis/generate_nominal_trpo_plots.py` | `nominal/training_log.jsonl` |
| `single_target_adam_vs_nominal_trpo` | values typed into the figure | — | Adam: 70 ns row of `adam_noise/horizon_search.csv`; TRPO: `nominal/summary.json` |
| `adam_single_target_horizon_sweep` | `adam_results/adam_horizon_sweep.csv` | `analysis/generate_adam_plots.py` | `adam_noise/horizon_search.csv` |
| `adam_family_sweep` | `adam_results/adam_family_sweep.csv` | `analysis/generate_adam_plots.py` | `adam_runtime_sweep/runtime_summary.csv` |
| `architecture_sweep_*` | `nn_size_sweep_results/sweep_complete_metrics.csv` | `analysis/analyze_nn_size_sweep.py` | `nn_size_sweep/` |
| `ufo_weight_sweep_*` | `cost_function_sweep_results/sweep_complete_metrics.csv` | `analysis/analyze_cost_sweep.py` | `cost_function_sweep/` |
| `runtime_vs_alpha_curriculum` | `runtime_results/phase_final_results_log.csv` | `analysis/analyze_runtime.py` | `runtime/` |
| `avg_fidelity_vs_noise`, `fidelity_variance_vs_noise` | `robustness_analysis/combined_ewma_data.csv` | `analysis/export_ewma_data.py` | `robustness_analysis/{noise,nominal,adam_noise}/robustness_curve.csv` |

The boxplot figures (`architecture_sweep_robustness`, `architecture_sweep_leakage`,
`ufo_weight_sweep_robustness`, `ufo_weight_sweep_leakage`) carry pre-computed box statistics in their
TikZ source. They match `sweep_complete_metrics.csv` exactly: minimum, median and maximum per group,
with quartiles computed by linear interpolation for the architectures and as Tukey hinges for the UFO
weights. `adam_vs_nominal_results/adam_vs_nominal_comparison.csv` holds an earlier comparison against
the 60 ns Adam solution and is not plotted.

## Retained runs

Every run directory written by a training script contains `args.json` (the exact command-line
arguments), `summary.json`, `training_log.jsonl` (one JSON record per iteration), `checkpoints/`
(`.pt` agent states), `plans/` (`.npz` control plans) and `robustness/` (per-iteration robustness at
σ = 1 MHz). Load a control plan with `uqc.eval.ControlPlan.load`.

Unless stated otherwise, the runs target `N(2.2, 2.2, π/2)`. Every run with an `args.json` uses
`dt = 2 ns`, a 180 ns horizon, and (outside the weight sweep) the UFO weights (χ, β, μ, κ) = (10, 10, 0.2, 0.1).

| Directory | Content | Script and key settings |
| --- | --- | --- |
| `nominal/` | TRPO trained without noise | `train_trpo_single_target.py`, 115 iterations × 5000 episodes, termination cost 0.3, seed 1 (see `args.json`) |
| `noise/` | TRPO trained with 1 MHz Gaussian noise | curriculum TRPO, 130 iterations. No `args.json` was stored. Its checkpoint paths point into `runs/runtime/`, and the author's note `runtime/Ricordo params` lists the cost thresholds used in successive training stages ("fino a 100": 0.0, "fino a 115": 0.2, "fino a 130": 0.18). |
| `adam_noise/` | Adam baseline optimized under 1 MHz noise; horizon search 40–180 ns | `train_adam_baseline.py`, 400 steps, lr 0.03, seed 2 |
| `adam_test/` | second Adam plan, evaluated in `robustness_analysis_2/` | summary and plan only |
| `adam_runtime_sweep/` | Adam over the family `N(α, α, π/2)`, α = 0 … π in steps of 0.1 | `train_adam_runtime.py` |
| `runtime/` | curriculum TRPO over `N(α, α, π/2)`, α = 0 … 3.14 in steps of 0.157, trained with noise | `train_trpo_runtime.py`, ≤ 150 iterations × 4000 episodes per α, termination and advance threshold 0.4, seed 2 |
| `cost_function_sweep/` | UFO-weight sweep: χ ∈ {5, 10, 20}, μ ∈ {0.1, 0.2, 0.4}, κ ∈ {0.05, 0.1, 0.2, 0.4}, 2 seeds (72 runs) | `run_param_search.py --mode trpo`, noise-optimized, 5 iterations × 10 000 episodes |
| `nn_size_sweep/` | actor–critic architectures 64-32-32, 64-64-64, 64-64-64-32, 128-64-32, 128-64-32-32, 3 seeds (15 runs) | `run_param_search.py --mode trpo`, noise-optimized, 20 iterations × 5000 episodes |

## Robustness evaluation

`robustness_analysis/` holds the unmodified output of `benchmark_robustness.py` for the best control
plans of `noise/`, `nominal/` and `adam_noise/`: 3401 noise levels σ = 0.1 … 3.5 MHz in steps of
0.001 MHz, 60 Monte Carlo rollouts per level, seed 1. `robustness_analysis_2/` repeats the evaluation on
a coarser grid (341 levels in steps of 0.01 MHz) with the `adam_test/` plan in place of `adam_noise/`.
Each `robustness_curve.csv` has the columns `method, sigma_mhz, num_samples, average_fidelity,
average_gate_fidelity, fidelity_variance`. `combined_ewma_data.csv` adds an exponentially weighted
moving average along σ (span 50 for the fidelity, 80 for the variance) next to the raw columns.

Earlier post-processed variants of these curves were removed from the repository in commit `f37f909`.
The figures in the thesis and the preprint use only the unmodified data described here.
