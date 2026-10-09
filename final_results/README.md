# final_results — data behind the thesis and the preprint

Licensed under CC BY 4.0 (see [`../LICENSE-DATA`](../LICENSE-DATA)).

This folder holds the training runs retained for the thesis and the preprint, and the processed data
derived from them. It contains two kinds of directories:

- **Run directories** (`nominal/`, `trpo_noise_alpha_2.2/`, `adam_noise/`, `runtime/`, the two sweeps, …) —
  raw output of the training and evaluation scripts at the repository root, copied here from the working
  folder `runs/`.
- **`*_results` and `robustness_analysis*` directories** — CSVs and reference PNGs written by the scripts in
  [`../analysis/`](../analysis/) and by `benchmark_robustness.py`. Every CSV the thesis or the preprint plots
  is a byte-identical copy of a file here (it lives under `data/` in the
  [thesis repository](https://github.com/FrancescoMinisini/Document)). Re-running the scripts regenerates
  each of these CSVs byte-for-byte.

## Two runs that do not target N(2.2, 2.2, π/2)

- **`noise/`** was presented in the thesis as the noise-trained TRPO controller for `N(2.2, 2.2, π/2)`.
  It is the α = 0 phase of the curriculum sweep (`runtime/gamma_1.570796/alpha_0.000000/`): its control
  plans and checkpoints target `N(0, 0, π/2) = i σᶻσᶻ`, and its best plan reaches fidelity 0.117 on the
  α = 2.2 gate. Its `summary.json` and `training_log.jsonl` nevertheless carry `alpha: 2.2`.
  `benchmark_robustness.py` scores each plan against its own target, so the `noise` curves in
  `robustness_analysis/` and `robustness_analysis_2/` describe a controller for a different, local gate.
  The directory is kept because the submitted thesis and the slides plot it.
- **`adam_test/`**, the Adam plan of `robustness_analysis_2/`, targets `N(0.2, 0.2, π/2)`.

The preprint uses neither. Its noise-trained controller is `trpo_noise_alpha_2.2/`, and its robustness data
are in `robustness_analysis_3/`, where every plan targets `N(2.2, 2.2, π/2)`.

## From figure to data

Thesis figure sources are the `plots/*.tex` files of the thesis repository; preprint figure sources are
`paper/figsrc/*.tex` there.

| Figure | Data file | Written by | Computed from |
| --- | --- | --- | --- |
| thesis `nominal_trpo_training_curves` | `trpo_nominal_results/nominal_trpo_training_curves.csv` | `analysis/generate_nominal_trpo_plots.py` | `nominal/training_log.jsonl` |
| thesis `single_target_adam_vs_nominal_trpo` | values typed into the figure | — | Adam: 70 ns row of `adam_noise/horizon_search.csv`; TRPO: `nominal/summary.json` |
| thesis and preprint `adam_single_target_horizon_sweep` | `adam_results/adam_horizon_sweep.csv` | `analysis/generate_adam_plots.py` | `adam_noise/horizon_search.csv` |
| thesis and preprint `adam_family_sweep` | `adam_results/adam_family_sweep.csv` | `analysis/generate_adam_plots.py` | `adam_runtime_sweep/runtime_summary.csv` |
| `architecture_sweep_*` | `nn_size_sweep_results/sweep_complete_metrics.csv` | `analysis/analyze_nn_size_sweep.py` | `nn_size_sweep/` |
| `ufo_weight_sweep_*` | `cost_function_sweep_results/sweep_complete_metrics.csv` | `analysis/analyze_cost_sweep.py` | `cost_function_sweep/` |
| thesis `runtime_vs_alpha_curriculum` | `runtime_results/phase_final_results_log.csv` | `analysis/analyze_runtime.py` | `runtime/` |
| thesis `avg_fidelity_vs_noise`, `fidelity_variance_vs_noise` | `robustness_analysis/combined_ewma_data.csv` | `analysis/export_ewma_data.py` | `robustness_analysis/{noise,nominal,adam_noise}/robustness_curve.csv` (the `noise` plan is for α = 0, see above) |
| preprint Table I (and `single_target_comparison_v3`, no longer included in the preprint) | values typed into the figure | — | nominal re-evaluation of `adam_noise_70ns/`, `adam_noise/`, `nominal/` and `trpo_noise_alpha_2.2/plans/iter_000011_control_plan.npz` |
| preprint `robustness_v3` (Fig. 1) | `robustness_analysis_3/combined_ewma_data.csv` | `analysis/export_ewma_data.py` | `robustness_analysis_3/*/robustness_curve.csv` |
| preprint Table II and robustness numbers | `robustness_analysis_3/representative_values.csv`, `crossovers.csv` | `analysis/summarize_robustness_3.py` | `robustness_analysis_3/combined_ewma_data.csv` |
| preprint `noise_training_mechanism` (Fig. 2b) | `trajectory_robustness_results/trajectory_robustness_summary.csv` | `analysis/robustness_training_trajectories.py` | `nominal/plans/`, `trpo_noise_alpha_2.2/plans/` (iterations 1–100) |
| preprint matched comparison repeated over three seeds with the leakage bound from the noise-free controls (Sec. V D) | `matched_retraining_results/{seed_summary,per_seed_summary,plan_statistics,trajectory_robustness}.csv` | `analysis/matched_retraining_robustness.py` | `matched_retraining/*/plans/` (iterations 1–100 of six runs) |
| preprint `noise_training_mechanism` (Fig. 2c) | `closed_loop_results/closed_loop_vs_open_loop_summary.csv` | `analysis/closed_loop_vs_open_loop.py` | checkpoints and plans of the same two runs |
| preprint `noise_training_mechanism` (Fig. 2a), Eq. (18) | `white_noise_results/excess_vs_duration.csv`, `first_order_rate.csv` | `analysis/white_noise_first_order.py` | the trajectory evaluation and `robustness_analysis_3/` |
| preprint numbers on the noisy leakage bound (Appendix B) | `noisy_leakage_results/noisy_leakage_bound.csv` | `analysis/analyze_noisy_leakage.py` | the four single-target plans |
| preprint scaling of the noisy leakage bound with the time step (Appendix B) | `noisy_leakage_results/noisy_leakage_time_step.csv` | `analysis/noisy_leakage_time_step.py` | the four single-target plans, replayed on grids 1, 2, 4 and 8 times finer |
| preprint conditional-phase numbers (Supplemental Material, Sec. S7) | `gate_structure_results/conditional_phase.csv` | `analysis/conditional_phase.py` | the noise-free plans of `adam_noise_models/` and four single-target plans |
| preprint `runtime_vs_alpha_v4` (Fig. 4), runtime table (Supplemental Material, Table S3) | `gate_structure_results/runtime_vs_gate_structure.csv` | `analysis/analyze_gate_structure.py` | `runtime_results/phase_final_results_log.csv`, `adam_results/adam_family_sweep.csv` |
| preprint `runtime_vs_alpha_v4`, grey band, and the bandwidth-limit numbers | `gate_structure_results/exchange_area_band.csv`, `runtime_vs_exchange_area.csv`, `reference_rotation_times.csv`, `exchange_area_capacity.csv` | `analysis/exchange_area_bound.py` | the two runtime sweeps and their stored plans |
| preprint `noise_with_memory` (Fig. 3) and Table IV | `noise_memory_results/{filter_functions,correlation_time,plan_sensitivity,adam_noise_models,adam_noise_models_nominal_leakage,matched_histories}.csv` | `analysis/noise_with_memory.py` | `adam_noise_models/`, `adam_noise_models_nominal_leakage/` (table rows marked with an asterisk), the five single-target plans, `nominal/plans/`, `trpo_noise_alpha_2.2/plans/` |
| preprint per-channel table, exact first-order rate, noise-interval and pre-filter numbers | `white_noise_results/{channel_resolved,exact_rate,noise_interval,prefilter_noise}.csv` | `analysis/white_noise_checks.py` | the five single-target plans and both plan histories |
| preprint simulator-step counts | `compute_budget_results/compute_budget.csv` | `analysis/compute_budget.py` | training logs of `nominal/`, `trpo_noise_alpha_2.2/`, `adam_noise/` |
| preprint mirror-symmetry numbers (Sec. VI C) | `gate_structure_results/curriculum_mirror_check.csv` | `analysis/analyze_mirror_symmetry.py` | the final-rollout plans in `runtime/` |

The boxplot figures (`architecture_sweep_robustness`, `architecture_sweep_leakage`,
`ufo_weight_sweep_robustness`, `ufo_weight_sweep_leakage`) carry pre-computed box statistics in their
TikZ source. They match `sweep_complete_metrics.csv` exactly: minimum, median and maximum per group,
with quartiles computed by linear interpolation for the architectures and as Tukey hinges for the UFO
weights. `adam_vs_nominal_results/adam_vs_nominal_comparison.csv` holds an earlier comparison against
the 60 ns Adam solution and is not plotted.

## Retained runs

Every run directory written by a training script contains `args.json` (the arguments of its last
invocation), `summary.json`, `training_log.jsonl` (one JSON record per iteration), `checkpoints/`
(`.pt` agent states), `plans/` (`.npz` control plans) and `robustness/` (per-iteration robustness at
σ = 1 MHz). Load a control plan with `uqc.eval.ControlPlan.load`. The checkpoint and plan of iteration
*k* belong to the same agent.

Unless stated otherwise, the runs target `N(2.2, 2.2, π/2)`. Every run with an `args.json` uses
`dt = 2 ns`, a 180 ns horizon, and (outside the weight sweep) the UFO weights (χ, β, μ, κ) = (10, 10, 0.2, 0.1).

| Directory | Content | Script and key settings |
| --- | --- | --- |
| `nominal/` | TRPO trained without noise | `train_trpo_single_target.py`, seed 1. Iterations 1–100 were run with 10 000 episodes per batch and termination cost 0.4 (in `runs/trpo_nominal_alpha_2.2_gamma_pi2_10k_episodes_.4ct`, whose plans 1–100 are copied here); iterations 101–115 with 5000 episodes and termination cost 0.3 (`args.json` records only this last stage). Best plan: iteration 111. |
| `trpo_noise_alpha_2.2/` | TRPO trained with 1 MHz Gaussian noise | `train_trpo_single_target.py --noise-optimized`, seed 1, 10 000 episodes per batch. Iterations 1–100 with termination cost 0.4 (resumed once from iteration 53), iterations 101–105 with 0.3. Copied from `runs/trpo_noise_alpha_2.2_gamma_pi2_10k_episodes_2nd_0.4ec_2nd`. `best_control_plan.npz` is iteration 91, because the resume restarted the best-tracking; the lowest nominal cost of the whole run is at iteration 11, the plan the preprint reports. |
| `matched_retraining/` | six TRPO runs: `matched_nominal_seed{1,2,3}` trained without noise and `matched_noise_nomleak_seed{1,2,3}` trained with 1 MHz Gaussian noise and the leakage bound of the training cost accumulated from the noise-free controls. Within a seed the two differ only in the training environment | `run_matched_retraining.py`: `train_trpo_single_target.py --engine batched --update-device cuda` (and `--noise-optimized --leakage-bound nominal`), 100 iterations × 10 000 episodes, termination cost 0.4; copied from `runs/matched_*`. The batched engine has the physics of the scalar one (`analysis/batched_env_checks.py`) but different random streams, so seed 1 here is not the seed 1 of `nominal/` and `trpo_noise_alpha_2.2/` |
| `noise/` | **not an α = 2.2 controller** (see above) | curriculum TRPO, α = 0 phase, 130 iterations. No `args.json` was stored. The author's note `runtime/Ricordo params` lists the cost thresholds used in successive training stages ("fino a 100": 0.0, "fino a 115": 0.2, "fino a 130": 0.18). |
| `adam_noise/` | Adam baseline optimized under 1 MHz noise (one noisy trajectory per step); horizon search 40–180 ns | `train_adam_baseline.py`, 400 steps, lr 0.03, seed 2. Only the plan with the lowest sampled loss (60 ns) is stored. |
| `adam_noise_70ns/` | the 70 ns horizon of `adam_noise/`, re-run alone to store its plan | same command with `--horizons-ns 70`; its `horizon_search.csv` matches the 70 ns row of `adam_noise/` to 13 significant digits |
| `adam_noise_models/` | Adam at fixed horizons (70, 100, 130 ns) trained without noise, under 1 MHz white noise and under 0.3 MHz quasi-static noise (one offset per channel and trajectory); 8 seeds per stochastic model, 51 plans in `plans/` | `train_adam_noise_models.py`, 400 steps, lr 0.03; copied from `runs/adam_noise_models` |
| `adam_noise_models_nominal_leakage/` | the same 51 runs with the leakage bound of the objective accumulated from the noise-free controls instead of the noisy Hamiltonian | `train_adam_noise_models.py --leakage-bound nominal`; copied from `runs/adam_noise_models_nominal_leakage`. The `args.json` of `adam_noise_models/` predates this option, whose default (`noisy`) reproduces that run |
| `adam_test/` | second Adam plan, evaluated in `robustness_analysis_2/`; **targets α = 0.2** | summary and plan only |
| `adam_runtime_sweep/` | Adam over the family `N(α, α, π/2)`, α = 0 … π in steps of 0.1, trained under 1 MHz noise, horizon chosen by the lowest sampled loss | `train_adam_runtime.py` |
| `runtime/` | curriculum TRPO over `N(α, α, π/2)`, α = 0 … 3.14 in steps of 0.157, trained with noise | `train_trpo_runtime.py`, ≤ 150 iterations × 4000 episodes per α, termination and advance threshold 0.4, seed 2 |
| `cost_function_sweep/` | UFO-weight sweep: χ ∈ {5, 10, 20}, μ ∈ {0.1, 0.2, 0.4}, κ ∈ {0.05, 0.1, 0.2, 0.4}, 2 seeds (72 runs) | `run_param_search.py --mode trpo`, noise-optimized, 5 iterations × 10 000 episodes, termination cost 0.15 |
| `nn_size_sweep/` | actor–critic architectures 64-32-32, 64-64-64, 64-64-64-32, 128-64-32, 128-64-32-32, 3 seeds (15 runs) | `run_param_search.py --mode trpo`, noise-optimized, 20 iterations × 5000 episodes, termination cost 0.4 |

## Robustness evaluation

All robustness curves are unmodified output of `benchmark_robustness.py` on 3401 noise levels
σ = 0.1 … 3.5 MHz in steps of 0.001 MHz, 60 Monte Carlo rollouts per level, seed 1 (except
`robustness_analysis_2/`, see below). Each `robustness_curve.csv` has the columns `method, sigma_mhz,
num_samples, average_fidelity, average_gate_fidelity, fidelity_variance`; `summary.json` records the source
plan and its target. `combined_ewma_data.csv` adds an exponentially weighted moving average along σ (span
50 for the fidelity, 80 for the variance) next to the raw columns.

- `robustness_analysis_3/` (preprint): `adam_70ns`, `adam_60ns`, `trpo_nominal`, `trpo_noise` (iteration 11
  of `trpo_noise_alpha_2.2/`) and `trpo_noise_it091`. The `trpo_nominal` and `adam_60ns` curves are
  numerically identical to the `nominal` and `adam_noise` curves of `robustness_analysis/`.
- `robustness_analysis/` (thesis): `noise`, `nominal`, `adam_noise`. The `noise` plan targets α = 0.
- `robustness_analysis_2/`: a coarser repeat (341 levels in steps of 0.01 MHz) with `adam_test/`, which
  targets α = 0.2, in place of `adam_noise/`.

The other preprint evaluations are in `trajectory_robustness_results/` (every plan of iterations 1–100 of
both single-target TRPO runs, 11 noise levels, 200 rollouts each), `closed_loop_results/` (checkpoints in
closed loop versus their open-loop plans at 1 MHz, 100 rollouts), `white_noise_results/`,
`noisy_leakage_results/`, `gate_structure_results/`, `noise_memory_results/` (quasi-static and exponentially
correlated noise: first-order prediction of `uqc/noise_response.py` and Monte Carlo replay, 2000 shots per plan
with the same offsets for every plan) and `compute_budget_results/`.

Earlier post-processed variants of the robustness curves were removed from the repository in commit
`f37f909`. The figures in the thesis and the preprint use only unmodified data.
