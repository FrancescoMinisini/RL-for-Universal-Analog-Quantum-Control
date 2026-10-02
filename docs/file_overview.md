# Workspace File Overview

This document provides a description of each file in the repository, along with instructions on how to run the scripts and their associated parameters.

## Root Directory Scripts

| File | Description | Run Command | Parameters |
| :--- | :--- | :--- | :--- |
| `train_trpo_single_target.py` | Trains a single-target TRPO controller for the Universal Functional Optimization (UFO) experiment. | `python train_trpo_single_target.py --alpha <alpha> --gamma <gamma> --out <dir>` | `--alpha`: Target alpha (e.g., '2.2', 'pi/2'). <br> `--gamma`: Target gamma (default: 'pi/2'). <br> `--noise-optimized`: Train in stochastic environment. <br> `--iterations`: Number of iterations. <br> `--out`: Output directory. |
| `train_trpo_runtime.py` | Performs a curriculum TRPO sweep to generate runtime curves (similar to Figure 3 in the paper). | `python train_trpo_runtime.py --out <dir>` | `--gammas`: Comma-separated target gammas. <br> `--alpha-start/stop/step`: Alpha sweep range. <br> `--advance-cost-threshold`: Threshold to move to next alpha. <br> `--out`: Output directory. |
| `train_transition_experiments.py` | Conducts checkpoint transfer/transition experiments across different target gates to test adaptation. | `python train_transition_experiments.py --targets <specs> --out <dir>` | `--source-checkpoints`: Paths to starting checkpoints. <br> `--targets`: Comma-separated alpha[:gamma] specs. <br> `--include-scratch`: Include training from scratch for comparison. <br> `--out`: Output directory. |
| `train_adam_baseline.py` | Runs an Adam-based gradient optimization baseline for comparison with TRPO. | `python train_adam_baseline.py --alpha <alpha> --out <dir>` | `--alpha`: Target alpha. <br> `--lr`: Learning rate. <br> `--adam-iters`: Number of optimization steps. <br> `--horizons-ns`: Search horizons in nanoseconds. <br> `--out`: Output directory. |
| `train_adam_noise_models.py` | Adam at fixed horizons under three training-noise models (none, white, quasi-static), several seeds, every plan stored. | `python train_adam_noise_models.py --out <dir>` | `--horizons-ns`, `--seeds`: grid of runs. <br> `--white-noise-std`, `--quasi-static-noise-std`: noise strengths in MHz. <br> `--leakage-bound`: `noisy` (default, as in the TRPO environment) or `nominal` (leakage bound from the noise-free controls). <br> `--num-workers`: parallel runs. |
| `evaluate_controls.py` | Evaluates control plans using the average-fidelity metric (similar to Figure 4 in the paper). | `python evaluate_controls.py --plans <files> --out <dir>` | `--plans`: List of `.npz` control plan files. <br> `--noise-min/max/step`: Robustness sweep range. <br> `--samples`: Monte Carlo samples per point. <br> `--out`: Output directory. |
| `plot_paper_repro.py` | Generates paper-style figures for runtime and robustness from CSV summaries. | `python plot_paper_repro.py --out <dir>` | `--runtime-csv`: Path to `runtime_summary.csv`. <br> `--robustness-csv`: Path to `robustness.csv`. <br> `--out`: Output directory for images. |
| `train_fast_thesis_suite.py` | A wrapper script that runs a "fast" suite of experiments (Adam, TRPO, and Transitions) in minutes. | `python train_fast_thesis_suite.py --out <dir>` | `--alpha`: Focal target alpha. <br> `--transition-targets`: Targets for adaptation test. <br> `--seed`: Random seed. <br> `--out`: Root output directory. |

## Library Components (`uqc/` directory)

| Component | Description |
| :--- | :--- |
| `uqc/env.py` | Implementation of the `QuantumControlEnv` (Gym-like interface for quantum control). |
| `uqc/physics.py` | Core physics logic, including the Gmon Hamiltonian, TSWT leakage bounds, and UFO cost functions. |
| `uqc/trpo.py` | Implementation of the Trust Region Policy Optimization (TRPO) agent and algorithm. |
| `uqc/baseline_adam.py`| Logic for the Adam gradient-based optimization baseline. |
| `uqc/eval.py` | Utilities for deterministic rollout, control plan management, and robustness evaluation. |
| `uqc/noise_response.py` | First-order (filter-function) response of a plan to control noise with any covariance, and a Monte Carlo replay with correlated noise. |
| `uqc/operators.py` | Definitions of quantum operators (Pauli matrices, projection operators). |
| `uqc/batched_env.py` | `BatchedQuantumControlEnv`: the environment of `uqc/env.py` for many episodes at once (torch, double precision), used by `train_trpo_single_target.py --engine batched`. |
| `uqc/parallel.py` | Multi-processed batch collection for RL training, with the scalar or the batched environment. |
| `uqc/utils.py` | Logging, directory handling, and math expression parsing. |

## Analysis Scripts (`analysis/` directory)

These turn the runs stored in `final_results/<exp>/` into the CSVs and figures in `final_results/<exp>_results/` that feed the thesis. They resolve paths relative to the repository root, so they can be launched from any working directory, e.g. `python analysis/export_ewma_data.py`.

| File | Description |
| :--- | :--- |
| `analysis/analyze_runtime.py` | Runtime sweep (`final_results/runtime`) -> `final_results/runtime_results/`. |
| `analysis/analyze_cost_sweep.py` | UFO-weight sweep (`final_results/cost_function_sweep`) -> `final_results/cost_function_sweep_results/`. |
| `analysis/analyze_nn_size_sweep.py` | Architecture sweep (`final_results/nn_size_sweep`) -> `final_results/nn_size_sweep_results/`. |
| `analysis/export_ewma_data.py` | Merges the per-method robustness curves of `final_results/robustness_analysis*/` into `combined_ewma_data.csv` (raw columns plus EWMA smoothing). |
| `analysis/analyze_crossover.py` | Prints where the robustness curves of `final_results/robustness_analysis/` cross. |
| `analysis/generate_*_plots.py` | Adam sweeps, Adam vs nominal TRPO, nominal TRPO training curves -> `final_results/{adam_results,adam_vs_nominal_results,trpo_nominal_results}/`. |
| `analysis/summarize_robustness_3.py` | Representative values and crossovers of the preprint's robustness curves (`final_results/robustness_analysis_3/`). |
| `analysis/robustness_training_trajectories.py` | Robustness of every per-iteration plan of the matched nominal and noise-trained TRPO runs -> `final_results/trajectory_robustness_results/`. Monte Carlo, parallel (`--workers`); `--summary-only` rebuilds the summary. |
| `analysis/matched_retraining_robustness.py` | The same evaluation on the six runs of `run_matched_retraining.py` (three seeds, noise-trained agents with the leakage bound from the noise-free controls), with per-seed and across-seed summaries -> `final_results/matched_retraining_results/`. Parallel (`--workers`), about 40 minutes; `--summary-only` rebuilds the summaries. |
| `analysis/closed_loop_vs_open_loop.py` | Checkpoints of both runs acting in closed loop on the noisy propagator versus their open-loop plans -> `final_results/closed_loop_results/`. |
| `analysis/white_noise_first_order.py` | First-order infidelity rate of the white-noise model, computed from the operators, against the measured rates -> `final_results/white_noise_results/`. |
| `analysis/white_noise_checks.py` | Exact first-order rate of every plan; Monte Carlo checks per channel, per noise interval and for noise injected before the filter -> `final_results/white_noise_results/`. About ten minutes. |
| `analysis/batched_env_checks.py` | Compares the batched environment with the scalar one step by step, replays batched rollouts in the scalar environment, and checks the vectorized advantage computation. Prints the differences; writes no files. About a minute. |
| `analysis/noise_with_memory.py` | Quasi-static and exponentially correlated noise on every stored plan and on the Adam noise-model pulses -> `final_results/noise_memory_results/`. Parallel (`--workers`). |
| `analysis/exchange_area_bound.py` | Shortest time in which the filtered, bounded coupling delivers the exchange area of each target (linear program) -> `final_results/gate_structure_results/`. |
| `analysis/compute_budget.py` | Simulator steps spent by the single-target controllers -> `final_results/compute_budget_results/`. |
| `analysis/analyze_noisy_leakage.py` | UFO cost terms, leakage bound and leakage population under the training noise -> `final_results/noisy_leakage_results/`. |
| `analysis/noisy_leakage_time_step.py` | Growth of the leakage bound under white noise on finer time grids (it scales as dt^-4) -> `final_results/noisy_leakage_results/`. |
| `analysis/conditional_phase.py` | Conditional phase from the second excited levels, `(4/|eta|) int g^2 dt`, and the infidelity it leaves, against the simulated infidelity of the stored pulses -> `final_results/gate_structure_results/`. |
| `analysis/analyze_gate_structure.py` | Exchange angle, Weyl coordinate, CNOT count and synthesis times of `N(a, a, pi/2)`, merged with the runtime sweeps -> `final_results/gate_structure_results/`. |
| `analysis/analyze_mirror_symmetry.py` | Curriculum plans mirrored onto `pi - alpha` (parity symmetry) -> `final_results/gate_structure_results/`. |

## Results Directories

| Directory | Content |
| :--- | :--- |
| `runs/` | Working output of the training scripts (not version-controlled). |
| `final_results/<exp>/` | Runs retained for the thesis, copied from `runs/`; described in `final_results/README.md`. |
| `final_results/<exp>_results/` | Analysis outputs, copied to `Document/data/<exp>_results/` in the thesis repository. |
| `final_results/robustness_analysis/` | Robustness curves used by the thesis (`noise`, `nominal`, `adam_noise`), as written by `benchmark_robustness.py`. The `noise` plan targets `N(0, 0, pi/2)`. |
| `final_results/robustness_analysis_2/` | Second, coarser robustness sweep using the `adam_test` Adam plan (which targets `N(0.2, 0.2, pi/2)`). |
| `final_results/robustness_analysis_3/` | Robustness curves used by the preprint: five plans, all for `N(2.2, 2.2, pi/2)`. |

## Documentation and Data Files

| File | Purpose |
| :--- | :--- |
| `README.md` | General project overview, installation, and example run commands. |
| `docs/REPRO_NOTES.md` | Technical details on changes made from the original draft to ensure paper fidelity. |
| `docs/FAST_RESULTS.md` | Summary of results obtained from the "fast" experiment suite. |
| `docs/PILOT_RESULTS.md` | Preliminary results from pilot runs. |
| `requirements.txt` | List of Python dependencies (NumPy, SciPy, Torch, Matplotlib, Pandas). |
