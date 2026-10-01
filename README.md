# RL for Universal Analog Quantum Control — an independent reproduction

Code and data for an independent reproduction of

> M. Y. Niu, S. Boixo, V. N. Smelyanskiy, H. Neven, *Universal quantum control through deep
> reinforcement learning*, npj Quantum Information **5**, 33 (2019),
> [doi:10.1038/s41534-019-0141-3](https://doi.org/10.1038/s41534-019-0141-3)

carried out for the bachelor's thesis *Reinforcement-Learning Framework for Robust and Universal
Analog Quantum Control on Superconducting Qubits* (F. G. Minisini, University of Milan, 2026). The
thesis, the defense slides and the accompanying preprint *Robust analog two-qubit gates from deep
reinforcement learning: an independent reproduction and family-resolved assessment* are in the
[thesis repository](https://github.com/FrancescoMinisini/Document).

A TRPO agent learns piecewise-constant analog controls for a two-transmon (gmon) system truncated to
three levels per transmon, targeting the two-qubit gate family `N(α, α, γ)`. It minimizes the UFO cost:
gate infidelity, a leakage bound, boundary regularity and runtime. The implementation uses only
NumPy, SciPy and PyTorch and includes:

- the gmon Hamiltonian in the rotating frame, propagated exactly per time step;
- the second-order time-dependent Schrieffer–Wolff (TSWT) leakage bound;
- the double-exponential bandwidth filter of the paper's Appendix C;
- noise-aware training with per-step Gaussian perturbations on η, g, δⱼ and fⱼ;
- the Appendix D average-fidelity estimator for Monte Carlo robustness evaluation;
- a curriculum runtime sweep over the family `N(α, α, π/2)` with transfer between neighboring targets;
- a differentiable Adam baseline sharing the same physics and cost;
- parallel rollout collection and per-iteration checkpoints and control plans.

## Installation

Python 3.12; CPU only.

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

Run the scripts from the repository root so that the `uqc` package is importable. The TRPO and Adam
scripts force CPU-only, single-threaded Torch and BLAS through environment variables. Parallelism
comes from `--num-workers` rollout processes instead.

A training run that finishes in about ten seconds, useful as a smoke test:

```bash
python train_trpo_single_target.py --alpha 2.2 --iterations 2 --episodes-per-batch 4 --dt-ns 10 --max-time-ns 60 --robustness-sigmas= --out runs/smoke
```

## Repository layout

| Path | Content |
| --- | --- |
| `uqc/` | library: physics (`physics.py`), environment (`env.py`), TRPO agent (`trpo.py`), Adam baseline (`baseline_adam.py`), evaluation and control plans (`eval.py`), parallel rollouts (`parallel.py`) |
| `train_*.py`, `benchmark_robustness.py`, `run_param_search.py` | experiment entry points |
| `analysis/` | turn `final_results/` into the CSVs and figures used by the thesis |
| `final_results/` | the runs retained for the thesis and their processed data — see [`final_results/README.md`](final_results/README.md) |
| `docs/` | reproduction notes (`REPRO_NOTES.md`), a per-file overview, and historical pilot results |
| `evaluate_controls.py`, `plot_paper_repro.py`, `train_fast_thesis_suite.py`, `dashboard.py` | helpers from the first scaffold: simple robustness sweep, paper-style plots, a fast demo suite, and a Streamlit dashboard (needs `pip install streamlit`) |

`runs/` is the default output folder for new experiments and is not version-controlled.

## Running experiments

Angles accept expressions such as `pi/2`. The training and evaluation scripts list their options with `--help`.

```bash
# single target, trained with 1 MHz control noise
python train_trpo_single_target.py --alpha 2.2 --gamma pi/2 --noise-optimized --out runs/trpo_2p2_noise

# curriculum runtime sweep over N(alpha, alpha, gamma)
python train_trpo_runtime.py --gammas pi/2 --out runs/runtime_sweep

# Adam baseline with a horizon search
python train_adam_baseline.py --alpha 2.2 --gamma pi/2 --out runs/adam_2p2

# Adam at fixed horizons without noise, under white noise and under quasi-static noise (about 20 minutes)
python train_adam_noise_models.py --alpha 2.2 --out runs/adam_noise_models
# the same with the leakage bound of the objective taken from the noise-free controls
python train_adam_noise_models.py --alpha 2.2 --leakage-bound nominal --out runs/adam_noise_models_nominal_leakage

# transfer from a trained checkpoint to nearby targets, compared with training from scratch
python train_transition_experiments.py --source-checkpoints runs/trpo_2p2_noise/best_agent.pt \
  --source-labels from_2p2 --targets 2.4:pi/2,2.6:pi/2 --include-scratch --out runs/transitions

# average fidelity and fidelity variance versus noise strength for several control plans
python benchmark_robustness.py --inputs runs/trpo_2p2_noise runs/adam_2p2 --labels noise adam --out runs/robustness --plot
```

Each training run writes `args.json`, `training_log.jsonl`, `summary.json`, per-iteration
`checkpoints/` and `plans/`, and `best_*` / `final_*` agents and control plans. `train_trpo_single_target.py
--resume <run_dir>` continues an interrupted run. `train_trpo_runtime.py` and `train_adam_runtime.py` take
`--resume` with the same `--out`.

## Reproducing the thesis and preprint results

1. **Retrain (optional, hours to days on a CPU).** The arguments of every retained run are stored in
   its `final_results/<run>/args.json`. [`final_results/README.md`](final_results/README.md) lists the
   script and key settings of each run.
2. **Recompute the plotted data from the stored runs (minutes).**

   ```bash
   python analysis/analyze_runtime.py
   python analysis/analyze_cost_sweep.py
   python analysis/analyze_nn_size_sweep.py
   python analysis/generate_adam_plots.py
   python analysis/generate_comparison_plots.py
   python analysis/generate_nominal_trpo_plots.py
   python analysis/export_ewma_data.py
   ```

   Each script writes into `final_results/<experiment>_results/`. The CSVs they produce are
   byte-identical to the ones plotted in the thesis and the preprint. The preprint's additional
   analyses are recomputed with

   ```bash
   python analysis/analyze_gate_structure.py      # exchange-angle / Weyl / CNOT structure of N(a, a, pi/2)
   python analysis/analyze_mirror_symmetry.py     # curriculum pulses mirrored onto pi - alpha
   python analysis/analyze_noisy_leakage.py       # UFO cost terms under the training noise (about a minute)
   python analysis/summarize_robustness_3.py      # numbers of the preprint's robustness table
   python analysis/white_noise_first_order.py     # first-order noise rate versus the measured rates
   python analysis/white_noise_checks.py          # exact rate per plan, per-channel / interval / pre-filter checks (ten minutes)
   python analysis/noise_with_memory.py           # quasi-static and correlated noise (a few minutes, --workers)
   python analysis/exchange_area_bound.py         # bandwidth-limited exchange time across the family
   python analysis/compute_budget.py              # simulator steps per controller
   python analysis/conditional_phase.py           # conditional phase of the higher levels versus the nominal infidelity
   python analysis/noisy_leakage_time_step.py     # noisy leakage bound on finer time grids (about a minute)
   ```

3. **Redo the robustness evaluations (hours).** The evaluations are seeded, so they reproduce the
   stored CSVs exactly. The preprint uses `final_results/robustness_analysis_3/`, one call per plan:

   ```bash
   python benchmark_robustness.py --inputs final_results/trpo_noise_alpha_2.2/plans/iter_000011_control_plan.npz \
     --labels trpo_noise --noise-min 0.1 --noise-max 3.5 --noise-step 0.001 --samples 60 --seed 1 \
     --out final_results/robustness_analysis_3
   ```

   and likewise for `final_results/nominal/best_control_plan.npz` (`trpo_nominal`),
   `final_results/trpo_noise_alpha_2.2/best_control_plan.npz` (`trpo_noise_it091`),
   `final_results/adam_noise/best_control_plan.npz` (`adam_60ns`) and
   `final_results/adam_noise_70ns/best_control_plan.npz` (`adam_70ns`), about 20 minutes each on one
   core. The matched-history and closed-loop evaluations run in parallel:

   ```bash
   python analysis/robustness_training_trajectories.py --workers 6   # about an hour
   python analysis/closed_loop_vs_open_loop.py --workers 6           # about ten minutes
   ```

   The thesis used `final_results/robustness_analysis/` (`--inputs final_results/noise
   final_results/nominal final_results/adam_noise --labels noise nominal adam_noise`). Its `noise`
   plan targets `N(0, 0, pi/2)`, not `N(2.2, 2.2, pi/2)`; see
   [`final_results/README.md`](final_results/README.md).

## Modeling assumptions

The paper specifies the UFO cost weights, filter bandwidth, target family, transfer-learning
curriculum and robustness metric, but not every engineering detail. The following are therefore
exposed as command-line arguments rather than hard-coded:

- `runtime_norm_ns`: the time normalization that makes the `κT` term commensurate with the others;
- `termination_cost` / `advance_cost_threshold`: thresholds for early stopping and curriculum advancement;
- TRPO settings such as `max_kl`, `lam`, `value_epochs` and the initial exploration scale;
- the time step `dt_ns` and the runtime budget `max_time_ns`. The published paper says only that its pulses
  have "around one thousand time steps"; the retained runs use 2 ns steps (30 to 90 per pulse). The step does
  not change the noise-free control problem, but it changes what a given noise strength means; the preprint
  quantifies this.

[`docs/REPRO_NOTES.md`](docs/REPRO_NOTES.md) lists the implementation choices in detail.

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite the original paper above.

## License

Code: MIT ([`LICENSE`](LICENSE)). Data and figures in `final_results/`: CC BY 4.0 ([`LICENSE-DATA`](LICENSE-DATA)).
