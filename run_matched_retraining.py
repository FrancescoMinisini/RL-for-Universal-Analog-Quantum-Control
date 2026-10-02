"""Matched nominal / noise-trained TRPO runs, several seeds, restartable.

For every seed two agents are trained on N(2.2, 2.2, pi/2) with identical settings (10 000
episodes per batch, 100 iterations, stopping cost 0.40, batched engine):

  <out-root>/matched_nominal_seed<N>          no training noise
  <out-root>/matched_noise_nomleak_seed<N>    1 MHz training noise, with the leakage bound of
                                              the training cost taken from the noise-free controls

The runs are launched one after the other through train_trpo_single_target.py. Launch the
script again after an interruption: runs with a summary.json are skipped, a run with
checkpoints is resumed from the newest one (at most one iteration is lost), the others start
from scratch.

    py -3.12 run_matched_retraining.py              # needs a CUDA build of torch
    py -3.12 run_matched_retraining.py --status     # what is done, without running anything
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

ARMS = {
    "matched_nominal": [],
    "matched_noise_nomleak": ["--noise-optimized", "--train-noise-std", "1.0", "--leakage-bound", "nominal"],
}


def last_iteration(run_dir: str) -> int:
    files = glob.glob(os.path.join(run_dir, "checkpoints", "iter_*.pt"))
    return max((int(os.path.basename(f)[5:-3]) for f in files), default=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Matched nominal / noise-trained TRPO runs, restartable.")
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--episodes-per-batch", type=int, default=10000)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--update-device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--out-root", type=str, default="runs")
    ap.add_argument("--status", action="store_true", help="Print the state of every run and exit.")
    args = ap.parse_args()

    os.chdir(ROOT)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    jobs = [(f"{arm}_seed{seed}", seed, extra) for seed in seeds for arm, extra in ARMS.items()]

    if args.status:
        for name, _, _ in jobs:
            run_dir = os.path.join(args.out_root, name)
            done = os.path.exists(os.path.join(run_dir, "summary.json"))
            print(f"{name:<34} {'done' if done else f'iteration {last_iteration(run_dir)}'}")
        return

    if args.update_device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit(
                f"{sys.executable} has no CUDA build of torch. Launch with the interpreter that has one "
                "(py -3.12 run_matched_retraining.py) or pass --update-device cpu (about six times slower)."
            )

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")  # uqc/__pycache__ is tracked in git
    for index, (name, seed, extra) in enumerate(jobs, start=1):
        run_dir = os.path.join(args.out_root, name)
        if os.path.exists(os.path.join(run_dir, "summary.json")):
            print(f"[{index}/{len(jobs)}] {name}: already complete, skipped", flush=True)
            continue

        cmd = [sys.executable, "train_trpo_single_target.py"]
        start_iter = last_iteration(run_dir)
        if start_iter > 0:
            # Settings come from the run's args.json; the device flag must be on the command line.
            cmd += ["--resume", run_dir, "--update-device", args.update_device, "--num-workers", str(args.num_workers)]
            print(f"[{index}/{len(jobs)}] {name}: resuming (checkpoints up to iteration {start_iter})", flush=True)
        else:
            cmd += [
                "--alpha", "2.2", "--gamma", "pi/2", "--seed", str(seed),
                "--iterations", str(args.iterations), "--episodes-per-batch", str(args.episodes_per_batch),
                "--max-time-ns", "180", "--termination-cost", "0.4",
                "--engine", "batched", "--update-device", args.update_device,
                "--num-workers", str(args.num_workers), "--out", run_dir,
            ] + extra
            print(f"[{index}/{len(jobs)}] {name}: starting", flush=True)

        os.makedirs(run_dir, exist_ok=True)
        t0 = time.time()
        # The per-iteration records are already in training_log.jsonl; keep the console short.
        with open(os.path.join(run_dir, "console.log"), "a", encoding="utf-8") as console:
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert proc.stdout is not None
            first_iter = None
            try:
                for line in proc.stdout:
                    console.write(line)
                    console.flush()
                    if line.startswith('{"iteration"'):
                        it = int(line.split(",", 1)[0].split(":")[1])
                        first_iter = it if first_iter is None else first_iter
                        per_iter = (time.time() - t0) / (it - first_iter + 1)
                        print(f"    iteration {it}  ({per_iter:.0f} s per iteration)", flush=True)
                code = proc.wait()
            finally:
                if proc.poll() is None:  # this script is being stopped: do not leave the training running
                    if os.name == "nt":  # with its rollout workers, which outlive a killed parent on Windows
                        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
                    else:
                        proc.kill()
        if code != 0:
            raise SystemExit(f"{name} stopped with exit code {code}; see {os.path.join(run_dir, 'console.log')}. "
                             "Launch this script again to continue.")
        print(f"[{index}/{len(jobs)}] {name}: complete in {(time.time() - t0) / 60:.0f} min", flush=True)

    print("all runs complete")


if __name__ == "__main__":
    main()
