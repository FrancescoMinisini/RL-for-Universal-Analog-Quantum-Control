from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import os
from dataclasses import replace
from typing import Dict, List, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

from .batched_env import BatchedQuantumControlEnv
from .env import EnvConfig, QuantumControlEnv
from .physics import GmonSystem, GmonSystemConfig
from .trpo import PolicyNetwork, TRPOAgent, TRPOConfig


torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def _worker_collect(
    system_config: GmonSystemConfig,
    env_config: EnvConfig,
    trpo_config: TRPOConfig,
    policy_state: Dict[str, torch.Tensor],
    episodes: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, float]]]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if env_config.seed is not None:
        np.random.seed(int(env_config.seed))
        torch.manual_seed(int(env_config.seed))
    system = GmonSystem(system_config)
    env = QuantumControlEnv(system, env_config)
    agent = TRPOAgent(env.observation_dim, env.action_dim, config=trpo_config, device="cpu")
    agent.policy.load_state_dict(policy_state)
    agent.policy.eval()
    transitions: List[Dict[str, object]] = []
    finals: List[Dict[str, float]] = []
    for _ in range(int(episodes)):
        rollout = env.rollout(agent.get_action, deterministic=False)
        for tr in rollout["transitions"]:
            transitions.append(
                {
                    "obs": tr["obs"],
                    "raw_action": tr["raw_action"],
                    "reward": tr["reward"],
                    "mask": tr["mask"],
                }
            )
        finals.append(rollout["final_info"])
    return transitions, finals


BATCHED_EPISODES_PER_TASK = 250


def _worker_collect_batched(
    system_config: GmonSystemConfig,
    env_config: EnvConfig,
    trpo_config: TRPOConfig,
    policy_state: Dict[str, torch.Tensor],
    episodes: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    env = BatchedQuantumControlEnv(GmonSystem(system_config), env_config, int(episodes))
    policy = PolicyNetwork(env.observation_dim, env.action_dim, trpo_config.hidden_sizes, trpo_config.init_log_std)
    policy.load_state_dict(policy_state)
    policy.eval()
    return env.rollout(policy)


class ParallelBatchCollector:
    def __init__(
        self,
        *,
        system_config: GmonSystemConfig,
        env_config: EnvConfig,
        trpo_config: TRPOConfig,
        num_workers: int,
        episodes_per_task: int | None = None,
        seed: int = 1,
    ):
        self.system_config = system_config
        self.env_config = env_config
        self.trpo_config = trpo_config
        self.num_workers = int(max(1, num_workers))
        self.episodes_per_task = episodes_per_task
        self.seed = int(seed)
        self.batch_index = 0
        start_method = "fork" if os.name == "posix" else "spawn"
        self.mp_context = mp.get_context(start_method)
        self.executor: concurrent.futures.ProcessPoolExecutor | None = None

    def __enter__(self) -> "ParallelBatchCollector":
        if self.num_workers > 1 and self.executor is None:
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers, mp_context=self.mp_context)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=False)
            self.executor = None

    def _task_sizes(self, episodes_per_batch: int) -> List[int]:
        if episodes_per_batch <= 0:
            return []
        chunk = int(self.episodes_per_task) if (self.episodes_per_task is not None and self.episodes_per_task > 0) else max(1, int(np.ceil(episodes_per_batch / max(self.num_workers, 1))))
        sizes: List[int] = []
        remaining = int(episodes_per_batch)
        while remaining > 0:
            take = min(chunk, remaining)
            sizes.append(take)
            remaining -= take
        return sizes

    def collect(self, agent: TRPOAgent, episodes_per_batch: int) -> Tuple[List[Dict[str, object]], List[Dict[str, float]]]:
        if self.num_workers <= 1:
            env = QuantumControlEnv(GmonSystem(self.system_config), self.env_config)
            transitions: List[Dict[str, object]] = []
            finals: List[Dict[str, float]] = []
            for _ in range(int(episodes_per_batch)):
                rollout = env.rollout(agent.get_action, deterministic=False)
                for tr in rollout["transitions"]:
                    transitions.append({"obs": tr["obs"], "raw_action": tr["raw_action"], "reward": tr["reward"], "mask": tr["mask"]})
                finals.append(rollout["final_info"])
            return transitions, finals
        if self.executor is None:
            self.__enter__()
        assert self.executor is not None
        policy_state = {k: v.detach().cpu() for k, v in agent.policy.state_dict().items()}
        futures = []
        for task_id, episodes in enumerate(self._task_sizes(int(episodes_per_batch))):
            task_seed = self.seed + self.batch_index * 100_000 + task_id * 1_000
            futures.append(self.executor.submit(_worker_collect, self.system_config, replace(self.env_config, seed=task_seed), self.trpo_config, policy_state, episodes))
        self.batch_index += 1
        transitions: List[Dict[str, object]] = []
        finals: List[Dict[str, float]] = []
        for fut in concurrent.futures.as_completed(futures):
            tr, fi = fut.result()
            transitions.extend(tr)
            finals.extend(fi)
        return transitions, finals

    def collect_batched(self, agent: TRPOAgent, episodes_per_batch: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Same batch as collect(), from BatchedQuantumControlEnv: every task steps all of its
        episodes together. Returns flat arrays for TRPOAgent.update_from_batch and the final
        infos as arrays, in task order.

        The task size does not depend on the number of workers (BATCHED_EPISODES_PER_TASK unless
        episodes_per_task is set), so a batch depends only on the seed. It also keeps the result
        of a task small: Windows pipes fail on messages of about 100 MB.
        """
        policy_state = {k: v.detach().cpu() for k, v in agent.policy.state_dict().items()}
        chunk = int(self.episodes_per_task) if (self.episodes_per_task is not None and self.episodes_per_task > 0) else BATCHED_EPISODES_PER_TASK
        sizes = [min(chunk, int(episodes_per_batch) - i) for i in range(0, int(episodes_per_batch), chunk)]
        if len(sizes) > 100:
            raise ValueError("More than 100 tasks per batch: their seeds would overlap with the next batch. Raise episodes_per_task.")
        tasks = []
        for task_id, episodes in enumerate(sizes):
            task_seed = self.seed + self.batch_index * 100_000 + task_id * 1_000
            tasks.append((self.system_config, replace(self.env_config, seed=task_seed), self.trpo_config, policy_state, episodes))
        self.batch_index += 1
        if self.num_workers <= 1:
            results = [_worker_collect_batched(*task) for task in tasks]
        else:
            if self.executor is None:
                self.__enter__()
            assert self.executor is not None
            futures = [self.executor.submit(_worker_collect_batched, *task) for task in tasks]
            results = [fut.result() for fut in futures]
        batch = {k: np.concatenate([r[0][k] for r in results]) for k in results[0][0]}
        finals = {k: np.concatenate([r[1][k] for r in results]) for k in results[0][1]}
        return batch, finals
