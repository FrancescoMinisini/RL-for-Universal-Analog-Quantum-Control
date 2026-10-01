from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


Tensor = torch.Tensor


@dataclass
class TRPOConfig:
    gamma: float = 0.99
    lam: float = 0.97
    max_kl: float = 0.01
    damping: float = 0.1
    cg_iters: int = 10
    residual_tol: float = 1e-10
    backtrack_iters: int = 10
    backtrack_coeff: float = 0.5
    value_lr: float = 1e-3
    value_epochs: int = 50
    init_log_std: float = -0.5
    hidden_sizes: Tuple[int, ...] = (64, 32, 32)



def flat_grad(grads: Sequence[Tensor | None], params: Sequence[nn.Parameter]) -> Tensor:
    flat: List[Tensor] = []
    for grad, p in zip(grads, params):
        if grad is None:
            flat.append(torch.zeros_like(p).view(-1))
        else:
            flat.append(grad.contiguous().view(-1))
    return torch.cat(flat)



def flat_params(params: Sequence[nn.Parameter]) -> Tensor:
    return torch.cat([p.data.view(-1) for p in params])



def set_params(params: Sequence[nn.Parameter], flat: Tensor) -> None:
    idx = 0
    for p in params:
        num = p.numel()
        p.data.copy_(flat[idx : idx + num].view_as(p))
        idx += num



def conjugate_gradient(
    f_ax: Callable[[Tensor], Tensor],
    b: Tensor,
    cg_iters: int,
    residual_tol: float,
) -> Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(cg_iters):
        z = f_ax(p)
        denom = torch.dot(p, z)
        if torch.abs(denom) < 1e-12:
            break
        v = rdotr / denom
        x = x + v * p
        r = r - v * z
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        mu = new_rdotr / (rdotr + 1e-12)
        p = r + mu * p
        rdotr = new_rdotr
    return x


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            linear = nn.Linear(prev, h)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.Tanh())
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int], init_log_std: float):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_sizes)
        self.mean = nn.Linear(self.backbone.out_dim, act_dim)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        self.log_std = nn.Parameter(torch.ones(act_dim) * float(init_log_std))

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        if obs.dtype != torch.float32:
            obs = obs.float()
        feat = self.backbone(obs)
        mean = self.mean(feat)
        std = torch.exp(self.log_std)
        return mean, std


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_sizes)
        self.value = nn.Linear(self.backbone.out_dim, 1)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, obs: Tensor) -> Tensor:
        if obs.dtype != torch.float32:
            obs = obs.float()
        feat = self.backbone(obs)
        return self.value(feat).squeeze(-1)


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return obj


class TRPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: TRPOConfig | None = None,
        device: str = "cpu",
        update_chunk_size: int | None = None,
    ):
        self.config = config or TRPOConfig()
        self.device = torch.device(device)
        # With a chunk size, the update accumulates every batch average over chunks of that many
        # transitions, which bounds its memory (needed on a small GPU). The result differs from
        # the unchunked one only by the order of the floating-point sums.
        self.update_chunk_size = update_chunk_size
        self.policy = PolicyNetwork(obs_dim, act_dim, self.config.hidden_sizes, self.config.init_log_std).to(self.device)
        self.value_net = ValueNetwork(obs_dim, self.config.hidden_sizes).to(self.device)
        self.value_optim = optim.Adam(self.value_net.parameters(), lr=self.config.value_lr)

    def policy_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.policy(obs_t)
            dist = torch.distributions.Normal(mean, std)
            if deterministic:
                raw = mean
            else:
                raw = dist.sample()
            action = torch.tanh(raw)
        return action.cpu().numpy()[0], raw.cpu().numpy()[0]

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        return self.policy_action(obs, deterministic=deterministic)

    def _prepare_batch(self, rollouts: Sequence[Dict[str, object]]) -> Dict[str, Tensor]:
        obs = torch.as_tensor(np.asarray([r["obs"] for r in rollouts], dtype=np.float32), device=self.device)
        raw_actions = torch.as_tensor(np.asarray([r["raw_action"] for r in rollouts], dtype=np.float32), device=self.device)
        rewards = torch.as_tensor(np.asarray([r["reward"] for r in rollouts], dtype=np.float32), device=self.device)
        masks = torch.as_tensor(np.asarray([r["mask"] for r in rollouts], dtype=np.float32), device=self.device)
        return {"obs": obs, "raw_actions": raw_actions, "rewards": rewards, "masks": masks}

    def compute_returns_advantages(self, rewards: Tensor, values: Tensor, masks: Tensor) -> Tuple[Tensor, Tensor]:
        # The backward recurrence restarts at every mask == 0, so the episodes are independent:
        # lay them out as columns of a [time, episode] array and run it over time for all of them
        # at once. Padding has reward, value and mask 0 and leaves the recurrence at zero.
        n = rewards.shape[0]
        ends = masks == 0
        ends[-1] = True
        end_idx = torch.nonzero(ends).squeeze(-1)
        start_idx = torch.cat([end_idx.new_zeros(1), end_idx[:-1] + 1])
        episode = torch.cumsum(ends.to(torch.long), dim=0) - ends.to(torch.long)
        pos = torch.arange(n, device=rewards.device) - start_idx[episode]
        horizon = int((end_idx - start_idx).max().item()) + 1

        def padded(x: Tensor) -> Tensor:
            out = torch.zeros((horizon, end_idx.shape[0]), dtype=x.dtype, device=x.device)
            out[pos, episode] = x
            return out

        r, v, m = padded(rewards), padded(values), padded(masks)
        ret = torch.zeros_like(r)
        adv_p = torch.zeros_like(r)
        gae = torch.zeros_like(r[0])
        running_return = torch.zeros_like(r[0])
        next_value = torch.zeros_like(r[0])
        for t in reversed(range(horizon)):
            running_return = r[t] + self.config.gamma * running_return * m[t]
            ret[t] = running_return
            delta = r[t] + self.config.gamma * next_value * m[t] - v[t]
            gae = delta + self.config.gamma * self.config.lam * m[t] * gae
            adv_p[t] = gae
            next_value = v[t]
        returns = ret[pos, episode]
        adv = adv_p[pos, episode]

        adv_std = adv.std(unbiased=False)
        if not torch.isfinite(adv_std) or float(adv_std.item()) < 1e-12:
            adv = adv - adv.mean()
        else:
            adv = (adv - adv.mean()) / (adv_std + 1e-8)
        return returns, adv

    def update(self, rollouts: Sequence[Dict[str, object]]) -> Dict[str, float]:
        if not rollouts:
            return {"updated": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0}
        return self.update_from_batch(self._prepare_batch(rollouts))

    def update_from_batch(self, batch: Dict[str, object]) -> Dict[str, float]:
        """batch: flat arrays or tensors "obs", "raw_actions", "rewards", "masks", episode after episode."""
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        raw_actions = torch.as_tensor(batch["raw_actions"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(batch["masks"], dtype=torch.float32, device=self.device)
        n = obs.shape[0]
        if n == 0:
            return {"updated": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0}

        chunk = self.update_chunk_size
        if chunk is None or chunk <= 0 or chunk >= n:
            slices = [slice(0, n)]
        else:
            slices = [slice(i, min(i + chunk, n)) for i in range(0, n, chunk)]

        def batch_mean(per_sample: Tensor) -> Tensor:
            # Contribution of one chunk to the average over the whole batch.
            return per_sample.mean() if len(slices) == 1 else per_sample.sum() / n

        with torch.no_grad():
            values_old = torch.cat([self.value_net(obs[sl]) for sl in slices])
            returns, advantages = self.compute_returns_advantages(rewards, values_old, masks)
            old_mean = torch.cat([self.policy(obs[sl])[0] for sl in slices])
            old_std = torch.exp(self.policy.log_std).clone()
            old_log_probs = torch.distributions.Normal(old_mean, old_std).log_prob(raw_actions).sum(dim=-1)

        def surrogate_terms(sl: slice) -> Tensor:
            mean, std = self.policy(obs[sl])
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(raw_actions[sl]).sum(dim=-1)
            ratio = torch.exp(log_probs - old_log_probs[sl])
            return -(ratio * advantages[sl])

        def kl_terms(sl: slice) -> Tensor:
            mean, std = self.policy(obs[sl])
            new_dist = torch.distributions.Normal(mean, std)
            old_dist = torch.distributions.Normal(old_mean[sl], old_std)
            return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)

        def evaluate_no_grad(terms: Callable[[slice], Tensor]) -> float:
            with torch.no_grad():
                return float(sum(batch_mean(terms(sl)) for sl in slices).item())

        params = list(self.policy.parameters())
        old_loss = 0.0
        loss_grad = torch.zeros_like(flat_params(params))
        for sl in slices:
            loss = batch_mean(surrogate_terms(sl))
            loss_grad = loss_grad + flat_grad(torch.autograd.grad(loss, params), params)
            old_loss += float(loss.item())

        if torch.norm(loss_grad) < 1e-12:
            return {"updated": 0.0, "policy_loss": old_loss, "value_loss": 0.0, "kl": 0.0}

        def fisher_vector_product(v: Tensor) -> Tensor:
            out = torch.zeros_like(v)
            for sl in slices:
                kl = batch_mean(kl_terms(sl))
                grad_kl = flat_grad(torch.autograd.grad(kl, params, create_graph=True), params)
                kl_v = (grad_kl * v).sum()
                out = out + flat_grad(torch.autograd.grad(kl_v, params), params)
            return out + self.config.damping * v

        step_dir = conjugate_gradient(
            fisher_vector_product,
            -loss_grad,
            cg_iters=self.config.cg_iters,
            residual_tol=self.config.residual_tol,
        )

        fvp_step = fisher_vector_product(step_dir)
        shs = 0.5 * (step_dir * fvp_step).sum()
        if torch.isnan(shs) or shs <= 0:
            return {"updated": 0.0, "policy_loss": old_loss, "value_loss": 0.0, "kl": 0.0}

        scale = torch.sqrt(shs / self.config.max_kl)
        full_step = step_dir / (scale + 1e-12)
        old_params = flat_params(params).clone()

        accepted = False
        final_kl = 0.0
        for j in range(self.config.backtrack_iters):
            stepfrac = self.config.backtrack_coeff ** j
            new_params = old_params + stepfrac * full_step
            set_params(params, new_params)
            new_loss = evaluate_no_grad(surrogate_terms)
            new_kl = evaluate_no_grad(kl_terms)
            improvement = old_loss - new_loss
            if improvement > 0 and new_kl <= self.config.max_kl:
                accepted = True
                final_kl = new_kl
                break
        if not accepted:
            set_params(params, old_params)
            final_kl = evaluate_no_grad(kl_terms)

        value_loss_scalar = 0.0
        for _ in range(self.config.value_epochs):
            self.value_optim.zero_grad(set_to_none=True)
            value_loss_scalar = 0.0
            for sl in slices:
                pred = self.value_net(obs[sl])
                value_loss = batch_mean((pred - returns[sl]) ** 2)
                value_loss.backward()
                value_loss_scalar += float(value_loss.item())
            self.value_optim.step()

        return {
            "updated": 1.0 if accepted else 0.0,
            "policy_loss": old_loss,
            "value_loss": value_loss_scalar,
            "kl": final_kl,
        }

    def state_dict(self) -> Dict[str, object]:
        # Always on the CPU, so that checkpoints do not depend on the device of the update.
        return {
            "policy": _to_cpu(self.policy.state_dict()),
            "value_net": _to_cpu(self.value_net.state_dict()),
            "value_optim": _to_cpu(self.value_optim.state_dict()),
            "config": self.config.__dict__,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.policy.load_state_dict(state["policy"])
        self.value_net.load_state_dict(state["value_net"])
        self.value_optim.load_state_dict(state["value_optim"])
