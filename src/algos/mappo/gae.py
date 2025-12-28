"""GAE computation.

TorchRL has built-in value estimators (GAE) that can be used directly.
This local implementation keeps you unblocked and makes the MAPPO loop explicit.

Batch is assumed to be a rollout tensor with time dimension T and batch dimension B.
"""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,         # [T, B, n_agents, 1] or [T, B, 1]
    values: torch.Tensor,          # [T, B, n_agents, 1] or [T, B, 1]
    dones: torch.Tensor,           # [T, B, 1]
    gamma: float,
    lmbda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (advantages, value_targets).

    Advantages are computed per time step. Terminal transitions are handled via `dones`.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)

    last_adv = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t].to(rewards.dtype)
        # Bootstrap with V_{t+1}; for last step, treat V_{T} = 0
        next_value = values[t + 1] if t + 1 < T else torch.zeros_like(values[t])
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_adv = delta + gamma * lmbda * not_done * last_adv
        adv[t] = last_adv

    value_targets = adv + values
    return adv, value_targets
