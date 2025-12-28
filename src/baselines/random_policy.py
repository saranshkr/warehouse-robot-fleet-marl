from __future__ import annotations

from typing import Optional

import torch


def random_valid_actions(
    action_mask: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a random valid action per agent (uniform over valid actions).

    Args:
        action_mask: bool tensor of shape [..., n_agents, n_actions]
        generator: optional torch.Generator to make sampling reproducible / isolated
                   from the global RNG state.

    Returns:
        actions: int64 tensor of shape [..., n_agents] on the same device as action_mask
    """
    if action_mask.dtype != torch.bool:
        action_mask = action_mask.bool()

    device = action_mask.device
    *prefix, n_agents, n_actions = action_mask.shape

    flat = action_mask.reshape(-1, n_agents, n_actions)  # [B, A, N]
    B = flat.shape[0]

    # Robust fallback: if an agent has zero valid actions, allow all actions for that agent.
    valid_counts = flat.sum(dim=-1)  # [B, A]
    no_valid = valid_counts == 0
    if no_valid.any():
        flat = flat.clone()
        flat[no_valid] = True

    # Uniform sampling over valid actions (weights are 1 for valid, 0 for invalid)
    weights = flat.to(torch.float32)  # [B, A, N]
    actions = torch.multinomial(
        weights.view(B * n_agents, n_actions),
        num_samples=1,
        replacement=True,
        generator=generator,
    ).view(B, n_agents)

    return actions.reshape(*prefix, n_agents).to(device=device, dtype=torch.int64)
