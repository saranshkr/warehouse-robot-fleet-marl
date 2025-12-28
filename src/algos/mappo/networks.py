"""Network builders for MAPPO (TorchRL primitives).

This file intentionally focuses on:
- **Decentralized actors**: per-agent observations -> action distribution (with masking).
- **Centralized critic**: global state (concat obs) -> value.

You can implement either:
- separate actors per role (agv / picker), OR
- one shared actor with a role embedding appended to the observation.

Recommended v1: **separate actors per role** + **centralized critic**.

Refs:
- TorchRL multi-agent PPO tutorial discusses centralized critic input as concatenated agent observations.https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int = 256, n_layers: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_size), nn.ReLU()]
            d = hidden_size
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ActorCriticModules:
    # These are Torch modules; you'll wrap them in TensorDictModule/ProbabilisticActor later.
    actor_agv: nn.Module
    actor_picker: nn.Module
    critic: nn.Module


def build_modules(
    obs_dim: int,
    n_actions: int,
    n_agents_total: int,
    hidden_size: int = 256,
    n_layers: int = 2,
) -> ActorCriticModules:
    """Build plain torch modules.

    - actor_*: maps per-agent obs -> logits over macro actions
    - critic: maps concatenated obs of all agents -> scalar value (global)

    For MAPPO, you can either:
    - output a single scalar V(s) and broadcast it per agent, OR
    - output V_i(s) for each agent i (shape n_agents).

    Start with scalar V(s): simpler + often stable.
    """
    actor_agv = MLP(obs_dim, n_actions, hidden_size=hidden_size, n_layers=n_layers)
    actor_picker = MLP(obs_dim, n_actions, hidden_size=hidden_size, n_layers=n_layers)

    critic_in = n_agents_total * obs_dim
    critic = MLP(critic_in, 1, hidden_size=hidden_size, n_layers=n_layers)

    return ActorCriticModules(actor_agv=actor_agv, actor_picker=actor_picker, critic=critic)
