from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int, *, act=nn.Tanh) -> nn.Sequential:
    layers = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)

@dataclass(frozen=True)
class ActorCriticSpecs:
    obs_dim: int
    act_dim: int
    num_agents: int
    num_roles: int

class SharedDiscreteActor(nn.Module):
    """
    Parameter-shared actor for discrete actions.

    Inputs:
      obs: [B, obs_dim]
      agent_idx: [B] int64 (optional)
      role_id: [B] int64 (optional)

    Output:
      logits: [B, act_dim]
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        num_agents: int,
        num_roles: int = 2,
        hidden: Tuple[int, ...] = (256, 256),
        use_agent_id: bool = True,
        use_role_id: bool = True,
        embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.use_agent_id = bool(use_agent_id)
        self.use_role_id = bool(use_role_id)

        extra = 0
        if self.use_agent_id:
            self.agent_emb = nn.Embedding(int(num_agents), int(embed_dim))
            extra += int(embed_dim)
        else:
            self.agent_emb = None

        if self.use_role_id:
            self.role_emb = nn.Embedding(int(num_roles), int(embed_dim))
            extra += int(embed_dim)
        else:
            self.role_emb = None

        self.net = mlp(self.obs_dim + extra, hidden, self.act_dim)

    def forward(self, obs: torch.Tensor, agent_idx: Optional[torch.Tensor] = None, role_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(f"obs must have shape [B, {self.obs_dim}], got {tuple(obs.shape)}")
        x = obs
        parts = [x]
        if self.use_agent_id:
            if agent_idx is None:
                raise ValueError("agent_idx is required when use_agent_id=True")
            parts.append(self.agent_emb(agent_idx))
        if self.use_role_id:
            if role_id is None:
                raise ValueError("role_id is required when use_role_id=True")
            parts.append(self.role_emb(role_id))
        x = torch.cat(parts, dim=-1)
        return self.net(x)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, agent_idx: torch.Tensor, role_id: Optional[torch.Tensor] = None, *, greedy: bool = False):
        logits = self.forward(obs, agent_idx=agent_idx, role_id=role_id)
        dist = torch.distributions.Categorical(logits=logits)
        if greedy:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()
        logp = dist.log_prob(actions)
        return actions, logp

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, agent_idx: torch.Tensor, role_id: Optional[torch.Tensor] = None):
        logits = self.forward(obs, agent_idx=agent_idx, role_id=role_id)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, entropy

class CentralValueCritic(nn.Module):
    """
    Centralized critic V(s) using flattened joint observation.

    Input:
      joint_obs: [B, N, obs_dim]  -> flattened to [B, N*obs_dim]
    Output:
      V: [B]
    """
    def __init__(self, obs_dim: int, num_agents: int, hidden: Tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.num_agents = int(num_agents)
        self.net = mlp(self.num_agents * self.obs_dim, hidden, 1)

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        if joint_obs.ndim != 3 or joint_obs.shape[1] != self.num_agents or joint_obs.shape[2] != self.obs_dim:
            raise ValueError(
                f"joint_obs must have shape [B, {self.num_agents}, {self.obs_dim}], got {tuple(joint_obs.shape)}"
            )
        x = joint_obs.reshape(joint_obs.shape[0], -1)
        v = self.net(x).squeeze(-1)
        return v
