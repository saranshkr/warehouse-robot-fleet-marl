from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)

@dataclass
class RolloutBatch:
    # actor (flattened over T*E*N)
    obs_actor: torch.Tensor        # [B_a, obs_dim]
    act: torch.Tensor             # [B_a]
    logp_old: torch.Tensor        # [B_a]
    adv: torch.Tensor             # [B_a]
    agent_idx: torch.Tensor       # [B_a]
    role_id: Optional[torch.Tensor]  # [B_a] or None

    # critic (flattened over T*E)
    obs_critic: torch.Tensor      # [B_c, N, obs_dim]
    ret: torch.Tensor             # [B_c]
    val_old: torch.Tensor         # [B_c]

class RolloutBuffer:
    """
    Stores on-policy rollouts for MAPPO (team reward recommended).

    Shapes:
      obs:      [T, E, N, obs_dim]
      actions:  [T, E, N]
      logp:     [T, E, N]
      rewards:  [T, E] (team scalar per step)
      dones:    [T, E] bool
      values:   [T, E] critic values before action
    """
    def __init__(self, T: int, E: int, N: int, obs_dim: int, device: torch.device) -> None:
        self.T, self.E, self.N, self.obs_dim = int(T), int(E), int(N), int(obs_dim)
        self.device = device

        self.obs = torch.zeros((T, E, N, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((T, E, N), dtype=torch.int64, device=device)
        self.logp = torch.zeros((T, E, N), dtype=torch.float32, device=device)

        self.rewards = torch.zeros((T, E), dtype=torch.float32, device=device)
        self.dones = torch.zeros((T, E), dtype=torch.bool, device=device)

        self.values = torch.zeros((T, E), dtype=torch.float32, device=device)

        self._t = 0

    def add(
        self,
        obs_t: torch.Tensor,          # [E, N, obs_dim]
        actions_t: torch.Tensor,      # [E, N]
        logp_t: torch.Tensor,         # [E, N]
        reward_team_t: torch.Tensor,  # [E]
        done_t: torch.Tensor,         # [E]
        value_t: torch.Tensor,        # [E]
    ) -> None:
        _require(self._t < self.T, f"RolloutBuffer overflow: tried to add step {self._t} with T={self.T}")
        self.obs[self._t] = obs_t
        self.actions[self._t] = actions_t
        self.logp[self._t] = logp_t
        self.rewards[self._t] = reward_team_t
        self.dones[self._t] = done_t
        self.values[self._t] = value_t
        self._t += 1

    def is_full(self) -> bool:
        return self._t >= self.T

    def compute_gae(self, *, gamma: float, gae_lambda: float, next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          advantages: [T, E]
          returns:    [T, E]
        """
        _require(self._t == self.T, f"compute_gae requires full buffer. Have {self._t}/{self.T} steps.")
        gamma = float(gamma)
        lam = float(gae_lambda)

        adv = torch.zeros((self.T, self.E), dtype=torch.float32, device=self.device)
        last_gae = torch.zeros((self.E,), dtype=torch.float32, device=self.device)

        for t in reversed(range(self.T)):
            not_done = (~self.dones[t]).float()
            v_next = next_value if t == self.T - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * v_next * not_done - self.values[t]
            last_gae = delta + gamma * lam * not_done * last_gae
            adv[t] = last_gae

        ret = adv + self.values
        return adv, ret

    def get_minibatches(
        self,
        advantages: torch.Tensor,   # [T, E]
        returns: torch.Tensor,      # [T, E]
        *,
        agent_ids: torch.Tensor,    # [N] int64
        role_ids: Optional[torch.Tensor],  # [N] int64
        minibatch_size: int,
        shuffle: bool = True,
    ):
        """
        Yields RolloutBatch.
        """
        _require(advantages.shape == (self.T, self.E), "advantages shape mismatch")
        _require(returns.shape == (self.T, self.E), "returns shape mismatch")

        # Normalize advantages across T*E for stability (team advantage)
        adv_flat = advantages.reshape(-1)
        adv_mean, adv_std = adv_flat.mean(), adv_flat.std(unbiased=False).clamp_min(1e-8)
        advantages_norm = (advantages - adv_mean) / adv_std

        # Actor batch: flatten (T,E,N)
        obs_a = self.obs.reshape(self.T * self.E * self.N, self.obs_dim)
        act_a = self.actions.reshape(-1)
        logp_old_a = self.logp.reshape(-1)

        # Repeat team advantage across agents
        adv_team = advantages_norm.reshape(self.T * self.E, 1).repeat(1, self.N).reshape(-1)

        # Agent indices repeated over (T,E)
        agent_idx = agent_ids.view(1, 1, self.N).repeat(self.T, self.E, 1).reshape(-1)
        role_idx = None
        if role_ids is not None:
            role_idx = role_ids.view(1, 1, self.N).repeat(self.T, self.E, 1).reshape(-1)

        # Critic batch: flatten (T,E)
        obs_c = self.obs.reshape(self.T * self.E, self.N, self.obs_dim)
        ret_c = returns.reshape(-1)
        val_old_c = self.values.reshape(-1)

        B = obs_a.shape[0]
        idx = torch.randperm(B, device=self.device) if shuffle else torch.arange(B, device=self.device)
        mb = int(minibatch_size)
        _require(mb > 0 and mb <= B, f"minibatch_size must be in (0, {B}], got {mb}")

        for start in range(0, B, mb):
            j = idx[start : start + mb]
            # Critic minibatch aligns on (t,e) not (t,e,n) -> we sample independently for critic later
            # For simplicity and minimalism, we train critic on full batch each epoch in the trainer.
            yield RolloutBatch(
                obs_actor=obs_a[j],
                act=act_a[j],
                logp_old=logp_old_a[j],
                adv=adv_team[j],
                agent_idx=agent_idx[j],
                role_id=None if role_idx is None else role_idx[j],
                obs_critic=obs_c,     # full critic batch
                ret=ret_c,
                val_old=val_old_c,
            )
