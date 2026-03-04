from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

@dataclass
class PpoLossOut:
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor
    approx_kl: float
    clip_frac: float
    explained_var: float

def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    var_y = torch.var(y_true)
    if var_y.item() < 1e-12:
        return 0.0
    return float(1.0 - torch.var(y_true - y_pred) / (var_y + 1e-12))

def ppo_loss_discrete(
    *,
    logp: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    value_clip: bool = False,
    values_old: Optional[torch.Tensor] = None,
) -> PpoLossOut:
    """
    Standard PPO losses for discrete action actor and scalar critic.
    """
    # Policy
    ratio = torch.exp(logp - logp_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.mean(torch.minimum(surr1, surr2))

    # KL + clip frac (approx)
    approx_kl = torch.mean(logp_old - logp).item()
    clip_frac = torch.mean((torch.abs(ratio - 1.0) > clip_eps).float()).item()

    # Value
    if value_clip:
        if values_old is None:
            raise ValueError("values_old is required when value_clip=True")
        v_clipped = values_old + torch.clamp(values - values_old, -clip_eps, clip_eps)
        v_loss1 = (values - returns) ** 2
        v_loss2 = (v_clipped - returns) ** 2
        value_loss = 0.5 * torch.mean(torch.maximum(v_loss1, v_loss2))
    else:
        value_loss = 0.5 * torch.mean((values - returns) ** 2)

    # Entropy bonus (maximize entropy -> subtract)
    entropy_loss = -torch.mean(entropy)

    ev = explained_variance(values.detach(), returns.detach())

    return PpoLossOut(
        policy_loss=policy_loss,
        value_loss=value_loss * value_coef,
        entropy_loss=entropy_loss * entropy_coef,
        approx_kl=float(approx_kl),
        clip_frac=float(clip_frac),
        explained_var=float(ev),
    )
