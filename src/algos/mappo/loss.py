"""Custom MAPPO loss (PPO-style) operating on TensorDict batches.

If you prefer, you can replace this with TorchRL's built-in ClipPPOLoss / PPOLoss.
TorchRL provides PPO loss modules that expect the actor to output actions + log-probs, and a critic that outputs a scalar value. See docs for PPOLoss.https://docs.pytorch.org/rl/main/reference/generated/torchrl.objectives.PPOLoss.html

Why keep a custom loss?
- Easier to debug keys/shapes in a custom multi-agent environment.
- Lets you add MAPPO-specific quirks (shared reward, broadcast value, role-wise losses).

Expected keys in the input TensorDict (common pattern):
- ('agents','advantage')
- ('agents','value_target')
- ('agents','action')
- ('agents','old_log_prob')
- ('agents','log_prob')  (from current actor)
- ('agents','entropy')   (from current dist, optional)
- ('state_value') or ('agents','state_value') (from critic)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PPOLossOutput:
    loss_total: torch.Tensor
    loss_policy: torch.Tensor
    loss_value: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


def ppo_clipped_loss(
    *,
    advantage: torch.Tensor,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    value_pred: torch.Tensor,
    value_target: torch.Tensor,
    clip_epsilon: float,
    entropy: torch.Tensor | None = None,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
) -> PPOLossOutput:
    """Compute PPO clipped objective (supports batched multi-agent tensors).

    All tensors should already be broadcastable to the same shape.
    """
    # Normalize advantage per minibatch (common PPO trick)
    adv = advantage
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    ratio = torch.exp(log_prob - old_log_prob)
    ratio_clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    surrogate1 = ratio * adv
    surrogate2 = ratio_clipped * adv
    loss_policy = -torch.minimum(surrogate1, surrogate2).mean()

    # Value loss (MSE)
    loss_value = 0.5 * torch.mean((value_pred - value_target) ** 2)

    # Diagnostics
    approx_kl = 0.5 * torch.mean((log_prob - old_log_prob) ** 2)
    clip_fraction = torch.mean((ratio != ratio_clipped).float())

    if entropy is None:
        entropy = torch.zeros((), device=loss_policy.device)

    loss_total = loss_policy + value_coef * loss_value - entropy_coef * entropy.mean()

    return PPOLossOutput(
        loss_total=loss_total,
        loss_policy=loss_policy,
        loss_value=loss_value,
        entropy=entropy.mean(),
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
    )


class PPOClipLoss(torch.nn.Module):
    """Small nn.Module wrapper around `ppo_clipped_loss`.

    This is convenient because it gives you a stateful object with fixed hyperparams,
    while keeping the actual PPO math transparent.
    """

    def __init__(self, clip_epsilon: float, entropy_coef: float, value_coef: float):
        super().__init__()
        self.clip_epsilon = float(clip_epsilon)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)

    def forward(
        self,
        *,
        advantage: torch.Tensor,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        value_pred: torch.Tensor,
        value_target: torch.Tensor,
        entropy: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = ppo_clipped_loss(
            advantage=advantage,
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            value_pred=value_pred,
            value_target=value_target,
            clip_epsilon=self.clip_epsilon,
            entropy=entropy,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
        )
        return {
            "loss_total": out.loss_total,
            "loss_policy": out.loss_policy,
            "loss_value": out.loss_value,
            "entropy": out.entropy,
            "approx_kl": out.approx_kl,
            "clip_fraction": out.clip_fraction,
        }
