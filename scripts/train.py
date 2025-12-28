"""Train MAPPO on TA-RWARE using TorchRL primitives.

This script is intentionally minimal:
- build EnvBase wrapper
- collect on-policy rollouts
- compute GAE
- optimize PPO-style objective

You will likely adapt the following to your TorchRL version:
- collector usage (SyncDataCollector vs other)
- ReplayBuffer/ storage (optional; for on-policy you can train directly on the batch)
- ProbabilisticActor wrappers

Usage:
  python scripts/train.py --env configs/env_default.yaml --algo configs/mappo_default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import json
import time

from src.utils.config import load_yaml, merge_dicts
from src.utils.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, type=str)
    p.add_argument("--algo", required=True, type=str)
    p.add_argument("--out", default="runs", type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env_cfg = load_yaml(args.env)
    algo_cfg = load_yaml(args.algo)
    cfg = merge_dicts(env_cfg, {"algo": algo_cfg})

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "configs").mkdir(exist_ok=True)

    # Save configs
    (out_dir / "configs" / "env.yaml").write_text(Path(args.env).read_text())
    (out_dir / "configs" / "algo.yaml").write_text(Path(args.algo).read_text())

    seed_everything(int(algo_cfg.get("seed", env_cfg.get("seed", 0))))

    try:
        import torch
    except Exception as e:
        raise RuntimeError("torch is required") from e

    # TorchRL imports (fail fast with clear error)
    try:
        from torchrl.collectors import SyncDataCollector
        from torchrl.envs.utils import check_env_specs
        from tensordict.nn import TensorDictModule
        from torchrl.modules import ProbabilisticActor
        from torchrl.modules.distributions import MaskedCategorical
    except Exception as e:
        raise RuntimeError(
            "TorchRL / TensorDict not installed (or version mismatch). "
            "Install from requirements.txt and ensure torchrl imports work."
        ) from e

    from src.envs.tarware_env import TARWareTorchRLEnv
    from src.algos.mappo.networks import build_modules
    from src.algos.mappo.loss import PPOClipLoss

    device = torch.device(env_cfg.get("device", "cpu"))

    env = TARWareTorchRLEnv(env_cfg, device=device)
    check_env_specs(env)

    # Specs
    n_agents = env.n_agents
    obs_dim = env.obs_dim
    n_actions = env.n_actions

    modules = build_modules(
        obs_dim=obs_dim,
        n_actions=n_actions,
        n_agents_total=n_agents,
        hidden_size=int(algo_cfg.get("hidden_size", 256)),
        n_layers=int(algo_cfg.get("n_layers", 2)),
    )

    # TODO(role split): map agent indices -> role groups based on your TA-RWARE env.
    # For now, treat *all* agents as using the same actor (swap later).

    actor_net = modules.actor_agv.to(device)
    critic_net = modules.critic.to(device)

    # Actor wrapper: obs -> logits
    actor_td = TensorDictModule(
        actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "logits")],
    )

    # Probabilistic actor: logits (+ mask) -> action + log_prob
    policy = ProbabilisticActor(
        module=actor_td,
        in_keys=[("agents", "logits"), ("agents", "action_mask")],
        distribution_class=MaskedCategorical,
        distribution_kwargs={"mask": ("agents", "action_mask")},
        return_log_prob=True,
        out_keys=[("agents", "action")],
    )

    # Critic wrapper: concat obs -> state_value
    def _concat_obs(td):
        obs = td.get(("agents", "observation"))  # [..., n_agents, obs_dim]
        flat = obs.reshape(*obs.shape[:-2], -1)
        return flat

    class CriticWrapper(torch.nn.Module):
        def __init__(self, critic):
            super().__init__()
            self.critic = critic

        def forward(self, obs_flat):
            return self.critic(obs_flat)

    critic_td = TensorDictModule(
        CriticWrapper(critic_net),
        in_keys=[("_global", "obs_flat")],
        out_keys=[("_global", "state_value")],
    )

    # Loss + optim
    loss_fn = PPOClipLoss(
        clip_epsilon=float(algo_cfg.get("clip_epsilon", 0.2)),
        entropy_coef=float(algo_cfg.get("entropy_coef", 0.01)),
        value_coef=float(algo_cfg.get("value_coef", 0.5)),
    )

    optim = torch.optim.Adam(list(actor_net.parameters()) + list(critic_net.parameters()), lr=float(algo_cfg.get("lr", 3e-4)))

    frames_per_batch = int(algo_cfg.get("frames_per_batch", 6000))

    # Collector: rollout with current policy
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch * 10_000,  # effectively "infinite"; you stop manually
        device=device,
        storing_device=device,
    )

    # Main loop (stop with Ctrl+C; add n_iters if you want a finite run)
    step = 0
    for batch in collector:
        step += 1

        # Batch is a rollout TensorDict. You must:
        # 1) compute value estimates with critic
        # 2) compute advantages + value targets (GAE)
        # 3) run PPO epochs over minibatches

        # ---- build critic inputs
        obs_flat = _concat_obs(batch)
        batch.set(("_global", "obs_flat"), obs_flat)
        critic_td(batch)

        # TODO: compute GAE (use TorchRL ValueEstimators.GAE or local compute_gae)
        # For now, place-holders:
        batch.set(("agents", "advantage"), torch.zeros_like(batch.get(("agents", "reward"))))
        batch.set(("agents", "value_target"), torch.zeros_like(batch.get(("agents", "reward"))))

        # PPO epochs (placeholder; implement minibatching)
        optim.zero_grad(set_to_none=True)
        # Recompute new log_probs under current policy
        policy(batch)

        losses = loss_fn(
            log_prob=batch.get(("agents", "sample_log_prob")),
            old_log_prob=batch.get(("agents", "sample_log_prob")),  # TODO: store behavior logp before update
            advantage=batch.get(("agents", "advantage")),
            value_pred=batch.get(("_global", "state_value")),
            value_target=batch.get(("agents", "value_target")),
            entropy=batch.get(("agents", "entropy")) if batch.has(("agents", "entropy")) else None,
        )
        losses["loss_total"].backward()
        torch.nn.utils.clip_grad_norm_(list(actor_net.parameters()) + list(critic_net.parameters()), float(algo_cfg.get("max_grad_norm", 1.0)))
        optim.step()

        if step % int(algo_cfg.get("log_every", 1)) == 0:
            print({k: float(v.detach().cpu()) for k, v in losses.items()})

        # TODO: evaluation, checkpointing, TensorBoard


if __name__ == "__main__":
    main()
