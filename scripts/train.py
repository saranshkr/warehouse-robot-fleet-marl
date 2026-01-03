"""Train MAPPO (custom) on TA-RWARE using TorchRL EnvBase wrapper.

Minimal, practical MAPPO v1:
- 2 actors: AGV + Picker
- 1 centralized critic: V(s) where s = concat(all obs)
- Team reward: mean over agents (you can switch to sum)
- Action masking on day 1
- On-policy rollout buffer + PPO update

Usage:
  python -m scripts.train --env configs/env_default.yaml --algo configs/mappo_default.yaml
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torchrl.envs.utils import check_env_specs
from src.envs.tarware_env import TARWareTorchRLEnv, cfg_from_dict

from src.utils.config import load_yaml, merge_dicts
from src.utils.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, type=str)
    p.add_argument("--algo", required=True, type=str)
    p.add_argument("--out", default="runs", type=str)
    return p.parse_args()


def _get_step_fields(td):
    """TorchRL env.step may return either plain keys or put them under 'next'."""
    if td.get(("next", "agents", "observation")) is not None:
        nxt = td.get("next")
        return (
            nxt.get(("agents", "observation")),
            nxt.get(("agents", "action_mask")),
            nxt.get(("agents", "reward")),
            nxt.get("done"),
            nxt,
        )
    return (
        td.get(("agents", "observation")),
        td.get(("agents", "action_mask")),
        td.get(("agents", "reward")),
        td.get("done"),
        td,
    )


def _mask_logits(logits, mask):
    # mask: True means valid
    return logits.masked_fill(~mask, -1e9)


def compute_gae(
    rewards: "torch.Tensor",     # [T]
    values: "torch.Tensor",      # [T]
    dones: "torch.Tensor",       # [T] bool or {0,1}
    last_value: "torch.Tensor",  # scalar
    gamma: float,
    lam: float,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """GAE for scalar team reward + scalar V(s). Returns (adv, returns), both [T]."""

    T = rewards.shape[0]
    adv = torch.zeros(T, device=rewards.device, dtype=torch.float32)
    gae = torch.tensor(0.0, device=rewards.device)

    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t].float()
        v_next = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nonterminal * v_next - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae

    ret = adv + values
    return adv, ret


@dataclass
class ActorCritic:
    actor_agv: "torch.nn.Module"
    actor_picker: "torch.nn.Module"
    critic: "torch.nn.Module"


def build_actor(obs_dim: int, n_actions: int, hidden: int, n_layers: int):
    layers = []
    in_dim = obs_dim
    for _ in range(n_layers):
        layers.append(torch.nn.Linear(in_dim, hidden))
        layers.append(torch.nn.ReLU())
        in_dim = hidden
    layers.append(torch.nn.Linear(in_dim, n_actions))
    return torch.nn.Sequential(*layers)


def build_critic(state_dim: int, hidden: int, n_layers: int):
    layers = []
    in_dim = state_dim
    for _ in range(n_layers):
        layers.append(torch.nn.Linear(in_dim, hidden))
        layers.append(torch.nn.ReLU())
        in_dim = hidden
    layers.append(torch.nn.Linear(in_dim, 1))
    return torch.nn.Sequential(*layers)


def main() -> None:
    args = parse_args()
    env_cfg = load_yaml(args.env)
    algo_cfg = load_yaml(args.algo)
    cfg = merge_dicts(env_cfg, {"algo": algo_cfg})

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    (out_dir / "configs").mkdir(exist_ok=True)
    (out_dir / "configs" / "env.yaml").write_text(Path(args.env).read_text())
    (out_dir / "configs" / "algo.yaml").write_text(Path(args.algo).read_text())
    
    save_every = int(algo_cfg.get("save_every", 50))

    seed_everything(int(algo_cfg.get("seed", env_cfg.get("seed", 0))))

    device = torch.device(env_cfg.get("device", "cpu"))

    env = TARWareTorchRLEnv(cfg_from_dict(env_cfg), device=device)
    check_env_specs(env)

    n_agents = env.n_agents
    obs_dim = env.obs_dim
    n_actions = env.n_actions

    n_agv = int(env_cfg.get("n_agv", n_agents))
    n_picker = int(env_cfg.get("n_picker", max(0, n_agents - n_agv)))
    assert n_agv + n_picker == n_agents, f"n_agv+n_picker must equal n_agents ({n_agents})"

    agv_idx = torch.arange(0, n_agv, device=device)
    picker_idx = torch.arange(n_agv, n_agents, device=device)

    hidden = int(algo_cfg.get("hidden_size", 256))
    n_layers = int(algo_cfg.get("n_layers", 2))

    model = ActorCritic(
        actor_agv=build_actor(obs_dim, n_actions, hidden, n_layers).to(device),
        actor_picker=build_actor(obs_dim, n_actions, hidden, n_layers).to(device),
        critic=build_critic(n_agents * obs_dim, hidden, n_layers).to(device),
    )

    lr = float(algo_cfg.get("lr", 3e-4))
    optim = torch.optim.Adam(
        list(model.actor_agv.parameters())
        + list(model.actor_picker.parameters())
        + list(model.critic.parameters()),
        lr=lr,
    )

    # MAPPO hyperparams
    rollout_T = int(algo_cfg.get("rollout_T", 256))
    ppo_epochs = int(algo_cfg.get("ppo_epochs", 4))
    minibatch_size = int(algo_cfg.get("minibatch_size", 64))
    gamma = float(algo_cfg.get("gamma", 0.99))
    lam = float(algo_cfg.get("gae_lambda", 0.95))
    clip_eps = float(algo_cfg.get("clip_epsilon", 0.2))
    entropy_coef = float(algo_cfg.get("entropy_coef", 0.01))
    value_coef = float(algo_cfg.get("value_coef", 0.5))
    max_grad_norm = float(algo_cfg.get("max_grad_norm", 0.5))
    log_every = int(algo_cfg.get("log_every", 1))
    total_iters = int(algo_cfg.get("iters", 2000))

    # Optional: reproducible sampling stream
    gen = torch.Generator(device=device)
    gen.manual_seed(int(algo_cfg.get("seed", 0)))

    def policy_act(obs, mask):
        """obs: [A, obs_dim], mask: [A, n_actions] -> action [A], logp [A], entropy [A]"""
        logits = torch.zeros((n_agents, n_actions), device=device, dtype=torch.float32)

        if n_agv > 0:
            logits_agv = model.actor_agv(obs.index_select(0, agv_idx))
            logits.index_copy_(0, agv_idx, logits_agv)

        if n_picker > 0:
            logits_pick = model.actor_picker(obs.index_select(0, picker_idx))
            logits.index_copy_(0, picker_idx, logits_pick)

        logits = _mask_logits(logits, mask)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        ent = dist.entropy()
        return action.to(torch.int64), logp, ent

    def critic_value(obs):
        """obs: [A, obs_dim] -> scalar V"""
        flat = obs.reshape(-1)  # [A*obs_dim]
        v = model.critic(flat).squeeze(-1)  # scalar
        return v

    # ---------------- training loop ----------------
    it = 0
    td = env.reset()

    for it in range(1, total_iters + 1):
        # Rollout storage
        obs_buf = torch.zeros((rollout_T, n_agents, obs_dim), device=device)
        mask_buf = torch.zeros((rollout_T, n_agents, n_actions), device=device, dtype=torch.bool)
        act_buf = torch.zeros((rollout_T, n_agents), device=device, dtype=torch.int64)
        logp_buf = torch.zeros((rollout_T, n_agents), device=device)
        ent_buf = torch.zeros((rollout_T, n_agents), device=device)
        done_buf = torch.zeros((rollout_T,), device=device, dtype=torch.bool)
        rew_team_buf = torch.zeros((rollout_T,), device=device)
        val_buf = torch.zeros((rollout_T,), device=device)

        # collect rollout
        for t in range(rollout_T):
            obs = td.get(("agents", "observation"))
            mask = td.get(("agents", "action_mask"))

            obs_buf[t] = obs
            mask_buf[t] = mask

            with torch.no_grad():
                v_t = critic_value(obs)
                action, logp, ent = policy_act(obs, mask)

            val_buf[t] = v_t
            act_buf[t] = action
            logp_buf[t] = logp
            ent_buf[t] = ent

            td.set(("agents", "action"), action)
            td_step = env.step(td)

            obs_next, mask_next, reward_next, done_next, td_next = _get_step_fields(td_step)

            # reward_next: [A,1] -> team scalar
            r_team = reward_next.mean().squeeze(-1)
            rew_team_buf[t] = r_team

            done_bool = bool(done_next.item())
            done_buf[t] = torch.tensor(done_bool, device=device)

            # advance
            td = td_next
            if done_bool:
                td = env.reset()

        # bootstrap
        with torch.no_grad():
            last_obs = td.get(("agents", "observation"))
            last_value = critic_value(last_obs)

        adv, ret = compute_gae(
            rewards=rew_team_buf,
            values=val_buf,
            dones=done_buf,
            last_value=last_value,
            gamma=gamma,
            lam=lam,
        )
        # normalize adv
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO update
        T = rollout_T
        idx = torch.arange(T, device=device)

        losses_acc = {"loss_total": 0.0, "loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0}

        for _epoch in range(ppo_epochs):
            perm = idx[torch.randperm(T, generator=gen)]
            for start in range(0, T, minibatch_size):
                mb = perm[start : start + minibatch_size]
                obs_mb = obs_buf.index_select(0, mb)        # [B, A, obs_dim]
                mask_mb = mask_buf.index_select(0, mb)      # [B, A, n_actions]
                act_mb = act_buf.index_select(0, mb)        # [B, A]
                old_logp_mb = logp_buf.index_select(0, mb)  # [B, A]
                adv_mb = adv.index_select(0, mb)            # [B]
                ret_mb = ret.index_select(0, mb)            # [B]

                # recompute logits for each timestep and role, then logp under current policy
                B = obs_mb.shape[0]
                logits_mb = torch.zeros((B, n_agents, n_actions), device=device)

                if n_agv > 0:
                    logits_agv = model.actor_agv(obs_mb[:, :n_agv, :].reshape(-1, obs_dim))
                    logits_agv = logits_agv.view(B, n_agv, n_actions)
                    logits_mb[:, :n_agv, :] = logits_agv

                if n_picker > 0:
                    logits_pick = model.actor_picker(obs_mb[:, n_agv:, :].reshape(-1, obs_dim))
                    logits_pick = logits_pick.view(B, n_picker, n_actions)
                    logits_mb[:, n_agv:, :] = logits_pick

                logits_mb = _mask_logits(logits_mb, mask_mb)

                dist = torch.distributions.Categorical(logits=logits_mb)
                new_logp = dist.log_prob(act_mb)            # [B, A]
                entropy = dist.entropy().mean()             # scalar

                # team advantage broadcast to agents
                adv_agents = adv_mb.view(B, 1).expand(B, n_agents)

                ratio = torch.exp(new_logp - old_logp_mb)
                surr1 = ratio * adv_agents
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_agents
                loss_pi = -(torch.min(surr1, surr2)).mean()

                # value loss (centralized critic)
                obs_flat = obs_mb.reshape(B, -1)
                v_pred = model.critic(obs_flat).squeeze(-1)  # [B]
                loss_v = torch.mean((v_pred - ret_mb) ** 2)

                loss_total = loss_pi + value_coef * loss_v - entropy_coef * entropy

                optim.zero_grad(set_to_none=True)
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.actor_agv.parameters())
                    + list(model.actor_picker.parameters())
                    + list(model.critic.parameters()),
                    max_grad_norm,
                )
                optim.step()

                losses_acc["loss_total"] += float(loss_total.detach().cpu())
                losses_acc["loss_pi"] += float(loss_pi.detach().cpu())
                losses_acc["loss_v"] += float(loss_v.detach().cpu())
                losses_acc["entropy"] += float(entropy.detach().cpu())

        # logging
        if it % log_every == 0:
            denom = max(1, (ppo_epochs * ((T + minibatch_size - 1) // minibatch_size)))
            msg = {
                "iter": it,
                "team_reward_mean_rollout": float(rew_team_buf.mean().detach().cpu()),
                "done_frac": float(done_buf.float().mean().detach().cpu()),
                "loss_total": losses_acc["loss_total"] / denom,
                "loss_pi": losses_acc["loss_pi"] / denom,
                "loss_v": losses_acc["loss_v"] / denom,
                "entropy": losses_acc["entropy"] / denom,
            }
            print(msg)

        # TODO: checkpoint + eval hooks
        if it % save_every == 0:
            ckpt_path = ckpt_dir / f"iter_{it:06d}.pt"
            torch.save(
                {
                    "iter": it,
                    "actor_agv": model.actor_agv.state_dict(),
                    "actor_picker": model.actor_picker.state_dict(),
                    "critic": model.critic.state_dict(),
                    "optim": optim.state_dict(),
                    "env_cfg": env_cfg,
                    "algo_cfg": algo_cfg,
                },
                ckpt_path,
            )
            torch.save({"path": str(ckpt_path)}, ckpt_dir / "latest.pt")


if __name__ == "__main__":
    main()
