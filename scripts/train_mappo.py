from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from mappo_tarware.config import load_yaml, deep_update, parse_overrides
from mappo_tarware.seeding import set_global_seeds
from mappo_tarware.env_factory import cfg_from_dict, make_env
from mappo_tarware.wrapper import TARWareTensorWrapper
from mappo_tarware.networks import SharedDiscreteActor, CentralValueCritic
from mappo_tarware.buffer import RolloutBuffer
from mappo_tarware.logging_utils import MetricsLogger, make_run_dir, save_json
from mappo_tarware.checkpoint import save_checkpoint, load_checkpoint

def _team_reward(rew_vec: torch.Tensor, mode: str) -> torch.Tensor:
    # rew_vec: [E, N]
    if mode == "team_sum":
        return rew_vec.sum(dim=-1)
    if mode == "team_mean":
        return rew_vec.mean(dim=-1)
    raise ValueError(f"Unsupported reward aggregation for MAPPO trainer: {mode}")

@torch.no_grad()
def evaluate(
    *,
    actor: SharedDiscreteActor,
    env_cfg: Dict[str, Any],
    device: torch.device,
    episodes: int,
    seed: int,
    greedy: bool,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    env = make_env(cfg_from_dict(env_cfg))
    w = TARWareTensorWrapper(env, reward_mode=str(env_cfg.get("reward_mode", "team_sum")))
    w.seed(seed)

    specs = w.get_specs()
    agent_ids = torch.arange(specs.num_agents, device=device, dtype=torch.int64)
    role_ids = None
    if specs.role_ids is not None:
        role_ids = torch.as_tensor(specs.role_ids, device=device, dtype=torch.int64)

    returns = []
    deliveries = []
    lengths = []

    for ep in range(int(episodes)):
        obs = w.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        steps = 0
        ep_del = 0.0

        while not done:
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)  # [N, obs_dim]
            actions, _ = actor.sample(
                obs_t,
                agent_idx=agent_ids,
                role_id=role_ids,
                greedy=greedy,
            )
            next_obs, rew, done, info = w.step(actions.detach().cpu().numpy().astype(np.int64))
            # rew is [N] already; aggregate team metric for reporting:
            team = float(np.sum(rew)) if env_cfg.get("reward_mode", "team_sum") != "team_mean" else float(np.mean(rew))
            ep_ret += team
            steps += 1
            if "requested_deliveries_completed" in info:
                ep_del = float(info["requested_deliveries_completed"])
            elif "deliveries_completed" in info:
                ep_del = float(info["deliveries_completed"])

            obs = next_obs
            if max_steps is not None and steps >= int(max_steps):
                break

        returns.append(ep_ret)
        deliveries.append(ep_del)
        lengths.append(steps)

    out = {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "deliveries_mean": float(np.mean(deliveries)),
        "deliveries_std": float(np.std(deliveries)),
        "len_mean": float(np.mean(lengths)),
        "len_std": float(np.std(lengths)),
        "pick_rate_mean": float(np.mean([d / max(l, 1) for d, l in zip(deliveries, lengths)])),
    }
    return out

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config containing env/algo/run.")
    p.add_argument("--override", type=str, nargs="*", default=[], help="Overrides like algo.lr_actor=3e-4 run.seed=1")
    p.add_argument("--device", type=str, default=None, help="cpu|cuda|mps (optional)")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    cfg = deep_update(cfg, parse_overrides(args.override))

    run_cfg = cfg.get("run", {})
    env_cfg = cfg.get("env", {})
    algo_cfg = cfg.get("algo", {})

    seed = int(run_cfg.get("seed", 0))
    set_global_seeds(seed)

    device_str = args.device or run_cfg.get("device", "cpu")
    device = torch.device(device_str)

    run_dir = make_run_dir(run_cfg.get("root", "runs"), run_cfg.get("experiment", "tarware_mappo"))
    logger = MetricsLogger(run_dir, use_tensorboard=bool(run_cfg.get("tensorboard", True)))
    save_json(run_dir, "config_resolved.json", cfg)

    env = make_env(cfg_from_dict(env_cfg))
    reward_mode = str(env_cfg.get("reward_mode", "team_sum"))
    w = TARWareTensorWrapper(
        env,
        reward_mode=reward_mode,
        use_role_id=bool(algo_cfg.get("use_role_id", True)),
    )

    specs = w.get_specs()
    N, obs_dim, act_dim = specs.num_agents, specs.obs_dim, specs.act_dim
    num_agvs = getattr(specs, "num_agvs", None)
    if num_agvs is None and hasattr(w, "_num_agvs"):
        num_agvs = int(w._num_agvs)
    if num_agvs is None:
        # fallback: assume all agents are "AGVs" for delivery heuristic (worst-case)
        num_agvs = N

    # IDs for embeddings
    agent_ids = torch.arange(N, device=device, dtype=torch.int64)
    role_ids = None
    if specs.role_ids is not None and bool(algo_cfg.get("use_role_id", True)):
        role_ids = torch.as_tensor(specs.role_ids, device=device, dtype=torch.int64)

    actor = SharedDiscreteActor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_agents=N,
        num_roles=int(algo_cfg.get("num_roles", 2)),
        hidden=tuple(algo_cfg.get("actor_hidden", [256, 256])),
        use_agent_id=bool(algo_cfg.get("use_agent_id", True)),
        use_role_id=bool(algo_cfg.get("use_role_id", True)),
        embed_dim=int(algo_cfg.get("embed_dim", 16)),
    ).to(device)

    critic = CentralValueCritic(
        obs_dim=obs_dim,
        num_agents=N,
        hidden=tuple(algo_cfg.get("critic_hidden", [256, 256])),
    ).to(device)

    opt_actor = torch.optim.Adam(actor.parameters(), lr=float(algo_cfg.get("lr_actor", 3e-4)))
    opt_critic = torch.optim.Adam(critic.parameters(), lr=float(algo_cfg.get("lr_critic", 3e-4)))

    step = 0
    best_metric = None

    if args.resume:
        payload = load_checkpoint(
            args.resume,
            actor=actor,
            critic=critic,
            opt_actor=opt_actor,
            opt_critic=opt_critic,
            map_location=device,
        )
        step = int(payload.get("step", 0))
        best_metric = payload.get("best_metric", None)

    # Hyperparams
    T = int(algo_cfg.get("rollout_len_T", 128))
    E = int(algo_cfg.get("num_envs_E", 1))
    if E != 1:
        raise NotImplementedError("Baseline stack supports num_envs_E=1. Add vectorized envs later.")
    gamma = float(algo_cfg.get("gamma", 0.99))
    lam = float(algo_cfg.get("gae_lambda", 0.95))
    clip_eps = float(algo_cfg.get("clip_eps", 0.2))
    entropy_coef = float(algo_cfg.get("entropy_coef", 0.01))
    value_coef = float(algo_cfg.get("value_coef", 0.5))
    max_grad_norm = float(algo_cfg.get("max_grad_norm", 0.5))
    ppo_epochs = int(algo_cfg.get("ppo_epochs_K", 4))
    minibatch_size = int(algo_cfg.get("minibatch_size", 256))
    value_clip = bool(algo_cfg.get("value_clip", False))

    actor_batch_size = T * E * N
    if minibatch_size > actor_batch_size:
        print(f"[warn] minibatch_size={minibatch_size} > actor_batch_size={actor_batch_size}. Clamping.")
        minibatch_size = actor_batch_size

    updates = int(run_cfg.get("updates", 2000))
    eval_every = int(run_cfg.get("eval_every", 50))
    ckpt_every = int(run_cfg.get("ckpt_every", 50))
    eval_episodes = int(run_cfg.get("eval_episodes", 10))

    def _deliveries_step(info: Dict[str, Any], rew_vec_np: np.ndarray) -> int:
        """
        Step-level delivery heuristic:
          - If info has per-step delivery signals, use them outside (we handle counters separately).
          - Else infer from per-agent rewards: AGV delivery reward is +1.0.
        """
        # If wrapper exposes raw per-agent rewards, prefer that
        r_raw = None
        if isinstance(info, dict) and ("reward_per_agent" in info):
            try:
                r_raw = np.asarray(info["reward_per_agent"], dtype=np.float32).reshape(-1)
            except Exception:
                r_raw = None

        if r_raw is None:
            # If reward_mode is per_agent, rew_vec is per-agent; otherwise may be broadcasted team reward.
            if reward_mode == "per_agent":
                r_raw = np.asarray(rew_vec_np, dtype=np.float32).reshape(-1)

        if r_raw is None or r_raw.size != N:
            return 0

        # count AGV deliveries (+1.0) only among AGVs
        agv_r = r_raw[: int(num_agvs)]
        return int(np.sum(np.isclose(agv_r, 1.0, atol=1e-6)))

    # Episode trackers
    obs = w.reset(seed=seed)
    ep_return = 0.0
    ep_len = 0
    ep_deliveries = 0
    ep_count = 0
    # For counter-style env metrics (cumulative within episode)
    prev_del = 0

    for update in range(updates):
        buf = RolloutBuffer(T=T, E=1, N=N, obs_dim=obs_dim, device=device)

        for t in range(T):
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)  # [1,N,obs_dim]
            v_t = critic(obs_t).detach()  # [1]
            obs_actor = obs_t.squeeze(0)  # [N,obs_dim]

            act, logp = actor.sample(
                obs_actor,
                agent_idx=agent_ids,
                role_id=role_ids,
                greedy=False,
            )
            act_np = act.detach().cpu().numpy().astype(np.int64)

            next_obs, rew_vec, done, info = w.step(act_np)
            if info is None:
                info = {}
            elif not isinstance(info, dict):
                raise TypeError(f"Env step info must be dict (or None). Got {type(info)}: {info!r}")

            # Compute correct team scalar reward for:
            #  - buffer (GAE/returns)
            #  - episode return tracking
            team_r = float(info.get("team_reward", float(np.sum(rew_vec))))
            team_t = torch.tensor([team_r], device=device, dtype=torch.float32)  # [E=1]

            done_t = torch.tensor([done], device=device, dtype=torch.bool)

            buf.add(
                obs_t=obs_t,
                actions_t=act.view(1, N),
                logp_t=logp.view(1, N),
                reward_team_t=team_t,
                done_t=done_t,
                value_t=v_t,
            )

            # ---- episode tracking ----
            ep_return += team_r
            ep_len += 1

            # Prefer env-provided counters; compute delta so we can sum per-episode
            if "shelf_deliveries" in info:
                if info['shelf_deliveries'] > 0:
                    print(f"shelf_deliveries: {info['shelf_deliveries']}")
                cur = int(info["shelf_deliveries"])
                ep_deliveries += max(0, cur - prev_del)
                prev_del = cur
            else:
                print("[warn] env info missing 'shelf_deliveries' counter.")
                # fallback heuristic from rewards (works if per-agent rewards are available)
                ep_deliveries += _deliveries_step(info, rew_vec)

            obs = next_obs

            if done:
                ep_count += 1
                logger.log(step, {
                    "train/episode_return": float(ep_return),
                    "train/episode_len": int(ep_len),
                    "train/deliveries": float(ep_deliveries),
                })
                # Reset episode trackers + counters
                ep_return = 0.0
                ep_len = 0
                ep_deliveries = 0
                prev_del = 0

                obs = w.reset(seed=seed + step + 1)

            if step < 300:  # just early, or gate with a flag
                print(f"t={ep_len} shelf_deliveries={info.get('shelf_deliveries')} done={done}")

            step += 1

        # Bootstrap value
        next_obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        next_value = critic(next_obs_t).detach()  # [1]

        adv, ret = buf.compute_gae(gamma=gamma, gae_lambda=lam, next_value=next_value)

        # PPO update
        obs_c = buf.obs.reshape(T * E, N, obs_dim)
        returns_c = ret.reshape(-1)
        values_old_c = buf.values.reshape(-1)

        last_stats = {}
        for epoch in range(ppo_epochs):
            # --- Critic ---
            v_pred = critic(obs_c)
            if value_clip:
                v_clipped = values_old_c + torch.clamp(v_pred - values_old_c, -clip_eps, clip_eps)
                v_loss1 = (v_pred - returns_c) ** 2
                v_loss2 = (v_clipped - returns_c) ** 2
                v_loss = 0.5 * torch.mean(torch.maximum(v_loss1, v_loss2))
            else:
                v_loss = 0.5 * torch.mean((v_pred - returns_c) ** 2)

            opt_critic.zero_grad(set_to_none=True)
            (v_loss * value_coef).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            opt_critic.step()

            # --- Actor ---
            approx_kl_sum = 0.0
            clip_frac_sum = 0.0
            entropy_sum = 0.0
            policy_loss_sum = 0.0
            n_mb = 0

            for batch in buf.get_minibatches(
                advantages=adv,
                returns=ret,
                agent_ids=agent_ids,
                role_ids=role_ids,
                minibatch_size=minibatch_size,
                shuffle=True,
            ):
                logp_new, ent = actor.evaluate_actions(
                    batch.obs_actor,
                    batch.act,
                    agent_idx=batch.agent_idx,
                    role_id=batch.role_id,
                )
                ratio = torch.exp(logp_new - batch.logp_old)
                surr1 = ratio * batch.adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch.adv
                p_loss = -torch.mean(torch.minimum(surr1, surr2))
                ent_loss = -torch.mean(ent)
                loss = p_loss + (entropy_coef * ent_loss)

                opt_actor.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                opt_actor.step()

                approx_kl = torch.mean(batch.logp_old - logp_new).item()
                clip_frac = torch.mean((torch.abs(ratio - 1.0) > clip_eps).float()).item()

                approx_kl_sum += float(approx_kl)
                clip_frac_sum += float(clip_frac)
                entropy_sum += float(torch.mean(ent).item())
                policy_loss_sum += float(p_loss.item())
                n_mb += 1

            if n_mb > 0:
                last_stats = {
                    "policy_loss": policy_loss_sum / n_mb,
                    "value_loss": float(v_loss.item()),
                    "entropy": entropy_sum / n_mb,
                    "approx_kl": approx_kl_sum / n_mb,
                    "clip_frac": clip_frac_sum / n_mb,
                }

        # Explained variance
        with torch.no_grad():
            v_pred = critic(obs_c)
            var_y = torch.var(returns_c)
            explained_var = 0.0 if var_y.item() < 1e-3 else float(
                1.0 - torch.var(returns_c - v_pred) / (var_y + 1e-8)
            )
            mse = torch.mean((returns_c - v_pred) ** 2).item()
            # print(f"EV={explained_var:.3f} var(returns)={var_y.item():.6f} MSE={mse:.6f} v_mean={v_pred.mean().item():.4f} ret_mean={returns_c.mean().item():.4f}")

        logger.log(step, {
            "train/policy_loss": float(last_stats.get("policy_loss", 0.0)),
            "train/value_loss": float(last_stats.get("value_loss", 0.0)) * value_coef,
            "train/entropy": float(last_stats.get("entropy", 0.0)),
            "train/approx_kl": float(last_stats.get("approx_kl", 0.0)),
            "train/clip_frac": float(last_stats.get("clip_frac", 0.0)),
            "train/explained_var": float(explained_var),
        })

        print(
            f"[upd {update+1:5d}/{updates} | step {step:8d}] "
        #     f"pi={last_stats.get('policy_loss', 0.0): .4f}  "
        #     f"v={last_stats.get('value_loss', 0.0): .4f}  "
        #     f"H={last_stats.get('entropy', 0.0): .4f}  "
        #     f"KL={last_stats.get('approx_kl', 0.0): .6f}  "
        #     f"clip={last_stats.get('clip_frac', 0.0): .3f}  "
        #     f"EV={float(explained_var): .3f}"
        )

        # Periodic evaluation + checkpointing
        if (update + 1) % eval_every == 0:
            eval_out = evaluate(
                actor=actor,
                env_cfg=env_cfg,
                device=device,
                episodes=eval_episodes,
                seed=seed + 10_000,
                greedy=bool(run_cfg.get("eval_greedy", False)),
            )
            logger.log(step, {f"eval/{k}": v for k, v in eval_out.items()})
            save_json(run_dir, "eval_latest.json", eval_out)

            metric = eval_out.get("deliveries_mean", eval_out.get("return_mean"))
            if metric is not None and (best_metric is None or metric > best_metric):
                best_metric = metric
                save_checkpoint(
                    run_dir / "checkpoints" / "best.pt",
                    actor=actor,
                    critic=critic,
                    opt_actor=opt_actor,
                    opt_critic=opt_critic,
                    step=step,
                    cfg=cfg,
                    best_metric=best_metric,
                    save_rng=True,
                )

        if (update + 1) % ckpt_every == 0:
            save_checkpoint(
                run_dir / "checkpoints" / "latest.pt",
                actor=actor,
                critic=critic,
                opt_actor=opt_actor,
                opt_critic=opt_critic,
                step=step,
                cfg=cfg,
                best_metric=best_metric,
                save_rng=True,
            )

    # Final save
    save_checkpoint(
        run_dir / "checkpoints" / "latest.pt",
        actor=actor,
        critic=critic,
        opt_actor=opt_actor,
        opt_critic=opt_critic,
        step=step,
        cfg=cfg,
        best_metric=best_metric,
        save_rng=True,
    )
    logger.close()
    print(f"Done. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
