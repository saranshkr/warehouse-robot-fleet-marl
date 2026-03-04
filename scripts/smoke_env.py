"""Quick env sanity script.

- builds TA-RWARE TorchRL env wrapper
- runs a random valid policy for one episode
- prints basic counters

Usage:
  python scripts/smoke_env.py --config configs/env_default.yaml
"""

from __future__ import annotations

import argparse
import torch
from torchrl.envs.utils import check_env_specs

from src.envs.tarware_env import TARWareEnvConfig, TARWareTorchRLEnv, cfg_from_dict
from src.baselines.random_policy import random_valid_actions

from src.utils.config import load_yaml
from src.utils.seeding import seed_everything


def parse_seeds(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--max_steps", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    # allow both:
    # - cfg["seed"] at top-level
    # - cfg["env"]["seed"] nested
    base_seed = int(cfg.get("seed", cfg.get("env", {}).get("seed", 0)))
    seed_everything(base_seed)

    # allow both:
    # - cfg["device"] at top-level
    # - cfg["env"]["device"] nested
    device_str = cfg.get("device", cfg.get("env", {}).get("device", "cpu"))
    device = torch.device(device_str)

    # allow nested env config, else treat top-level as env config
    env_cfg_dict = cfg.get("env", cfg)

    # Build dataclass safely (ignore unrelated keys like "device", "algo", etc.)
    allowed = set(TARWareEnvConfig.__dataclass_fields__.keys())
    env_cfg_clean = {k: v for k, v in env_cfg_dict.items() if k in allowed}
    env_cfg = TARWareEnvConfig(**env_cfg_clean)

    # Parse sweep params
    seeds = parse_seeds(args.seeds)
    episodes_per_seed = int(args.episodes)

    # hard cap to avoid infinite loops in early debugging
    max_steps = int(args.max_steps) if args.max_steps is not None else int(getattr(env_cfg, "max_steps", 500))

    print(f"[diag] seeds={seeds} episodes_per_seed={episodes_per_seed} max_steps={max_steps}")

    # Create env ONCE (that’s fine); we pass seed via reset tensordict.
    env = TARWareTorchRLEnv(cfg_from_dict(env_cfg), device=device)
    # print(f"{env.keys()}")
    check_env_specs(env)

    # One-time reset prints (shape sanity)
    td0 = env.reset()
    obs0 = td0.get(("agents", "observation"))
    mask0 = td0.get(("agents", "action_mask"))
    print(f"[reset] obs shape={tuple(obs0.shape)} dtype={obs0.dtype} device={obs0.device}")
    print(f"[reset] mask shape={tuple(mask0.shape)} dtype={mask0.dtype} device={mask0.device}")
    print(f"[env] n_agents={env.n_agents} obs_dim={env.obs_dim} n_actions={env.n_actions}")

    from tensordict import TensorDict  # local import ok

    # Run sweep: seeds x episodes
    for seed in seeds:
        for ep in range(episodes_per_seed):
            # Use TorchRL-style: pass seed via input tensordict so env._reset can read it
            td_seed = TensorDict({"seed": torch.tensor([seed], dtype=torch.int64)}, batch_size=[])
            td = env.reset(td_seed)

            done = False
            steps = 0
            total_reward = 0.0

            # Track deliveries each episode
            d0 = td.get(("stats", "shelf_deliveries"), None)
            deliveries_last = float(d0.item()) if d0 is not None else 0.0

            while not done and steps < max_steps:
                mask = td.get(("agents", "action_mask"))
                actions = random_valid_actions(mask)
                td.set(("agents", "action"), actions)

                td_step = env.step(td)
                td = td_step.get("next", td_step)  # advance state (TorchRL style)

                r = td.get(("agents", "reward"))  # [n_agents, 1]
                total_reward += float(r.mean().item())
                steps += 1
                done = bool(td.get("done").item())

                d = td.get(("stats", "shelf_deliveries"), None)
                if d is not None:
                    deliveries_last = float(d.item())

                # Optional: periodic debug prints (keep lightweight)
                if steps % 200 == 0:
                    clashes_t = td.get(("stats", "clashes"), None)
                    stucks_t = td.get(("stats", "stucks"), None)
                    clashes = float(clashes_t.item()) if clashes_t is not None else 0.0
                    stucks = float(stucks_t.item()) if stucks_t is not None else 0.0
                    print(f"  seed={seed} ep={ep+1} step={steps} deliveries={deliveries_last} clashes={clashes} stucks={stucks}")

            status = "done" if done else "hit max_steps"
            print(
                f"seed={seed} ep={ep+1}/{episodes_per_seed} "
                f"{status} steps={steps} mean-reward-sum={total_reward:.3f} deliveries_last={deliveries_last}"
            )

    env.close()


if __name__ == "__main__":
    main()
