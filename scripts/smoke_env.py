"""Quick env sanity script.

- builds TA-RWARE TorchRL env wrapper
- runs a random valid policy for one episode
- prints basic counters

Usage:
  python scripts/smoke_env.py --config configs/env_default.yaml
"""

from __future__ import annotations

import argparse

from src.utils.config import load_yaml
from src.utils.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    # allow both:
    # - cfg["seed"] at top-level
    # - cfg["env"]["seed"] nested
    seed = int(cfg.get("seed", cfg.get("env", {}).get("seed", 0)))
    seed_everything(seed)

    import torch
    from torchrl.envs.utils import check_env_specs

    from src.envs.tarware_env import TARWareEnvConfig, TARWareTorchRLEnv, cfg_from_dict
    from src.baselines.random_policy import random_valid_actions

    # allow both:
    # - cfg["device"] at top-level
    # - cfg["env"]["device"] (if you choose)
    device_str = cfg.get("device", cfg.get("env", {}).get("device", "cpu"))
    device = torch.device(device_str)

    # allow nested env config, else treat top-level as env config
    env_cfg_dict = cfg.get("env", cfg)

    # Build dataclass safely (ignore unrelated keys like "device", "algo", etc.)
    allowed = set(TARWareEnvConfig.__dataclass_fields__.keys())
    env_cfg_clean = {k: v for k, v in env_cfg_dict.items() if k in allowed}
    env_cfg = TARWareEnvConfig(**env_cfg_clean)

    env = TARWareTorchRLEnv(cfg_from_dict(env_cfg), device=device)
    check_env_specs(env)

    td = env.reset()

    obs = td.get(("agents", "observation"))
    mask = td.get(("agents", "action_mask"))
    print(f"[reset] obs shape={tuple(obs.shape)} dtype={obs.dtype} device={obs.device}")
    print(f"[reset] mask shape={tuple(mask.shape)} dtype={mask.dtype} device={mask.device}")
    print(f"[env] n_agents={env.n_agents} obs_dim={env.obs_dim} n_actions={env.n_actions}")

    done = False
    steps = 0
    total_reward = 0.0

    # hard cap to avoid infinite loops in early debugging
    max_steps = int(getattr(env_cfg, "max_steps", 500))

    while not done and steps < max_steps:
        mask = td.get(("agents", "action_mask"))
        actions = random_valid_actions(mask)  # should return int64 [n_agents] on same device
        td.set(("agents", "action"), actions)

        td = env.step(td)

        if td.get(("next", "agents", "reward")) is not None:
            r = td.get(("next", "agents", "reward"))
            done = bool(td.get(("next", "done")).item())
            td = td.get("next")  # advance state
        else:
            r = td.get(("agents", "reward"))
            done = bool(td.get("done").item())

        total_reward += float(r.mean().item())
        steps += 1

        # r = td.get(("agents", "reward"))  # [n_agents, 1]
        # total_reward += float(r.mean().item())
        # steps += 1
        done = bool(td.get("done").item())

    env.close()

    status = "done" if done else "hit max_steps"
    print(f"Episode {status} in {steps} steps. mean-reward-sum={total_reward:.3f}")


if __name__ == "__main__":
    main()
