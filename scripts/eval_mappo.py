from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mappo_tarware.config import load_yaml
from mappo_tarware.env_factory import cfg_from_dict
from mappo_tarware.networks import SharedDiscreteActor
from mappo_tarware.checkpoint import load_checkpoint
from scripts.train_mappo import evaluate  # reuse eval logic

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, required=True, help="Same YAML used for training (for env/algo specs).")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    env_cfg = cfg.get("env", {})
    algo_cfg = cfg.get("algo", {})

    # Build actor with correct dims from checkpoint config if available
    device = torch.device(args.device)

    # We can't infer obs_dim/act_dim without env, so create env once.
    from mappo_tarware.env_factory import make_env
    from mappo_tarware.wrapper import TARWareTensorWrapper
    env = make_env(cfg_from_dict(env_cfg))
    w = TARWareTensorWrapper(env, reward_mode=str(env_cfg.get("reward_mode", "team_sum")))
    specs = w.get_specs()
    actor = SharedDiscreteActor(
        obs_dim=specs.obs_dim,
        act_dim=specs.act_dim,
        num_agents=specs.num_agents,
        num_roles=int(algo_cfg.get("num_roles", 2)),
        hidden=tuple(algo_cfg.get("actor_hidden", [256, 256])),
        use_agent_id=bool(algo_cfg.get("use_agent_id", True)),
        use_role_id=bool(algo_cfg.get("use_role_id", True)),
        embed_dim=int(algo_cfg.get("embed_dim", 16)),
    ).to(device)

    load_checkpoint(args.checkpoint, actor=actor, critic=None, map_location=device, restore_rng=False)

    out = evaluate(
        actor=actor,
        env_cfg=env_cfg,
        device=device,
        episodes=args.episodes,
        seed=args.seed,
        greedy=args.greedy,
    )
    print(json.dumps(out, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
