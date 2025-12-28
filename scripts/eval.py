"""Evaluate a saved policy across seeds/scenarios.

This is a placeholder harness: it runs rollouts and aggregates the operational metrics
(picks, pick rate, reward). Extend it with queue/idle/congestion metrics once your TA-RWARE wrapper exposes those.

Usage:
  python scripts/eval.py --run runs/<RUN_ID> --eval configs/eval_default.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.utils.config import load_yaml
from src.utils.seeding import seed_everything
from src.eval.metrics import EpisodeMetrics, summarize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=str)
    p.add_argument("--eval", required=True, type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    eval_cfg = load_yaml(args.eval)

    # Load saved policy weights
    try:
        import torch
        from tensordict.nn import TensorDictModule
        from torchrl.modules import ProbabilisticActor
        from torchrl.modules.distributions import MaskedCategorical
    except Exception as e:
        raise RuntimeError("Install torchrl/tensordict to run evaluation") from e

    # Read config artifacts
    env_cfg = load_yaml(run_dir / "configs" / "env.yaml")
    algo_cfg = load_yaml(run_dir / "configs" / "algo.yaml")

    from src.envs.tarware_env import TARWareTorchRLEnv
    from src.algos.mappo.networks import build_modules

    device = torch.device(env_cfg.get("device", "cpu"))

    # Build env once; update scenario params per sweep item
    base_env = TARWareTorchRLEnv(env_cfg, device=device)
    n_agents, obs_dim, n_actions = base_env.n_agents, base_env.obs_dim, base_env.n_actions

    modules = build_modules(obs_dim=obs_dim, n_actions=n_actions, n_agents_total=n_agents)
    actor_net = modules.actor_agv.to(device)

    # Wrap policy
    actor_td = TensorDictModule(actor_net, in_keys=[("agents", "observation")], out_keys=[("agents", "logits")])
    policy = ProbabilisticActor(
        module=actor_td,
        in_keys=[("agents", "logits"), ("agents", "action_mask")],
        distribution_class=MaskedCategorical,
        distribution_kwargs={"mask": ("agents", "action_mask")},
        return_log_prob=False,
        out_keys=[("agents", "action")],
    )

    ckpt = torch.load(run_dir / "policy.pt", map_location=device)
    policy.load_state_dict(ckpt["policy"])  # train.py saves this format
    policy.eval()

    results: Dict[str, Any] = {"scenarios": {}}

    for scenario in eval_cfg.get("sweep", []):
        name = scenario.get("name", "scenario")
        # Merge scenario override into env config
        env_cfg_s = dict(env_cfg)
        env_cfg_s.update({k: v for k, v in scenario.items() if k != "name"})

        metrics_all: List[EpisodeMetrics] = []
        for seed in eval_cfg.get("seeds", [0]):
            seed_everything(int(seed))
            env = TARWareTorchRLEnv(env_cfg_s, device=device)
            for _ in range(int(eval_cfg.get("episodes_per_seed", 5))):
                td = env.reset()
                total_reward = 0.0
                steps = 0
                picks = 0
                while True:
                    with torch.no_grad():
                        td = policy(td)
                    td = env.step(td)
                    # Optional: env could emit picks in info; add once available.
                    r = td.get(("agents", "reward")).mean().item()
                    total_reward += float(r)
                    steps += 1
                    done = td.get("done").item()
                    if done:
                        break
                metrics_all.append(EpisodeMetrics(picks=picks, steps=steps, total_reward=total_reward))

        results["scenarios"][name] = summarize(metrics_all)

    (run_dir / "eval_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
