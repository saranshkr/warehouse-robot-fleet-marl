"""Evaluate a saved MAPPO policy (custom) on TA-RWARE.

- Loads checkpoint saved by scripts/train.py (actor_agv/actor_picker/critic)
- Runs greedy (argmax) actions with action masking
- Aggregates simple episode metrics (reward, steps)
- Scenario/seed sweeps supported via configs/eval_default.yaml

Usage:
  python -m scripts.eval --run runs/<RUN_ID> --eval configs/eval_default.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.config import load_yaml
from src.utils.seeding import seed_everything
from src.eval.metrics import EpisodeMetrics, summarize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=str, help="runs/<RUN_ID>")
    p.add_argument("--eval", required=True, type=str, help="configs/eval_default.yaml")
    p.add_argument(
        "--ckpt",
        default=None,
        type=str,
        help="Optional checkpoint path. If omitted, uses runs/<RUN_ID>/checkpoints/latest.pt",
    )
    return p.parse_args()


def _get_step_fields(td):
    """TorchRL env.step may return either plain keys or put them under 'next'."""
    nxt = td.get("next", None)
    if nxt is not None and nxt.get(("agents", "observation")) is not None:
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


def _masked_greedy_action(logits, mask):
    # mask True = valid
    logits = logits.masked_fill(~mask, -1e9)
    return torch.argmax(logits, dim=-1).to(torch.int64)


def _build_actor(obs_dim: int, n_actions: int, hidden: int, n_layers: int):
    import torch

    layers = []
    in_dim = obs_dim
    for _ in range(n_layers):
        layers.append(torch.nn.Linear(in_dim, hidden))
        layers.append(torch.nn.ReLU())
        in_dim = hidden
    layers.append(torch.nn.Linear(in_dim, n_actions))
    return torch.nn.Sequential(*layers)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    eval_cfg = load_yaml(args.eval)

    import torch

    # Load configs from run artifacts
    env_cfg = load_yaml(run_dir / "configs" / "env.yaml")
    algo_cfg = load_yaml(run_dir / "configs" / "algo.yaml")
    device = torch.device(env_cfg.get("device", "cpu"))

    # Resolve checkpoint
    ckpt_path: Optional[Path] = None
    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
    else:
        latest = run_dir / "checkpoints" / "latest.pt"
        if latest.exists():
            meta = torch.load(latest, map_location="cpu")
            ckpt_path = Path(meta["path"])
        else:
            raise FileNotFoundError(
                "No checkpoint provided and runs/<RUN_ID>/checkpoints/latest.pt not found. "
                "Save checkpoints from train.py first (or pass --ckpt)."
            )

    ckpt = torch.load(ckpt_path, map_location=device)

    from src.envs.tarware_env import TARWareTorchRLEnv, cfg_from_dict

    # Build a base env to infer dims
    base_env = TARWareTorchRLEnv(cfg_from_dict(env_cfg), device=device)
    n_agents, obs_dim, n_actions = base_env.n_agents, base_env.obs_dim, base_env.n_actions

    # Role split (same convention as train.py)
    n_agv = int(env_cfg.get("n_agv", n_agents))
    n_picker = int(env_cfg.get("n_picker", max(0, n_agents - n_agv)))
    assert n_agv + n_picker == n_agents, f"n_agv+n_picker must equal n_agents ({n_agents})"

    hidden = int(algo_cfg.get("hidden_size", 256))
    n_layers = int(algo_cfg.get("n_layers", 2))

    actor_agv = _build_actor(obs_dim, n_actions, hidden, n_layers).to(device)
    actor_picker = _build_actor(obs_dim, n_actions, hidden, n_layers).to(device)

    actor_agv.load_state_dict(ckpt["actor_agv"])
    actor_picker.load_state_dict(ckpt["actor_picker"])
    actor_agv.eval()
    actor_picker.eval()

    results: Dict[str, Any] = {"checkpoint": str(ckpt_path), "scenarios": {}}

    # Scenario sweep
    sweep = eval_cfg.get("sweep", [])
    if not sweep:
        sweep = [{"name": "default"}]

    for scenario in sweep:
        name = scenario.get("name", "scenario")
        env_cfg_s = dict(env_cfg)
        env_cfg_s.update({k: v for k, v in scenario.items() if k != "name"})

        metrics_all: List[EpisodeMetrics] = []

        for seed in eval_cfg.get("seeds", [0]):
            seed_everything(int(seed))

            env = TARWareTorchRLEnv(cfg_from_dict(env_cfg_s), device=device)
            episodes = int(eval_cfg.get("episodes_per_seed", 5))

            for _ in range(episodes):
                td = env.reset()
                total_reward = 0.0
                steps = 0
                picks = 0  # placeholder until you expose picks in info

                while True:
                    obs = td.get(("agents", "observation"))       # [A, obs_dim]
                    mask = td.get(("agents", "action_mask"))      # [A, n_actions]

                    logits = torch.zeros((n_agents, n_actions), device=device, dtype=torch.float32)
                    if n_agv > 0:
                        logits[:n_agv] = actor_agv(obs[:n_agv])
                    if n_picker > 0:
                        logits[n_agv:] = actor_picker(obs[n_agv:])

                    action = _masked_greedy_action(logits, mask)

                    td.set(("agents", "action"), action)
                    td_step = env.step(td)

                    obs_next, mask_next, reward_next, done_next, td_next = _get_step_fields(td_step)

                    # reward_next: [A,1] -> mean team reward per step
                    r = float(reward_next.mean().item())
                    total_reward += r
                    steps += 1

                    done = bool(done_next.item())
                    td = td_next
                    if done:
                        break

                metrics_all.append(EpisodeMetrics(picks=picks, steps=steps, total_reward=total_reward))

        results["scenarios"][name] = summarize(metrics_all)

    (run_dir / "eval_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
