from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

def capture_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])

def save_checkpoint(
    path: Path,
    *,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    opt_actor: torch.optim.Optimizer,
    opt_critic: torch.optim.Optimizer,
    step: int,
    cfg: Dict[str, Any],
    best_metric: Optional[float] = None,
    save_rng: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "opt_actor": opt_actor.state_dict(),
        "opt_critic": opt_critic.state_dict(),
        "step": int(step),
        "cfg": cfg,
        "best_metric": best_metric,
    }
    if save_rng:
        payload["rng_state"] = capture_rng_state()

    torch.save(payload, path)

def load_checkpoint(
    path: Path,
    *,
    actor: Optional[torch.nn.Module] = None,
    critic: Optional[torch.nn.Module] = None,
    opt_actor: Optional[torch.optim.Optimizer] = None,
    opt_critic: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = False,
) -> Dict[str, Any]:
    """
    Load a checkpoint. Pass actor/critic/optimizers you want restored; any can be None.
    """
    path = Path(path)
    payload = torch.load(path, map_location=map_location)

    if actor is not None and "actor" in payload:
        actor.load_state_dict(payload["actor"])
    if critic is not None and "critic" in payload:
        critic.load_state_dict(payload["critic"])

    if opt_actor is not None and "opt_actor" in payload:
        opt_actor.load_state_dict(payload["opt_actor"])
    if opt_critic is not None and "opt_critic" in payload:
        opt_critic.load_state_dict(payload["opt_critic"])

    if restore_rng and payload.get("rng_state") is not None:
        restore_rng_state(payload["rng_state"])

    return payload
