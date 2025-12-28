from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class EpisodeMetrics:
    picks: int
    steps: int
    total_reward: float
    mean_queue: float | None = None
    mean_idle: float | None = None

    @property
    def pick_rate_per_1k_steps(self) -> float:
        if self.steps <= 0:
            return 0.0
        return 1000.0 * float(self.picks) / float(self.steps)


def summarize(metrics: List[EpisodeMetrics]) -> Dict[str, float]:
    arr = {
        "picks": np.array([m.picks for m in metrics], dtype=float),
        "steps": np.array([m.steps for m in metrics], dtype=float),
        "total_reward": np.array([m.total_reward for m in metrics], dtype=float),
        "pick_rate_per_1k_steps": np.array([m.pick_rate_per_1k_steps for m in metrics], dtype=float),
    }

    out: Dict[str, float] = {}
    for k, v in arr.items():
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return out
