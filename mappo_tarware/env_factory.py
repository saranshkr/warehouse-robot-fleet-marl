from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Dict, Optional

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)

@dataclass(frozen=True)
class EnvConfig:
    """
    Minimal config for TA-RWARE creation.

    Notes:
    - TA-RWARE env ids typically encode layout + agents + obs mode, e.g.
      "tarware-tiny-3agvs-2pickers-partialobs-v1".
    - This stack keeps env construction centralized. Training code should not
      call gym.make(...) directly.
    """
    env_id: str
    seed: int = 0
    episode_horizon: Optional[int] = None  # optional TimeLimit wrapper
    render_mode: Optional[str] = None

    # Reward aggregation handled by wrapper (not env)
    reward_mode: str = "team_sum"  # "team_sum" | "team_mean" | "per_agent"

    # Optional extra kwargs passed to gym.make
    env_kwargs: Optional[Dict[str, Any]] = None

def validate_env_config(cfg: EnvConfig) -> None:
    _require(isinstance(cfg.env_id, str) and cfg.env_id.strip(), "env_id must be a non-empty string.")
    _require(cfg.reward_mode in {"team_sum", "team_mean", "per_agent"}, "reward_mode must be one of: team_sum, team_mean, per_agent.")
    _require(isinstance(cfg.seed, int), "seed must be an int.")
    if cfg.episode_horizon is not None:
        _require(isinstance(cfg.episode_horizon, int) and cfg.episode_horizon > 0, "episode_horizon must be a positive int if set.")
    if cfg.env_kwargs is not None:
        _require(isinstance(cfg.env_kwargs, dict), "env_kwargs must be a dict if provided.")

def cfg_from_dict(d: Any) -> EnvConfig:
    if isinstance(d, EnvConfig):
        return d
    if is_dataclass(d):
        d = asdict(d)
    if not isinstance(d, dict):
        raise TypeError(f"Expected dict or EnvConfig, got {type(d)}")
    # allow legacy keys from your PRD:
    # layout_id/num_agvs/num_pickers/etc. can be encoded in env_id upstream
    env_id = d.get("env_id") or d.get("gym_id")
    if not env_id:
        raise KeyError("Missing env_id (or gym_id) in env config.")
    return EnvConfig(
        env_id=str(env_id),
        seed=int(d.get("seed", 0)),
        episode_horizon=d.get("episode_horizon", None),
        render_mode=d.get("render_mode", None),
        reward_mode=str(d.get("reward_mode", "team_sum")),
        env_kwargs=d.get("env_kwargs", None),
    )

def make_env(cfg: EnvConfig):
    """
    Create the raw gymnasium environment.
    """
    validate_env_config(cfg)

    try:
        import gymnasium as gym
    except Exception as e:  # pragma: no cover
        raise ImportError("gymnasium is required. Install with: pip install gymnasium") from e

    # TA-RWARE is typically registered on import tarware.
    # We import lazily so this module can be imported without tarware installed.
    try:
        import tarware  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "tarware is required but not importable. Install/enable TA-RWARE first."
        ) from e

    kwargs = dict(cfg.env_kwargs or {})
    if cfg.render_mode is not None:
        kwargs["render_mode"] = cfg.render_mode

    try:
        env = gym.make(cfg.env_id, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to gym.make(env_id={cfg.env_id!r}). "
            "Check that TA-RWARE is installed and the env_id is correct."
        ) from e

    # Optional TimeLimit wrapper (if env doesn't already enforce it)
    if cfg.episode_horizon is not None:
        try:
            from gymnasium.wrappers import TimeLimit
            env = TimeLimit(env, max_episode_steps=int(cfg.episode_horizon))
        except Exception as e:
            raise RuntimeError("Failed to apply gymnasium.wrappers.TimeLimit.") from e

    # Ensure deterministic reset through wrapper seed; also set here if supported.
    try:
        env.reset(seed=int(cfg.seed))
    except TypeError:
        # Some envs accept seed only via wrapper call; ignore here.
        pass

    return env
