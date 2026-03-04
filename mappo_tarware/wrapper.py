from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _is_dict_like(x: Any) -> bool:
    return isinstance(x, dict)


def _flatten_obs(x: Any) -> np.ndarray:
    """
    Convert an arbitrary obs object into a 1D float32 vector.
    TA-RWARE obs is typically already a 1D numeric array.
    """
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False).reshape(-1)
    # allow list/tuple of numbers
    return np.asarray(x, dtype=np.float32).reshape(-1)


@dataclass(frozen=True)
class WrapperSpecs:
    num_agents: int
    num_agvs: int
    num_pickers: int
    obs_dim: int
    act_dim: int
    agent_ids: List[str]
    role_ids: Optional[np.ndarray]  # shape [N], int64 (0=AGV, 1=PICKER)


class TARWareTensorWrapper:
    """
    Minimal tensor-friendly wrapper for TA-RWARE (Gymnasium-like API).
    Outputs numpy arrays; training code converts to torch tensors.

    API:
      reset() -> obs: [N, obs_dim] float32
      step(actions: [N] int64) -> (obs, rewards: [N] float32, done: bool, info: dict)

    Agent ordering:
      - Derived from env.spec.kwargs['num_agvs'], ['num_pickers'].
      - Assumes env uses AGVs first then pickers (confirmed from upstream repo).

    Rewards:
      - per_agent: [N] (env reward vector)
      - team_sum/team_mean: compute team scalar and return as [N] (broadcast) so downstream stays [N]
        (also stored in info['team_reward']).
    """

    def __init__(
        self,
        env: Any,
        *,
        reward_mode: str = "team_sum",
        use_role_id: bool = True,
        explicit_agent_ids: Optional[Sequence[str]] = None,
        explicit_role_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.env = env
        _require(
            reward_mode in {"per_agent", "team_sum", "team_mean"},
            "reward_mode must be one of: per_agent | team_sum | team_mean",
        )
        self.reward_mode = reward_mode
        self._episode_step = 0

        # ---- infer counts (preferred) ----
        num_agvs, num_pickers = self._infer_counts_from_spec()
        self._num_agvs = num_agvs
        self._num_pickers = num_pickers
        self._num_agents = num_agvs + num_pickers

        # ---- ids / roles (deterministic, no guessing from obs structure) ----
        self._agent_ids = self._build_agent_ids(explicit_agent_ids)
        self._agent_index = {aid: i for i, aid in enumerate(self._agent_ids)}

        self._role_ids: Optional[np.ndarray] = None
        if use_role_id:
            self._role_ids = self._build_role_ids(explicit_role_ids)

        # ---- infer dims ----
        obs0 = self._reset_raw(seed=None)
        obs_arr = self._obs_to_array(obs0)
        self._obs_dim = int(obs_arr.shape[1])
        self._act_dim = self._infer_act_dim()

        self._specs = WrapperSpecs(
            num_agents=self._num_agents,
            num_agvs=self._num_agvs,
            num_pickers=self._num_pickers,
            obs_dim=self._obs_dim,
            act_dim=self._act_dim,
            agent_ids=list(self._agent_ids),
            role_ids=None if self._role_ids is None else self._role_ids.copy(),
        )

    def get_specs(self) -> WrapperSpecs:
        return self._specs

    def seed(self, seed: int) -> None:
        # Gymnasium-style
        try:
            _ = self.env.reset(seed=int(seed))
            return
        except TypeError:
            pass
        # Fallbacks
        if hasattr(self.env, "seed"):
            self.env.seed(int(seed))
            return
        raise RuntimeError(
            "Env does not support seeding via reset(seed=...) and has no env.seed()."
        )
    
    def _done_all(self, x: Any) -> bool:
        """
        Helper to interpret various done formats.
        """
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if isinstance(x, dict):
            return all(bool(v) for v in x.values())
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=np.bool_)
            return bool(arr.all())
        raise TypeError(f"Unsupported done type: {type(x)}")

    # ---------------- Gym adapters ----------------

    def reset(self, *, seed: Optional[int] = None) -> np.ndarray:
        self._episode_step = 0
        obs = self._reset_raw(seed=seed)
        return self._obs_to_array(obs)

    def step(self, action_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        actions = self._actions_from_tensor(action_tensor)

        out = self.env.step(actions)
        if not isinstance(out, tuple):
            raise RuntimeError(f"Env.step returned {type(out)}; expected a tuple.")

        if len(out) == 5:
            obs, rew, terminated, truncated, info = out          
            done = self._done_all(terminated) or self._done_all(truncated)
        elif len(out) == 4:
            obs, rew, done, info = out
            done = self._done_all(done)
        else:
            raise RuntimeError(f"Env.step returned tuple of len={len(out)}; expected 4 or 5.")

        self._episode_step += 1

        obs_arr = self._obs_to_array(obs)
        rew_vec = self._reward_to_vector(rew)

        info = dict(info or {})
        info.setdefault("episode_step", self._episode_step)
        info.setdefault("agent_ids", list(self._agent_ids))
        if self._role_ids is not None:
            info.setdefault("role_ids", self._role_ids.copy())

        # Always include team reward if applicable
        if 'team_reward' not in info:
            if self.reward_mode == "team_sum":
                info["team_reward"] = float(np.sum(rew_vec))
            elif self.reward_mode == "team_mean":
                info["team_reward"] = float(np.mean(rew_vec))

        return obs_arr, rew_vec.astype(np.float32, copy=False), done, info

    # ---------------- internals ----------------

    def _infer_counts_from_spec(self) -> Tuple[int, int]:
        spec = getattr(self.env, "spec", None)
        kwargs = getattr(spec, "kwargs", None) if spec is not None else None
        if not isinstance(kwargs, dict):
            raise RuntimeError(
                "TA-RWARE wrapper expects env.spec.kwargs to be a dict containing "
                "'num_agvs' and 'num_pickers'. (Got missing/non-dict spec.kwargs.)"
            )
        if "num_agvs" not in kwargs or "num_pickers" not in kwargs:
            raise RuntimeError(
                f"env.spec.kwargs missing required keys. Found keys={list(kwargs.keys())}, "
                "required: ['num_agvs','num_pickers']"
            )
        num_agvs = int(kwargs["num_agvs"])
        num_pickers = int(kwargs["num_pickers"])
        _require(num_agvs >= 0 and num_pickers >= 0, "num_agvs/num_pickers must be non-negative.")
        _require(num_agvs + num_pickers > 0, "num_agents must be > 0.")
        return num_agvs, num_pickers

    def _build_agent_ids(self, explicit_agent_ids: Optional[Sequence[str]]) -> List[str]:
        if explicit_agent_ids is not None:
            ids = [str(x) for x in explicit_agent_ids]
            _require(
                len(ids) == self._num_agents,
                f"explicit_agent_ids len={len(ids)} must equal num_agents={self._num_agents} "
                f"(num_agvs={self._num_agvs}, num_pickers={self._num_pickers}).",
            )
            return ids

        # Deterministic + matches env internal ordering (AGVs first, then pickers)
        return (
            [f"agv_{i}" for i in range(self._num_agvs)]
            + [f"picker_{i}" for i in range(self._num_pickers)]
        )

    def _build_role_ids(self, explicit_role_ids: Optional[Sequence[int]]) -> np.ndarray:
        if explicit_role_ids is not None:
            arr = np.asarray(explicit_role_ids, dtype=np.int64).reshape(-1)
            _require(arr.shape[0] == self._num_agents, f"explicit_role_ids must have shape [N={self._num_agents}].")
            return arr

        # 0=AGV, 1=PICKER, aligned to confirmed ordering
        return np.asarray([0] * self._num_agvs + [1] * self._num_pickers, dtype=np.int64)

    def _reset_raw(self, seed: Optional[int]) -> Any:
        # Call reset with or without seed (Gymnasium-style preferred)
        if seed is None:
            out = self.env.reset()
        else:
            try:
                out = self.env.reset(seed=int(seed))
            except TypeError:
                # If reset(seed=...) unsupported, seed separately then reset()
                self.seed(int(seed))
                out = self.env.reset()

        # Case A: Gymnasium reset returns (obs, info)
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            return out[0]

        # Case B: Multi-agent reset returns (obs_0, obs_1, ..., obs_{N-1})
        # Your env seems to do this.
        if isinstance(out, (list, tuple)) and len(out) == self._num_agents:
            return out

        # Case C: reset returns obs directly (ndarray, dict, etc.)
        return out


    def _obs_to_array(self, obs: Any) -> np.ndarray:
        """
        Returns [N, obs_dim] float32, aligned to AGV-first ordering.
        Supports:
          - np.ndarray shape (N, obs_dim)
          - list/tuple of N vectors
          - dict-like mapping agent_id -> vector (must contain our agent_ids)
        """
        if _is_dict_like(obs):
            missing = [aid for aid in self._agent_ids if aid not in obs]
            if missing:
                raise KeyError(
                    f"Obs dict missing keys {missing}. Got keys={list(obs.keys())}."
                )
            rows = [_flatten_obs(obs[aid]) for aid in self._agent_ids]
            arr = np.stack(rows, axis=0).astype(np.float32, copy=False)
            return self._pad_or_check_obs(arr)

        if isinstance(obs, np.ndarray):
            if obs.ndim == 2:
                _require(
                    obs.shape[0] == self._num_agents,
                    f"Obs array first dim N={obs.shape[0]} != num_agents={self._num_agents}.",
                )
                return self._pad_or_check_obs(obs.astype(np.float32, copy=False))
            if obs.ndim == 1:
                # single-agent fallback (shouldn't happen for TA-RWARE multi-agent)
                raise ValueError(
                    f"Expected multi-agent obs array [N, obs_dim], got 1D array shape={obs.shape}."
                )
            raise ValueError(f"Unsupported obs ndarray ndim={obs.ndim}; expected 2.")

        if isinstance(obs, (list, tuple)):
            _require(
                len(obs) == self._num_agents,
                f"Obs list/tuple len={len(obs)} != num_agents={self._num_agents}.",
            )

            rows = [_flatten_obs(x) for x in obs]
            lens = [r.size for r in rows]
            maxd = int(max(lens))
            _require(maxd > 0, f"Invalid obs lengths: {lens}")

            # During __init__ we haven't set self._obs_dim yet; after init we enforce/pad to self._obs_dim
            target_dim = maxd if not hasattr(self, "_obs_dim") else int(self._obs_dim)

            # If a later call returns larger obs than we initialized with, that's a real bug.
            if hasattr(self, "_obs_dim") and maxd > target_dim:
                raise ValueError(f"Obs dim increased from {target_dim} to {maxd} (unexpected).")

            out = np.zeros((self._num_agents, target_dim), dtype=np.float32)
            for i, r in enumerate(rows):
                d = min(r.size, target_dim)
                out[i, :d] = r[:d]
            return out

    def _pad_or_check_obs(self, arr: np.ndarray) -> np.ndarray:
        # During __init__, we don't yet know obs_dim. So we allow any 2D array and set obs_dim later.
        if not hasattr(self, "_obs_dim"):
            _require(arr.ndim == 2 and arr.shape[0] == self._num_agents, "Invalid initial obs array shape.")
            return arr

        # After init, ensure correct obs_dim (pad only if env sometimes returns shorter vectors).
        _require(arr.ndim == 2 and arr.shape[0] == self._num_agents, "Invalid obs array shape.")
        if arr.shape[1] == self._obs_dim:
            return arr
        if arr.shape[1] > self._obs_dim:
            raise ValueError(f"Obs dim increased from {self._obs_dim} to {arr.shape[1]} (unexpected).")
        # pad right with zeros
        out = np.zeros((self._num_agents, self._obs_dim), dtype=np.float32)
        out[:, : arr.shape[1]] = arr
        return out

    def _infer_act_dim(self) -> int:
        asp = getattr(self.env, "action_space", None)
        if asp is None:
            raise AttributeError("Env has no action_space; cannot infer act_dim.")

        # Discrete
        if hasattr(asp, "n"):
            return int(asp.n)

        # Tuple/Dict spaces
        if hasattr(asp, "spaces"):
            spaces = asp.spaces
            if isinstance(spaces, (list, tuple)):
                ns = []
                for s in spaces:
                    if not hasattr(s, "n"):
                        raise TypeError(f"Expected Discrete in action_space tuple; got {type(s)}")
                    ns.append(int(s.n))
                return int(max(ns))
            if isinstance(spaces, dict):
                ns = []
                for s in spaces.values():
                    if not hasattr(s, "n"):
                        raise TypeError(f"Expected Discrete in action_space dict; got {type(s)}")
                    ns.append(int(s.n))
                return int(max(ns))

        raise NotImplementedError(
            f"Unsupported action_space type: {type(asp)}. Expected Discrete or Tuple/Dict of Discrete."
        )

    def _actions_from_tensor(self, action_tensor: np.ndarray) -> Any:
        if not isinstance(action_tensor, np.ndarray):
            action_tensor = np.asarray(action_tensor)

        _require(
            action_tensor.dtype.kind in {"i", "u"},
            f"actions must be integer dtype, got {action_tensor.dtype}",
        )

        a = action_tensor.reshape(-1)
        _require(
            a.shape[0] == self._num_agents,
            f"actions shape must be [N={self._num_agents}], got {a.shape}",
        )

        asp = getattr(self.env, "action_space", None)
        if asp is None:
            return [int(x) for x in a.tolist()]

        if hasattr(asp, "spaces") and isinstance(asp.spaces, dict):
            # map our deterministic ids to env's dict-style API if it expects dict
            return {aid: int(a[self._agent_index[aid]]) for aid in self._agent_ids}

        # Default: list/tuple style API (common in TA-RWARE gym envs)
        return [int(x) for x in a.tolist()]

    def _reward_to_vector(self, rew: Any) -> np.ndarray:
        n = self._num_agents

        if _is_dict_like(rew):
            missing = [aid for aid in self._agent_ids if aid not in rew]
            if missing:
                raise KeyError(f"Reward dict missing keys {missing}. Got keys={list(rew.keys())}.")
            vec = np.asarray([float(rew[aid]) for aid in self._agent_ids], dtype=np.float32)

        elif isinstance(rew, np.ndarray):
            if rew.ndim == 0:
                vec = np.full((n,), float(rew), dtype=np.float32)
            else:
                _require(rew.shape[0] == n, f"Reward array len={rew.shape[0]} != num_agents={n}.")
                vec = rew.astype(np.float32, copy=False).reshape(n)

        elif isinstance(rew, (list, tuple)):
            _require(len(rew) == n, f"Reward sequence len={len(rew)} != num_agents={n}.")
            vec = np.asarray(rew, dtype=np.float32).reshape(n)

        else:
            # scalar reward
            vec = np.full((n,), float(rew), dtype=np.float32)

        if self.reward_mode == "per_agent":
            return vec

        # compute team scalar and broadcast (keeps training code simple: always [N])
        team = float(np.sum(vec)) if self.reward_mode == "team_sum" else float(np.mean(vec))
        return np.full((n,), team, dtype=np.float32)
