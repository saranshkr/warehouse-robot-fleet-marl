"""TorchRL EnvBase wrapper for TA-RWARE.

Goal:
- Convert TA-RWARE multi-agent tuple I/O into TorchRL/TensorDict I/O.
- Expose action masks for invalid macro-actions (targets/tasks).

Conventions:
- Per-agent tensors live under the "agents" sub-tensordict.
- Shapes are `[n_agents, ...]` for single-environment (non-batched) usage.

TA-RWARE (tarware) is typically used via Gymnasium:
    import tarware
    import gymnasium as gym
    env = gym.make("tarware-tiny-3agvs-2pickers-partialobs-v1")

Obs is typically a tuple (one obs per agent).
Actions are typically a tuple (one action per agent).
Reward/terminated/truncated may be per-agent lists; we reduce to episode-level done.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Composite, Categorical, Unbounded


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _bool_any(x: Any) -> bool:
    """Robust 'any' for bool / list[bool] / np.ndarray."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    arr = np.asarray(x, dtype=bool)
    return bool(arr.any())


@dataclass
class TARWareEnvConfig:
    # Provide either a full env_id OR the pieces to construct it.
    env_id: Optional[str] = None

    # Naming scheme pieces (used if env_id is None)
    layout: str = "tiny"           # e.g. tiny/small/medium (depends on your TA-RWARE install)
    obs_mode: str = "partialobs"   # e.g. partialobs/globalobs (depends on install)
    version: str = "v1"            # usually v1

    seed: int = 0
    max_steps: int = 500

    n_agv: int = 2
    n_picker: int = 2

    # Placeholder for your later stress tests / demand bursts
    order_rate: float = 1.0
    burst: Optional[Dict[str, Any]] = None

    # If None, infer maximum action count across agents and pad masks to that.
    n_actions: Optional[int] = None


class TARWareTorchRLEnv(EnvBase):
    """TA-RWARE wrapper exposing TorchRL specs + TensorDict I/O.

    Exposed keys:
    - ("agents", "observation") : float [n_agents, obs_dim]
    - ("agents", "action_mask") : bool  [n_agents, n_actions] (True => valid)
    - ("agents", "action")      : int64 [n_agents]
    - ("agents", "reward")      : float [n_agents, 1]
    - ("done",)                 : bool  [1]
    """

    def __init__(self, cfg: TARWareEnvConfig, device: Union[torch.device, str] = "cpu"):
        super().__init__(device=torch.device(device))
        self.cfg = cfg

        self._env = self._build_underlying_env(cfg)

        # Infer dims/specs using a dummy reset
        obs, info = self._reset_underlying(seed=cfg.seed)
        obs_t = self._extract_obs(obs)

        self.n_agents = int(obs_t.shape[0])
        self.obs_dim = int(obs_t.shape[1])

        # Infer per-agent action sizes (Tuple(Discrete, Discrete, ...))
        self._per_agent_n_actions = self._infer_per_agent_n_actions()

        # Pad to a single max action dim for all agents (so tensor shapes are fixed)
        if cfg.n_actions is not None:
            self.n_actions = int(cfg.n_actions)
        else:
            self.n_actions = int(max(self._per_agent_n_actions))

        self._make_specs()

    # ----------------------- TorchRL required overrides -----------------------

    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        seed = self.cfg.seed
        if tensordict is not None and tensordict.get("seed", None) is not None:
            seed = int(tensordict.get("seed").item())

        obs, info = self._reset_underlying(seed=seed)
        obs_t = self._extract_obs(obs)
        mask_t = self._extract_action_mask(obs=obs, info=info)

        if tensordict is None:
            tensordict = TensorDict({}, batch_size=[], device=self.device)
        else:
            tensordict = tensordict.empty()

        tensordict.set(("agents", "observation"), obs_t)
        tensordict.set(("agents", "action_mask"), mask_t)
        tensordict.set("done", torch.zeros(1, dtype=torch.bool, device=self.device))
        return tensordict


    def _step(self, tensordict: TensorDict) -> TensorDict:
        actions = tensordict.get(("agents", "action"))
        actions_np = actions.detach().cpu().numpy().reshape(-1)
        actions_tuple = tuple(int(a) for a in actions_np.tolist())

        step_out = self._env.step(actions_tuple)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
        elif len(step_out) == 4:
            obs, reward, done, info = step_out
            terminated, truncated = done, False
        else:
            raise RuntimeError(f"Unexpected env.step output length: {len(step_out)}")

        obs_t = self._extract_obs(obs)
        mask_t = self._extract_action_mask(obs=obs, info=info if isinstance(info, dict) else {})

        # --- Reward handling ---
        # If you wrapped TA-RWARE to satisfy Gymnasium checker, reward may be scalar,
        # and per-agent rewards will be stored in info["reward_per_agent"].
        rew_source = None
        if isinstance(info, dict) and "reward_per_agent" in info:
            rew_source = info["reward_per_agent"]
        else:
            rew_source = reward

        rew_arr = np.asarray(rew_source, dtype=np.float32)

        # If scalar, broadcast to all agents
        if rew_arr.ndim == 0:
            rew_arr = np.full((self.n_agents,), float(rew_arr), dtype=np.float32)

        # Ensure shape [n_agents, 1]
        if rew_arr.ndim == 1:
            rew_arr = rew_arr[:, None]

        rew_t = torch.as_tensor(rew_arr, dtype=torch.float32, device=self.device)

        # --- Done handling ---
        # If wrapped, terminated/truncated may be bool scalars, while per-agent lists
        # are stored in info["terminated_per_agent"] / info["truncated_per_agent"].
        term_source = terminated
        trunc_source = truncated
        if isinstance(info, dict):
            term_source = info.get("terminated_per_agent", term_source)
            trunc_source = info.get("truncated_per_agent", trunc_source)

        done_any = _bool_any(term_source) or _bool_any(trunc_source)

        # Outplace: overwrite contents
        out = tensordict.empty()
        out.set(("agents", "observation"), obs_t)
        out.set(("agents", "action_mask"), mask_t)
        out.set(("agents", "reward"), rew_t)
        out.set("done", torch.tensor([done_any], dtype=torch.bool, device=self.device))
        return out


    def _set_seed(self, seed: Optional[int] = None) -> int:
        """TorchRL hook: set the env seed and return the seed used."""
        if seed is None:
            seed = int(self.cfg.seed)
        else:
            seed = int(seed)

        # Persist on cfg for reproducibility
        self.cfg.seed = seed

        # Best-effort: Gymnasium environments generally accept reset(seed=...)
        try:
            self._env.reset(seed=seed)
        except TypeError:
            # Some envs might expose seed() instead
            if hasattr(self._env, "seed"):
                try:
                    self._env.seed(seed)
                except Exception:
                    pass
        except Exception:
            # Don't hard-fail if seeding is not supported in your fork
            pass

        return seed

    def _reset_underlying(self, seed: int) -> Tuple[Any, Dict[str, Any]]:
        out = self._env.reset(seed=seed)
        # Gymnasium standard: (obs, info). Some envs return obs only.
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            obs, info = out
        else:
            obs, info = out, {}
        return obs, info

    # ------------------------------ Specs ------------------------------------

    def _make_specs(self) -> None:
        # Observations (NO done here)
        self.observation_spec = Composite(
            agents=Composite(
                observation=Unbounded(
                    shape=(self.n_agents, self.obs_dim),
                    device=self.device,
                    dtype=torch.float32,
                ),
                # action_mask is just a boolean tensor; Unbounded works fine here
                action_mask=Unbounded(
                    shape=(self.n_agents, self.n_actions),
                    device=self.device,
                    dtype=torch.bool,
                ),
                device=self.device,
            ),
            device=self.device,
        )

        # Discrete actions per agent
        self.action_spec = Composite(
            agents=Composite(
                action=Categorical(
                    n=self.n_actions,
                    shape=(self.n_agents,),
                    device=self.device,
                    dtype=torch.int64,
                ),
                device=self.device,
            ),
            device=self.device,
        )

        # Per-agent reward scalar
        self.reward_spec = Composite(
            agents=Composite(
                reward=Unbounded(
                    shape=(self.n_agents, 1),
                    device=self.device,
                    dtype=torch.float32,
                ),
                device=self.device,
            ),
            device=self.device,
        )

        # Done spec lives separately; EnvBase examples use Categorical(dtype=bool)
        self.done_spec = Categorical(
            n=2,
            shape=(1,),
            device=self.device,
            dtype=torch.bool,
        )

    # ---------------------------- TA-RWARE hooks -----------------------------

    def _build_underlying_env(self, cfg: TARWareEnvConfig):
        """Instantiate TA-RWARE env via Gymnasium and wrap it to satisfy Gymnasium checker."""
        import gymnasium as gym  # you already have it
        import tarware  # noqa: F401

        env_id = cfg.env_id
        if env_id is None:
            env_id = f"tarware-{cfg.layout}-{cfg.n_agv}agvs-{cfg.n_picker}pickers-{cfg.obs_mode}-{cfg.version}"

        # optional: disables Gymnasium's passive checker entirely (extra safety)
        # but we won't rely on this alone — we also wrap to be compliant.
        env = gym.make(env_id, disable_env_checker=True)

        class _TarwareGymnasiumCompat(gym.Wrapper):
            """Convert TA-RWARE multi-agent signals into Gymnasium-compatible scalars.

            - reset() -> (obs, info)
            - step()  -> reward scalar, terminated bool, truncated bool
            - preserve original lists in info so the TorchRL wrapper can still use them
            """

            def reset(self, **kwargs):
                out = self.env.reset(**kwargs)
                if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
                    obs, info = out
                else:
                    obs, info = out, {}
                return obs, info

            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)

                # Preserve originals
                if info is None:
                    info = {}
                info["reward_per_agent"] = reward
                info["terminated_per_agent"] = terminated
                info["truncated_per_agent"] = truncated

                # Convert to Gymnasium-friendly scalars
                reward_scalar = float(np.mean(np.asarray(reward, dtype=np.float32)))
                terminated_bool = bool(np.any(np.asarray(terminated, dtype=bool)))
                truncated_bool = bool(np.any(np.asarray(truncated, dtype=bool)))

                return obs, reward_scalar, terminated_bool, truncated_bool, info

        env = _TarwareGymnasiumCompat(env)

        # Optional TimeLimit
        try:
            from gymnasium.wrappers import TimeLimit
            if cfg.max_steps is not None and getattr(env, "_max_episode_steps", None) is None:
                env = TimeLimit(env, max_episode_steps=int(cfg.max_steps))
        except Exception:
            pass

        return env


    def _infer_per_agent_n_actions(self) -> Sequence[int]:
        """Infer per-agent action counts from underlying env.action_space.

        TA-RWARE often uses a Tuple action space: one Discrete per agent.
        """
        asp = getattr(self._env, "action_space", None)
        if asp is None:
            raise NotImplementedError("Underlying env has no action_space; set cfg.n_actions manually.")

        spaces = getattr(asp, "spaces", None)
        if spaces is not None:
            per = []
            for s in spaces:
                if hasattr(s, "n"):
                    per.append(int(s.n))
                else:
                    raise NotImplementedError(
                        f"Unsupported action space element type {type(s)}; expected Discrete."
                    )
            return per

        # Single-agent fallback
        if hasattr(asp, "n"):
            return [int(asp.n)] * int(getattr(self._env, "n_agents", 1))

        raise NotImplementedError("Could not infer per-agent action counts; set cfg.n_actions.")


    def _extract_obs(self, obs: Any) -> torch.Tensor:
        """Convert TA-RWARE obs into float32 tensor [n_agents, obs_dim].

        If obs dims differ per agent/role, we pad to max dim so we can stack.
        """
        if isinstance(obs, dict):
            # Unlikely for TA-RWARE, but support it anyway
            keys = list(obs.keys())
            obs_list = [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in keys]
        elif isinstance(obs, (tuple, list)):
            obs_list = [np.asarray(o, dtype=np.float32).reshape(-1) for o in obs]
        else:
            arr = np.asarray(obs, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

        n_agents = len(obs_list)
        max_dim = max(int(o.shape[0]) for o in obs_list)
        obs_mat = np.zeros((n_agents, max_dim), dtype=np.float32)
        for i, o in enumerate(obs_list):
            obs_mat[i, : o.shape[0]] = o

        return torch.as_tensor(obs_mat, dtype=torch.float32, device=self.device)


    def _extract_action_mask(self, obs: Any, info: Dict[str, Any]) -> torch.Tensor:
        """Return bool mask [n_agents, n_actions] where True means action is valid.

        Priority:
        1) If info provides masks, use them.
        2) If env exposes a method, call it.
        3) Fallback: static mask per-agent based on Discrete action space size (padded).
        """
        # 1) info-based common keys
        for k in (
            "action_mask",
            "action_masks",
            "avail_actions",
            "available_actions",
            "valid_actions_mask",
        ):
            if k in info:
                m = np.asarray(info[k], dtype=bool)
                if m.ndim == 1:
                    m = np.tile(m[None, :], (self.n_agents, 1))
                mask = np.zeros((self.n_agents, self.n_actions), dtype=bool)
                mask[:, : min(self.n_actions, m.shape[1])] = m[:, : self.n_actions]
                return torch.as_tensor(mask, dtype=torch.bool, device=self.device)

        # 2) env method-based
        for meth in ("get_action_mask", "get_action_masks", "action_masks", "get_avail_actions"):
            fn = getattr(self._env, meth, None)
            if callable(fn):
                m = np.asarray(fn(), dtype=bool)
                if m.ndim == 1:
                    m = np.tile(m[None, :], (self.n_agents, 1))
                mask = np.zeros((self.n_agents, self.n_actions), dtype=bool)
                mask[:, : min(self.n_actions, m.shape[1])] = m[:, : self.n_actions]
                return torch.as_tensor(mask, dtype=torch.bool, device=self.device)

        # 3) static fallback: only first n_i actions are valid for agent i
        mask = np.zeros((self.n_agents, self.n_actions), dtype=bool)
        for i, n_i in enumerate(self._per_agent_n_actions):
            mask[i, : min(int(n_i), self.n_actions)] = True
        return torch.as_tensor(mask, dtype=torch.bool, device=self.device)


    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass
