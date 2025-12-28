# TA-RWARE Robot Fleet Orchestration (TorchRL + custom MAPPO)

Starter scaffold for a **multi-agent fleet orchestration** project on **TA-RWARE**, using **TorchRL** and a **custom MAPPO (CTDE)** training loop.

## What you get here
- A suggested **repo layout** that matches the master plan (configs / env wrapper / MAPPO / baselines / eval / demo).
- A **TorchRL-style contract** for the TA-RWARE environment wrapper (TensorDict in/out + action masking).
- A **MAPPO wiring plan** (actors per role + centralized critic) that you can implement incrementally.

This scaffold intentionally leaves some TODOs where TA-RWARE-specific API details differ across forks.

## Quickstart (once you implement the env wrapper)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# sanity check: environment specs + 1 random episode
python scripts/smoke_env.py --config configs/env_default.yaml

# train MAPPO
python scripts/train.py --env configs/env_default.yaml --algo configs/mappo_default.yaml

# evaluate on multiple seeds / scenarios
python scripts/eval.py --run runs/<RUN_ID> --eval configs/eval_default.yaml
```

## Conventions (TensorDict keys)
We follow TorchRL multi-agent conventions: per-agent tensors live under an `"agents"` sub-tensordict.

Minimum keys:
- `("agents", "observation")`: `[..., n_agents, obs_dim]`
- `("agents", "action_mask")`: `[..., n_agents, n_actions]` (bool)
- `("agents", "action")`: `[..., n_agents]` (int64)
- `("agents", "reward")`: `[..., n_agents, 1]` (float)
- `("done")` and/or `("terminated")`: `[..., 1]` (bool)

## What to implement first
1. `src/envs/tarware_env.py`: TA-RWARE → `EnvBase` wrapper + specs + action masks
2. `scripts/smoke_env.py`: env reset/step loop and `check_env_specs`
3. `src/baselines/random_policy.py`: random valid action baseline (uses masks)
4. MAPPO v1 (feed-forward): `src/algos/mappo/` + `scripts/train.py`

## License
Choose MIT/Apache-2.0 when you publish.
