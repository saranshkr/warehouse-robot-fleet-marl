# MAPPO.API.md — TA‑RWARE + MAPPO (Baseline Reward)

This project trains a **cooperative multi‑agent policy** in **TA‑RWARE** using **MAPPO** (Multi‑Agent PPO) with **CTDE** (Centralized Training, Decentralized Execution).

Baseline reward (unchanged):
- **AGVs:** +1 when a **requested shelf** is delivered to a goal location.
- **Pickers:** +0.1 whenever they help an AGV load/unload a shelf.
- Sparse rewards are expected.

---

## 1. Algorithm Overview (MAPPO / CTDE)

**Execution-time:** each agent acts using only its own observation (decentralized).  
**Training-time:** a centralized critic sees a joint representation of all agents’ observations.

### Actor (parameter-shared)
A single policy network is shared across all agents:
- input: per-agent observation `obs_i`
- optional embeddings: `agent_id` and/or `role_id` (recommended for heterogeneous roles)
- output: discrete action logits `logits_i`

### Centralized critic
Value function **V(s)** is estimated from the **joint observation**:
- `joint_obs` shape `[N, obs_dim]`
- flattened to `[N * obs_dim]`
- output: scalar value `V_t`

### Rollout + GAE
Collect `T` steps of on-policy experience, then compute:
- Advantages with GAE(λ)
- Returns = Advantages + Values

### PPO Update
Optimize with clipped PPO objective:
- policy loss: clipped surrogate objective
- value loss: MSE (optional value clipping)
- entropy bonus: improves exploration

---

## 2. Project Interface

### 2.1 Environment factory
`mappo_tarware/env_factory.py`
- `make_env(cfg: EnvConfig) -> env`
- centralizes `gym.make(env_id, **kwargs)`
- validates config and applies optional `TimeLimit`

### 2.2 Tensor wrapper
`mappo_tarware/wrapper.py`

API:
- `reset(seed=None) -> obs: [N, obs_dim] float32`
- `step(actions: [N] int64) -> (next_obs, rewards: [N] float32, done: bool, info: dict)`
- `seed(seed_int)`

Key properties:
- stable agent ordering (`agent_ids` stored in `info["agent_ids"]`)
- observations are padded to a consistent `obs_dim` if roles differ
- `reward_mode`:
  - `per_agent`: rewards per agent
  - `team_sum` / `team_mean`: wrapper returns `[N]` filled with the team reward each step (keeps PPO code simple)

### 2.3 Training CLI
`scripts/train_mappo.py`

Run:
```bash
python -m scripts.train_mappo --config configs/default.yaml
```

Override config values:
```bash
python -m scripts.train_mappo --config configs/default.yaml \
  --override run.seed=1 algo.rollout_len_T=256 algo.lr_actor=3e-4
```

Artifacts:
- `runs/<experiment>/<timestamp>/`
  - `metrics.csv`
  - `tb/` (TensorBoard, if available)
  - `checkpoints/latest.pt`, `checkpoints/best.pt`
  - `config_resolved.json`
  - `eval_latest.json`

### 2.4 Evaluation CLI
`scripts/eval_mappo.py`

```bash
python -m scripts.eval_mappo --checkpoint runs/.../checkpoints/best.pt --config configs/default.yaml --episodes 20 --greedy
```

Outputs:
- mean/std episodic return
- mean/std deliveries (if env exposes it)
- pick rate proxy: deliveries / episode length

---

## 3. Reproducibility Notes

This stack aims for reproducible runs by:
- setting Python / NumPy / Torch seeds (`run.seed`)
- seeding env resets via wrapper
- saving full resolved config
- checkpointing model + optimizer states
- optional RNG state saving in checkpoints (enabled)

---

## 4. Baselines

Minimum supported baselines for comparison:
- **Random policy:** sample actions uniformly from the discrete space
- (Optional later) simple heuristics if TA‑RWARE exposes enough state
