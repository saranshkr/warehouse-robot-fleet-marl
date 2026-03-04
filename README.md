# TA-RWARE + MAPPO (baseline reward) — minimal PyTorch stack

## What this is
A small, self-contained implementation of MAPPO (CTDE) for **discrete-action** TA-RWARE environments:

- **Shared actor** (parameter sharing across agents) with optional `agent_id` / `role_id` embeddings
- **Centralized critic** `V(s)` where `s = concat(all obs)`
- PPO-style clipped objective + GAE
- CSV + optional TensorBoard logging
- Checkpointing (`latest.pt`, `best.pt`)
- Evaluation script

Docs: see `MAPPO.API.md`.

## Install
You need TA-RWARE installed/available in your python env.

Minimum python deps:
- `torch`
- `numpy`
- `gymnasium`
- `pyyaml`
- `tarware` (TA-RWARE)

## Run training
From this repo root:

```bash
python -m scripts.train_mappo --config configs/default.yaml
```

Override config values:

```bash
python -m scripts.train_mappo --config configs/default.yaml \
  --override run.seed=1 run.device=cuda algo.rollout_len_T=256 algo.minibatch_size=512
```

Artifacts go to `runs/<experiment>/<timestamp>/`.

## Run evaluation
```bash
python -m scripts.eval_mappo --checkpoint runs/<...>/checkpoints/best.pt --config configs/default.yaml --episodes 20 --greedy
```
