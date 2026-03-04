# TA‑RWARE Multi‑Agent Task Assignment (TorchRL MAPPO)

## 0) Project snapshot
**Goal:** Train and evaluate a Multi‑Agent PPO variant (MAPPO, CTDE) on **TA‑RWARE** (Task Assignment Robotic Warehouse), demonstrating improved warehouse throughput/efficiency vs dispatch baselines (random + heuristic + OR‑style matching), with robust evaluation and strong portfolio‑grade artifacts.

**Core deliverables:**
- A TorchRL‑compatible environment wrapper (action masking, per‑agent obs, stats)
- MAPPO training script (reproducible, checkpointed)
- Evaluation harness across **seeds** + **scenario sweeps**
- Baselines (random + heuristic; optional greedy/Hungarian)
- Clear metrics + plots + demo replay

---

## Progress so far (current repo status)
**Repo & workflow**
- [x] Created repo and committed foundational files to `main`
- [x] Created development branch: `feature/mappo-training`
- [x] TA‑RWARE installed in editable mode (cloned repo + `pip install -e .`) and import works

**Environment integration**
- [x] Implemented `TARWareTorchRLEnv` (TorchRL `EnvBase`) wrapper
- [x] Added Gymnasium compatibility wrapper around TA‑RWARE to avoid checker warnings (reset returns `(obs, info)`, step returns scalar reward/booleans while preserving per‑agent lists in `info`)
- [x] Added robust `_extract_obs` and `_extract_action_mask`
- [x] Fixed spec/key issues (done keys, deprecated spec classes) and got `check_env_specs` passing
- [x] Added basic `stats` extraction from TA‑RWARE `info` (deliveries, clashes, stucks, utilization, distance, idle time)

**Debug & diagnostics**
- [x] `scripts/smoke_env.py` runs multi‑seed × multi‑episode random rollouts and prints periodic stats
- [x] Confirmed event reward signal can be **positive** on rare events (per‑agent reward spikes; deliveries sometimes increment)
- [x] Confirmed action mask dtype/shape correctness in resets

**Training & evaluation scaffolding**
- [x] `scripts/train.py` exists and runs MAPPO‑style rollout + PPO update loop
- [x] `configs/mappo_default.yaml` exists (seed, rollout, PPO/GAE, logging, checkpoint cadence)
- [x] Checkpoint saving wired (iter checkpoints + latest pointer)
- [x] `scripts/eval.py` harness exists (loads saved policy and runs rollouts)
- [ ] Evaluation outputs + plots still to be formalized (mean±std tables, robustness curves)

**Known current issues (to address next)**
- [ ] Training appears to converge to low‑entropy behavior without improving deliveries (deliveries often remain 0)
- [ ] Need a stronger diagnostic for whether the policy sees enough learning signal (reward shaping / longer horizon / baseline comparisons)
- [ ] Need to verify/standardize episode length handling (TA‑RWARE may have a built‑in termination around 500 steps in your current env IDs)

---

## 1) Problem framing (for README + demo)
### What TA‑RWARE is (and what it isn’t)
- TA‑RWARE is primarily a **task assignment / dispatch** environment.
- Robot motion is typically handled by built‑in routing/heuristics; learning focuses on **which agent does which job, and when**.

### What you’ll optimize
Pick 2–4 primary KPIs:
- **Pick/Delivery throughput**: items delivered per time (or per episode)
- **Pick rate**: deliveries per step (or per minute)
- **Order latency**: time from order arrival to completion (p50/p95)
- **Utilization**: fraction of time robots are busy vs idle
- Optional: congestion proxy (clashes/stucks), travel distance

---

## 2) Key modeling choice: CTDE MAPPO
### Why MAPPO here
- Agents are partially observed and coordination is required
- Centralized critic stabilizes learning
- Decentralized actors remain deployable (each agent acts on its own obs)

### Role heterogeneity (AGV vs Picker)
Two common approaches:
1. **Separate actors per role** (agv actor + picker actor)
2. **Single shared actor + role embedding**

Start with (1) for stability, then optionally move to (2) to reduce params.

---

## 3) Environment wrapper contract (TorchRL)
### Required outputs per step
Your TorchRL env wrapper should emit (at minimum):
- `("agents","observation")`: float32 tensor `[n_agents, obs_dim]`
- `("agents","action_mask")`: bool tensor `[n_agents, n_actions]` (True = valid)
- `("agents","reward")`: float32 tensor `[n_agents, 1]` (or team reward broadcast)
- `"done"`: bool tensor `[1]`
- Optional diagnostic signals:
  - `("stats","shelf_deliveries")`, `("stats","vehicles_busy")`, etc.

### Episode termination
Ensure wrapper consistently sets:
- `done=True` when environment terminates
- (optional) `terminated` / `truncated` if you want Gymnasium‑style separation

---

## 4) Baselines (must have)
At least 2 baselines are expected; 3–4 is ideal.

### Minimum baselines
1. **Random valid action** (already implemented)
2. **Heuristic dispatcher** from TA‑RWARE repo (if exposed)

### Strong baselines (high value)
3. **Greedy assignment**
   - assign available jobs to nearest eligible robot
4. **Hungarian / min‑cost matching**
   - cost = travel distance + congestion/idle penalties

---

## 5) Reward design & learning signal
### What to confirm early
- Does the environment reward deliver/pick events strongly enough?
- Is there a dense shaping signal (e.g., +0.1 for sub‑task progress) or is it sparse?
- Are episodes long enough for random to ever succeed?

### If reward is too sparse
Consider:
- Extending horizon (if possible)
- Using curriculum (small layouts, fewer robots)
- Reward shaping (distance, queue reduction, avoid stuck/clash)

---

## 6) MAPPO training pipeline (end‑to‑end)
### Data collection
- On‑policy rollout for `T` steps
- Store for each timestep:
  - obs, action, mask
  - logp(action)
  - reward (team or per agent)
  - done
  - value estimate V(s)

### Advantage estimation
- Compute **GAE(λ)** with `(gamma, lambda)`
- Normalize advantages per batch

### PPO update
- For K epochs:
  - shuffle timesteps into minibatches
  - recompute logp under current policy
  - compute ratio and clipped surrogate objective
  - update centralized critic on value targets
  - add entropy bonus

### Checkpointing
- Save every `save_every` iterations:
  - actor(s), critic, optimizer, config, iter
- Maintain `latest.pt` pointer

---

## 7) Evaluation harness (do not skip)
### What evaluation must answer
- Does MAPPO beat baselines on **throughput** and/or **latency**?
- Is performance consistent across **seeds**?
- Does it generalize to scenario changes?

### Standard evaluation protocol
- Train with N seeds (e.g., 3)
- Evaluate with M seeds (e.g., 5) and report mean ± std
- Scenario sweep:
  - in‑distribution layout/config
  - more robots
  - higher order rate
  - demand burst

### Reported metrics
- total deliveries (per episode)
- pick rate (deliveries / steps)
- mean reward (team)
- utilization proxy (vehicles_busy)
- collisions/stucks

---

## 8) Artifacts & reporting (portfolio quality)
### Must‑have artifacts
- `README.md` with quickstart + results table
- `results.json` / `eval_results.json`
- Plots:
  - deliveries over time
  - pick rate vs scenario
  - reward curves

### Recommended artifacts
- `report.md` per run: configs + plots + summary table
- Short demo video with overlays

---

## 9) Stress tests / robustness (highly impressive, TA‑RWARE‑friendly)
Because TA‑RWARE doesn’t learn routing, focus disruptions on dispatch realism:

### Stress tests
1. **Demand bursts**
   - spike order arrival for a window of time
2. **Robot failure**
   - freeze/remove a robot mid‑episode; measure recovery
3. **Observation noise/dropout**
   - corrupt queue lengths or task availability signals
4. **Action delay**
   - delay assignment actions by 1–3 steps
5. **Station capacity reduction**
   - reduce throughput of a station (simulate outage/slowdown)
6. **Role imbalance**
   - fewer pickers than AGVs (or vice versa) to test coordination under constraints

### Robustness scoring
- Plot pick rate vs severity level
- Plot latency tail (p95) vs severity
- Summarize as area‑under‑curve or worst‑case performance

---

## 10) Ablations (show you understand what matters)
Run 4–6 focused ablations:
1. **No centralized critic** (independent PPO) vs MAPPO
2. **No congestion term**
3. **No idle penalty**
4. **No recurrence** (MLP vs GRU)
5. **No role embedding** (forces shared policy to “forget” heterogeneity)
6. **No reassignment penalty** (tests thrashing behavior)

---

## 11) Interpretability / analysis (another senior signal)
### Coordination analysis
- **Utilization breakdown** per role: idle vs working vs blocked
- **Assignment entropy**: how diverse/decisive assignments are
- **Reassignment count**: too high indicates thrashing
- **Bottleneck heatmaps**: occupancy and waiting time

### Failure mode catalog (include screenshots/GIFs)
- thrashing assignments during bursts
- station overload / queue collapse
- imbalance (AGVs waiting for pickers, or vice versa)
- cascading delays due to a single failure

---

## 12) Deployment & demo plan (how to make TA‑RWARE demos pop)
Because traversal is heuristic, you must overlay metrics.

### A) Demo video storyboard (60 seconds)
1. 0–10s: problem statement + KPI list
2. 10–25s: heuristic dispatcher replay with **live cumulative picks** chart
3. 25–50s: MARL replay with same seed + same chart (clear separation)
4. 50–60s: results table + robustness curves thumbnail

**Overlays to include**
- cumulative picks vs time (baseline vs MARL)
- current queue lengths at stations
- utilization % (idle/blocked) per role

### B) Streamlit dashboard (recommended)
Pages:
1. **Scenario Runner**
   - layout, #AGVs, #pickers, order rate, burst toggle
   - choose policy: Random / Heuristic / Greedy / Hungarian / Independent / MARL
   - run + show replay + live KPIs
2. **Metrics**
   - KPI table + plots
3. **Robustness**
   - stress test curves
4. **Ablations**
   - bar charts for pick rate/latency
5. **About**
   - method + reproducibility + links

### C) Inference packaging
- Save actor weights + config
- Provide:
  - `export_policy.py` (TorchScript/ONNX)
  - `infer.py` (loads exported model and runs an episode)
- Optional: FastAPI service
  - `/act` for assignment decisions
  - `/health` readiness

### D) Reproducible container
- `Dockerfile` for training and demo
- `make train`, `make eval`, `make demo`

---

## 13) Repo layout (suggested)
```
ta-rware-marl/
  README.md
  LICENSE
  pyproject.toml (or requirements.txt)
  Dockerfile
  Makefile
  configs/
    env_default.yaml
    mappo_default.yaml
    eval_default.yaml
  src/
    envs/
      tarware_env.py
      wrappers.py
    algos/
      mappo/
        actor.py
        critic.py
        buffer.py
        train_step.py
    baselines/
      random_policy.py
      tarware_heuristic.py
      greedy_dispatch.py
      hungarian_dispatch.py
      independent_ppo.py
    eval/
      evaluator.py
      metrics.py
    demo/
      replay.py
      render.py
      streamlit_app.py
    utils/
      seeding.py
      logging.py
      config.py
  scripts/
    train.py
    eval.py
    sweep.py
    make_replays.py
  reports/
    results.md
    ablations.md
```

---

## 14) Milestones (practical build order)
### Phase 1: Foundations (get to a replay fast)
- Setup TA‑RWARE env + wrapper (roles + action masking)
- Implement heuristic baseline + random baseline
- Build replay renderer (GIF/MP4) with KPI overlay (cumulative picks)

### Phase 2: MARL v1 (working training)
- Implement MAPPO for heterogeneous roles (separate actors or role‑embedded actor)
- Train on small layouts and confirm learning signal
- Add evaluation script (mean/std across seeds)

### Phase 3: Senior upgrades
- Implement greedy + Hungarian dispatch baselines
- Add stress tests (bursts, failures, noise, delay)
- Run ablations
- Add interpretability visuals (utilization + congestion heatmaps)
- Start Streamlit dashboard

### Phase 4: Polish + ship
- Export policy (TorchScript/ONNX)
- Dockerize training + demo
- Record final video and update README
- Write short results report + failure modes

---

## 15) README checklist (what to include)
- 6‑line overview + gif
- “Why this matters” (throughput/pick rate, latency, utilization)
- Quickstart (train/eval/demo)
- Baselines and metrics explained (include OR baseline)
- Results table + robustness plot
- Demo video
- Reproducibility details (seeds, configs, versions)
- Limitations + next steps

---

## 16) Resume bullet template (save for later)
Use 2–3 bullets like:
- Built a multi‑agent **task assignment/orchestration** system for heterogeneous warehouse robots, improving **pick rate by X%** and reducing **p95 order latency by Y%** vs FIFO and min‑cost matching baselines across N layouts.
- Implemented **CTDE MAPPO** with role‑conditioned policies and a reproducible evaluation harness featuring **stress tests** (demand bursts, robot failures, delayed observations) and robustness curves.
- Shipped a **Streamlit replay dashboard** with KPI overlays and an exportable **TorchScript/ONNX policy** for fast inference and reproducible demos (Dockerized).

---

## 17) Optional stretch ideas (if you want “wow”)
- Curriculum learning (low demand → bursty demand; small → large layouts)
- Graph neural network critic (model spatial/role relations)
- Multi‑objective optimization (Pareto: pick rate vs congestion vs fairness)
- Offline RL or imitation warm‑start using heuristic trajectories
- Uncertainty‑aware dispatch under noisy queue signals

---

## 18) Final “definition of done”
You are done when you can say:
1. “Here’s a video showing MARL beating dispatch baselines on the same seed with cumulative picks overlay.”
2. “Here’s a table with mean±std across multiple seeds and scenario sweeps.”
3. “Here are robustness curves under demand bursts, failures, noise, and delays.”
4. “Here’s a Streamlit app to replay and compare policies.”
5. “Here’s a Docker command that reproduces results.”

---

## Next actions (recommended order)
1. **Confirm episode horizon behavior** for your chosen env ID(s) (why it terminates at ~500) and decide whether to keep it or extend it.
2. **Baseline benchmarking**: run heuristic (and/or greedy) baseline to establish a non‑zero deliveries target.
3. **Training signal audit**:
   - verify deliveries happen in baseline runs
   - confirm reward correlates with deliveries
4. **Upgrade evaluation**: write `eval_results.json` + summary table (mean±std)
5. **Add one strong OR baseline** (Hungarian) and one stress test (demand burst)

*Keep this as the master plan; update progress boxes as you implement each phase.*

