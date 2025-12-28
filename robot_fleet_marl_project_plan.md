# TA-RWARE Robot Fleet Orchestration with Multi-Agent Reinforcement Learning (MARL)
*A reusable, end-to-end project plan optimized for “senior” signals on a resume + GitHub.*

---

## 0) One-sentence pitch (use everywhere)
Build a **multi-agent fleet orchestration system** for warehouse automation where heterogeneous robots coordinate **task assignment** to maximize **pick rate / throughput** while minimizing **idle time, congestion, and delay**, with a **reproducible training pipeline**, rigorous baselines (heuristics + OR), and a **replay dashboard**.

> **Why TA-RWARE (vs RWARE)?** TA-RWARE focuses on **macro decision-making**: agents choose **target locations / tasks**, while low-level traversal is handled by a **predefined pathfinding heuristic** (e.g., A*). This makes the project read like real dispatch/orchestration (MRTA), and typically reduces training pain vs learning navigation end-to-end.

---

## 1) Goals and “seniority signals”
### Primary goals
1. **Coordination**: heterogeneous agents learn to coordinate assignments and timing (who does what, when).
2. **Operational performance**: maximize **pick rate** (fulfilled order lines per unit time/steps) and minimize order latency.
3. **Reliability**: robust to demand spikes, partial observability/noise, agent failures, and distribution shift (more robots, larger layouts).
4. **Reproducibility**: deterministic runs, experiment tracking, Dockerized execution.
5. **Interpretability** (optional): explain coordination quality (utilization, fairness, congestion, reassignment behavior).

### What will read as “senior” on GitHub/resume
- Clean **system design**: env wrappers + training + evaluation + replay + export.
- **Baselines + ablations**: include both heuristic and optimization-style baselines.
- **Stress/robustness tests**: failure injection + demand bursts + delayed observations.
- **Operational KPIs**: pick rate, latency, utilization, congestion, fairness—not just reward.
- **Deployable artifact**: replay viewer + exportable policy (TorchScript/ONNX) + simple inference API (optional).
- **Failure analysis**: a short “what failed / why” section with examples.

---

## 2) TA-RWARE concept mapping (domain clarity)
### Key terms
- **Pick**: an order line (item request) that becomes **fulfilled** (e.g., delivered/processed at an output/packing station).
- **Pick rate**: picks per unit time. Report as:
  - `picks / episode`
  - `picks / 1,000 steps`
  - (optional) `picks / minute` if you simulate wall-clock.

### What changes vs RWARE
- Your policy controls **assignment decisions** (macro actions), not low-level navigation.
- Movement is handled by a deterministic heuristic → demos are best when paired with **metric overlays** (cumulative picks, utilization, queueing).

---

## 3) Project deliverables (what you’ll ship)
### Must-have (portfolio-grade)
1. **GitHub repo** with:
   - reproducible training scripts
   - evaluation harness + metrics
   - baseline implementations (heuristic + OR + ML baseline)
   - replay generation utilities
2. **Demo assets**
   - 30–60s highlight video (baseline vs MARL side-by-side)
   - 3–5 GIFs for README
3. **Replay dashboard (Streamlit recommended)**
   - scenario selection
   - policy selection
   - replay + KPI overlays
4. **Policy export**
   - TorchScript or ONNX export + inference snippet
5. **Short technical report** (Markdown)
   - experimental setup, results table, ablations, stress tests, failure analysis

### Nice-to-have
- Simple inference API (FastAPI) for “dispatch” endpoints
- Docker image for training + demo
- CI (lint/tests) + minimal unit tests for wrappers/metrics

---

## 4) Metrics (what you log and show)
### Core operational metrics (TA-RWARE friendly)
- **Pick rate / throughput**
  - picks per episode
  - picks per 1,000 steps
- **Mean order completion time** (latency)
- **Idle time %** per agent role (AGV vs picker)
- **Empty travel ratio**: distance traveled without carrying / without progress
- **Queueing metrics**
  - avg/max queue length at stations
  - time in queue
- **Congestion**
  - bottleneck occupancy
  - average “blocked waiting” time (agents stuck due to others)
- **Fairness / balance**
  - work done per agent
  - optional Gini coefficient over work distribution

### ML/training stability metrics
- reward curve (train/eval)
- entropy (exploration)
- value loss / policy loss
- gradient norms

### Robustness metrics (report curves)
- pick rate vs number of robots
- pick rate vs demand burst severity
- pick rate vs observation noise / dropout
- pick rate vs action delay
- pick rate vs robot failure rate

---

## 5) Baselines (crucial for credibility in TA-RWARE)
You want baselines that match macro-action task assignment, since traversal is heuristic.

### Minimum recommended set (strong and fair)
1. **Built-in TA-RWARE heuristic (FIFO/greedy dispatcher)**
   - Use the env/repo-provided default policy if available (document it).
2. **Random valid target selection**
   - Random among valid targets; include action masking.
3. **Greedy cost-based assignment**
   - Score tasks by estimated travel time + queue penalty + congestion penalty.
4. **Min-cost matching (Hungarian) baseline**
   - Model assignment as bipartite matching (robots ↔ tasks or robot-pairs ↔ tasks).
5. **Independent learners (no CTDE)**
   - Per-role PPO (or shared params) **without centralized critic**.

### Why these matter
- (1) is the “everyone must beat this” baseline.
- (4) signals “I know OR / dispatch algorithms,” which reads senior.

---

## 6) Main algorithm plan (what to build and why)

### Recommended training algorithm (decision)
**Use MAPPO (CTDE) as the primary algorithm.** It is a strong, widely used cooperative MARL baseline for discrete/partially-observed settings and fits TA‑RWARE’s “dispatch/task assignment” nature.

**Why MAPPO here**
- **CTDE**: decentralized actors (deployable) + centralized critic (stabilizes multi-agent learning).
- **Heterogeneous roles**: naturally supported via either separate role policies or a role-conditioned shared policy.
- **Action masking friendly**: TA‑RWARE typically benefits from masking invalid targets; MAPPO works well with masks.

**Implementation defaults (keep it simple)**
- Start with **two actors (one per role)** + **one centralized critic** (or 1 shared actor + role embedding).
- Critic input (v1): **concatenate all agents’ observations** (works even if the env doesn’t expose a special “global state”).
- Add **action masking** from day 1. Consider **top‑K candidate reduction** (e.g., nearest K targets) if the action space is very large.

### Baseline RL algorithms (for credible comparisons)
Include at least one “ML baseline” besides your main MAPPO:

1) **IPPO (Independent PPO)** — *no centralized critic*  
   - Same PPO code, but each agent learns independently (or shared parameters) with **local critic inputs only**.  
   - This cleanly shows the value of CTDE.

2) **QMIX / VDN (optional, strong second family)**  
   - Value-decomposition methods often perform well on cooperative discrete tasks.  
   - Use only if you want an extra comparison line; it adds implementation/tuning work.

> Recommendation: Ship with **MAPPO + IPPO** first. Add **QMIX** later only if you have time and want a second algorithm family in the report.


### Recommended mainline: MAPPO (CTDE) adapted for TA-RWARE
Why:
- strong default for cooperative tasks and partial observability
- centralized critic helps resolve coordination

**Key TA-RWARE implementation choices**
- **Heterogeneous roles**:
  - Option A: separate actors per role (AGV policy + picker policy)
  - Option B: single shared actor + **role embedding** + agent ID embedding
- **Centralized critic**:
  - input: concatenated per-agent observations (and/or global state if exposed)
  - output: value estimate for joint state
- **Recurrent policy** (optional):
  - GRU to help with partial observability and dynamic queues

### Alternative algorithms (optional comparative)
- **QMIX / VDN**: value factorization; strong for cooperative discrete macro actions.
- **Hierarchical / manager-worker** (stretch): manager assigns tasks; workers execute.

### Reward design (be explicit + ablate)
Use a reward aligned with warehouse KPIs:
- +R_pick per fulfilled pick
- −c_delay per step (or per unfulfilled order)
- −c_idle for idle time (role-specific)
- −c_congestion for blocked waiting / bottleneck occupancy
- −c_reassign for excessive reassignment thrash (optional)

**Ablations:** remove congestion penalty / idle penalty / reassignment penalty and show impact.

---

## 7) Environment engineering (senior signal: clean wrappers + reproducibility)
### Wrapper responsibilities
- Standardize observation tensor shapes:
  - per-role grouping if needed
  - output: `obs: Dict[role, Tensor[num_agents_role, obs_dim]]` or a padded tensor
- Action tensors per role:
  - `actions_agv: [n_agv]`, `actions_picker: [n_picker]`
- Add:
  - action masking for invalid targets
  - conversion utilities between env dict I/O and tensors
- Provide:
  - `reset(seed) -> obs, info`
  - `step(actions) -> obs, rewards, done, truncated, info`

### Scenario generator (evaluation is everything)
Parameterize:
- layout size / topology
- number of AGVs and pickers
- station count / capacity
- order arrival rate (including bursty distributions)
- obstacle/bottleneck settings if supported

### Determinism checklist
- seed Python, NumPy, torch
- seed env reset
- log all config in run folder
- record env version/commit and dependency versions

---

## 8) Training pipeline (MLOps vibe)
### Experiment structure
- `configs/` (YAML): env + algo + training
- `train.py`: reads config, launches training, writes artifacts
- `eval.py`: batch evaluation across seeds + scenarios
- `sweep.py`: optional grid search runner
- `replay.py`: generate GIF/MP4 for episodes

### Logging & tracking
Minimum:
- TensorBoard for losses + KPIs
- `results.json` with final evaluation summary
Recommended:
- produce a `report.md` artifact per run with plots + tables

### Evaluation protocol (do not skip)
- Train with N seeds (e.g., 3)
- Evaluate:
  - in-distribution layouts
  - out-of-distribution: more robots, different layouts, higher order rates
- Report mean ± std for pick rate, latency, utilization, congestion

---

## 9) Stress tests / robustness (highly impressive, TA-RWARE-friendly)
Because TA-RWARE doesn’t learn routing, focus disruptions on dispatch realism:

### Stress tests
1. **Demand bursts**
   - spike order arrival for a window of time
2. **Robot failure**
   - freeze/remove a robot mid-episode; measure recovery
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
- Summarize as area-under-curve or worst-case performance

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

## 12) Deployment & demo plan (how to make TA-RWARE demos pop)
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
- Setup TA-RWARE env + wrapper (roles + action masking)
- Implement heuristic baseline + random baseline
- Build replay renderer (GIF/MP4) with KPI overlay (cumulative picks)

### Phase 2: MARL v1 (working training)
- Implement MAPPO for heterogeneous roles (separate actors or role-embedded actor)
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
- 6-line overview + gif
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
- Built a multi-agent **task assignment/orchestration** system for heterogeneous warehouse robots, improving **pick rate by X%** and reducing **p95 order latency by Y%** vs FIFO and min-cost matching baselines across N layouts.
- Implemented **CTDE MAPPO** with role-conditioned policies and a reproducible evaluation harness featuring **stress tests** (demand bursts, robot failures, delayed observations) and robustness curves.
- Shipped a **Streamlit replay dashboard** with KPI overlays and an exportable **TorchScript/ONNX policy** for fast inference and reproducible demos (Dockerized).

---

## 17) Optional stretch ideas (if you want “wow”)
- Curriculum learning (low demand → bursty demand; small → large layouts)
- Graph neural network critic (model spatial/role relations)
- Multi-objective optimization (Pareto: pick rate vs congestion vs fairness)
- Offline RL or imitation warm-start using heuristic trajectories
- Uncertainty-aware dispatch under noisy queue signals

---

## 18) Final “definition of done”
You are done when you can say:
1. “Here’s a video showing MARL beating dispatch baselines on the same seed with cumulative picks overlay.”
2. “Here’s a table with mean±std across multiple seeds and scenario sweeps.”
3. “Here are robustness curves under demand bursts, failures, noise, and delays.”
4. “Here’s a Streamlit app to replay and compare policies.”
5. “Here’s a Docker command that reproduces results.”

---

*Keep this as the master plan; prune later once scope is locked.*
