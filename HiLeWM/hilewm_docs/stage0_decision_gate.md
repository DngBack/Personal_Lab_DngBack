# Stage 0 Decision Gate

## Purpose

This document decides whether the project is ready to move from research framing to implementation.

## Stage 0 decision

Decision: **GO to Stage 1**

Reason:

The research direction is now sufficiently narrow:

**Improve LeWorldModel long-horizon plannability using event-level temporal abstraction and reachability-aligned latent structure.**

## Required acceptance criteria

### Criterion 1 — Thesis clarity

The project has a clear thesis:

**Stable local prediction is not sufficient for long-horizon planning; event-level controllability structure is needed.**

Status: PASS

### Criterion 2 — Method clarity

The method has a minimum viable implementation:

1. Frozen LeWM encoder.
2. Cached fast latents.
3. Fixed-length event segments.
4. Event aggregator.
5. Reachability head.
6. Hierarchical subgoal planner.
7. Short-horizon low-level CEM.

Status: PASS

### Criterion 3 — Experiment clarity

The first target experiment is clear:

**Hard TwoRoom: HiLeWM-fixed versus LeWM+CEM and LeWM+short-CEM.**

The second target experiment is clear:

**PushT long-horizon: transfer the same event-planning idea.**

Status: PASS

### Criterion 4 — Evidence clarity

The paper claims have matching evidence requirements:

- Success versus horizon.
- Reachability AUC.
- Ablation without event latent.
- Ablation without reachability loss.
- Planning time.
- Boundary/event visualization.

Status: PASS

### Criterion 5 — Compute feasibility

The plan is compute-aware:

- Start from checkpoints.
- Cache latents.
- Freeze backbone.
- Train small event modules.
- Use 1× L40S.

Status: PASS

### Criterion 6 — Timeline feasibility

The plan is feasible only if Stage 1–4 are completed quickly.

Hard requirement:

**By 2026-07-02, there must be a first hierarchical-planner result on Hard TwoRoom.**

Status: CONDITIONAL PASS

## Stage 1 immediate tasks

### Task 1 — Create repository structure

Create:

```text
hilewm/
  configs/
  hilewm/
  scripts/
  results/
  caches/
  checkpoints/
  docs/
```

### Task 2 — Add Stage 0 docs

Copy these Stage 0 files into:

```text
docs/stage0/
```

### Task 3 — Reproduce LeWM baseline

Start with:

```text
TwoRoom → PushT
```

Do not start with all environments.

### Task 4 — Define baseline logging

Every baseline run must log:

```text
env_name
method
checkpoint
seed
success_rate
planning_time_per_step
cem_horizon
cem_candidates
gpu_name
peak_gpu_memory
wall_clock_time
git_commit
config_path
```

### Task 5 — Create first result file

Target output:

```text
results/stage1_baseline/baseline_tworooms_seed0.json
```

## Go/no-go after Stage 1

Proceed to Stage 2 only if:

1. LeWM checkpoint can be evaluated.
2. Baseline metrics are logged.
3. At least one environment runs end-to-end.
4. You understand where baseline fails or saturates.

If Stage 1 fails:

1. Fix environment/checkpoint issues.
2. Reduce to a simpler controlled environment.
3. Do not implement HiLeWM modules yet.

## Go/no-go after Stage 4

This is the real paper decision.

Proceed to full paper experiments if:

1. HiLeWM-fixed improves Hard TwoRoom success.
2. Event reachability AUC is clearly above baseline.
3. Short-CEM-only is worse than HiLeWM-fixed.
4. The improvement is repeatable over at least 3 seeds.

Pivot if:

1. No planning improvement.
2. Reachability AUC is weak.
3. Baseline already saturates.
4. Planner improvement comes only from engineering tricks.

## Stage 0 final statement

The project should now move to implementation.

The first coding objective is not to build the full HiLeWM system.

The first coding objective is:

**Reproduce LeWM, cache latents, and prove that fixed event-level reachability helps one hard long-horizon task.**
