# Risk Register

## Purpose

This document lists major risks before implementation begins. Each risk has symptoms, mitigation, and pivot triggers.

## Risk 1 — Baseline reproduction fails

### Description

The official LeWM setup may not run immediately due to dependency, dataset, or environment issues.

### Symptoms

- Checkpoint cannot be loaded.
- Evaluation numbers are far from paper/repo.
- Environment version mismatch.
- CEM planning produces unstable behavior.

### Mitigation

1. Start from official checkpoints.
2. Reproduce only one environment first: TwoRoom.
3. Save exact environment versions.
4. Use Docker or conda lockfile if needed.
5. Do not modify baseline code before saving baseline results.

### Pivot trigger

If baseline cannot be reproduced within 5–7 days, switch to:

- a simpler local reproduction using cached observations,
- or a smaller controlled navigation environment.

---

## Risk 2 — LeWM already solves the chosen hard tasks

### Description

If baseline success is already high, there is little room for improvement.

### Symptoms

- LeWM+CEM success >90% on all selected tasks.
- Horizon stress test does not reduce performance.
- Raw latent distance selects good subgoals.

### Mitigation

1. Increase goal distance.
2. Add Hard TwoRoom/MultiRoom.
3. Reduce CEM horizon/candidate budget to expose planning weakness.
4. Use planning cost as the main axis if success is saturated.

### Pivot trigger

If all long-horizon tasks are saturated, pivot to efficiency:

**Event-level planning achieves similar success with fewer rollouts and lower latency.**

---

## Risk 3 — Event latent does not improve reachability

### Description

The event abstraction may not improve reachability over fast latent.

### Symptoms

- Event reachability AUC close to fast-latent AUC.
- Segment length ablation has no effect.
- Event latent clusters are not meaningful.

### Mitigation

1. Try K=4/8/16.
2. Add action summaries.
3. Use attention pooling instead of mean pooling.
4. Normalize event latent separately.
5. Add hard negatives.

### Pivot trigger

If event latent does not improve reachability after basic tuning, pivot to:

**Reachability metric learning on fast LeWM latent.**

---

## Risk 4 — Reachability improves but planning does not

### Description

The representation metric may improve without translating to control performance.

### Symptoms

- Reachability AUC high.
- Subgoals look plausible.
- Actual policy/planner still fails.

### Mitigation

1. Improve low-level controller.
2. Use shorter subgoal distances.
3. Add subgoal feasibility filtering.
4. Use top-k re-planning rather than single subgoal.
5. Add action-sequence retrieval from nearest training segment.

### Pivot trigger

If planning fails but reachability is strong, reframe paper as:

**Learning controllability-aware latent geometry for world models**

This is weaker unless accompanied by strong analysis.

---

## Risk 5 — Learned boundary is noisy

### Description

BoundaryNet may learn prediction-error artifacts rather than meaningful events.

### Symptoms

- Boundaries appear everywhere.
- Boundaries do not align with door/contact/phase changes.
- Learned boundary worse than fixed K.

### Mitigation

1. Keep fixed segmentation as main method.
2. Regularize average segment length.
3. Use simple pseudo-boundary from prediction error and latent velocity.
4. Visualize early.
5. Do not overfit to boundary novelty.

### Pivot trigger

If learned boundary does not improve by the paper deadline, remove it from main method and keep as future work.

---

## Risk 6 — Joint training destabilizes LeWM

### Description

Adding event losses may damage the stable LeWM representation.

### Symptoms

- Fast latent collapse.
- Baseline local prediction loss worsens.
- Planning performance drops.
- SIGReg statistics become unstable.

### Mitigation

1. Keep frozen-backbone result as main.
2. Only unfreeze predictor/projection layers.
3. Use small learning rate.
4. Use gradient clipping.
5. Monitor latent standard deviation.

### Pivot trigger

If joint training is unstable after 2–3 attempts, remove it from main experiments.

---

## Risk 7 — Novelty is seen as incremental

### Description

Reviewers may view the method as "LeWM plus reachability head."

### Symptoms

- No clear conceptual difference from metric-learning baselines.
- Ablations show reachability head alone explains most gains.
- Method description overemphasizes planner instead of representation.

### Mitigation

1. Emphasize temporal abstraction and event structure.
2. Include event-level analysis and boundary visualization.
3. Compare against fast-latent reachability.
4. Show horizon scaling, not only final success.
5. Make the paper thesis about plannability, not just performance.

### Pivot trigger

If fast-latent reachability matches event method, rename/reframe to a metric-learning paper or pivot to AdaSIGReg.

---

## Risk 8 — Compute budget is exceeded

### Description

Training and evaluation may become too expensive for 1× L40S.

### Symptoms

- Full image training is too slow.
- Too many seeds/environments.
- Planner evaluation is slow.
- Joint training consumes time.

### Mitigation

1. Cache latents.
2. Freeze backbone first.
3. Use TwoRoom before PushT.
4. Run 3 seeds initially.
5. Postpone Cube/Reacher.
6. Use shorter evaluation episodes during development.

### Pivot trigger

If compute is the blocker, paper claim becomes:

**A frozen-backbone, cached-latent planning extension to LeWM.**

---

## Risk 9 — Timeline misses AAAI-27

### Description

AAAI-27 deadlines are close.

### Symptoms

- No Stage 4 result by early July.
- Method still changing near abstract deadline.
- No main table by mid-July.

### Mitigation

1. Freeze MVP by Stage 4.
2. Write paper skeleton immediately.
3. Do not add optional environments before main result.
4. Do not pursue joint training unless Stage 4 is strong.

### Pivot trigger

If no planning improvement by 2026-07-02, switch to a smaller but cleaner paper direction:

1. AdaSIGReg-LeWM.
2. Fast-latent reachability metric.
3. Diagnostic study of LeWM plannability.

---

## Overall project risk level

Current risk level: high but manageable.

Reason:

- The idea is publishable only if planning improvement is clear.
- The implementation is feasible because frozen latent training is lightweight.
- The deadline is tight, so the method must be kept narrow.

## Recommended risk strategy

Use a two-track strategy:

### Main track

HiLeWM-fixed → event reachability → hierarchical planner.

### Backup track

Fast-latent reachability metric or AdaSIGReg.

Do not wait until the end to pivot.
