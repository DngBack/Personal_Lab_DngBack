# Experiment Scope

## Purpose

This document defines the experimental scope for HiLeWM. The goal is to prevent scope creep and ensure that every experiment directly supports the paper thesis.

## Core experimental objective

Demonstrate that event-level latent abstraction improves long-horizon planning from pixels over LeWorldModel-style short-horizon latent MPC.

## Primary environments

### Environment group A — Long-horizon navigation/topology

Priority: highest.

Recommended environments:

1. TwoRoom
2. Hard TwoRoom
3. MultiRoom, if available or easy to implement

Why this group matters:

- These tasks expose topological bottlenecks.
- Raw terminal latent distance can be misleading.
- Subgoal/event planning should provide a clear advantage.
- Results are easy to visualize.

Main expected phenomenon:

A baseline planner may move toward the goal in latent distance but fail to choose the correct doorway or intermediate room transition. HiLeWM should discover or select event-level subgoals corresponding to useful transitions.

### Environment group B — Contact-rich manipulation

Priority: high.

Recommended environment:

1. PushT
2. PushT-Go50
3. PushT-Go75
4. PushT-Go100, if implemented

Why this group matters:

- It tests whether event abstraction is useful beyond navigation.
- Contact-rich control naturally has phases: approach object, contact, push, align, finish.
- Event boundaries can be visually and physically interpretable.

Main expected phenomenon:

HiLeWM should identify control-relevant phases and use intermediate subgoals to reduce long-horizon failure.

### Environment group C — Optional continuous 3D/control task

Priority: optional.

Candidate environments:

1. Cube
2. Reacher

Use only after TwoRoom and PushT show strong results.

Why optional:

- They improve generality.
- But they can consume too much time.
- They are not necessary for the first strong result if navigation and PushT are convincing.

## Excluded environments for the first submission

Do not include:

- Large-scale robotic manipulation requiring expensive simulation.
- Real robot data.
- High-resolution video datasets without actions.
- Atari-style benchmarks unless directly integrated with LeWM.
- Foundation-model-based video planning benchmarks.

Reason:

The research claim is about hierarchical plannability under limited compute, not broad benchmark domination.

## Baselines

### Required baseline 1 — LeWM + CEM

This is the main baseline.

It tests:

- Original local latent MPC.
- Whether the event hierarchy improves over the default LeWM planning setup.

Required logs:

- success rate,
- planning time per step,
- number of candidate action sequences,
- CEM horizon,
- seeds,
- GPU memory.

### Required baseline 2 — LeWM + short-horizon CEM

This baseline is necessary because HiLeWM low-level control may use short-horizon CEM.

It tests:

- Whether the gain comes from event-level subgoal selection rather than merely shortening the planning horizon.

### Required baseline 3 — LeWM fast latent distance

This is a representation baseline.

It tests:

- Whether raw fast latent Euclidean distance is sufficient for long-horizon subgoal ranking.

### Required baseline 4 — Reachability head without event latent

This is an ablation-style baseline.

It tests:

- Whether the gain comes only from adding reachability supervision, or from event-level abstraction.

### Optional baseline 5 — Goal-conditioned inverse dynamics

Use if time permits.

It tests:

- Whether amortized low-level control can reduce planning cost.
- It should not be the central novelty.

### Optional baseline 6 — TRM-style reachability metric

Use if time permits and implementation is straightforward.

It tests:

- Whether HiLeWM outperforms a post-hoc reachability metric.

## Methods to evaluate

### Method A — HiLeWM-fixed

Frozen LeWM backbone, fixed-length segments, event aggregator, reachability head, hierarchical planner.

This is the MVP method.

### Method B — HiLeWM-learned-boundary

Frozen LeWM backbone, learned event boundary module, event aggregator, reachability head, hierarchical planner.

This is the stronger method if time permits.

### Method C — HiLeWM-joint

Light joint fine-tuning of predictor/projection layers with event objectives.

This is optional and should only be attempted after Method A works.

## Main metrics

### Planning success rate

Primary metric.

Report:

- mean success rate,
- standard deviation or standard error,
- number of seeds,
- number of episodes per seed.

### Success rate versus horizon

Critical for paper thesis.

Plot:

- x-axis: horizon or goal distance,
- y-axis: success rate.

Expected result:

- LeWM drops faster as horizon increases.
- HiLeWM remains more robust.

### Planning wall-clock time

Secondary but important.

Report:

- time per decision,
- total planning time per episode,
- GPU memory if relevant.

### Reachability AUC

Representation metric.

Compare:

- fast latent Euclidean score,
- fast latent reachability head,
- event latent reachability head.

### Subgoal ranking quality

Optional but useful.

Metric:

- whether top-k selected event subgoals lie on successful trajectories,
- recall@k for useful subgoals,
- distance-to-successful-path.

### Boundary interpretability

Qualitative metric.

Examples:

- boundary near doorway crossing,
- boundary near contact onset,
- boundary near object alignment,
- boundary near change in motion mode.

## Required ablations

### Ablation 1 — No event latent

Use fast latent with the same reachability head.

Question:

Does temporal abstraction matter?

### Ablation 2 — No reachability loss

Use event prediction only.

Question:

Does reachability alignment matter?

### Ablation 3 — Fixed segment versus learned boundary

Question:

Is learned segmentation useful, or is fixed temporal abstraction enough?

### Ablation 4 — Euclidean event cost versus reachability score

Question:

Is the planner benefit caused by event latent alone, or by the reachability metric?

### Ablation 5 — Segment length

Test:

- K=4,
- K=8,
- K=16.

Question:

What temporal scale is best?

### Ablation 6 — Frozen versus light joint fine-tuning

Question:

Can the method work as a lightweight add-on, and does joint training improve further?

## Required figures

### Figure 1 — Method overview

Show:

- observation,
- LeWM encoder,
- fast latent,
- event segmentation,
- event latent,
- reachability planner,
- low-level controller.

### Figure 2 — Failure mode of raw LeWM latent planning

Show:

- direct terminal latent cost prefers a bad path or fails at bottleneck,
- event-level planner selects a better intermediate subgoal.

### Figure 3 — Success versus horizon

This is likely the most important quantitative plot.

### Figure 4 — Event boundary examples

Show:

- TwoRoom door-crossing boundary,
- PushT contact/alignment boundary.

### Figure 5 — Reachability heatmap or latent topology

Show that event latent better reflects controllable transitions.

## Required tables

### Table 1 — Main results

Columns:

- Method,
- TwoRoom-Hard success,
- PushT-Go75 success,
- PushT-Go100 success,
- planning time,
- GPU memory or train time.

### Table 2 — Ablation

Columns:

- Variant,
- success,
- reachability AUC,
- planning time.

### Table 3 — Compute budget

Columns:

- method,
- train time,
- GPU,
- parameters added,
- frozen or joint.

## Minimum result threshold

A result is paper-worthy if at least one of these holds:

1. HiLeWM improves long-horizon success by a large absolute margin, ideally >15–20 points.
2. HiLeWM matches success while reducing planning time substantially.
3. HiLeWM strongly improves reachability prediction and this correlates with planning improvement.
4. HiLeWM shows clear qualitative event discovery with quantitative gain.

## Strong result threshold

A result is strong enough for a main-track submission if:

1. Improvement appears in both navigation and PushT.
2. Ablations show event latent and reachability loss are both necessary.
3. Success versus horizon clearly supports the thesis.
4. Runtime remains practical on a single L40S.
5. Visualizations make the failure mode and solution obvious.

## Scope guardrails

Do not spend time on:

- training very large visual encoders,
- adding language models,
- real robot demonstrations,
- too many environments,
- excessive planner engineering,
- beautiful but non-diagnostic visualizations,
- joint training before frozen-backbone results work.

## Stage 0 experiment conclusion

The first empirical target is:

**Show that HiLeWM-fixed improves over LeWM+CEM and LeWM+short-CEM on Hard TwoRoom.**

The second target is:

**Show that the same idea transfers to PushT long-horizon.**
