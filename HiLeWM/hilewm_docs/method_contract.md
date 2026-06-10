# Method Contract

## Purpose

This document defines what the HiLeWM method is and is not. It prevents the implementation from drifting into a large, hard-to-debug system.

## Method name

**HiLeWM**

Full name:

**Hierarchical Event-Structured LeWorldModel**

## One-line definition

HiLeWM adds a slow event-level latent abstraction and reachability-guided hierarchical planner on top of a fast LeWorldModel latent dynamics backbone.

## Backbone

The backbone is LeWM.

Inputs:

- observation `o_t`,
- action `a_t`.

Outputs:

- fast latent `z_t`,
- predicted next fast latent `z_{t+1}`.

The backbone should initially be frozen.

## Fast latent

Definition:

`z_t = Encoder(o_t)`

Role:

- encode current visual state,
- support local dynamics,
- support low-level control.

Fast latent should not be expected to directly solve long-horizon topology.

## Event latent

Definition:

`e_k = EventAggregator(z_{t_start:t_end})`

Role:

- summarize a temporally extended segment,
- represent control-relevant events,
- support high-level planning.

Event latent is slower than fast latent.

## Segmentation

### MVP segmentation

Fixed-length segments:

- K=4,
- K=8,
- K=16.

This must be implemented first.

### Advanced segmentation

Learned boundary module:

`b_t = BoundaryNet(z_{t-w:t+w}, a_{t-w:t+w})`

This is optional until the fixed-segment planner works.

## Event aggregator

Initial implementation:

1. Mean pooling.
2. MLP over mean-pooled latent.
3. Attention pooling, only if needed.

Do not start with a large Transformer.

## Event dynamics

Predict next event latent:

`e_hat_{k+1} = EventDynamics(e_k, action_summary_k)`

Action summary:

`action_summary_k = mean(a_t ... a_{t+K-1})`

Loss:

`L_event_pred = || e_hat_{k+1} - e_{k+1} ||^2`

## Reachability head

Predict whether one event state can reach another under a budget.

Input:

`R(e_i, e_j, h)`

Output:

logit for reachability.

Budget set:

`h ∈ {1, 2, 4, 8}`

Positive examples:

- same episode,
- temporal distance within budget.

Negative examples:

- same episode but too far,
- different episode,
- same episode but unreachable if environment labels allow it.

Loss:

`BCEWithLogitsLoss`

## Total MVP loss

For frozen-backbone MVP:

`L = L_event_pred + λ_reach L_reach + λ_reg L_event_reg`

Initial weights:

- `λ_reach = 1.0`
- `λ_reg = 0.01–0.05`

Do not include too many loss terms in the first implementation.

## Joint training loss

Only after MVP works:

`L_total = L_LeWM + λ_event L_event_pred + λ_reach L_reach + λ_sig_event L_event_sigreg + λ_boundary L_boundary`

Initial weights:

- `λ_event = 1.0`
- `λ_reach = 1.0`
- `λ_sig_event = 0.05`
- `λ_boundary = 0.1`

## Planner

### High-level planner

Input:

- current event latent,
- goal event latent,
- candidate event memory.

Candidate selection:

`score = R(e_current, e_candidate, h1) + R(e_candidate, e_goal, h2)`

Select top-k candidates.

### Low-level controller

MVP:

- short-horizon CEM using LeWM.

Optional:

- goal-conditioned inverse dynamics.

## Candidate event memory

Build from cached training trajectories.

Each candidate stores:

- event latent,
- original episode id,
- start time,
- end time,
- representative observation,
- associated fast latent,
- action sequence if available.

## Main method variants

### HiLeWM-fixed

Frozen LeWM + fixed-length event segments + reachability planner.

This is the required MVP.

### HiLeWM-boundary

Frozen LeWM + learned event boundaries + reachability planner.

This is the preferred full method.

### HiLeWM-joint

Light joint fine-tuning.

This is optional.

## Non-goals

Do not implement these in the first version:

- large pretrained vision encoders,
- language-conditioned planning,
- diffusion planners,
- transformer world models with hundreds of millions of parameters,
- real robot integration,
- foundation-model comparison unless scripts are already available.

## Implementation order

1. LeWM baseline.
2. Latent cache.
3. Fixed event dataset.
4. Event aggregator.
5. Reachability head.
6. Hierarchical planner.
7. Learned boundary.
8. Joint fine-tuning.

## Method pass condition

The method passes the MVP stage if:

1. Event reachability AUC is meaningfully above fast-latent baseline.
2. HiLeWM-fixed improves long-horizon success over LeWM+short-CEM.
3. HiLeWM-fixed is competitive with or better than LeWM+full-CEM on at least one hard setting.

## Method pivot condition

Pivot if:

1. Event reachability AUC is close to random.
2. Hierarchical planner does not improve any long-horizon setting.
3. Fixed segmentation and learned segmentation both fail.
4. Baseline LeWM already solves all chosen environments.

Possible pivots:

1. AdaSIGReg-LeWM.
2. Reachability-only metric learning for LeWM.
3. Action-free or latent-action LeWM.
