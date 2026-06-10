# Claims and Required Evidence

## Purpose

This document maps each intended paper claim to the evidence required to support it. If a claim cannot be supported by evidence, it must be removed or weakened.

## Claim 1

### Claim

**LeWorldModel-style local latent prediction is not sufficient for long-horizon plannability.**

### Required evidence

1. LeWM baseline performs well on short-horizon tasks but degrades on longer-horizon variants.
2. Raw terminal latent distance fails to rank useful intermediate subgoals in at least one environment.
3. Visualization shows a failure case where the direct latent-distance objective is misleading.

### Required experiments

- LeWM + CEM on TwoRoom short versus Hard TwoRoom.
- LeWM + CEM on PushT-Go50/Go75/Go100.
- Subgoal ranking using raw fast latent Euclidean distance.

### Required plots

- Success versus horizon.
- Example trajectory failure.
- Latent distance heatmap or subgoal score map.

### Failure condition

If LeWM+CEM already solves the long-horizon task reliably, this claim is weak. In that case, increase task difficulty or pivot to planning cost.

---

## Claim 2

### Claim

**Event-level latent abstraction improves long-horizon reachability modeling.**

### Required evidence

1. Event latent reachability AUC > fast latent reachability AUC.
2. Event latent distance/reachability better identifies useful subgoals.
3. Segment length ablation shows a meaningful temporal scale.

### Required experiments

- Train reachability head on fast latent.
- Train reachability head on fixed event latent.
- Train reachability head on learned-boundary event latent, if available.
- Compare AUC and top-k subgoal ranking.

### Required metrics

- Reachability AUC.
- Recall@k for useful subgoals.
- Ranking correlation with successful trajectories.

### Failure condition

If event latent does not improve reachability AUC or subgoal ranking, the hierarchy is not justified.

---

## Claim 3

### Claim

**Hierarchical event-level planning improves long-horizon control from pixels.**

### Required evidence

1. HiLeWM improves success rate over LeWM+CEM on Hard TwoRoom or equivalent.
2. HiLeWM improves or matches LeWM+CEM on PushT long-horizon.
3. Short-horizon CEM alone does not explain the improvement.

### Required experiments

Compare:

1. LeWM + full CEM.
2. LeWM + short CEM.
3. HiLeWM-fixed + short CEM.
4. HiLeWM-learned-boundary + short CEM, if available.

### Required metrics

- success rate,
- planning time,
- number of rollouts,
- horizon sensitivity.

### Failure condition

If HiLeWM only improves reachability AUC but not planning success, the paper becomes a representation analysis paper rather than a control paper. This is weaker.

---

## Claim 4

### Claim

**The improvement comes from event-level temporal abstraction and reachability alignment, not merely from a stronger planner.**

### Required evidence

Ablations must show degradation when removing key components.

### Required ablations

1. No event latent.
2. No reachability loss.
3. Fixed boundary only.
4. Learned boundary.
5. Euclidean event distance instead of reachability score.
6. Short CEM only.

### Required table

Ablation table with:

- success rate,
- reachability AUC,
- planning time.

### Failure condition

If the full method is not better than "reachability head on fast latent", the event contribution is weak.

---

## Claim 5

### Claim

**HiLeWM remains lightweight and suitable for limited compute.**

### Required evidence

1. Training runs on 1× L40S.
2. Frozen-backbone version works.
3. Added parameters are small relative to LeWM.
4. Latent caching reduces training cost.

### Required logs

- GPU type,
- peak GPU memory,
- training wall-clock time,
- number of parameters,
- dataset size,
- number of seeds.

### Required table

Compute table:

| Component | Trainable? | Params | Train time | GPU memory |
|---|---:|---:|---:|---:|
| LeWM backbone | frozen/joint | ... | ... | ... |
| Event aggregator | yes | ... | ... | ... |
| BoundaryNet | optional | ... | ... | ... |
| Reachability head | yes | ... | ... | ... |

### Failure condition

If the method requires multiple GPUs or long training cycles, it no longer matches the project constraint.

---

## Claim 6

### Claim

**Learned event boundaries correspond to control-relevant changes.**

This claim is optional. Use it only if Stage 5 succeeds.

### Required evidence

1. Boundary visualizations align with meaningful events.
2. Learned boundary improves over fixed segmentation.
3. Boundary density is not degenerate.

### Required experiments

- Fixed K=4/8/16.
- Learned boundary.
- BoundaryNet visualization on TwoRoom and PushT.

### Required metrics

- success rate,
- reachability AUC,
- average segment length,
- boundary entropy,
- qualitative examples.

### Failure condition

If learned boundary does not improve over fixed segmentation, weaken the claim:

**Fixed temporal abstraction is already sufficient to improve long-horizon planning in our tested settings.**

---

## Final claim hierarchy

### Minimum claim set

Use this if experiments are modest:

1. LeWM local planning degrades under long-horizon stress.
2. A lightweight event-level reachability layer improves planning.
3. The method works with frozen LeWM latents on a single L40S.

### Strong claim set

Use this if experiments are strong:

1. Local prediction is not sufficient for plannability.
2. Event-structured latent abstraction improves reachability geometry.
3. Hierarchical event planning improves long-horizon pixel control.
4. Learned event boundaries discover control-relevant phase transitions.
5. The method is compute-efficient and reproducible.

## Stage 0 evidence checklist

Before writing the final paper, every claim in the abstract must map to:

- one main table result,
- one ablation result,
- one diagnostic plot,
- or one visualization.

If a sentence in the abstract cannot be mapped to evidence, remove it.
