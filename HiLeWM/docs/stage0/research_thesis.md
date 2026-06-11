# Research Thesis

## Working title

**HiLeWM: Hierarchical Event-Structured JEPA World Models for Long-Horizon Pixel Planning**

Alternative titles:

1. **From Local Prediction to Event-Level Planning in Latent World Models**
2. **Temporally Abstract JEPA World Models for Long-Horizon Control from Pixels**
3. **Learning Event-Structured Latent Spaces for Plannable World Models**

Recommended title for the first draft:

**HiLeWM: Hierarchical Event-Structured JEPA World Models for Long-Horizon Pixel Planning**

## Background

LeWorldModel, or LeWM, proposes a stable end-to-end Joint-Embedding Predictive Architecture from raw pixels. The method is intentionally lightweight: it trains a latent world model using a next-embedding prediction loss and a Gaussian latent regularizer, without pixel reconstruction, pretrained visual encoders, EMA target networks, reward labels, or complex auxiliary losses.

This is important because LeWM makes a class of world models practical under limited compute. The official paper and repository state that the model is small, trains on a single GPU, and supports planning in learned latent space. This matches the hardware constraint of this project: 1× NVIDIA L40S.

However, LeWM is primarily a **local predictive latent model**. It predicts the next latent state and uses model-predictive control over short candidate action sequences. This is suitable for short-horizon control, but it does not by itself guarantee that the learned latent space has a good geometry for long-horizon planning.

## Central problem

The central problem is:

**A latent representation can be predictive without being plannable.**

A next-step predictive world model may encode physical state information, but long-horizon control requires more than state encoding. It requires a latent geometry where reachability, bottlenecks, subgoals, and temporally extended events are represented in a way that helps planning.

In other words:

- LeWM can learn **what happens next**.
- Long-horizon planning needs to know **which intermediate events make a distant goal reachable**.

This gap becomes visible in tasks such as:

- navigating through rooms connected by narrow doors,
- pushing or manipulating objects through contact-rich phases,
- reaching goals that require intermediate subgoals,
- planning where the direct Euclidean latent distance to the final goal is misleading.

## Main thesis

The proposed thesis is:

**Stable local latent prediction is not sufficient for long-horizon pixel planning. World models also need temporally abstract, controllability-aligned latent structure.**

The proposed solution is:

**Add an event-level latent abstraction on top of LeWorldModel. The fast latent models local dynamics; the slow event latent models temporally extended, control-relevant transitions. Planning is then performed through event/subgoal states instead of only through short action rollouts.**

## Why this is not merely an engineering extension

A weak version of this work would be:

> Add another reachability head to LeWM.

That is not enough for a strong AAAI paper.

The stronger version is:

> Learn a temporally abstract latent structure inside or on top of a JEPA world model, and show that this structure changes long-horizon planning behavior.

The novelty should be framed around **representation geometry for planning**, not just planner engineering.

## Proposed method in one paragraph

HiLeWM keeps the LeWM fast latent backbone for local dynamics. It then constructs a slow event latent by segmenting latent trajectories into temporally extended events and aggregating the fast latent sequence inside each event. The event representation is trained with event prediction and budget-conditioned reachability objectives. At test time, the planner first selects event-level subgoals using reachability scores, then uses a short-horizon low-level controller to execute each event. This reduces the burden on short-horizon MPC and makes planning more robust to long-horizon bottlenecks.

## Proposed method in one sentence

**HiLeWM turns a locally predictive JEPA world model into a hierarchical plannable world model by learning event-level latent abstractions aligned with reachability.**

## Target contribution list

The paper should claim three main contributions.

### Contribution 1 — Event-structured latent abstraction

We introduce an event-level latent layer for LeWorldModel that summarizes temporally extended transitions rather than individual frames.

Expected evidence:

- Event latents cluster meaningful behavior phases.
- Event boundaries or fixed segments correspond to control-relevant changes.
- Event latents improve long-horizon reachability prediction over fast latent states.

### Contribution 2 — Reachability-aligned event representation

We train event latents with budget-conditioned reachability and event prediction objectives, making the representation more suitable for planning than raw latent Euclidean distance.

Expected evidence:

- Higher reachability AUC than fast latent distance.
- Better subgoal ranking.
- Better performance under long-horizon stress tests.

### Contribution 3 — Lightweight hierarchical planning

We show that event-level planning improves long-horizon success while remaining compatible with limited compute.

Expected evidence:

- Runs on 1× L40S.
- Uses cached LeWM latents.
- Improves success rate on long-horizon tasks.
- Does not require foundation-model-scale compute.

## Expected paper story

The ideal paper narrative is:

1. JEPA world models such as LeWM make stable pixel-based latent world modeling practical.
2. But local prediction does not imply long-horizon plannability.
3. We identify that the missing structure is temporal abstraction and reachability-aligned geometry.
4. We introduce HiLeWM, an event-structured extension to LeWM.
5. HiLeWM improves long-horizon planning on navigation and manipulation tasks while remaining lightweight.
6. Ablations show that the gain comes from event-level reachability structure, not merely from a stronger planner.

## Minimum viable paper claim

If time is limited, the minimum viable claim is:

**A lightweight event-level planning layer over frozen LeWorldModel improves long-horizon pixel planning without retraining the full world model.**

This version is weaker but still useful.

## Strong paper claim

If experiments are strong, the full claim is:

**Learning event-structured latent abstractions jointly with a JEPA world model yields controllability-aligned representations that significantly improve long-horizon pixel planning under limited compute.**

This is the target claim for AAAI.

## What the paper must not claim

Avoid overclaiming:

- Do not claim general embodied intelligence.
- Do not claim universal world modeling.
- Do not claim human-like event cognition unless directly evaluated.
- Do not claim state-of-the-art across all world-model benchmarks unless all baselines are reproduced carefully.
- Do not claim that LeWM is fundamentally flawed; frame it as a strong local predictive baseline whose latent geometry can be improved for long-horizon planning.

## Research question

Primary research question:

**Can a lightweight event-level latent abstraction turn a locally predictive pixel world model into a more effective long-horizon planner?**

Secondary research questions:

1. Does event-level reachability improve subgoal selection compared with raw latent Euclidean distance?
2. Is learned temporal abstraction better than fixed-length segmentation?
3. Can the method improve planning without expensive full-model retraining?
4. Does the method remain practical on a single L40S?

## Hypotheses

### H1 — Event latent improves reachability prediction

Event latents should predict long-horizon reachability better than fast latents because they abstract over local frame-level variation.

### H2 — Event-level planning improves long-horizon success

A planner that selects event-level subgoals should outperform short-horizon CEM when the goal requires intermediate bottlenecks.

### H3 — The improvement is strongest in topological or contact-rich tasks

The gain should be largest in TwoRoom/Hard-TwoRoom/MultiRoom and PushT long-horizon variants.

### H4 — Frozen-backbone training is sufficient for the first paper result

A frozen LeWM encoder plus event/reachability modules should already provide a meaningful improvement, reducing compute risk.

## Stage 0 conclusion

Proceed to Stage 1 only if the project accepts this narrow thesis:

**We are not trying to build a larger LeWM. We are trying to make LeWM more plannable over long horizons by adding event-level temporal abstraction and reachability structure.**
