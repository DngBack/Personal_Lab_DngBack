# HiLeWM Stage 0 — Research Framing Package

Date: 2026-06-09  
Project: HiLeWM — Hierarchical Event-Structured LeWorldModel for Long-Horizon Pixel Planning  
Target venue: AAAI-27 Main Technical Track  
Hardware constraint: 1× NVIDIA L40S as the primary development target

This folder contains the completed Stage 0 research planning package.

## Files

1. `research_thesis.md`  
   Defines the core research thesis, problem framing, novelty boundary, and paper narrative.

2. `experiment_scope.md`  
   Defines environments, baselines, metrics, artifact requirements, and what must be excluded to avoid scope creep.

3. `claims_and_required_evidence.md`  
   Converts each paper claim into required empirical evidence, ablations, plots, and failure conditions.

4. `method_contract.md`  
   Specifies the minimum method that should be implemented in Stage 1–4.

5. `risk_register.md`  
   Lists technical, novelty, timeline, and compute risks, with mitigation and pivot triggers.

6. `stage0_decision_gate.md`  
   Final Stage 0 go/no-go checklist before coding.

## Stage 0 outcome

The recommended main direction is:

**HiLeWM: Hierarchical Event-Structured JEPA World Models for Long-Horizon Pixel Planning**

Core thesis:

**Stable local latent prediction is not sufficient for long-horizon pixel planning. A world model also needs temporally abstract, controllability-aligned latent structure.**

The immediate next stage is:

**Stage 1 — reproduce LeWorldModel baseline using the official codebase/checkpoints, starting with TwoRoom and PushT.**
