# Stage 1 Go/No-Go — LeWM Baseline Reproduction

Date: 2026-06-11

## Completed

| Check | Status |
|-------|--------|
| LeWM checkpoint eval works | ✅ |
| TwoRoom end-to-end (4 seeds) | ✅ |
| PushT end-to-end (3 seeds) | ✅ |
| Baseline JSON artifacts | ✅ `results/stage1_baseline/` |

## Baseline summary (LeWM + full CEM)

| Environment | Seeds | Mean SR | Range |
|-------------|-------|---------|-------|
| TwoRoom | 42, 0, 1, 2 | **88.5%** | 84–94% |
| PushT | 0, 1, 2 | **83.3%** | 78–90% |

CEM config: horizon=5, candidates=300, n_steps=30, goal_offset=25, eval_budget=50.

## Interpretation

1. **Standard TwoRoom saturates** (~85–95%). Baseline is strong; HiLeWM gain is unlikely on this easy setting alone.
2. **PushT is slightly harder** (~78–90%) but still high. Long-horizon PushT variants (Go75/Go100) are needed later.
3. **Risk R2 triggered** (see `docs/stage0/risk_register.md`): baseline may already solve selected tasks.

## Required next experiments (still Stage 1 / pre-Stage-2)

Run **Hard TwoRoom** and **short-CEM** baselines before implementing HiLeWM modules:

```bash
conda activate personal_lab
export STABLEWM_HOME=~/.stable-wm
bash HiLeWM/scripts/run_baseline_eval.sh tworoom_hard
bash HiLeWM/scripts/run_baseline_eval.sh tworoom_short_cem
```

Configs:
- `le-wm/config/eval/tworoom_hard.yaml` — goal_offset=50, eval_budget=75
- `le-wm/config/eval/solver/cem_short.yaml` — 50 candidates, 10 CEM steps

## Stage 2 entry criteria

Proceed to latent cache + event modules only if:

1. ✅ Checkpoint + eval pipeline stable
2. ✅ Metrics logged to JSON
3. ⬜ Hard setting shows **lower** baseline success (target: expose bottleneck failures)
4. ⬜ Short-CEM baseline logged (required comparison for paper)

## Stage 2 first task

```bash
export STABLEWM_HOME=~/.stable-wm
python HiLeWM/scripts/cache_latents.py --dataset tworoom --max-episodes 200
```

Output: `HiLeWM/caches/latents/tworoom/ep_*.npz`

## Decision

**GO to Stage 2 prep** (latent cache + hard-baseline runs).  
**NO-GO on HiLeWM module coding** until hard-baseline numbers are collected.
