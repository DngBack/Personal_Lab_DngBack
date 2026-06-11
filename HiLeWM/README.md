# HiLeWM

Hierarchical Event-Structured extension of [LeWorldModel](le-wm/).

## Layout

```text
HiLeWM/
  le-wm/              # official LeWM codebase (baseline)
  scripts/            # eval helpers, result collector, latent cache
  results/            # JSON baseline artifacts
  caches/             # cached fast latents (Stage 2)
  docs/stage0/        # research framing package
  hilewm/             # HiLeWM modules (Stage 2+)
```

## Quick start

```bash
conda activate personal_lab
export STABLEWM_HOME=~/.stable-wm
cd HiLeWM/le-wm
```

### Collect existing baseline results

```bash
python ../scripts/collect_baseline_results.py
```

### Run next baseline experiments

```bash
# Hard TwoRoom (longer horizon)
bash ../scripts/run_baseline_eval.sh tworoom_hard

# Short-CEM ablation on hard setting
bash ../scripts/run_baseline_eval.sh tworoom_short_cem
```

### Stage 2 — cache latents

```bash
python ../scripts/cache_latents.py --dataset tworoom --max-episodes 200
```

## Stage status

- **Stage 1 (complete):** [docs/stage1_report.md](docs/stage1_report.md)
- Stage 1 summary: [docs/stage1_go_nogo.md](docs/stage1_go_nogo.md)
