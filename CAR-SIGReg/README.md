# CAR-SIGReg

**Controllability-Aware Rank-Adaptive SIGReg**

A drop-in replacement for the SIGReg regularizer in [LeWorldModel (LeWM)](https://github.com/quentinll/le-wm)
that adaptively selects which latent subspace to regularize based on both
covariance variance and action-response sensitivity.

---

## Idea in One Paragraph

LeWM applies SIGReg uniformly to all latent directions — pushing the entire
embedding to look Gaussian regardless of whether those directions are
controllable by actions.  Sub-JEPA improves this by splitting the space into
multiple fixed random subspaces, but still treats all subspaces equally.
CAR-SIGReg tracks the *joint* score of each PCA direction:

```
q_i = var_i^alpha * ctrl_i^beta
```

where `var_i` is the variance-fraction and `ctrl_i` is the average action
sensitivity in that direction (via finite difference on `act_emb`).
SIGReg is then applied **only on the top-k active directions** (threshold
controlled by `tau`), while inactive directions are compressed with a cheap
L2 penalty.  This concentrates regularization where it matters for planning.

---

## Claimed Contributions

1. **Adaptive active subspace**: rank automatically adjusts every
   `update_basis_every` steps; no fixed number of subspaces to tune.
2. **Controllability-aware selection**: alpha/beta allow clean ablations
   separating variance-driven vs. action-driven subspace selection.
3. **Diagnostic metrics**: `eff_rank`, `active_rank`, `ctrl_align`,
   `inactive_energy` are logged every step — paper evidence lives in the
   metrics, not just the final success rate.

---

## File Structure

```
CAR-SIGReg/
├── car_sigreg.py            # CARSIGReg + SketchSIGReg — no le-wm dependencies
├── train.py                 # Modified lejepa_forward; run from repo root
├── config/train/
│   ├── lewm.yaml            # Main Hydra config (defaults: data=tworoom)
│   ├── launcher/local.yaml  # WandB / launcher config
│   ├── model/lewm.yaml      # JEPA model config (identical to le-wm)
│   └── data/tworoom.yaml    # TwoRoom dataset config (missing from le-wm)
└── tools/
    └── latent_diagnostics.py  # Eff-rank, ctrl-align, covariance analysis CLI
```

---

## Requirements

- The `ca-lewm/third_party/le-wm/` Python environment (`stable-worldmodel`,
  `stable-pretraining`, `lightning`, `hydra-core`, `wandb`).
- Python >= 3.10 (union type hints `X | Y` in `car_sigreg.py`).
- TwoRoom dataset at `$STABLEWM_HOME/datasets/tworoom.h5` (~12 GB).

---

## Setup

```bash
# From repo root: Personal_Lab_DngBack/
source ca-lewm/third_party/le-wm/.venv/bin/activate

export STABLEWM_HOME=~/.stable-wm        # or ~/.stable_worldmodel
export PYTHONPATH=CAR-SIGReg:ca-lewm/third_party/le-wm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Verify the dataset is reachable:
```bash
swm inspect tworoom
```

---

## Training Recipes

All commands assume the setup above.  Replace `CUDA_VISIBLE_DEVICES=0` with
the GPU index you want to use.

### Stage A — Smoke test (1 epoch, basis update debug)

Reduces `warmup_steps` and `update_basis_every` so basis updates fire within
one epoch.  Check that `train/active_rank` moves away from the warmup default
(32) after step 10.

```bash
CUDA_VISIBLE_DEVICES=0 python CAR-SIGReg/train.py \
  data=tworoom \
  wandb.enabled=false \
  trainer.max_epochs=1 \
  trainer.devices=1 \
  loader.batch_size=64 \
  loss.sigreg.kwargs.warmup_steps=10 \
  loss.sigreg.kwargs.update_basis_every=10 \
  subdir=tworoom/car_smoke
```

Expected logs each step:
```
train/pred_loss  train/sigreg_loss  train/inactive_loss
train/eff_rank   train/active_rank  train/inactive_energy
```

### Stage B — 5-epoch diagnostic

```bash
CUDA_VISIBLE_DEVICES=0 python CAR-SIGReg/train.py \
  data=tworoom \
  wandb.enabled=false \
  trainer.max_epochs=5 \
  trainer.devices=1 \
  loader.batch_size=64 \
  subdir=tworoom/car_diag_5ep
```

Watch for:
- `active_rank` converging to a stable value (ideally 8–40)
- `inactive_energy` decreasing over time
- `pred_loss` not blowing up (should stay near LeWM baseline)

---

## Ablation Matrix (paper Table 1)

Run all 6 configurations to 10 epochs, then eval each.

### A0 — LeWM baseline (original SIGReg)

Use `ca-lewm/third_party/le-wm/train.py` directly.

```bash
cd ca-lewm/third_party/le-wm
CUDA_VISIBLE_DEVICES=0 python train.py \
  data=pusht \
  data.dataset.name=tworoom.h5 \
  wandb.enabled=false \
  trainer.max_epochs=10 \
  trainer.devices=1 \
  loader.batch_size=64 \
  subdir=tworoom/lewm_baseline_10ep
```

### A2 — PCA-Rank SIGReg (covariance only, beta=0)

```bash
CUDA_VISIBLE_DEVICES=0 python CAR-SIGReg/train.py \
  data=tworoom \
  wandb.enabled=false \
  trainer.max_epochs=10 \
  trainer.devices=1 \
  loader.batch_size=64 \
  loss.sigreg.kwargs.alpha=1.0 \
  loss.sigreg.kwargs.beta=0.0 \
  subdir=tworoom/pca_rank_10ep
```

### A3 — Ctrl-Rank SIGReg (controllability only, alpha=0)

```bash
CUDA_VISIBLE_DEVICES=0 python CAR-SIGReg/train.py \
  data=tworoom \
  wandb.enabled=false \
  trainer.max_epochs=10 \
  trainer.devices=1 \
  loader.batch_size=64 \
  loss.sigreg.kwargs.alpha=0.0 \
  loss.sigreg.kwargs.beta=1.0 \
  subdir=tworoom/ctrl_rank_10ep
```

### A4 — CAR-SIGReg, no ctrl loss (main method V1)

```bash
CUDA_VISIBLE_DEVICES=0 python CAR-SIGReg/train.py \
  data=tworoom \
  wandb.enabled=false \
  trainer.max_epochs=10 \
  trainer.devices=1 \
  loader.batch_size=64 \
  loss.sigreg.use_ctrl_loss=false \
  loss.sigreg.ctrl_weight=0.0 \
  subdir=tworoom/car_select_10ep
```

### A5 — CAR-SIGReg + ctrl loss

```bash
CUDA_VISIBLE_DEVICES=0 python CAR-SIGReg/train.py \
  data=tworoom \
  wandb.enabled=false \
  trainer.max_epochs=10 \
  trainer.devices=1 \
  loader.batch_size=64 \
  loss.sigreg.use_ctrl_loss=true \
  loss.sigreg.ctrl_weight=0.01 \
  subdir=tworoom/car_full_10ep
```

---

## Evaluation

```bash
cd ca-lewm/third_party/le-wm

# Single seed (fast check)
python eval.py --config-name=tworoom.yaml \
  policy=tworoom/car_select_10ep/lewm \
  seed=42

# Multi-seed
python eval.py --config-name=tworoom.yaml \
  policy=tworoom/car_select_10ep/lewm \
  seed=42,100,2026 \
  --multirun
```

---

## Latent Diagnostics

After training, run the diagnostic script to compute controllability alignment
and effective rank — the key claims of the paper.

```bash
python CAR-SIGReg/tools/latent_diagnostics.py \
  --checkpoint tworoom/car_select_10ep/lewm_object.ckpt \
  --dataset tworoom \
  --label car_select \
  --out results/car_select_diagnostics.json

python CAR-SIGReg/tools/latent_diagnostics.py \
  --checkpoint tworoom/lewm_baseline_10ep/lewm_object.ckpt \
  --dataset tworoom \
  --label lewm_baseline \
  --out results/lewm_baseline_diagnostics.json
```

---

## Hyperparameter Troubleshooting

| Symptom | Fix |
|---------|-----|
| `pred_loss` NaN | Lower `loss.sigreg.weight` to `0.05`, `inactive_weight` to `0.001` |
| `active_rank` stuck at warmup value (32) | Reduce `warmup_steps=10`, `update_basis_every=10` for smoke test |
| `active_rank` always = `r_max` (64) | Lower `tau` to `0.90` or `r_max` to `48` |
| `active_rank` always = `r_min` (4) | Raise `tau` to `0.98` or `r_min` to `8` |
| Stage A5 too slow | Use `use_ctrl_loss=false`; basis selection already uses ctrl (no-grad) |
| OOM on A30 | `loader.batch_size=64`, `loss.sigreg.kwargs.num_proj=256` |
| `RuntimeError: scalar type Float but found Half` | Already handled in `car_sigreg.py` via `.to(emb.dtype)` casts |

---

## Expected Paper Table

| Method | Success ↑ | Pred Loss ↓ | Eff. Rank ↓ | Active Rank | Ctrl Align ↑ |
|--------|-----------|-------------|-------------|-------------|--------------|
| LeWM | | | | — | |
| Sub-JEPA* | | | | — | |
| PCA-Rank SIGReg (A2) | | | | | |
| Ctrl-Rank SIGReg (A3) | | | | | |
| CAR-SIGReg (A4) | | | | | |
| CAR-SIGReg + Ctrl (A5) | | | | | |

*Sub-JEPA must be cloned separately from https://github.com/intcomp/Sub-JEPA.

---

## Go / No-Go Criteria

**GO** if:
- CAR-SIGReg (A4) improves `eval/success` vs. LeWM at same epoch budget, OR
- `ctrl_align` is higher for CAR than PCA-Rank, with `inactive_energy` lower.

**NO-GO / Pivot** if:
- A4 loses to Sub-JEPA by more than 5pp with no interpretability advantage.
- `pred_loss` is consistently worse than LeWM baseline.

If A5 does not beat A4, use A4 as the main method and frame A5 as an
"aggressive variant" with note on compute cost.
