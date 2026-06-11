# Stage 1 — Báo cáo kết quả triển khai LeWM Baseline

**Dự án:** HiLeWM — Hierarchical Event-Structured LeWorldModel  
**Giai đoạn:** Stage 1 — Reproduce LeWM baseline  
**Ngày hoàn thành:** 2026-06-11  
**Trạng thái:** ✅ **HOÀN THÀNH** — sẵn sàng chuyển Stage 2

---

## 1. Mục tiêu Stage 1

Theo `docs/stage0/stage0_decision_gate.md`, Stage 1 yêu cầu:

1. Reproduce LeWM checkpoint trên môi trường chính thức (TwoRoom → PushT).
2. Chạy eval end-to-end với CEM planning.
3. Log đầy đủ metrics baseline (success rate, planning time, CEM config, seed, …).
4. Xác định setting mà baseline **chưa saturate** — làm nền cho HiLeWM.

**Không làm trong Stage 1:** code HiLeWM modules, event latent, reachability head.

---

## 2. Môi trường triển khai

| Hạng mục | Giá trị |
|----------|---------|
| Codebase | `HiLeWM/le-wm` (official LeWM) |
| Conda env | `personal_lab` (Python 3.12) |
| Cache | `STABLEWM_HOME=~/.stable-wm` |
| GPU | NVIDIA A30 (×2 visible) |
| Renderer | `MUJOCO_GL=egl` |
| Dependencies bổ sung | `hdf5plugin` (bắt buộc cho HDF5Dataset) |

### Dataset đã tải

| Dataset | File | Kích thước |
|---------|------|------------|
| TwoRoom | `~/.stable-wm/datasets/tworoom.h5` | ~12 GB |
| PushT | `~/.stable-wm/datasets/pusht_expert_train.h5` | ~44 GB |

### Checkpoint (HuggingFace, auto-download)

| Env | Repo | Local cache |
|-----|------|-------------|
| TwoRoom | `quentinll/lewm-tworooms` | `~/.stable-wm/checkpoints/models--quentinll--lewm-tworooms/` |
| PushT | `quentinll/lewm-pusht` | `~/.stable-wm/checkpoints/models--quentinll--lewm-pusht/` |

---

## 3. Cấu hình eval

### 3.1 CEM planner (mặc định — Full CEM)

File: `le-wm/config/eval/solver/cem.yaml`

| Tham số | Giá trị |
|---------|---------|
| `num_samples` | 300 |
| `n_steps` | 30 |
| `topk` | 30 |
| `plan_config.horizon` | 5 |
| `plan_config.action_block` | 5 (frameskip) |
| `num_eval` | 50 episode / run |

### 3.2 Short CEM (ablation)

File: `le-wm/config/eval/solver/cem_short.yaml`

| Tham số | Giá trị |
|---------|---------|
| `num_samples` | 50 |
| `n_steps` | 10 |
| `topk` | 10 |

### 3.3 Các setting eval

| Tên | Config | `goal_offset` | `eval_budget` | Mục đích |
|-----|--------|---------------|---------------|----------|
| TwoRoom (standard) | `tworoom.yaml` | 25 | 50 | Sanity / reproduce paper |
| PushT | `pusht.yaml` | 25 | 50 | Manipulation baseline |
| **TwoRoom Hard** | `tworoom_hard.yaml` | **50** | **75** | Long-horizon / topology stress test |

---

## 4. Kết quả chi tiết

### 4.1 TwoRoom — Standard (LeWM + Full CEM)

| Seed | Success rate | Successes | Wall-clock (s) | ~s/step |
|------|--------------|-----------|----------------|---------|
| 42 | 86.0% | 43/50 | 329.1 | 0.13 |
| 0 | 90.0% | 45/50 | 282.8 | 0.11 |
| 1 | 94.0% | 47/50 | 325.9 | 0.13 |
| 2 | 84.0% | 42/50 | 351.2 | 0.14 |
| **Mean** | **88.5%** | — | **322.3** | **0.13** |

**Nhận xét:** Baseline rất mạnh, gần saturate. Không phù hợp làm benchmark chính cho HiLeWM.

---

### 4.2 PushT — Standard (LeWM + Full CEM)

| Seed | Success rate | Successes | Wall-clock (s) | ~s/step |
|------|--------------|-----------|----------------|---------|
| 0 | 90.0% | 45/50 | 205.1 | 0.08 |
| 1 | 82.0% | 41/50 | 208.6 | 0.08 |
| 2 | 78.0% | 39/50 | 218.9 | 0.09 |
| **Mean** | **83.3%** | — | **210.9** | **0.08** |

**Nhận xét:** Khó hơn TwoRoom standard một chút nhưng vẫn cao. PushT long-horizon (Go75/Go100) để lại Stage 3+.

---

### 4.3 TwoRoom Hard — LeWM + Full CEM (Baseline chính — upper bound)

Setting: `goal_offset=50`, `eval_budget=75`, CEM 300×30.

| Seed | Success rate | Successes | Wall-clock (s) | ~s/step |
|------|--------------|-----------|----------------|---------|
| 42 | 46.0% | 23/50 | 4567.0 | 1.22 |
| 0 | 52.0% | 26/50 | 4484.1 | 1.20 |
| 1 | 60.0% | 30/50 | 4306.2 | 1.15 |
| 2 | 46.0% | 23/50 | 4800.5 | 1.28 |
| **Mean** | **51.0%** | — | **4539.5** | **1.21** |

---

### 4.4 TwoRoom Hard — LeWM + Short CEM (Ablation — lower bound)

Cùng Hard setting, CEM 50×10.

| Seed | Success rate | Successes | Wall-clock (s) | ~s/step |
|------|--------------|-----------|----------------|---------|
| 42 | 38.0% | 19/50 | 4669.2 | 1.25 |
| 0 | 34.0% | 17/50 | 4764.0 | 1.27 |
| 1 | 44.0% | 22/50 | 4523.0 | 1.21 |
| 2 | 42.0% | 21/50 | 4199.8 | 1.12 |
| **Mean** | **39.5%** | — | **4539.0** | **1.21** |

---

## 5. Bảng so sánh tổng hợp

| Setting | Method | Mean SR | Δ vs Easy TwoRoom |
|---------|--------|---------|-------------------|
| TwoRoom standard | Full CEM | **88.5%** | — |
| PushT standard | Full CEM | **83.3%** | — |
| **TwoRoom Hard** | **Full CEM** | **51.0%** | **−37.5 pp** |
| **TwoRoom Hard** | **Short CEM** | **39.5%** | **−49.0 pp** |
| Hard: Full vs Short | — | **+11.5 pp** | — |

```text
TwoRoom easy     ████████████████████  88.5%
Hard + Full CEM  ██████████            51.0%
Hard + Short CEM ████████              39.5%
```

---

## 6. Phân tích & kết luận

### 6.1 Pipeline reproduce thành công

- LeWM checkpoint load và eval ổn định qua `eval.py` + `policy=quentinll/lewm-*`.
- Dataset HDF5, CEM solver, và multi-seed eval đều chạy end-to-end.
- Artifacts JSON và raw log được lưu tự động.

### 6.2 Hard TwoRoom đạt mục tiêu Stage 1

- Success giảm từ ~88% xuống ~51% → **expose failure mode** rõ ràng.
- Baseline **không saturate** trên Hard setting → có không gian cho HiLeWM cải thiện.
- Gap Full vs Short CEM (~11.5 pp) hợp lệ cho ablation paper.

### 6.3 Risk đã xử lý

| Risk (stage0) | Trạng thái |
|---------------|------------|
| R1 — Env/checkpoint lỗi | ✅ Đã fix (`hdf5plugin`, dataset path) |
| R2 — Baseline saturate trên mọi task | ⚠️ Standard TwoRoom/PushT saturate; **Hard TwoRoom không** |
| R3 — Thiếu short-CEM ablation | ✅ Đã log |

### 6.4 Benchmark chính cho Stage 2+

**Primary:** `tworoom_hard.yaml` + LeWM frozen backbone.

**Targets cho HiLeWM-fixed + Short CEM:**

| Mức | Success rate | Ý nghĩa |
|-----|--------------|---------|
| Tối thiểu | > 39.5% | Vượt short-CEM |
| Paper-worthy | **> 55%** (+15 pp vs short) | Claim chính |
| Mạnh | ≥ 51% (match Full CEM) | Competitive với upper bound |

---

## 7. Artifacts

### 7.1 Raw results (append log)

```
~/.stable-wm/quentinll/tworoom_results.txt
~/.stable-wm/quentinll/pusht_results.txt
~/.stable-wm/quentinll/tworoom_hard_results.txt   # 8 runs (4 full + 4 short)
```

### 7.2 JSON per-run

```
HiLeWM/results/stage1_baseline/
  baseline_tworoom_seed{0,1,2,42}.json
  baseline_pusht_seed{0,1,2}.json
  baseline_tworoom_hard_seed{0,1,2,42}.json      # chỉ short-CEM (xem §7.3)
  summary.json                                      # chưa cập nhật hard runs
```

### 7.3 Scripts & configs đã tạo

```
HiLeWM/scripts/collect_baseline_results.py
HiLeWM/scripts/run_baseline_eval.sh
HiLeWM/scripts/cache_latents.py          # Stage 2 prep
HiLeWM/le-wm/config/eval/tworoom_hard.yaml
HiLeWM/le-wm/config/eval/solver/cem_short.yaml
```

### 7.4 Known issue

`eval.py` ghi JSON theo `baseline_{env}_seed{N}.json` — khi chạy Full rồi Short trên cùng seed, **file JSON bị ghi đè** bởi run sau. Số liệu Full CEM Hard vẫn đầy đủ trong `tworoom_hard_results.txt`.  
**Khuyến nghị Stage 2:** đổi tên JSON thành `baseline_{env}_{solver}_seed{N}.json`.

---

## 8. Lệnh reproduce

```bash
conda activate personal_lab
export STABLEWM_HOME=~/.stable-wm
cd HiLeWM/le-wm

# Standard
python eval.py --config-name=tworoom.yaml policy=quentinll/lewm-tworooms seed=42
python eval.py --config-name=pusht.yaml policy=quentinll/lewm-pusht seed=0

# Hard baselines
python eval.py --config-name=tworoom_hard.yaml policy=quentinll/lewm-tworooms seed=42
python eval.py --config-name=tworoom_hard.yaml policy=quentinll/lewm-tworooms solver=cem_short seed=42

# Collect JSON
python ../scripts/collect_baseline_results.py
```

---

## 9. Go / No-Go — Stage 2

| Tiêu chí | Kết quả |
|----------|---------|
| LeWM checkpoint eval được | ✅ |
| Baseline metrics logged | ✅ |
| ≥1 environment end-to-end | ✅ (TwoRoom, PushT, Hard TwoRoom) |
| Hiểu baseline fail / saturate ở đâu | ✅ (easy saturate; hard ~51%) |
| Short-CEM ablation | ✅ (~39.5%) |

### Quyết định: **GO → Stage 2**

**Task đầu tiên Stage 2:**

```bash
export STABLEWM_HOME=~/.stable-wm
python HiLeWM/scripts/cache_latents.py --dataset tworoom --max-episodes 500
```

Sau đó: fixed event segments (K=8) → reachability head → hierarchical planner v0.

**Không làm:** joint fine-tune LeWM, learned boundaries, thêm environment mới — cho đến khi HiLeWM-fixed có gain trên Hard TwoRoom.

---

## 10. Tham chiếu

- `docs/stage0/stage0_decision_gate.md` — checklist Stage 1
- `docs/stage0/method_contract.md` — implementation order Stage 2+
- `docs/stage0/experiment_scope.md` — paper targets
- `docs/stage1_go_nogo.md` — bản tóm tắt ngắn (có thể deprecated bởi file này)
