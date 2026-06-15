# Hướng dẫn chạy LeWM (PushT) — Personal Lab

Tài liệu này tổng hợp cách setup, chạy training, và các thay đổi đã làm **ngoài README gốc** của `le-wm`.

Repo LeWM nằm tại: `ca-lewm/third_party/le-wm/`

---

## 1. Cấu trúc thư mục

```
Personal_Lab_DngBack/
└── ca-lewm/
    ├── LEWM_SETUP.md                  ← file này
    └── third_party/le-wm/
        ├── .venv/                     ← môi trường Python
        ├── data/
        │   └── pusht_expert_train.h5    ← data đã giải nén (~44 GB)
        └── train.py                   ← đã sửa (xem mục 7)
```

---

## 2. Cài đặt môi trường (chỉ làm 1 lần)

```bash
cd ca-lewm/third_party/le-wm

uv venv --python=3.10
source .venv/bin/activate

# Lưu ý zsh: phải quote dấu []
uv pip install 'stable-worldmodel[train,env,format]'
```

**Vì sao thêm `format`?**  
README gốc chỉ cài `[train,env]`, nhưng file data `.h5` cần extra `format` (gói `h5py`, `hdf5plugin`) mới đọc được.

---

## 3. Chuẩn bị data

### Giải nén (nếu còn file `.zst`)

```bash
cd ca-lewm/third_party/le-wm/data
zstd -d pusht_expert_train.h5.zst

# Xóa file nén sau khi giải nén thành công (tùy chọn):
# zstd -d --rm pusht_expert_train.h5.zst
```

> **Lưu ý:** Đây là file `.h5.zst` (zstd nén trực tiếp), **không** phải tar archive.  
> Lệnh `tar --zstd` trong README chỉ dùng cho `.tar.zst`.

### Đặt data đúng chỗ training đọc

Training **không** đọc từ `data/` trong repo, mà từ:

```
~/.stable_worldmodel/datasets/
```

**Symlink (khuyên dùng — tiết kiệm disk):**

```bash
mkdir -p ~/.stable_worldmodel/datasets
ln -sf "$(pwd)/ca-lewm/third_party/le-wm/data/pusht_expert_train.h5" \
       ~/.stable_worldmodel/datasets/pusht_expert_train.h5
```

**Kiểm tra:**

```bash
source ca-lewm/third_party/le-wm/.venv/bin/activate
swm inspect pusht_expert_train
```

---

## 4. Chạy training

### Lệnh khuyên dùng

```bash
cd ca-lewm/third_party/le-wm
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=1 python train.py \
  data=pusht \
  data.dataset.name=pusht_expert_train.h5 \
  trainer.devices=1 \
  loader.batch_size=64
```

### Giải thích từng phần

| Tham số | Lý do |
|---------|--------|
| `CUDA_VISIBLE_DEVICES=1` | GPU 0 thường bị process khác chiếm ~11 GB |
| `trainer.devices=1` | Tránh DDP 2 GPU (dễ OOM / NCCL lỗi) |
| `data.dataset.name=pusht_expert_train.h5` | Config gốc trỏ `.lance`, ta dùng file `.h5` |
| `loader.batch_size=64` | Giảm OOM (mặc định 128) |
| `PYTORCH_CUDA_ALLOC_CONF=...` | Giảm lỗi phân mảnh VRAM |

### Nếu vẫn OOM

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  data=pusht \
  data.dataset.name=pusht_expert_train.h5 \
  trainer.devices=1 \
  loader.batch_size=32
```

### Test nhanh (1 epoch)

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  data=pusht \
  data.dataset.name=pusht_expert_train.h5 \
  trainer.devices=1 \
  trainer.max_epochs=1 \
  loader.batch_size=32
```

---

## 5. Output sau khi train

| Loại | Đường dẫn |
|------|-----------|
| Weights LeWM | `~/.stable_worldmodel/checkpoints/lewm/weights_epoch_*.pt` |
| Config model | `~/.stable_worldmodel/checkpoints/lewm/config.json` |
| Log / checkpoint Lightning | `~/.cache/stable-pretraining/runs/<date>/<run_id>/` |
| Hydra config run | `ca-lewm/third_party/le-wm/outputs/<date>/<time>/config.yaml` |

**Xem checkpoint:**

```bash
swm checkpoints
```

---

## 6. Đánh giá sau training

```bash
cd ca-lewm/third_party/le-wm
source .venv/bin/activate

python eval.py --config-name=pusht.yaml policy=lewm
```

`policy=lewm` trỏ tới `~/.stable_worldmodel/checkpoints/lewm/`.

---

## 7. Những gì đã sửa / làm thêm (ngoài README)

### A. Sửa code — `third_party/le-wm/train.py`

Thêm sau `import torch`:

```python
# Avoid CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH (torch 2.12+cu130 vs system cuDNN).
torch.backends.cudnn.enabled = False
```

**Lý do:** PyTorch 2.12+cu130 xung đột cuDNN hệ thống → lỗi `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` khi chạy ViT conv2d. Tắt cuDNN vẫn train trên GPU bình thường.

### B. Cài thêm package (không có trong README)

```bash
uv pip install 'stable-worldmodel[format]'
```

Để đọc file `.h5`.

### C. Setup data (không có trong README)

- Giải nén `.h5.zst` bằng `zstd -d`.
- Symlink `.h5` vào `~/.stable_worldmodel/datasets/`.

### D. Override khi chạy (không sửa file config)

| Vấn đề gặp phải | Cách xử lý |
|-----------------|------------|
| `zsh: no matches found: stable-worldmodel[train,env]` | Quote: `'stable-worldmodel[train,env,format]'` |
| NCCL OOM khi DDP 2 GPU | `trainer.devices=1` + `CUDA_VISIBLE_DEVICES=1` |
| CUDA OOM khi train | `loader.batch_size=64` hoặc `32` |
| Config trỏ `.lance` nhưng chỉ có `.h5` | `data.dataset.name=pusht_expert_train.h5` |

### E. Không sửa

- File config YAML gốc (`config/train/...`) — chỉ override qua command line.
- README và các file khác trong repo `le-wm` (trừ `train.py`).

---

## 8. One-liner đầy đủ

```bash
cd ca-lewm/third_party/le-wm && \
source .venv/bin/activate && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
CUDA_VISIBLE_DEVICES=1 python train.py \
  data=pusht \
  data.dataset.name=pusht_expert_train.h5 \
  trainer.devices=1 \
  loader.batch_size=64
```

---

## 9. Ghi chú thêm

- **WandB** mặc định tắt (`config/train/launcher/local.yaml`). Bật bằng:
  ```bash
  wandb.enabled=true wandb.config.entity=YOUR_ENTITY wandb.config.project=YOUR_PROJECT
  ```
- **100 epoch** mất khá lâu (~5–6 it/s). Loss train ~0.07 / val ~0.12 ở epoch 12+ là bình thường.
- **GPU 0** thường bị chiếm bởi process khác — luôn ưu tiên `CUDA_VISIBLE_DEVICES=1`.
- Cache mặc định: `STABLEWM_HOME` không set → dùng `~/.stable_worldmodel` (khác với `~/.stable-wm/` ghi trong README).
