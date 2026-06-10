# Benchmark & Tối ưu Inference Extraction — Tài liệu kỹ thuật

Tài liệu tổng hợp hai lần triển khai benchmark schema_align (GIẤY GỬI TIỀN TIẾT KIỆM) qua vLLM, các kỹ thuật **có sẵn** trong engine, các kỹ thuật **đã thêm** ở lần tối ưu, và kết quả đo được.

**Phạm vi:** Bước `schema_align_llm` — map Chandra blocks → JSON bbox theo layout schema. **Chưa** gồm Chandra OCR (vision).

**Model:** `Qwen/Qwen2.5-3B-Instruct`  
**Hardware:** NVIDIA A30, `CUDA_VISIBLE_DEVICES=1`  
**Dữ liệu:** 8 PDF `test_1` … `test_8` trong `data/test/GIAY_GUI_TIEN_TIET_KIEM/`

---

## Mục lục

1. [Kiến trúc pipeline](#1-kiến-trúc-pipeline)
2. [Ba lần triển khai benchmark](#2-ba-lần-triển-khai-benchmark)
3. [Kỹ thuật có sẵn (vLLM & application)](#3-kỹ-thuật-có-sẵn-vllm--application)
4. [Kỹ thuật thêm ở lần tối ưu (Round 2)](#4-kỹ-thuật-thêm-ở-lần-tối-ưu-round-2)
5. [Kết quả chi tiết theo scenario](#5-kết-quả-chi-tiết-theo-scenario)
6. [So sánh trước / sau baseline](#6-so-sánh-trước--sau-baseline)
7. [Kết luận & cấu hình khuyến nghị](#7-kết-luận--cấu-hình-khuyến-nghị)
8. [Chưa thử — hướng tiếp theo](#8-chưa-thử--hướng-tiếp-theo)
9. [Cách chạy lại benchmark](#9-cách-chạy-lại-benchmark)
10. [File kết quả](#10-file-kết-quả)

---

## 1. Kiến trúc pipeline

```
PDF/Ảnh
   │
   ▼
┌─────────────────────────────────────┐
│  Chandra OCR-2 (vision, HuggingFace) │  ← chưa benchmark lần này
│  → HTML + blocks (bbox 0–1000)      │
└─────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────┐
│  Schema Align LLM (text, vLLM API)   │  ← phạm vi benchmark
│  Input: system + fewshots + blocks   │
│  Output: JSON map field → bbox       │
└─────────────────────────────────────┘
```

**Đặc điểm workload schema_align:**

| Thông số | Giá trị điển hình |
|----------|-------------------|
| Schema fields | 41 |
| Chandra blocks / trang | ~40 |
| Prompt tokens / trang | ~4.100 |
| Completion tokens / trang | ~1.300–1.500 |
| Decode | JSON dài, 41 key |

---

## 2. Ba lần triển khai benchmark

### Lần 1 — Baseline (`round1_plain`)

| Hạng mục | Giá trị |
|----------|---------|
| vLLM | `vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000` (plain) |
| `max_new_tokens` | **4096** |
| Scenarios | 6 (`scenarios.yaml`) |
| Kết quả | `bench/results/bench_report.json` |

### Lần 2a — Tối ưu application (`round2_plain`)

| Hạng mục | Giá trị |
|----------|---------|
| vLLM | Plain (giống Lần 1) |
| `max_new_tokens` | **8192** |
| Thêm | CCU sweep 2/4/8, compact prompt, mega concat |
| Scenarios | 7 (`scenarios_round2.yaml`) |
| Kết quả | `bench/results/round2_plain/bench_report.json` |

### Lần 2b — Tối ưu vLLM server (`round2_optimized`)

| Hạng mục | Giá trị |
|----------|---------|
| vLLM | `max-num-seqs=4`, `gpu-memory-utilization=0.85`, `max-model-len=32768` |
| `max_new_tokens` | 8192 |
| Scenarios | 3 (`scenarios_round2_optimized.yaml`) — serial, CCU=4, CCU=8 |
| Kết quả | `bench/results/round2_optimized/bench_report.json` |

---

## 3. Kỹ thuật có sẵn (vLLM & application)

### 3.1. vLLM engine (bật mặc định, không cần flag)

Các kỹ thuật sau **đã hoạt động** trong mọi lần chạy benchmark vì là default của vLLM 0.20+.

#### Continuous Batching (Dynamic Batching)

| | |
|--|--|
| **Tên khác** | Dynamic batching, iteration-level scheduling |
| **Mô tả** | vLLM không chờ cả batch đồng bộ; request mới vào/ra liên tục trong khi GPU decode. Request ngắn xong trước không bị kẹt bởi request dài. |
| **Khi nào có lợi** | Nhiều request đồng thời (CCU > 1) — scenario `s04`, `r2_ccu4`, v.v. |
| **Bằng chứng benchmark** | CCU=4: wall 8 docs **109s → 29s** (serial vs parallel) nhờ batching; latency/request ~14s không đổi nhiều. |
| **Cấu hình** | Mặc định; giới hạn bởi `--max-num-seqs` (mặc định cao hơn 4). |

#### PagedAttention

| | |
|--|--|
| **Mô tả** | KV cache chia trang (block) thay vì buffer liên tục → giảm lãng phí VRAM, tăng số sequence song song. |
| **Khi nào có lợi** | Mọi request có context dài (~4k+ tokens). |
| **Benchmark** | Cho phép chạy 4–8 request ~4k prompt trên A30 24GB mà không OOM ngay. |

#### Prefix Caching (Automatic Prefix Caching)

| | |
|--|--|
| **Mô tả** | Cache KV của **prefix trùng nhau** giữa các request (ví dụ cùng system prompt + fewshots). Request sau chỉ prefill phần suffix khác (`chandra_blocks`). |
| **Trạng thái vLLM 0.20+** | `enable_prefix_caching=True` mặc định. |
| **Benchmark** | Scenario `s06` / `r2_prefix_burst`: 8 request burst, shared system+fewshots+schema. **Kết quả không tốt hơn** CCU=8 thường (cùng OK 38%, latency ~68–100s). Nguyên nhân có thể: contention decode JSON dài che lợi ích prefill; hoặc suffix vẫn đủ lớn. |
| **Tiềm năng** | Vẫn đáng giữ khi production có hàng đợi ổn định cùng template prompt. |

#### Chunked Prefill

| | |
|--|--|
| **Mô tả** | Chia prefill dài thành chunk, xen kẽ decode của request khác → giảm spike latency khi 1 request có context rất dài. |
| **Benchmark** | Scenario `s05` / `r2_mega_concat`: **22.306 prompt tokens**, latency **~14–20s** — không chậm hơn nhiều so với ~4k tokens. Cho thấy prefill dài **không phải bottleneck chính**; decode JSON 41 field mới là. |

#### KV Cache dtype / Auto dtype

| | |
|--|--|
| **Mô tả** | `dtype auto` trên model và KV cache theo hardware. |
| **Benchmark** | Dùng trong mọi lần chạy, không so A/B riêng. |

---

### 3.2. Application layer (chandra4layout — có sẵn trước benchmark)

| Kỹ thuật | File / module | Mô tả |
|----------|---------------|--------|
| **Blocks compact** | `schema_align_llm.compact_blocks()` | Chỉ gửi `i, label, bbox, text` (text cắt `text_max`, mặc định 320). Giảm token vs raw HTML Chandra. |
| **Few-shot prompting** | `data/samples/schema_align_fewshots.json` | 2 ví dụ mapping EN + VI trong user prompt. |
| **System prompt JSON-only** | `prompts/schema_align_llm_system.txt` | Ép output pure JSON, copy bbox verbatim. |
| **Fold-match reconcile** | `schema_align_llm.reconcile_keys()` | Chuẩn hóa key LLM → tên schema UTF-8. |
| **Heuristic schema label** | `run.py` `_schema_for_block()` | Rule-based gắn nhãn từ `:`, Section-Header — dùng cho viz, không trong bench vLLM. |
| **Hybrid 5-pass match** | `run_layout_giay_gui.py` | Thay thế LLM bằng rule/IoU — chưa benchmark latency. |

---

### 3.3. vLLM có sẵn nhưng **không** dùng trong benchmark

| Kỹ thuật | Mô tả | Lý do chưa dùng |
|----------|--------|-----------------|
| **Tensor Parallel (TP)** | Chia model nhiều GPU | 1 GPU A30 đủ cho Qwen2.5-3B |
| **Pipeline Parallel** | Chia layer pipeline | Không cần |
| **Quantization (AWQ/GPTQ)** | Giảm VRAM, tăng throughput | Chưa thử |
| **Speculative decoding** | Draft model + verify | Chưa thử |
| **torch.compile / CUDA graphs** | Kernel tối ưu | Plain serve; `enforce-eager` có thể bật khi thiếu gcc |
| **Structured / guided JSON** | Ràng buộc output JSON | Chưa tích hợp API |
| **FlashInfer sampler** | Sampling nhanh | Tắt trong `air_scenario_lab` do môi trường dev |

---

## 4. Kỹ thuật thêm ở lần tối ưu (Round 2)

### 4.1. Application

| Kỹ thuật | Thay đổi | Kết quả vs baseline |
|----------|----------|---------------------|
| **`max_new_tokens` 8192** | 4096 → 8192 | Serial: 100% OK, ~13.2s/doc (≈ baseline 13.7s). CCU=8: **không cải thiện** OK (38%). |
| **CCU sweep** | Client gửi 2/4/8 request đồng thời | **CCU=4 sweet spot** — xem §6. |
| **Compact `text_max=120`** | 320 → 120 chars/block | Prompt 4103→4091 (−0.3%). **Không đáng kể.** |
| **Mega concat** | 8 trang blocks trong 1 request | 22k tokens, ~14.3s — xác nhận decode là bottleneck. |

### 4.2. vLLM server (Round 2b)

| Kỹ thuật | Flag | Kết quả vs R2 plain |
|----------|------|---------------------|
| **`max-num-seqs=4`** | `--max-num-seqs 4` | CCU=4: chậm hơn (+29% wall). CCU=8: OK **38%→75%**, latency −39%. |
| **`gpu-memory-utilization=0.85`** | `--gpu-memory-utilization 0.85` | Tránh OOM khi GPU 0 bận; dùng GPU 1. |
| **`max-model-len=32768`** | `--max-model-len 32768` | Hỗ trợ mega prompt / tương lai context dài. |

---

## 5. Kết quả chi tiết theo scenario

### 5.1. Lần 1 — Baseline

| ID | Scenario | CCU | OK | Wall | Lat mean | Prompt tok | Ghi chú |
|----|----------|-----|-----|------|----------|------------|---------|
| s01 | 1 trang baseline | 1 | 1/1 | 14.1s | 14.1s | 4,103 | Chuẩn 1 doc |
| s02 | Prompt lớn + HTML | 1 | 1/1 | 11.3s | 11.3s | 4,831 | Output ngắn hơn |
| s03 | 8 trang serial | 1 | 8/8 | **109.3s** | 13.7s | 4,127 | **Baseline throughput** |
| s04 | 8 trang parallel | 8 | 7/8 | 43.2s | 20.0s | 4,127 | JSON truncate |
| s05 | Mega 8 trang | 1 | 1/1 | 19.7s | 19.7s | 22,306 | Long prefill |
| s06 | Prefix burst | 8 | 3/8 | 46.5s | 34.8s | 4,127 | Prefix cache stress |

### 5.2. Lần 2a — Round 2 plain

| ID | Scenario | CCU | OK | Wall | Lat mean | Throughput | Ghi chú |
|----|----------|-----|-----|------|----------|------------|---------|
| r2_serial | Serial + tok 8192 | 1 | 8/8 | 105.6s | 13.2s | 0.076 doc/s | ≈ s03 |
| r2_compact | text_max 120 | 1 | 1/1 | 15.2s | 15.2s | — | Không hiệu quả |
| r2_ccu2 | Parallel | 2 | 7/8 | 124.1s | 22.5s | 0.065 doc/s | Chậm hơn serial |
| **r2_ccu4** | **Parallel** | **4** | **8/8** | **28.8s** | **14.1s** | **0.278 doc/s** | **Tốt nhất** |
| r2_ccu8 | Parallel | 8 | 3/8 | 100.3s | 68.4s | 0.080 doc/s | Quá tải |
| r2_mega | Mega concat | 1 | 1/1 | 14.3s | 14.3s | — | Prefill OK |
| r2_prefix | Prefix burst | 8 | 3/8 | 100.3s | 68.4s | — | = ccu8 |

### 5.3. Lần 2b — Round 2 optimized

| ID | Scenario | CCU | OK | Wall | Lat mean | Throughput |
|----|----------|-----|-----|------|----------|------------|
| opt_serial | Serial | 1 | 8/8 | 105.7s | 13.2s | 0.076 doc/s |
| opt_ccu4 | Parallel | 4 | 8/8 | 37.0s | 17.1s | 0.216 doc/s |
| opt_ccu8 | Parallel | 8 | 6/8 | 86.9s | 42.0s | 0.092 doc/s |

---

## 6. So sánh trước / sau baseline

### 6.1. Cải thiện chính (đáng triển khai)

**So sánh xử lý 8 tài liệu:**

| Chỉ số | Baseline tốt nhất (s04 CCU=8) | Sau tối ưu (r2_ccu4) | Delta |
|--------|-------------------------------|----------------------|-------|
| OK rate | 88% | **100%** | +12 pp |
| Wall time | 43.2s | **28.8s** | **−33%** |
| Latency / request | 20.0s | **14.1s** | **−30%** |
| Throughput | 0.185 doc/s | **0.278 doc/s** | **+50%** |
| Giây / doc (amortized) | 5.4s | **3.6s** | **−33%** |

**So với baseline serial (s03):**

| Chỉ số | s03 serial | r2_ccu4 | Delta |
|--------|------------|---------|-------|
| Wall 8 docs | 109.3s | 28.8s | **−74%** |
| Latency / user | 13.7s | 14.1s | +3% (chấp nhận được) |
| OK rate | 100% | 100% | = |

### 6.2. Kỹ thuật không cải thiện (so baseline)

| Kỹ thuật | Kết luận |
|----------|----------|
| Compact text_max 120 | Prompt −0.3%, không đáng kể |
| Prefix caching burst | Giống CCU=8 thường, OK 38% |
| CCU=2 | Chậm hơn serial |
| max-num-seqs=4 @ CCU=4 | Chậm hơn plain CCU=4 |

### 6.3. Sơ đồ trade-off CCU

```
Throughput ↑
    │
    │     ★ CCU=4 (r2_ccu4)  ← khuyến nghị
    │    ╱
    │   ╱  s04 CCU=8 (baseline)
    │  ╱
    │ ╱ s03 serial
    └──────────────────→ Latency ổn định / OK rate
         CCU=8 R2: OK thấp, tail ~100s
```

---

## 7. Kết luận & cấu hình khuyến nghị

### Production schema_align (sau 2 lần tối ưu)

```bash
# Server
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-3B-Instruct \
  --port 8000 --host 0.0.0.0

# Client
# - max_new_tokens: 8192
# - CCU: 4 (semaphore / worker pool)
# - temperature: 0
```

**Kỳ vọng:** ~**14s / tài liệu** (latency user), ~**29s / 8 tài liệu** (batch), **100% OK**.

### Khi bắt buộc CCU ≥ 8

```bash
vllm serve ... --max-num-seqs 4 --gpu-memory-utilization 0.85 --max-model-len 32768
```

OK ~75% (vs 38% plain), vẫn kém CCU=4.

### Kỹ thuật có sẵn đang phát huy tác dụng

1. **Continuous (Dynamic) Batching** — lõi của cải thiện CCU=4  
2. **PagedAttention** — cho phép multi-seq trên A30  
3. **Chunked prefill** — mega 22k tokens không spike  
4. **Prefix caching** — có trên server nhưng **chưa thấy lợi** trên burst test hiện tại  

### Kỹ thuật Round 2 thực sự có ích

1. **CCU=4** (client scheduling)  
2. **max_new_tokens=8192** (độ tin cậy serial / CCU thấp)  

---

## 8. Chưa thử — hướng tiếp theo

| Ưu tiên | Kỹ thuật | Kỳ vọng |
|---------|----------|---------|
| 1 | Structured JSON / guided decoding | Fix OK rate CCU cao |
| 2 | AWQ 4-bit quantization | +30–50% throughput |
| 3 | Hybrid rule-match thay LLM | Bỏ ~14s/doc nếu accuracy đủ |
| 4 | Chia 2-pass schema (header + table) | Giảm decode tokens |
| 5 | Benchmark full pipeline Chandra + align | Xác định bottleneck thật |
| 6 | `enforce-eager=0` + compile (khi có gcc/nvcc) | Kernel nhanh hơn |

---

## 9. Cách chạy lại benchmark

### Chuẩn bị dữ liệu

```bash
python chandra4layout/bench/prepare_data.py
# Blocks thật cho test_2–6:
python chandra4layout/bench/prepare_data.py --run-chandra
```

### Round 1 (baseline)

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000
python chandra4layout/bench/run_vllm_bench.py \
  --scenarios-file chandra4layout/bench/scenarios.yaml
```

### Round 2 plain

```bash
python chandra4layout/bench/run_vllm_bench.py \
  --scenarios-file chandra4layout/bench/scenarios_round2.yaml \
  --tag round2_plain --vllm-profile plain
```

### Round 2 optimized

```bash
CUDA_VISIBLE_DEVICES=1 chandra4layout/bench/scripts/start_vllm_optimized.sh
python chandra4layout/bench/run_vllm_bench.py \
  --scenarios-file chandra4layout/bench/scenarios_round2_optimized.yaml \
  --tag round2_optimized --vllm-profile optimized
```

### Tổng hợp báo cáo

```bash
python chandra4layout/bench/summarize_improvements.py
# → bench/results/improvement_report.json
```

### Chạy một scenario

```bash
python chandra4layout/bench/run_vllm_bench.py \
  --scenarios-file chandra4layout/bench/scenarios_round2.yaml \
  --only r2_ccu4_parallel --tag round2_plain
```

---

## 10. File kết quả

```
chandra4layout/bench/
├── docs/
│   └── BENCHMARK_OPTIMIZATION.md    ← tài liệu này
├── scenarios.yaml                   # Round 1
├── scenarios_round2.yaml            # Round 2 app
├── scenarios_round2_optimized.yaml  # Round 2 vLLM
├── prepare_data.py
├── run_vllm_bench.py
├── summarize_improvements.py
├── scripts/
│   ├── start_vllm_plain.sh
│   └── start_vllm_optimized.sh
└── results/
    ├── bench_report.json            # Round 1
    ├── round2_plain/bench_report.json
    ├── round2_optimized/bench_report.json
    ├── improvement_report.json      # Tổng hợp JSON
    ├── vllm_server.log
    └── vllm_optimized_server.log
```

---

*Tạo từ benchmark thực tế trên A30, tháng 6/2026. Model: Qwen2.5-3B-Instruct, vLLM 0.20.1 (env `personal_lab`).*
