# LFM2-2.6B swap — Report tổng hợp (2026-07-16)

Mục tiêu: thay Qwen3.5-2B bằng **LiquidAI/LFM2-2.6B** trên workload phase-1, đo ERS thật,
tối ưu tối đa, và audit độ tin cậy của hybrid cache.

Harness: `config_yml_aira/0705-0940/replay_trace_phase1.py` (real-ERS, γ=2, w=0.5,
TTFT floor/ceiling 100/1500ms, TPOT 20/45ms) trên trace-round1 (120 req / ~25s,
system prompt chung ~6.5k tok + user ~6.6k tok, max_tokens=200, temp=0).
Rig: H200 dùng chung (nhiễu), image `dngback/vllm-v0.24.0-ubuntu2404-opt-2:r12-prefillbias-v2`
(vLLM 0.24.0) — **không sửa image, 100% cấu hình runtime**.
Raw số liệu: `results/phase1_replay_summary.*.json`.

## 1. Bản chốt (WINNER)

File: [`../docker-compose-lfm2-2.6b-fullstack.yml`](../docker-compose-lfm2-2.6b-fullstack.yml)

**ERS primed ≈ 0.99 (0.988 / 0.990 / 0.996, 3 reps), TTFT p50 ~90–110ms / p95 ~130–170ms,
TBT p95 ~18–20ms, 120/120 pass SLO** — ngang Qwen3.5-2B shipped (~0.9949).

Thành phần:
- **bf16 backbone** (KHÔNG `--quantization=fp8` — xem §3)
- `--kv-cache-dtype=fp8_e4m3` (FA3 đọc fp8 native, không dequant)
- `--mamba-cache-dtype=bfloat16` (conv-state của LFM2; model KHÔNG có SSM state)
- `--enable-prefix-caching` + `AIR_PRIME_PACK=1` với `/app/warmup_prime_pack_full.jsonl`
  (prime 120/120 prompt → hit ~100%; pack gửi TEXT nên tự tokenize lại theo LFM2)
- `--async-scheduling`, `--enable-chunked-prefill`, `--max-num-batched-tokens=3072`,
  `--no-disable-cascade-attn`, đầy đủ `--prefill-bias-*`
- env: `VLLM_USE_RUST_FRONTEND=1`, `ENGPACE_RATE_MS=12`/`BURST_N=3` (KHÔNG nới — xem §3),
  `AIR_WARMUP=1`, `AIR_GC_FREEZE=1`, `AIR_LMHEAD_FP8=1`
- Cờ Qwen bị BỎ vì sai kiến trúc: `--language-model-only`, `--mamba-ssm-cache-dtype`,
  `VLLM_GDN_DIRECT_SCAN_OUTPUT`

Khi nộp bài trên 1 MIG riêng: đổi `--gpu-memory-utilization` 0.34 → **0.95**.
Weights: volume `lfm2_model` (daemon remote → stage bằng helper + `docker cp`, xem chú thích trong compose).

## 2. Bảng tổng kết mọi lần đo (ERS primed, trừ khi ghi khác)

| Cấu hình | ERS | TBT p95 | Kết luận |
|---|---|---|---|
| Qwen3.5-2B clean, cold | 0.661 | — | tham chiếu |
| LFM2 clean, cold | 0.642 | 39.9 | tham chiếu |
| Qwen3.5-2B clean, primed | 0.988 | 11.1 | tham chiếu |
| LFM2 clean, primed | 0.958 | 13.2 | thiếu full-stack |
| LFM2 full-stack all-fp8 | 0.986 / 0.996 | ~20 | ngang bf16 |
| **LFM2 full-stack bf16 (WINNER)** | **0.988 / 0.990 / 0.996** | 18.4–20.4 | **chốt** |
| LFM2 ENGPACE nới (4/12) | 0.876 / 0.972 / 0.996 | 25–34 | ❌ thundering-herd |
| LFM2 `--enforce-eager` | 0.853 / 0.862 | 26–29 | chẩn đoán: cudagraph tiết kiệm ~7ms/tok |
| LFM2 n-gram spec-decode | 0.916 / 0.946 / 0.939 | 33–34 | ❌ acceptance thấp + mất async |
| LFM2 FLASHINFER (bị ignore → FA3) | 0.988 | 20.8 | backend không phải đòn bẩy |

## 3. Vì sao chốt như vậy (các phát hiện chính)

1. **Primed = decode/scheduling-bound.** Hit ~100% → mỗi request chỉ prefill ~1 token.
   Mô hình băng thông: KV fp8 ~2.7ms + weights ~1.1ms/step @ batch 120, nhưng TBT đo ~15ms
   → phần lớn là launch-overhead, cudagraph `decode, FULL` đã xử lý tối đa
   (bằng chứng: enforce-eager TBT 22ms vs 15ms).
2. **fp8 weight-quant vô ích khi primed** (không compute-bound), còn nhích TBT — bỏ.
   Chỉ cân nhắc bật lại nếu BTC cấm prime (regime cold = prefill-bound).
3. **ENGPACE 12/3 là thiết yếu** — nới ra gây thundering-herd, ERS sập còn 0.88.
4. **Hopper đã kịch trần**: FA3 fp8 native cho 8 attn layer, fp8-KV, lm_head fp8 w8a8,
   cudagraph FULL cho decode. FlashInfer không chọn được với model hybrid (vLLM ép FA3).
5. **Spec-decode n-gram thất bại** (output hội thoại → acceptance thấp). Hướng còn lại duy nhất
   nếu muốn phá trần: LFM2-1.2B làm draft model (cùng họ) — chưa thử.
6. **ERS ~0.99 primed là trần thực tế** của LFM2-2.6B trên engine/workload này.

## 4. Audit hybrid cache (PASSED — không có bug)

Cơ chế: 2 nhóm cache — `FullAttentionManager` (8 attn layer, KV từng token, paged fp8) +
`MambaManager` (22 conv layer, **snapshot state tại ranh giới block**, bf16, mode `align`,
hit chỉ cần 1 block cuối khớp; hit cap tại num_prompt−1).

Kiểm chứng byte-compare (temp=0):
- Cùng server: cold → full-hit → hit lặp: **giống hệt từng byte** → restore bitwise-faithful.
- Partial-hit (khôi phục conv-state GIỮA prompt, prefill tiếp phần user): ~80 token đầu
  giống hệt cold rồi mới lệch tại 1 near-tie → state đúng (nếu sai phải lệch từ token 1).
- Mỗi đường 100% deterministic khi lặp.
- Text có thể khác giữa cold/partial/full-hit: **non-batch-invariance chuẩn của vLLM**
  (ranh giới chunk + GEMM shape → rounding khác → lật argmax near-tie; fp8-KV khuếch đại nhẹ).
  Không phải lỗi hybrid. Bài chấm primed = full-hit = deterministic.
- Tối ưu tầng cache đã soi và loại: retention_interval (chỉ tiết kiệm RAM, đang dư 200×),
  block-size lớn hơn (gain ≈0), batch-invariant kernels (phản latency).

## 5. Rủi ro còn treo (QUAN TRỌNG, chưa kiểm tra được)

**Cổng accuracy/delta**: FINAL = 100 × ERS × penalty(delta); delta ≥ 0.16 → 0 điểm.
LFM2 là model KHÁC Qwen → nếu delta so output với reference của Qwen thì đổi model
gần như chắc chắn zero điểm bất kể ERS. **Phải xác nhận thể lệ trước khi nộp LFM2.**
Chừng nào chưa rõ, bản nộp an toàn vẫn là Qwen r12 (`docker-compose-r12-prefillbias.yml`).

## 6. Trạng thái vận hành

- Container `lfm2_vllm` (:8137, GPU1) đang chạy đúng WINNER config.
- Image gốc từ Hub, không commit/rebuild gì.
- Weights LFM2-2.6B: volume `lfm2_model` + HF cache `/home/jovyan/scratch/cache/huggingface`.
