# Report — Tối ưu LFM2.5-1.2B-Instruct dưới luật MỚI (warmup hợp lệ) — 2026-07-16

## 0. TL;DR
- **KHÔNG cần build/đẩy image mới.** Mọi tối ưu là flag/env runtime trên image gốc
  `dngback/vllm-v0.24.0-ubuntu2404-opt-2:r12-prefillbias-v2` (đã có trên Hub, không commit).
- Cần triển khai: image gốc + volume `lfm25_model` (weights) + file compose. Hết.
- Cấu hình chốt: [`docker-compose-lfm25-1.2b-legal.yml`](../docker-compose-lfm25-1.2b-legal.yml).
- Kết quả: legal baseline **0.038 → 0.11-0.12** (~×3) dưới thông số chấm THẬT.

## 1. Hai thay đổi luật chơi (ảnh hưởng mọi con số cũ)
1. **Thông số chấm SIẾT:** F_ttft=10ms C_ttft=400ms F_tpot=1ms C_tpot=10ms
   (harness `replay_trace_phase1.py` vẫn in số theo tham số CŨ 100/1500/20/45 → SAI;
   phải rescore bằng [`rescore_real_params.py`](rescore_real_params.py)).
2. **Prime bằng đúng prompt bài thi = PHẠM LUẬT.** Chỉ được warm-up "dữ liệu tương
   đương, vừa đủ". → Mọi số "~0.99 / ~0.31 primed" trước đây vô hiệu.

## 2. Warm-up hợp lệ (đã dựng, đã verify)
- `AIR_WARMUP=1` — generic filler (JIT/cudagraph/fp8 autotune). Hợp lệ tuyệt đối.
- `AIR_PRIME_CACHE=1` — cache **system-prefix cố định** (system-prompt caching chuẩn, "vừa đủ").
- `AIR_PRIME_PACK=0` — TẮT replay 120 prompt test (phạm luật).
- Log xác nhận: `generic done` + `prime shared system prefix (38956 chars)` + `storm`, KHÔNG pack.

## 3. Đã tối ưu những gì (cold r1 hợp lệ, real params)

| Bước | Config | ERS_real | ttft50 | tpot50 |
|---|---|---|---|---|
| baseline | cascade ON, bf16 | 0.038 | 396 | 13.5 |
| +1 | **cascade OFF** | 0.069 | 346 | 9.0 |
| +2 | **fp8 weights + cascade OFF** ⭐ | **0.108 / 0.122** | ~250 | ~8.1 |

### Lever 1 — TẮT cascade attention (đòn bẩy KV-read/TPOT)
Đây chính là tối ưu "cách lưu/đọc KV" bạn yêu cầu, nhưng kết quả NGƯỢC trực giác:
- vLLM tự bật cascade (đọc shared-prefix 1 lần cho cả wave). Heuristic của nó *dự đoán*
  cascade thắng (tôi tính tay khớp: cascade_time 51 < flash_decoding 62).
- **Nhưng đo thực: two-kernel cascade CHẬM hơn** FlashDecoding cho pattern hybrid này.
  TBT median 13.5ms (on) → 8-9ms (off). Probe decode sạch (6.5k shared + 6.5k unique,
  20 concurrent) xác nhận cascade-off ~6.7ms.
- Exact, KHÔNG mất accuracy. Bật bằng `--disable-cascade-attn`.
- → **Kết quả negative đáng viết paper**: "cascade attention regress trên hybrid
  conv-attention (LFM2) — heuristic use_cascade của vLLM mis-predict".

### Lever 2 — fp8 weight-quant (đòn bẩy TTFT)
- Ở regime legal/cold, prefill 6.5k token user RIÊNG là **compute-bound** → fp8 GEMM
  (H200 tensor core) tăng gấp đôi throughput → TTFT 396→266ms.
- **Ngược hẳn regime primed** (decode-bound, fp8 vô ích) → regime quyết định fp8.
- Bật `--quantization=fp8` (giữ fp8-KV + AIR_LMHEAD_FP8).

## 4. Bức tranh & nút thắt hiện tại
- **TPOT coi như đã giải quyết**: ~8ms < ceiling 10ms (cascade-off + fp8-KV).
- **Nút thắt còn lại = TTFT** (prefill 6.5k token user riêng × 20 request/wave, blows
  ceiling 400ms). Đây KHÔNG phải vấn đề KV-read — phần user riêng không có prefix chung
  để khai thác. Giảm thêm chỉ bằng tăng tốc prefill (prefill-tile tuning, thêm fp8),
  không phải mẹo KV.
- Trần thực tế của LFM2-class trên workload 13k-context + ceiling siết ~0.1-0.2 hợp lệ.

## 5. Có cần đẩy image mới không? — KHÔNG
- Image gốc `r12-prefillbias-v2` (Hub) đủ chạy mọi thứ. Không commit/rebuild/push.
- Prime-pack file `/app/warmup_prime_pack_full.jsonl` nằm sẵn trong image nhưng ta
  KHÔNG dùng (AIR_PRIME_PACK=0). system-prompt để prime lấy từ image (warmup_system_prompt.txt).
- Triển khai = image gốc + `docker volume lfm25_model` + compose. Đổi util 0.12 → 0.9 khi nộp.

## 6. Rủi ro chưa kiểm (score-zeroing)
- **Cổng accuracy/delta**: LFM2.5 instruct + fp8×3 (weight+kv+lmhead). Chưa đo accuracy.
  Đây là rủi ro nhân-0 LỚN NHẤT còn lại. Phải kiểm trước khi tin con số ERS.
- `AIR_PRIME_CACHE` (cache system-prefix) — nếu ban tổ chức coi cả system prompt là "test
  data" thì phải tắt luôn; khi đó wave-1 chịu thêm prefill system → TTFT xấu hơn chút.

## 7. Việc tiếp theo (nếu muốn đẩy nữa)
- **TTFT**: prefill-tile sweep (BLOCK_M/BLOCK_N/num_stages của causal_conv1d) + tune
  max-num-batched-tokens/chunked-prefill → giảm TTFT (nút thắt chính).
- **Accuracy**: đo delta của LFM2.5 + fp8 so với reference trước khi chốt.
- **Paper**: viết up kết quả cascade-regress-on-hybrid (có số đo A/B sạch).
