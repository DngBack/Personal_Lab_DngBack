# LayoutAgent

LangGraph agent tối ưu prompt tự động cho pipeline trích xuất layout tài liệu ngân hàng tiếng Việt.

## Kiến trúc

```
PDF / Image
    │
    ▼
Chandra OCR-2 ──────────────────────────────► HTML layout (<div data-bbox data-label>)
(local HuggingFace hoặc API)                         │
                                                      ▼
                                           Qwen 3 VL 4B ──────► HTML schema (<div data-schema>)
                                           (local HuggingFace)         │
                                                                        ▼
                                                             merge_schemas (dedup)
                                                                        │
                                                                        ▼
                                                                   visualize
                                                                        │
                                                                        ▼
                                                          GPT-4o evaluate (coverage + visual judge)
                                                                        │
                                                            ┌───────────┴──────────┐
                                                         score OK              score low
                                                            │                      │
                                                           END          GPT-4o optimize_prompt
                                                                                   │
                                                                          loop → align_schema
```

**3 model roles:**

| Role | Model | Chạy ở đâu |
|---|---|---|
| Layout extraction | `datalab-to/chandra-ocr-2` | Local HuggingFace hoặc API |
| Schema alignment | `Qwen/Qwen3-VL-4B-Instruct` | Local HuggingFace hoặc API |
| Evaluate + Optimize | `gpt-4o` | OpenAI API (bắt buộc) |

## Cài đặt

```bash
cd LayoutAgent
pip install -r requirements.txt
```

**Yêu cầu thêm cho local mode:**
```bash
pip install torch transformers accelerate pymupdf
```

## Cấu hình `.env`

```bash
cp .env.sample .env
# Mở .env và điền OPENAI_API_KEY (bắt buộc cho evaluate + optimize)
```

Nội dung tối thiểu cho local mode:
```env
OPENAI_API_KEY=sk-...
```

## Cấu trúc thư mục

```
LayoutAgent/
├── main.py                  # CLI: full tuning loop (LangGraph)
├── run_inference.py         # CLI: single-shot inference (không tune)
├── requirements.txt
├── .env.sample
│
├── data/
│   ├── prompts/             # Prompt chung (dùng cho mọi doc type)
│   │   ├── chandra_general_template.txt   # Chandra prompt, có {schema_fields}
│   │   └── schema_align_general.txt       # Qwen alignment prompt
│   ├── samples/             # Dữ liệu mẫu có ground truth
│   │   └── GIAY_GUI_TIEN_TIET_KIEM/
│   │       ├── layout _GIAY_GUI_TIEN_TIET_KIEM.json   # Schema definition
│   │       ├── 20251018+MYNTT1_0001-p18.pdf            # Sample PDF
│   │       └── 20251018+MYNTT1_0001-p18_page01_schema_boxes.jpg  # Reference image
│   ├── test/                # File test để infer
│   └── tuning_runs/         # Output của mỗi lần tune (auto-generated)
│
└── src/
    ├── state.py             # AgentState TypedDict (LangGraph)
    ├── graph.py             # Wiring các node thành StateGraph
    ├── clients/
    │   ├── local_chandra.py     # Chandra via HuggingFace Transformers
    │   ├── local_qwen_vl.py     # Qwen VL via HuggingFace Transformers
    │   ├── chandra_api.py       # Chandra via OpenAI-compatible API
    │   └── openai_vision.py     # GPT-4o via OpenAI API
    ├── nodes/               # LangGraph nodes
    │   ├── setup.py         # Load schema, khởi tạo run dir
    │   ├── run_chandra.py   # Chạy Chandra OCR (có cache + template injection)
    │   ├── align_schema.py  # Qwen VL: map layout → schema fields
    │   ├── merge_schemas.py # Dedup duplicate schema divs
    │   ├── visualize.py     # Vẽ bbox lên ảnh, lưu JPEG
    │   ├── evaluate.py      # Coverage + GPT-4o visual judge
    │   ├── checkpoint.py    # Lưu artifacts mỗi iter
    │   └── optimize_prompt.py   # GPT-4o viết lại prompt
    └── utils/
        ├── schema_html.py   # Adapter sang layoutDectectionChan utils
        ├── image_io.py      # PDF rasterize, base64 encode, draw bbox
        └── scoring.py       # Coverage, composite score
```

---

## Sử dụng

### 1. Tuning prompt từ đầu (có ground truth)

```bash
cd LayoutAgent

python main.py \
  --pdf data/samples/GIAY_GUI_TIEN_TIET_KIEM/20251018+MYNTT1_0001-p18.pdf \
  --layout-json "data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout _GIAY_GUI_TIEN_TIET_KIEM.json" \
  --chandra-prompt data/prompts/chandra_general_template.txt \
  --initial-prompt data/prompts/schema_align_general.txt \
  --reference-image "data/samples/GIAY_GUI_TIEN_TIET_KIEM/20251018+MYNTT1_0001-p18_page01_schema_boxes.jpg" \
  --local \
  --qwen-device cuda:0 \
  --max-iterations 3 \
  --auto-resume
```

> **Kết quả:** `data/tuning_runs/<run_id>/best_prompt.txt`

### 2. Tuning nhanh hơn (reuse Chandra cache)

Nếu đã có Chandra log từ lần chạy trước:

```bash
python main.py \
  --pdf data/samples/GIAY_GUI_TIEN_TIET_KIEM/20251018+MYNTT1_0001-p18.pdf \
  --layout-json "data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout _GIAY_GUI_TIEN_TIET_KIEM.json" \
  --chandra-log data/samples/GIAY_GUI_TIEN_TIET_KIEM/20251018+MYNTT1_0001-p18_llm.log \
  --initial-prompt data/prompts/schema_align_general.txt \
  --reference-image "data/samples/GIAY_GUI_TIEN_TIET_KIEM/20251018+MYNTT1_0001-p18_page01_schema_boxes.jpg" \
  --local \
  --qwen-device cuda:0 \
  --max-iterations 3 \
  --auto-resume
```

### 3. Inference trên file test

```bash
python run_inference.py \
  --pdf data/test/GIAY_GUI_TIEN_TIET_KIEM/test_7.pdf \
  --layout-json "data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout _GIAY_GUI_TIEN_TIET_KIEM.json" \
  --chandra-log data/test/GIAY_GUI_TIET_KIEM/test_7_llm.log \
  --align-prompt data/tuning_runs/<run_id>/best_prompt.txt \
  --out-dir data/test/GIAY_GUI_TIEN_TIET_KIEM/out_test7 \
  --local \
  --qwen-device cuda:0
```

> Nếu chưa có Chandra log cho file test, thay `--chandra-log` bằng:
> ```
> --chandra-prompt data/prompts/chandra_general_template.txt
> ```

### 4. So sánh 2 prompt

```bash
python run_inference.py \
  --pdf data/test/GIAY_GUI_TIEN_TIET_KIEM/test_7.pdf \
  --layout-json "data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout _GIAY_GUI_TIEN_TIET_KIEM.json" \
  --chandra-log data/test/GIAY_GUI_TIET_KIEM/test_7_llm.log \
  --align-prompt data/tuning_runs/<run_id>/best_prompt.txt \
  --align-prompt-b data/prompts/schema_align_general.txt \
  --out-dir data/test/GIAY_GUI_TIEN_TIET_KIEM/out_compare \
  --local \
  --qwen-device cuda:0
```

---

## Thêm document type mới

1. Tạo thư mục sample:
   ```
   data/samples/<DOC_TYPE>/
       layout_<DOC_TYPE>.json      ← schema definition
       sample.pdf                   ← một trang mẫu
       sample_schema_boxes.jpg      ← ảnh reference có bbox vẽ sẵn (ground truth)
   ```

2. Chạy tuning — pipeline tự inject schema fields vào prompt:
   ```bash
   python main.py \
     --pdf data/samples/<DOC_TYPE>/sample.pdf \
     --layout-json data/samples/<DOC_TYPE>/layout_<DOC_TYPE>.json \
     --chandra-prompt data/prompts/chandra_general_template.txt \
     --initial-prompt data/prompts/schema_align_general.txt \
     --reference-image data/samples/<DOC_TYPE>/sample_schema_boxes.jpg \
     --local --qwen-device cuda:0 --max-iterations 3 --auto-resume
   ```

---

## Tham số CLI chính

### `main.py`

| Tham số | Mô tả | Default |
|---|---|---|
| `--pdf` | File PDF đầu vào | bắt buộc |
| `--layout-json` | JSON định nghĩa schema fields | bắt buộc |
| `--initial-prompt` | Prompt ban đầu cho Qwen VL | bắt buộc |
| `--chandra-prompt` | Prompt Chandra (có `{schema_fields}`) | một trong hai |
| `--chandra-log` | Reuse Chandra output có sẵn | một trong hai |
| `--reference-image` | Ảnh ground truth để judge | khuyến nghị |
| `--local` | Dùng HuggingFace Transformers local | `False` |
| `--qwen-device` | GPU cho Qwen VL | `cuda:0` |
| `--chandra-device` | GPU cho Chandra | `cuda:0` |
| `--max-iterations` | Số iter tối đa | `3` |
| `--stop-threshold` | Composite score để dừng sớm | `0.85` |
| `--coverage-threshold` | Coverage tối thiểu để dừng sớm | `0.95` |
| `--auto-resume` | Chạy tự động không hỏi | `False` |

### `run_inference.py`

| Tham số | Mô tả |
|---|---|
| `--pdf` | File PDF test |
| `--layout-json` | JSON schema |
| `--align-prompt` | Prompt A (bắt buộc) |
| `--align-prompt-b` | Prompt B để so sánh (tuỳ chọn) |
| `--chandra-log` | Reuse Chandra log |
| `--chandra-prompt` | Chạy Chandra on-the-fly |
| `--out-dir` | Thư mục output |
| `--local` | Local HuggingFace mode |
| `--qwen-device` | GPU cho Qwen VL |

---

## Stop condition

Agent dừng sớm khi **đồng thời** đạt:
- `composite_score >= stop_threshold` (default 0.85)
- `coverage >= coverage_threshold` (default 0.95)

Nếu chỉ đạt composite nhưng coverage thấp (còn field bị miss), agent tiếp tục tune.
