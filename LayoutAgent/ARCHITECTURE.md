# LayoutAgent — Kiến trúc & Flow chi tiết

## Tổng quan

LayoutAgent là một LangGraph agent tự động tối ưu prompt cho pipeline trích xuất layout tài liệu ngân hàng tiếng Việt. Agent hoạt động theo vòng lặp đánh giá → tối ưu, sử dụng 3 model với vai trò riêng biệt.

---

## 1. Ba model roles

```
┌─────────────────────────────────────────────────────────────────┐
│                        MODEL ROLES                              │
├──────────────────┬─────────────────────────┬───────────────────┤
│ Role             │ Model                   │ Chạy ở đâu        │
├──────────────────┼─────────────────────────┼───────────────────┤
│ Layout extract   │ datalab-to/chandra-ocr-2│ Local (HF) / API  │
│ Schema align     │ Qwen/Qwen3-VL-4B-Instruct│ Local (HF) / API │
│ Evaluate+Optimize│ gpt-4o                  │ OpenAI API        │
└──────────────────┴─────────────────────────┴───────────────────┘
```

**Tại sao tách 3 model?**
- **Chandra OCR-2**: Chuyên biệt cho layout detection — trả về HTML với bounding box chính xác
- **Qwen 3 VL 4B**: Multimodal VLM hiểu cả ảnh lẫn HTML — phù hợp schema alignment
- **GPT-4o**: Mạnh về reasoning — dùng để judge chất lượng và viết lại prompt

---

## 2. LangGraph StateGraph

### 2.1 Sơ đồ node

```
                    ┌─────────────────────────────────────┐
                    │              INIT ONCE               │
                    │                                     │
         START ────►│  setup ──► run_chandra              │
                    │               │                     │
                    └───────────────┼─────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────┐
                    │         TUNING LOOP (per iter)       │
                    │                                     │
                    │  align_schema                       │
                    │       │                             │
                    │  merge_schemas                      │
                    │       │                             │
                    │  visualize                          │
                    │       │                             │
                    │  evaluate                           │
                    │       │                             │
                    │  checkpoint ──┬────────────────────►│ END
                    │               │   score OK AND       │
                    │               │   coverage OK        │
                    │               │                     │
                    │               ▼  score NOT OK        │
                    │       [INTERRUPT] ◄── human review   │
                    │               │                     │
                    │       optimize_prompt               │
                    │               │                     │
                    │               └──► align_schema ◄───┘
                    │                    (next iter)      │
                    └─────────────────────────────────────┘
```

### 2.2 Phân loại node

| Node | Chạy | Dùng model | Mô tả ngắn |
|---|---|---|---|
| `setup` | 1 lần | — | Load schema, khởi tạo run |
| `run_chandra` | 1 lần | Chandra OCR-2 | Extract layout HTML |
| `align_schema` | mỗi iter | Qwen 3 VL 4B | Map layout → schema fields |
| `merge_schemas` | mỗi iter | — | Dedup duplicate divs (deterministic) |
| `visualize` | mỗi iter | — | Vẽ bbox lên ảnh |
| `evaluate` | mỗi iter | GPT-4o | Coverage + visual judge |
| `checkpoint` | mỗi iter | — | Lưu artifacts + cập nhật best |
| `optimize_prompt` | mỗi iter | GPT-4o | Viết lại prompt từ feedback |

---

## 3. Flow chi tiết từng node

### Node 1: `setup`

**Input:** `layout_json_path`, `initial_prompt_path`

**Xử lý:**
1. Đọc `layout_json` → duyệt DFS → lấy danh sách schema field names (có thứ tự)
2. Tạo thư mục `data/tuning_runs/<run_id>/`
3. Đọc initial prompt text vào `current_prompt`
4. Khởi tạo: `iteration=0`, `best_score=-1`, `eval_history=[]`

**Output state keys:** `schema_fields`, `run_id`, `output_dir`, `current_prompt`, `iteration`

---

### Node 2: `run_chandra`

**Input:** `chandra_prompt_path` hoặc `chandra_html` (pre-loaded), `page_image_path`

**Xử lý:**
```
chandra_html có sẵn trong state?
    ├── YES → skip, reuse (cache hit)
    └── NO  →
          Template injection: {schema_fields} trong prompt → inject danh sách field
          ├── use_local_models=True → LocalChandraClient (HuggingFace Transformers)
          └── use_local_models=False → ChandraApiClient (OpenAI-compatible endpoint)
          → Rasterize PDF page → Image → Chandra model → HTML
          → Lưu cache log (_llm.log)
```

**Output HTML mẫu:**
```html
<div data-bbox="23 42 340 68" data-label="Tên khách hàng"><p>NGUYEN VAN A</p></div>
<div data-bbox="350 42 680 68" data-label="CIF"><p>12345678</p></div>
```

**Lưu ý:** Bbox là tọa độ nguyên [0, 1000] tương đối theo chiều rộng/cao trang.

---

### Node 3: `align_schema`

**Input:** `chandra_html`, `current_prompt`, `schema_fields`, `page_image_path`

**Xử lý:**
```
use_local_models=True?
    ├── YES → LocalQwenVLClient
    │         └── QwenVLSchemaHtmlMerger (từ layoutDectectionChan)
    │             └── Qwen3VLForConditionalGeneration.generate()
    └── NO  → chat_vision() via OpenAI-compatible API
              └── QWEN_BASE_URL nếu có, hoặc OpenAI fallback

Input cho model:
    - system_prompt = current_prompt
    - user message = schema_fields JSON + chandra_html
    - images = [page_image] (+ reference_image nếu có)
```

**Output HTML mẫu:**
```html
<div data-bbox="23 42 340 68" data-label="Tên khách hàng" data-schema="Tên khách hàng">
  <p>NGUYEN VAN A</p>
</div>
<div data-bbox="628 716 888 866" data-label="Kiểm soát viên" data-schema="Kiểm soát viên">
  <p>Kiểm soát viên</p><br/><p>Võ Thị Phương Thủy</p>
</div>
```

**Điểm khác với Chandra output:** thêm `data-schema` attribute khớp với schema field name.

---

### Node 4: `merge_schemas`

**Input:** `merged_html` từ align_schema

**Xử lý (deterministic, không dùng LLM):**
```
Parse HTML → group by data-schema value
Với mỗi schema field có N div:
    → union bbox: x0=min(x0ᵢ), y0=min(y0ᵢ), x1=max(x1ᵢ), y1=max(y1ᵢ)
    → nối inner content bằng <br/>
    → thay N div bằng 1 div duy nhất
```

**Ví dụ merge:**
```
TRƯỚC (Qwen output):
  <div data-schema="Kiểm soát viên" data-bbox="628 716 733 729">label</div>
  <div data-schema="Kiểm soát viên" data-bbox="573 729 888 844">signature</div>
  <div data-schema="Kiểm soát viên" data-bbox="667 840 851 866">name</div>

SAU (merged):
  <div data-schema="Kiểm soát viên" data-bbox="573 716 888 866">label<br/>signature<br/>name</div>
```

---

### Node 5: `visualize`

**Input:** `merged_html`, `page_image_path`

**Xử lý:**
1. Rasterize PDF page → PIL Image
2. Parse `data-bbox` và `data-schema` từ mỗi div
3. Vẽ colored rectangle lên ảnh (mỗi schema field một màu ngẫu nhiên)
4. Annotate label text
5. Lưu JPEG → `iter_XX/schema_boxes.jpg`

---

### Node 6: `evaluate`

**Input:** `merged_html`, `schema_fields`, `viz_source_path`, `reference_image_path`

**Xử lý 2 bước:**

**Bước A — Coverage (không dùng LLM):**
```python
extracted = {div.data-schema for div in merged_html if data-schema != ""}
missing = [f for f in schema_fields if f not in extracted]
coverage = len(schema_fields - missing) / len(schema_fields)
```

**Bước B — Visual Judge (GPT-4o):**
```
Input: [reference_image, output_image] + missing fields hint
System prompt: strict scoring rubric
    - Start at 1.0
    - -0.05 per missing field
    - -0.03 per misaligned field
    - -0.02 per extra field
    - -0.05 nếu table sub-fields bị gom vào 1 box
Output JSON:
{
  "score": 0.85,
  "missing_fields_visual": [...],
  "misaligned_fields": [...],
  "extra_fields": [...],
  "feedback": "..."
}
```

**Composite score:**
```
composite = 0.5 × coverage + 0.5 × llm_judge_score
```

Hai metric được cân bằng bằng nhau — một metric cao không thể che khuất metric kia.

---

### Node 7: `checkpoint`

**Input:** toàn bộ state sau evaluate

**Xử lý:**
```
Tạo iter_XX/:
    ├── prompt.txt           ← current_prompt
    ├── schema_merged.html   ← merged_html
    ├── eval.json            ← coverage, judge, composite, missing, feedback
    ├── layout_values.json   ← {field_name: inner_text, ...}
    └── schema_boxes.jpg     ← visualization (copy từ visualize)

Cập nhật run-level:
    ├── summary.json         ← best_score, best_iteration, eval_history
    └── best_prompt.txt      ← prompt của iter có composite score cao nhất
    └── best_schema_merged.html
```

---

### Node 8: `optimize_prompt` ← INTERRUPT BEFORE

**Khi có human interrupt:** LangGraph dừng ở đây. Operator xem kết quả rồi nhấn `y/n` để tiếp tục.

**Khi auto-resume:** chạy thẳng.

**Input:** `current_prompt`, `composite_score`, `coverage`, `missing_fields`, `eval_feedback`, `viz_source_path`, `reference_image_path`

**Xử lý (GPT-4o):**
```
System: "You are an expert prompt engineer..."
User message:
    - Evaluation metrics (composite, coverage, judge, missing, duplicates)
    - Score history qua các iter trước
    - Visual feedback từ judge
    - Danh sách schema fields đầy đủ
    - Current prompt (wrapped in ```)
    - Hai ảnh: [reference, output_current_iter]

Output JSON:
{
  "improved_prompt": "...",
  "changes_summary": "..."
}
```

**Output:** `current_prompt` = `improved_prompt`, `iteration` += 1

---

### Routing: `should_continue_router`

Chạy sau `checkpoint`, quyết định tiếp hay dừng:

```python
# Dừng sớm: phải đạt CẢ HAI điều kiện
if composite >= stop_threshold AND coverage >= coverage_threshold:
    return END

# Dừng do hết iter
if iteration >= max_iterations:
    return END

# Tiếp tục
return "optimize_prompt"
```

**Default thresholds:**
- `stop_threshold = 0.85` (composite)
- `coverage_threshold = 0.95` (field coverage)

---

## 4. AgentState — luồng dữ liệu

State là một `TypedDict` được LangGraph duy trì qua toàn bộ graph. Mỗi node nhận state đầy đủ và trả về **partial update** (chỉ các key thay đổi).

```
                    FIXED (init once)              UPDATED (per iter)
                ┌─────────────────────┐        ┌──────────────────────┐
                │ page_image_path     │        │ current_prompt       │
                │ layout_json_path    │        │ merged_html          │
                │ schema_fields       │        │ coverage             │
                │ reference_image_path│        │ llm_judge            │
                │ chandra_html        │        │ composite_score      │
                │ openai_model        │        │ missing_fields       │
                │ openai_judge_model  │        │ eval_feedback        │
                │ optimizer_model     │        │ eval_history         │
                │ use_local_models    │        │ iteration            │
                │ chandra_device      │        │ viz_source_path      │
                │ qwen_device         │        │ best_score           │
                │ max_iterations      │        │ best_prompt          │
                │ stop_threshold      │        │ best_iteration       │
                │ coverage_threshold  │        └──────────────────────┘
                │ output_dir          │
                │ run_id              │
                └─────────────────────┘
```

---

## 5. Data formats

### 5.1 Layout JSON (schema definition)

```json
{
  "fields": [
    {"name": "Logo", "type": "image"},
    {"name": "Tên khách hàng", "type": "text"},
    {
      "name": "BẢNG KÊ TIỀN MẶT (CASH LIST)",
      "type": "section",
      "children": [
        {"name": "Mệnh giá (Denomination)", "type": "text"},
        {"name": "Số tờ (Quantity)", "type": "text"},
        {"name": "Thành tiền (Amount)", "type": "text"},
        {"name": "Tổng cộng (Total)", "type": "text"}
      ]
    }
  ]
}
```

DFS traversal → `["Logo", "Tên khách hàng", ..., "Mệnh giá (Denomination)", ...]`

### 5.2 Chandra output HTML

```html
<div data-bbox="23 42 340 68" data-label="Text"><p>Tên khách hàng: NGUYEN VAN A</p></div>
<div data-bbox="573 716 888 866" data-label="Text"><p><img alt="Signature"/></p></div>
```

### 5.3 Aligned + merged HTML (output của pipeline)

```html
<div data-bbox="23 42 340 68" data-label="Tên khách hàng" data-schema="Tên khách hàng">
  <p>NGUYEN VAN A</p>
</div>
<div data-bbox="573 716 888 866" data-label="Kiểm soát viên" data-schema="Kiểm soát viên">
  <p>Kiểm soát viên</p><br/><p><img alt="Signature of Võ Thị Phương Thủy"/></p><br/><p>Võ Thị Phương Thủy</p>
</div>
```

### 5.4 Layout values JSON (output cuối)

```json
{
  "Logo": "",
  "Tên khách hàng": "NGUYEN VAN A",
  "CIF": "10925530",
  "Kiểm soát viên": "Võ Thị Phương Thủy",
  "Mệnh giá (Denomination)": "",
  "Tổng cộng (Total)": ""
}
```

---

## 6. Prompt template injection

Chandra prompt hỗ trợ `{schema_fields}` placeholder — được inject tự động tại runtime:

```
chandra_general_template.txt:
    ...
    Item list (use these names verbatim as data-label):
    {schema_fields}          ← placeholder

→ inject tại run_chandra node:
    Item list (use these names verbatim as data-label):
    Logo
    GIẤY GỬI TIỀN TIẾT KIỆM
    Ngày
    Tên khách hàng
    CIF
    ...
```

Điều này cho phép dùng **một prompt template cho mọi doc type** — không cần sửa prompt thủ công khi thêm loại tài liệu mới.

---

## 7. Cơ chế cache

```
run_chandra:
    log_path = <pdf_stem>_llm.log
    ├── file tồn tại + force_rerun=False → đọc log, parse HTML → skip model call
    └── không có → chạy model, lưu log

align_schema:
    không cache (chạy lại mỗi iter vì prompt thay đổi)

checkpoint:
    lưu tất cả artifacts mỗi iter → có thể resume từ bất kỳ iter nào
```

---

## 8. Chế độ chạy

### 8.1 Interactive mode (default)

```
main.py → graph stream → INTERRUPT tại optimize_prompt
    → hiển thị evaluation summary
    → hỏi operator: "Continue? [y/n]"
    → y: resume → optimize_prompt chạy → loop
    → n: dừng, lưu best prompt
```

### 8.2 Auto-resume mode (`--auto-resume`)

```
main.py → graph stream → không interrupt
    → optimize_prompt chạy tự động
    → loop cho đến khi đạt threshold hoặc max_iterations
```

### 8.3 Inference only (`run_inference.py`)

```
run_pipeline():
    1. load_layout_names → schema_fields
    2. chandra_log hoặc run Chandra → chandra_html
    3. align_schema (Qwen local) → raw_html
    4. merge_schemas → merged_html
    5. visualize → schema_boxes.jpg
    6. compute_coverage → coverage, missing_fields
    7. lưu layout_values.json
    8. in summary

Không có evaluate (GPT-4o), không có optimize_prompt.
Hỗ trợ so sánh 2 prompt (--align-prompt + --align-prompt-b).
```

---

## 9. Dependency graph giữa modules

```
main.py / run_inference.py
    │
    ├── src/graph.py (StateGraph)
    │       └── src/state.py (AgentState TypedDict)
    │
    ├── src/nodes/
    │       ├── setup.py
    │       │     └── src/utils/schema_html.py ──► layoutDectectionChan/src/
    │       ├── run_chandra.py
    │       │     ├── src/clients/local_chandra.py ──► layoutDectectionChan/src/load_model.py
    │       │     │                                    layoutDectectionChan/src/layout_extract.py
    │       │     └── src/clients/chandra_api.py ───► src/clients/openai_vision.py
    │       ├── align_schema.py
    │       │     ├── src/clients/local_qwen_vl.py ──► layoutDectectionChan/src/schema_merge_qwen_vl.py
    │       │     └── src/clients/openai_vision.py
    │       ├── merge_schemas.py
    │       │     └── src/utils/schema_html.py
    │       ├── visualize.py
    │       │     └── src/utils/image_io.py ──► layoutDectectionChan/src/schema_viz.py
    │       ├── evaluate.py
    │       │     ├── src/utils/schema_html.py
    │       │     ├── src/utils/scoring.py
    │       │     └── src/clients/openai_vision.py (GPT-4o judge)
    │       ├── checkpoint.py
    │       └── optimize_prompt.py
    │             └── src/clients/openai_vision.py (GPT-4o optimizer)
    │
    └── layoutDectectionChan/src/ (reused via sys.path injection)
```

---

## 10. Mở rộng cho doc type mới

1. **Tạo sample data:**
   ```
   data/samples/<DOC_TYPE>/
       layout_<DOC_TYPE>.json     ← định nghĩa schema
       sample.pdf                  ← trang mẫu đại diện
       sample_schema_boxes.jpg     ← ảnh reference (ground truth visual)
   ```

2. **Chạy tuning** — pipeline tự inject schema fields:
   ```bash
   python main.py \
     --pdf data/samples/<DOC_TYPE>/sample.pdf \
     --layout-json data/samples/<DOC_TYPE>/layout_<DOC_TYPE>.json \
     --chandra-prompt data/prompts/chandra_general_template.txt \
     --initial-prompt data/prompts/schema_align_general.txt \
     --reference-image data/samples/<DOC_TYPE>/sample_schema_boxes.jpg \
     --local --qwen-device cuda:0 --max-iterations 3 --auto-resume
   ```

3. **Lấy best prompt:**
   ```
   data/tuning_runs/<run_id>/best_prompt.txt
   ```

4. **Inference:**
   ```bash
   python run_inference.py \
     --pdf data/test/<DOC_TYPE>/test_1.pdf \
     --layout-json data/samples/<DOC_TYPE>/layout_<DOC_TYPE>.json \
     --chandra-prompt data/prompts/chandra_general_template.txt \
     --align-prompt data/tuning_runs/<run_id>/best_prompt.txt \
     --out-dir data/test/<DOC_TYPE>/out_test1 \
     --local --qwen-device cuda:0
   ```
