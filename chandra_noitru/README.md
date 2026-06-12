# Chandra OCR 2 – Parse PDF nội trú

Pipeline OCR layout cho tài liệu bệnh án nội trú, dùng [datalab-to/chandra-ocr-2](https://huggingface.co/datalab-to/chandra-ocr-2).

## Chuẩn bị

1. Copy 4 PDF vào `data/pdfs/`:

   - `2300030376-6263.pdf`
   - `2300033911-5962.pdf`
   - `2500064072-3637.pdf`
   - `2500077856-33.pdf`

2. Cài dependency (dùng chung với `chandra4layout`):

   ```bash
   pip install -r chandra_noitru/requirements.txt
   ```

## Chạy

Parse **2 trang đầu** mỗi PDF (mặc định GPU `cuda:1`):

```bash
python chandra_noitru/run_parse.py --device-map cuda:1
```

Một file cụ thể:

```bash
python chandra_noitru/run_parse.py \
  --input-file /đường/dẫn/2300030376-6263.pdf \
  --max-pages 2 \
  --device-map cuda:1
```

## Đầu ra (`results/`)

| File | Mô tả |
|------|--------|
| `{stem}_page01_raw.html` | HTML layout từ Chandra |
| `{stem}_page01_blocks.json` | Blocks (bbox + text) |
| `{stem}_page01_layout.jpg` | Ảnh có vẽ bbox |
| `{stem}_page01.md` | Text thuần theo block |
| `summary.json` | Tổng hợp run |
