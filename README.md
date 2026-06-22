# Lab 18: Production RAG Pipeline

**MSSV:** 2A202600647  
**Họ tên:** Nguyễn Văn Chung (NguyenVanChung)  
**Repository:** [2A202600647-NguyenVanChung-Day18](https://github.com/vanchungnguyxn/2A202600647-NguyenVanChung-Day18)  
**Khóa học:** AICB-P2T3 · Ngày 18 · Production RAG  
**Giảng viên:** Trần Quang Thiện

---

## Tổng quan

Bài tập **cá nhân** — implement toàn bộ 5 modules production RAG trên corpus chính sách nội bộ tiếng Việt:

```
M1 Chunking → M5 Enrichment → M2 Hybrid Search → M3 Reranking → LLM Answer → M4 RAGAS Eval
```

| Module | File | Mô tả |
|--------|------|-------|
| M1 | `src/m1_chunking.py` | Semantic, hierarchical, structure-aware chunking |
| M2 | `src/m2_search.py` | BM25 + Dense (bge-m3) + RRF fusion |
| M3 | `src/m3_rerank.py` | CrossEncoder `BAAI/bge-reranker-v2-m3` |
| M4 | `src/m4_eval.py` | RAGAS 4 metrics + failure analysis |
| M5 | `src/m5_enrichment.py` | Combined enrichment (HyQA, metadata, contextual) |
| Pipeline | `src/pipeline.py` | Ghép M1–M5, latency tracking, parent context expansion |

---

## Kết quả

### RAGAS — Production vs Naive Baseline

| Metric | Naive | Production | Δ |
|--------|-------|------------|---|
| Faithfulness | 0.8833 | **0.9091** | +0.0258 |
| Answer Relevancy | 0.8033 | **0.8582** | +0.0549 |
| Context Precision | 0.9250 | **1.0000** | +0.0750 |
| Context Recall | 0.9083 | **0.9500** | +0.0417 |

Production vượt baseline trên cả 4 metrics. Chi tiết: [`reports/ragas_report.json`](reports/ragas_report.json).

### Latency Breakdown (205.66s)

| Bước | Thời gian | % |
|------|-----------|---|
| M1 Chunking | 0.22s | 0.1% |
| M5 Enrichment (cached) | 0.0s | 0% |
| M2 Indexing (BM25 + Dense) | 42.69s | 20.8% |
| M3 Reranker load | 0.0s | 0% |
| Eval 20 queries | 136.98s | 66.6% |
| M4 RAGAS | 25.76s | 12.5% |

Trung bình **6.85s/query**. Chi tiết: [`reports/latency_report.md`](reports/latency_report.md).

### Tests

```
37/37 passed (100%)
```

Chạy `py -3.12 check_lab.py` để kiểm tra deliverables trước khi nộp.

---

## Prerequisites

| Dependency | Bắt buộc? | Dùng cho |
|-----------|-----------|----------|
| Docker (Qdrant) | ✅ | M2 Dense Search |
| Python **3.12** | ✅ | RAGAS + sentence-transformers (tránh Python 3.14) |
| `OPENAI_API_KEY` | ✅ | M4 RAGAS, M5 Enrichment, LLM answer |

> **Lưu ý Windows:** Dùng `py -3.12`, không dùng `python` nếu máy có Python 3.14 (lỗi numpy).

**Pre-download models** (tránh timeout lần đầu):

```powershell
py -3.12 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
py -3.12 -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

---

## Quick Start

```powershell
git clone https://github.com/vanchungnguyxn/2A202600647-NguyenVanChung-Day18.git
cd 2A202600647-NguyenVanChung-Day18

docker compose up -d
py -3.12 -m pip install -r requirements.txt
copy .env.example .env          # Điền OPENAI_API_KEY

py -3.12 naive_baseline.py      # Chạy TRƯỚC để có baseline
py -3.12 src/pipeline.py        # Production pipeline + RAGAS + latency report
py -3.12 check_lab.py           # Kiểm tra trước khi nộp
```

Hoặc chạy toàn bộ (baseline + production + so sánh):

```powershell
py -3.12 main.py
```

---

## Deliverables

| File | Mô tả |
|------|-------|
| `src/m1_chunking.py` … `src/pipeline.py` | 5 modules + pipeline |
| `reports/ragas_report.json` | Kết quả RAGAS production |
| `reports/naive_baseline_report.json` | Kết quả naive baseline |
| `reports/latency_report.md` | Bảng thời gian từng bước |
| `analysis/failure_analysis.md` | Phân tích bottom-5 failures |
| `analysis/group_report.md` | Báo cáo nhóm (cá nhân) |
| `analysis/reflections/reflection_NguyenVanChung.md` | Reflection cá nhân |

---

## Kiến trúc pipeline

```
Documents (40 .md + PDFs)
    │
    ▼
M1: Hierarchical chunking (100 chunks, 26 parents)
    │
    ▼
M5: Enrichment (combined mode, cached)
    │
    ▼
M2: BM25 + Dense (bge-m3) + RRF → top-20
    │
    ▼
M3: CrossEncoder rerank → top-3
    │
    ▼
Parent context expansion + GPT-4o-mini answer
    │
    ▼
M4: RAGAS evaluation (4 metrics × 20 questions)
```

**Điểm nổi bật trong implementation:**

- **Parent ID namespaced** (`{source}::{parent_id}`) — tránh collision khi nhiều document
- **Parent context expansion** — retrieve child, trả parent text cho LLM/RAGAS
- **Enrichment cache** — `reports/enriched_cache.json` (local, không commit)
- **Latency tracking** — tự sinh `reports/latency_report.md` sau mỗi lần chạy

---

## Cấu trúc repo

```
├── README.md
├── ASSIGNMENT.md
├── RUBRIC.md
├── main.py
├── check_lab.py
├── naive_baseline.py
├── config.py
├── requirements.txt
├── docker-compose.yml
├── test_set.json
│
├── data/                       # Corpus tiếng Việt (40 .md + PDFs)
├── src/
│   ├── m1_chunking.py
│   ├── m2_search.py
│   ├── m3_rerank.py
│   ├── m4_eval.py
│   ├── m5_enrichment.py
│   └── pipeline.py
├── tests/                      # 37 auto-tests
├── analysis/
│   ├── failure_analysis.md
│   ├── group_report.md
│   └── reflections/
│       └── reflection_NguyenVanChung.md
└── reports/
    ├── ragas_report.json
    ├── naive_baseline_report.json
    └── latency_report.md
```

---

## Key Findings

1. **Cải thiện lớn nhất:** Answer Relevancy (+0.055) nhờ hybrid search + CrossEncoder rerank + parent context.
2. **Thách thức:** Câu numeric/multi-hop — LLM tự suy luận phép tính dù context đúng (faithfulness thấp).
3. **Bottleneck latency:** Eval queries (~67%) — search + rerank + OpenAI mỗi câu hỏi.

Xem thêm: [`analysis/failure_analysis.md`](analysis/failure_analysis.md) · [`analysis/group_report.md`](analysis/group_report.md)

---

## Tài liệu tham khảo

- [ASSIGNMENT.md](ASSIGNMENT.md) — Đề bài chi tiết
- [RUBRIC.md](RUBRIC.md) — Hệ thống chấm điểm
- Fork gốc: [VinUni-AI20k/Day18-Track3-Production-RAG-batch-2](https://github.com/VinUni-AI20k/Day18-Track3-Production-RAG-batch-2)
