# Group Report — Lab 18: Production RAG

**Nhóm:** Cá nhân  
**Ngày:** 2026-06-22

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| NguyenVanChung | M1: Chunking | ☑ | 13/13 |
| NguyenVanChung | M2: Hybrid Search | ☑ | 5/5 |
| NguyenVanChung | M3: Reranking | ☑ | 5/5 |
| NguyenVanChung | M4: Evaluation | ☑ | 4/4 |
| NguyenVanChung | M5: Enrichment | ☑ | 10/10 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.8833 | **0.9091** | +0.0258 |
| Answer Relevancy | 0.8033 | **0.8582** | +0.0549 |
| Context Precision | 0.9250 | **1.0000** | +0.0750 |
| Context Recall | 0.9083 | **0.9500** | +0.0417 |

## Latency Breakdown

Xem chi tiết: [`reports/latency_report.md`](../reports/latency_report.md)

| Bước | Thời gian | % |
|------|-----------|---|
| M1 Chunking | 0.22s | 0.1% |
| M5 Enrichment (cached) | 0.0s | 0% |
| M2 Indexing | 42.69s | 20.8% |
| M3 Reranker load | 0.0s | 0% |
| Eval 20 queries | 136.98s | 66.6% |
| M4 RAGAS | 25.76s | 12.5% |
| **Tổng** | **205.66s** | 100% |

## Key Findings

1. **Biggest improvement:** Answer Relevancy (+0.055) nhờ hybrid search + CrossEncoder rerank + parent context expansion.
2. **Biggest challenge:** Câu numeric/multi-hop (tạm ứng, pro-rata) — LLM vẫn tự tính sai dù context đúng.
3. **Surprise finding:** Sau khi pre-download model và sửa parent_id namespaced, production vượt baseline trên cả 4 metrics.

## Presentation Notes (5 phút)

1. RAGAS scores: production thắng baseline toàn diện; faithfulness **0.909**, context precision **1.0**.
2. Biggest win — M2 (BM25+Dense+RRF) + M3 rerank + M5 enrichment + hierarchical parent return.
3. Case study — tạm ứng 15M/20 ngày: context đúng nhưng LLM tính sai phạt → fix ở prompt generation.
4. Next optimization: prompt chặn tự tính toán + metadata version filter cho policy v1/v2.
