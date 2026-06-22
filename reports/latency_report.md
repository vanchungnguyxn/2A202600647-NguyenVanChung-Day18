# Latency Breakdown Report — Lab 18 Production RAG

**Ngày chạy:** 2026-06-22 15:19:26  
**Tổng thời gian:** 205.66s  
**Corpus:** 100 chunks · 20 câu hỏi eval

## Tóm tắt

| Bước | Module | Thời gian | % tổng |
|------|--------|-----------|--------|
| M1 Chunking (hierarchical) | M1 | 0.22s | 0.1% |
| M5 Enrichment | M5 | 0.0s | 0.0% |
| M2 Indexing (BM25 + Dense) | M2 | 42.69s | 20.8% |
| M3 Reranker load | M3 | 0.0s | 0.0% |
| Eval: 20 queries (search+rerank+LLM) | M2+M3+LLM | 136.98s | 66.6% |
| M4 RAGAS evaluation | M4 | 25.76s | 12.5% |
| **TỔNG** | | **205.66s** | **100%** |

**Trung bình mỗi query:** 6.85s (hybrid search + rerank + GPT-4o-mini)

## Môi trường

| Thành phần | Giá trị |
|------------|---------|
| Python | 3.12 |
| Embedding | BAAI/bge-m3 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | gpt-4o-mini |
| Enrichment cache | Có |

## Phân tích bottleneck

1. **Indexing (M2)** — embed 100 chunks bằng bge-m3 chiếm ~20% tổng thời gian.
2. **Eval queries** — 20 câu × (hybrid search + CrossEncoder rerank + OpenAI) chiếm phần lớn (~65%).
3. **RAGAS (M4)** — ~28s cho 80 metric evaluations.
4. **M1/M3/M5 (cached)** — gần như instant khi model đã tải và enrichment có cache.

## RAGAS scores (cùng lần chạy)

| Metric | Score |
|--------|-------|
| faithfulness | 0.9091 |
| answer_relevancy | 0.8582 |
| context_precision | 1.0000 |
| context_recall | 0.9500 |
