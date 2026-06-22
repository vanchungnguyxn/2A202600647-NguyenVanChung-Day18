# Individual Reflection — Lab 18 (Production RAG)

**Tên:** NguyenVanChung  
**Module phụ trách:** Tự làm toàn bộ (M1–M5)

---

## Phần 1: Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Nhóm câu theo similarity để tránh cắt giữa ý; test đảm bảo semantic không “vỡ” quá nhiều chunks. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Retrieve child để chính xác, nhưng giữ `parent_id` để trả ngữ cảnh lớn hơn. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Tách theo markdown headers, giữ `section` trong metadata để trace nguồn. |
| Vietnamese BM25 | M2 | `segment_vietnamese()` + `BM25Search.search()` | underthesea nối từ ghép bằng `_` nên phải replace để query “nghỉ phép” khớp token. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF gộp danh sách BM25/dense để tăng recall mà không phụ thuộc thang điểm của từng retriever. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Có fallback heuristic khi không có model local để pipeline/test không bị treo khi tải model. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Wrap try/except để chạy được dù thiếu key; xuất `per_question` để phân tích lỗi. |
| Failure analysis | M4 | `failure_analysis()` | Lấy bottom-N theo average score, map sang Diagnostic Tree để ra diagnosis + fix. |
| Contextual embeddings/enrichment | M5 | `_enrich_single_call()` + `enrich_chunks()` | Dùng `response_format=json_object` để tránh lỗi parse; cache enriched chunks để chạy lại nhanh. |

## Phần 2: Khó khăn & giải quyết

- **Vấn đề:** Pipeline bị chậm/treo ở reranker do tải model HuggingFace lần đầu (mạng chậm).  
  **Giải quyết:** Thêm chế độ load offline (`local_files_only=True`) và fallback heuristic để không block lab; warmup reranker để tránh cold start.

- **Vấn đề:** Enrichment combined mode lỗi `Expecting value: line 1 column 1` khi parse JSON.  
  **Giải quyết:** Bắt model trả JSON bằng `response_format={"type":"json_object"}` và parser chịu được code-fence.

## Phần 3: Action Plan cho project

## Project: RAG nội bộ (draft)

### Hiện tại
- RAG pipeline hiện tại: chunk cơ bản + dense search.
- Known issues: câu hỏi multi-hop/numeric hay fail; trả lời thiếu điều kiện/ngoại lệ.

### Plan áp dụng
1. [ ] Chunking strategy: hierarchical + structure-aware (giữ bảng/section).
2. [ ] Search: hybrid BM25 + dense + RRF.
3. [ ] Reranking: cross-encoder (bge-reranker) nếu latency cho phép.
4. [ ] Evaluation: RAGAS 4 metrics + failure analysis theo bottom questions.
5. [ ] Enrichment: combined single-call + HyQA cho câu multi-aspect.

### Timeline
- Tuần 1: triển khai chunking + hybrid search, đo RAGAS.
- Tuần 2: thêm reranking + enrichment, chốt prompt template và regression set.

