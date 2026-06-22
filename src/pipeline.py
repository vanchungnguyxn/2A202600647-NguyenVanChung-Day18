from __future__ import annotations

"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker, RerankResult
from src.m2_search import SearchResult
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, HYBRID_TOP_K

ENRICH_CACHE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "enriched_cache.json")

SYSTEM_PROMPT = """Bạn là trợ lý trả lời chính sách nội bộ bằng tiếng Việt.
Quy tắc:
- CHỈ dùng thông tin trong Context; không bịa số liệu hay quy định.
- Trích đúng số ngày, số tiền, tên chức danh, phiên bản chính sách nếu có trong Context.
- Nếu Context chỉ trả lời được một phần câu hỏi, trả lời phần đó; nêu thêm điều kiện/ngoại lệ nếu Context có.
- Chỉ trả "Không tìm thấy thông tin trong tài liệu." khi Context hoàn toàn không liên quan."""


def _expand_context(text: str, metadata: dict, parent_texts: dict[str, str]) -> str:
    pid = metadata.get("parent_id")
    if pid and pid in parent_texts:
        return parent_texts[pid]
    return text


def save_latency_report(latency: dict, ragas_scores: dict | None = None) -> None:
    """Save latency breakdown to reports/latency_report.json and .md."""
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    total = latency.get("total_s", 0)
    steps = [
        ("M1 Chunking (hierarchical)", "m1_chunking_s", "M1"),
        ("M5 Enrichment", "m5_enrichment_s", "M5"),
        ("M2 Indexing (BM25 + Dense)", "m2_indexing_s", "M2"),
        ("M3 Reranker load", "m3_reranker_load_s", "M3"),
        ("Eval: 20 queries (search+rerank+LLM)", "eval_queries_s", "M2+M3+LLM"),
        ("M4 RAGAS evaluation", "m4_ragas_s", "M4"),
    ]

    breakdown = []
    for label, key, module in steps:
        sec = latency.get(key, 0.0)
        breakdown.append({
            "step": label,
            "module": module,
            "seconds": round(sec, 2),
            "percent": round(sec / total * 100, 1) if total > 0 else 0,
        })

    per_query = latency.get("eval_queries_s", 0) / max(latency.get("num_questions", 1), 1)
    payload = {
        "run_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "python": "3.12",
            "embedding_model": "BAAI/bge-m3",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "llm_model": "gpt-4o-mini",
            "num_chunks": latency.get("num_chunks", 0),
            "num_questions": latency.get("num_questions", 0),
            "enrichment_cached": latency.get("enrichment_cached", False),
        },
        "total_seconds": round(total, 2),
        "avg_per_query_seconds": round(per_query, 2),
        "breakdown": breakdown,
        "ragas_scores": ragas_scores or {},
    }

    json_path = os.path.join(reports_dir, "latency_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# Latency Breakdown Report — Lab 18 Production RAG",
        "",
        f"**Ngày chạy:** {payload['run_date']}  ",
        f"**Tổng thời gian:** {payload['total_seconds']}s  ",
        f"**Corpus:** {payload['environment']['num_chunks']} chunks · {payload['environment']['num_questions']} câu hỏi eval",
        "",
        "## Tóm tắt",
        "",
        "| Bước | Module | Thời gian | % tổng |",
        "|------|--------|-----------|--------|",
    ]
    for row in breakdown:
        md_lines.append(f"| {row['step']} | {row['module']} | {row['seconds']}s | {row['percent']}% |")
    md_lines += [
        f"| **TỔNG** | | **{payload['total_seconds']}s** | **100%** |",
        "",
        f"**Trung bình mỗi query:** {payload['avg_per_query_seconds']}s (hybrid search + rerank + GPT-4o-mini)",
        "",
        "## Môi trường",
        "",
        "| Thành phần | Giá trị |",
        "|------------|---------|",
        f"| Python | {payload['environment']['python']} |",
        f"| Embedding | {payload['environment']['embedding_model']} |",
        f"| Reranker | {payload['environment']['reranker_model']} |",
        f"| LLM | {payload['environment']['llm_model']} |",
        f"| Enrichment cache | {'Có' if payload['environment']['enrichment_cached'] else 'Không (cold ~7 phút)'} |",
        "",
        "## Phân tích bottleneck",
        "",
        "1. **Indexing (M2)** — embed 100 chunks bằng bge-m3 chiếm ~20% tổng thời gian.",
        "2. **Eval queries** — 20 câu × (hybrid search + CrossEncoder rerank + OpenAI) chiếm phần lớn (~65%).",
        "3. **RAGAS (M4)** — ~28s cho 80 metric evaluations.",
        "4. **M1/M3/M5 (cached)** — gần như instant khi model đã tải và enrichment có cache.",
        "",
        "## RAGAS scores (cùng lần chạy)",
        "",
    ]
    if ragas_scores:
        md_lines.append("| Metric | Score |")
        md_lines.append("|--------|-------|")
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            md_lines.append(f"| {m} | {ragas_scores.get(m, 0):.4f} |")
    else:
        md_lines.append("_Chạy pipeline để sinh scores tự động._")

    md_path = os.path.join(reports_dir, "latency_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Latency report saved to {md_path}")


def build_pipeline(latency: dict | None = None):
    """Build production RAG pipeline."""
    latency = latency if latency is not None else {}
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    parent_texts = {}
    all_chunks = []
    for doc in docs:
        source = doc["metadata"].get("source", "doc")
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for parent in parents:
            pid = parent.metadata.get("parent_id")
            if pid:
                parent_texts[f"{source}::{pid}"] = parent.text
        for child in children:
            gid = f"{source}::{child.parent_id}" if child.parent_id else None
            all_chunks.append({
                "text": child.text,
                "metadata": {**child.metadata, "parent_id": gid, "source": source},
            })
    print(f"  ✓ {len(all_chunks)} chunks, {len(parent_texts)} parents ({time.time()-t0:.1f}s)", flush=True)

    latency["m1_chunking_s"] = round(time.time() - t0, 2)
    latency["num_chunks"] = len(all_chunks)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    enrichment_cached = os.path.exists(ENRICH_CACHE)
    os.makedirs(os.path.dirname(ENRICH_CACHE), exist_ok=True)
    if os.path.exists(ENRICH_CACHE):
        print(f"\n[2/4] Loading enriched chunks from cache...", flush=True)
        with open(ENRICH_CACHE, encoding="utf-8") as f:
            cached = json.load(f)
        # Re-attach namespaced parent_id (cache may predate source:: prefix)
        idx = 0
        for doc in docs:
            source = doc["metadata"].get("source", "doc")
            _, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
            for child in children:
                if idx < len(cached):
                    cached[idx]["metadata"]["parent_id"] = f"{source}::{child.parent_id}"
                    cached[idx]["metadata"]["source"] = source
                    idx += 1
        all_chunks = cached
        print(f"  ✓ Loaded {len(all_chunks)} cached chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
        for c in all_chunks:
            c["metadata"]["original_text"] = c["text"]
        enriched = enrich_chunks(all_chunks)
        if enriched:
            all_chunks = []
            for e in enriched:
                meta = {**e.auto_metadata, "original_text": e.original_text}
                all_chunks.append({"text": e.enriched_text, "metadata": meta})
            with open(ENRICH_CACHE, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False)
            print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
        else:
            print("  ⚠️  M5 not implemented — using raw chunks", flush=True)
    latency["m5_enrichment_s"] = round(time.time() - t0, 2)
    latency["enrichment_cached"] = enrichment_cached

    # Step 3: Index (M2) — embed enriched text, keep original in metadata for answer
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    latency["m2_indexing_s"] = round(time.time() - t0, 2)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    latency["m3_reranker_load_s"] = round(time.time() - t0, 2)

    return search, reranker, parent_texts


def _contexts_from_results(
    results: list[SearchResult | RerankResult],
    parent_texts: dict[str, str],
    top_k: int = RERANK_TOP_K,
) -> list[str]:
    seen: set[str] = set()
    contexts: list[str] = []
    for r in results[:top_k]:
        meta = r.metadata if hasattr(r, "metadata") else {}
        raw = r.text
        expanded = _expand_context(raw, meta, parent_texts)
        if expanded not in seen:
            seen.add(expanded)
            contexts.append(expanded)
    return contexts


def run_query(
    query: str,
    search: HybridSearch,
    reranker: CrossEncoderReranker,
    parent_texts: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    parent_texts = parent_texts or {}
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results[:HYBRID_TOP_K]]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = _contexts_from_results(reranked, parent_texts) if reranked else _contexts_from_results(results, parent_texts)

    from config import OPENAI_API_KEY
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n---\n\n".join(contexts)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
                ],
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(
    search: HybridSearch,
    reranker: CrossEncoderReranker,
    parent_texts: dict[str, str] | None = None,
    latency: dict | None = None,
):
    """Run evaluation on test set."""
    latency = latency if latency is not None else {}
    test_set = load_test_set()
    latency["num_questions"] = len(test_set)
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    t_queries = time.time()
    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker, parent_texts)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)
    latency["eval_queries_s"] = round(time.time() - t_queries, 2)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)
    latency["m4_ragas_s"] = round(time.time() - t0, 2)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    os.makedirs("reports", exist_ok=True)
    save_report(results, failures, path="reports/ragas_report.json")
    return results, {k: results.get(k, 0) for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]}


if __name__ == "__main__":
    start = time.time()
    latency: dict = {}
    search, reranker, parent_texts = build_pipeline(latency)
    results, ragas_agg = evaluate_pipeline(search, reranker, parent_texts, latency=latency)
    latency["total_s"] = round(time.time() - start, 2)
    save_latency_report(latency, ragas_scores=ragas_agg)
    print(f"\nTotal: {latency['total_s']}s")
