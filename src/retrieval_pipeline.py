"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp:
    - Task 5: semantic_search
    - Task 6: lexical_search / BM25
    - Task 7: RRF + lightweight rerank
    - Task 8: PageIndex/local vectorless fallback

Cải tiến:
    - Query expansion cho câu hỏi news/nghệ sĩ
    - Intent filter: news query ưu tiên news, legal query ưu tiên legal
    - Diversify source cho câu hỏi liệt kê để tránh lấy 5 chunks cùng 1 file
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


DEFAULT_TOP_K = 5
SCORE_THRESHOLD = 0.15
RERANK_METHOD = "query_overlap"


NEWS_TERMS = [
    "nghệ sĩ",
    "ca sĩ",
    "rapper",
    "diễn viên",
    "người nổi tiếng",
    "bị bắt",
    "dương tính",
    "sử dụng ma túy",
    "liên quan tới ma túy",
    "liên quan đến ma túy",
]

LEGAL_TERMS = [
    "nghị định",
    "quyết định",
    "thông tư",
    "pháp lệnh",
    "điều",
    "khoản",
    "danh mục",
    "tiền chất",
    "cai nghiện bắt buộc",
    "địa bàn trọng điểm",
    "pháp luật",
]


def is_news_query(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in NEWS_TERMS)


def is_legal_query(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in LEGAL_TERMS)


def is_broad_list_query(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in ["những", "các", "nào", "liệt kê", "danh sách", "ai"])


def expand_query(query: str) -> str:
    """
    Mở rộng query cho retrieval, đặc biệt với câu hỏi tổng hợp về nghệ sĩ.
    Không thay đổi query gốc dùng cho rerank/generation.
    """
    if is_news_query(query):
        return (
            query
            + " ca sĩ rapper diễn viên nghệ sĩ người nổi tiếng "
            + "bị bắt sử dụng ma túy dương tính ma túy khởi tố tạm giam"
        )

    return query


def filter_by_intent(results: list[dict], query: str, min_keep: int) -> list[dict]:
    """
    Lọc theo intent:
    - News query: ưu tiên metadata.type == news
    - Legal query: ưu tiên metadata.type == legal
    Nếu lọc ra quá ít thì giữ original để tránh mất recall.
    """
    if not results:
        return []

    if is_news_query(query):
        filtered = [
            item for item in results
            if item.get("metadata", {}).get("type") == "news"
        ]
        if len(filtered) >= min_keep:
            return filtered

    if is_legal_query(query) and not is_news_query(query):
        filtered = [
            item for item in results
            if item.get("metadata", {}).get("type") == "legal"
        ]
        if len(filtered) >= min_keep:
            return filtered

    return results


def diversify_by_source(results: list[dict], top_k: int, max_per_source: int = 1) -> list[dict]:
    """
    Với câu hỏi liệt kê, tránh việc top_k toàn chunks từ cùng 1 file.
    Lấy tối đa max_per_source chunks mỗi source trước, rồi fill phần còn thiếu.
    """
    selected = []
    source_counts = {}

    for item in results:
        source = item.get("metadata", {}).get("source", "unknown")
        count = source_counts.get(source, 0)

        if count < max_per_source:
            selected.append(item)
            source_counts[source] = count + 1

        if len(selected) >= top_k:
            return selected

    # Fill nếu chưa đủ top_k
    selected_keys = {
        item.get("metadata", {}).get("chunk_id", item.get("content", "")[:80])
        for item in selected
    }

    for item in results:
        key = item.get("metadata", {}).get("chunk_id", item.get("content", "")[:80])
        if key not in selected_keys:
            selected.append(item)

        if len(selected) >= top_k:
            break

    return selected[:top_k]


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh.

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str
        }
    """
    if not query or not query.strip():
        return []

    print(f"Retrieving for query: {query}")

    search_query = expand_query(query)

    # Lấy rộng hơn để câu hỏi tổng hợp có cơ hội bắt nhiều source khác nhau
    search_k = max(top_k * 6, 30)

    dense_results = semantic_search(search_query, top_k=search_k)
    sparse_results = lexical_search(search_query, top_k=search_k)

    print(f"  Dense results: {len(dense_results)}")
    print(f"  Sparse results: {len(sparse_results)}")

    merged_results = rerank_rrf(
        ranked_lists=[dense_results, sparse_results],
        top_k=search_k,
    )

    for item in merged_results:
        item["source"] = "hybrid"

    print(f"  Merged results: {len(merged_results)}")

    # Intent filter trước rerank để tránh câu hỏi news bị legal lấn top
    intent_filtered = filter_by_intent(
        results=merged_results,
        query=query,
        min_keep=top_k,
    )

    print(f"  After intent filter: {len(intent_filtered)}")

    if use_reranking and intent_filtered:
        reranked_results = rerank(
            query=query,
            candidates=intent_filtered,
            top_k=search_k,
            method=RERANK_METHOD,
        )
    else:
        reranked_results = intent_filtered

    # Nếu câu hỏi dạng liệt kê news, lấy đa dạng theo source
    if is_news_query(query) and is_broad_list_query(query):
        final_results = diversify_by_source(
            reranked_results,
            top_k=top_k,
            max_per_source=1,
        )
    else:
        final_results = reranked_results[:top_k]

    for item in final_results:
        item["source"] = "hybrid"

    best_score = final_results[0]["score"] if final_results else 0.0
    print(f"  Best hybrid score: {best_score:.4f}")

    if not final_results or best_score < score_threshold:
        print(
            f"  ⚠ Hybrid score thấp hơn threshold "
            f"({best_score:.4f} < {score_threshold:.4f}). "
            f"Fallback → PageIndex/local vectorless"
        )

        fallback_results = pageindex_search(query, top_k=top_k)

        for item in fallback_results:
            item["source"] = item.get("source", "pageindex_local")

        return fallback_results[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "danh mục chất ma túy và tiền chất theo Nghị định 28/2026",
        "tiêu chí xác định địa bàn trọng điểm phức tạp về ma túy",
        "cơ sở cai nghiện bắt buộc",
        "ca sĩ Miu Lê bị bắt sử dụng ma túy",
        "rapper Bình Gold dương tính ma túy",
        "Những nghệ sĩ nào trong dữ liệu bị bắt hoặc liên quan tới ma túy?",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print("-" * 80)

        results = retrieve(query, top_k=5)

        for i, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            score = result.get("score", 0.0)
            result_source = result.get("source", "unknown")
            doc_source = metadata.get("source", "unknown")
            doc_type = metadata.get("type", "unknown")
            preview = result.get("content", "")[:220].replace("\n", " ")

            print(
                f"{i}. score={score:.4f} | retrieval={result_source} "
                f"| type={doc_type} | source={doc_source}"
            )
            print(f"   {preview}...")