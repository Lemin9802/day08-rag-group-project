"""
Task 7 — Reranking Module.

Chọn phương pháp:
    - RRF (Reciprocal Rank Fusion): dùng để merge dense + lexical ranked lists
    - Lightweight query-aware rerank: dùng token overlap + score gốc

Lý do:
    - Không cần API key Jina/OpenAI
    - Không cần model nặng
    - Phù hợp với pipeline local ở Task 4, 5, 6
"""

import math
import re
from collections import defaultdict


# =============================================================================
# BASIC UTILS
# =============================================================================

def tokenize(text: str) -> list[str]:
    """Tokenize đơn giản cho tiếng Việt và số hiệu văn bản."""
    text = text.lower()
    return re.findall(r"[0-9a-zA-ZÀ-ỹ]+", text, flags=re.UNICODE)


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity cho MMR nếu candidate có embedding."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def candidate_key(item: dict) -> str:
    """
    Key ổn định để deduplicate candidate.
    Ưu tiên metadata.chunk_id do Task 4 tạo.
    """
    metadata = item.get("metadata", {})
    if metadata.get("chunk_id"):
        return metadata["chunk_id"]

    source = metadata.get("source", "unknown")
    chunk_index = metadata.get("chunk_index", "unknown")
    content = item.get("content", "")

    return f"{source}::{chunk_index}::{content[:80]}"


def min_max_normalize(values: list[float]) -> list[float]:
    """Normalize list score về [0, 1]."""
    if not values:
        return []

    min_v = min(values)
    max_v = max(values)

    if max_v == min_v:
        return [1.0 for _ in values]

    return [(v - min_v) / (max_v - min_v) for v in values]


def query_overlap_score(query: str, content: str) -> float:
    """
    Score nhẹ dựa trên overlap token giữa query và content.
    Dùng để rerank query-aware mà không cần cross-encoder.
    """
    query_tokens = set(tokenize(query))
    content_tokens = set(tokenize(content))

    if not query_tokens or not content_tokens:
        return 0.0

    overlap = query_tokens.intersection(content_tokens)
    return len(overlap) / len(query_tokens)


# =============================================================================
# OPTION 1 — CROSS ENCODER FALLBACK
# =============================================================================

def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Placeholder/fallback cho cross-encoder reranking.

    Vì bài cá nhân đang chạy local, không dùng Jina/Qwen model nặng.
    Hàm này dùng lightweight query overlap để không làm vỡ pipeline.
    """
    return rerank(query=query, candidates=candidates, top_k=top_k, method="query_overlap")


# =============================================================================
# OPTION 2 — MMR
# =============================================================================

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List candidate có key 'embedding'
        top_k: Số lượng kết quả
        lambda_param: Trade-off relevance/diversity

    Returns:
        List top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    candidates_with_embedding = [
        c for c in candidates
        if c.get("embedding") and len(c.get("embedding", [])) == len(query_embedding)
    ]

    if not candidates_with_embedding:
        return candidates[:top_k]

    selected_indices = []
    remaining_indices = list(range(len(candidates_with_embedding)))

    for _ in range(min(top_k, len(candidates_with_embedding))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining_indices:
            candidate = candidates_with_embedding[idx]
            candidate_embedding = candidate["embedding"]

            relevance = cosine_sim(query_embedding, candidate_embedding)

            max_sim_to_selected = 0.0
            for selected_idx in selected_indices:
                selected_embedding = candidates_with_embedding[selected_idx]["embedding"]
                sim = cosine_sim(candidate_embedding, selected_embedding)
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * max_sim_to_selected
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    results = []
    for idx in selected_indices:
        item = candidates_with_embedding[idx].copy()
        item["score"] = float(item.get("score", 0.0))
        item["rerank_score"] = float(item.get("score", 0.0))
        item["rerank_method"] = "mmr"
        results.append(item)

    return results


# =============================================================================
# OPTION 3 — RRF
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List các ranked result lists, ví dụ:
                      [dense_results, bm25_results]
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant, default 60

    Returns:
        List top_k candidates sorted by RRF score descending.
    """
    rrf_scores = defaultdict(float)
    item_map = {}
    original_scores = defaultdict(list)
    appeared_in = defaultdict(list)

    for list_idx, ranked_list in enumerate(ranked_lists):
        ranker_name = f"ranker_{list_idx + 1}"

        for rank, item in enumerate(ranked_list, start=1):
            key = candidate_key(item)

            rrf_scores[key] += 1.0 / (k + rank)
            item_map[key] = item
            original_scores[key].append(float(item.get("score", 0.0)))
            appeared_in[key].append(ranker_name)

    sorted_keys = sorted(
        rrf_scores.keys(),
        key=lambda key: rrf_scores[key],
        reverse=True,
    )

    results = []

    for key in sorted_keys[:top_k]:
        item = item_map[key].copy()

        item["score"] = float(rrf_scores[key])
        item["rerank_score"] = float(rrf_scores[key])
        item["rerank_method"] = "rrf"
        item["original_scores"] = original_scores[key]
        item["appeared_in"] = appeared_in[key]

        results.append(item)

    return results


# =============================================================================
# UNIFIED RERANK INTERFACE
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "query_overlap",
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method:
            - "query_overlap": lightweight query-aware rerank
            - "score": sort theo score gốc
            - "cross_encoder": fallback sang query_overlap
            - "rrf": nếu truyền 1 list candidate thì sort score gốc

    Returns:
        List top_k reranked candidates.
    """
    if not candidates:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)

    if method == "score" or method == "rrf":
        sorted_candidates = sorted(
            candidates,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        return sorted_candidates[:top_k]

    if method != "query_overlap":
        raise ValueError(f"Unknown rerank method: {method}")

    base_scores = [float(c.get("score", 0.0)) for c in candidates]
    normalized_scores = min_max_normalize(base_scores)

    reranked = []

    for candidate, normalized_score in zip(candidates, normalized_scores):
        overlap = query_overlap_score(query, candidate.get("content", ""))

        # Kết hợp score gốc và overlap với query.
        # 0.7 giữ ranking retrieval, 0.3 tăng candidate match đúng keyword query.
        rerank_score = 0.7 * normalized_score + 0.3 * overlap

        item = candidate.copy()
        item["original_score"] = float(candidate.get("score", 0.0))
        item["score"] = float(rerank_score)
        item["rerank_score"] = float(rerank_score)
        item["rerank_method"] = "query_overlap"

        reranked.append(item)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

    return reranked[:top_k]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    dense_results = [
        {
            "content": "Nghị định 28/2026 quy định danh mục chất ma túy và tiền chất.",
            "score": 0.82,
            "metadata": {"source": "nghi-dinh-28-2026.md", "chunk_id": "a"},
        },
        {
            "content": "Ca sĩ Miu Lê bị bắt quả tang sử dụng ma túy ở bãi biển.",
            "score": 0.78,
            "metadata": {"source": "article_03.md", "chunk_id": "b"},
        },
        {
            "content": "Tiêu chí xác định địa bàn trọng điểm phức tạp về ma túy.",
            "score": 0.70,
            "metadata": {"source": "quyet-dinh-28-2025.md", "chunk_id": "c"},
        },
    ]

    bm25_results = [
        {
            "content": "Ca sĩ Miu Lê bị bắt quả tang sử dụng ma túy ở bãi biển.",
            "score": 36.4,
            "metadata": {"source": "article_03.md", "chunk_id": "b"},
        },
        {
            "content": "Ca sĩ Long Nhật cùng Sơn Ngọc Minh bị bắt.",
            "score": 18.6,
            "metadata": {"source": "article_01.md", "chunk_id": "d"},
        },
        {
            "content": "Nghị định 28/2026 quy định danh mục chất ma túy và tiền chất.",
            "score": 15.2,
            "metadata": {"source": "nghi-dinh-28-2026.md", "chunk_id": "a"},
        },
    ]

    print("=" * 80)
    print("Test RRF")
    print("-" * 80)

    rrf_results = rerank_rrf([dense_results, bm25_results], top_k=5)
    for i, result in enumerate(rrf_results, 1):
        print(
            f"{i}. score={result['score']:.4f} "
            f"| source={result['metadata'].get('source')} "
            f"| method={result.get('rerank_method')}"
        )
        print(f"   {result['content']}")

    print("\n" + "=" * 80)
    print("Test query-aware rerank")
    print("-" * 80)

    reranked = rerank(
        query="ca sĩ Miu Lê sử dụng ma túy",
        candidates=rrf_results,
        top_k=3,
        method="query_overlap",
    )

    for i, result in enumerate(reranked, 1):
        print(
            f"{i}. score={result['score']:.4f} "
            f"| source={result['metadata'].get('source')} "
            f"| method={result.get('rerank_method')}"
        )
        print(f"   {result['content']}")