"""
Task 5 — Semantic Search Module.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Tương thích với embedding/index đã tạo ở Task 4

Cách làm:
    - Đọc local vector index từ data/index/chunks_with_embeddings.json
    - Embed query bằng cùng hàm embed_text ở Task 4
    - Tính cosine similarity giữa query vector và chunk vector
    - Return top_k chunks
"""

import json
import math
from pathlib import Path

from .task4_chunking_indexing import embed_text


INDEX_PATH = Path(__file__).parent.parent / "data" / "index" / "chunks_with_embeddings.json"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Tính cosine similarity giữa 2 vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def load_index() -> list[dict]:
    """Load local vector index được tạo từ Task 4."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy index: {INDEX_PATH}. "
            "Hãy chạy Task 4 trước: python -m src.task4_chunking_indexing"
        )

    chunks = json.loads(INDEX_PATH.read_text(encoding="utf-8"))

    if not chunks:
        raise RuntimeError("Index rỗng. Hãy chạy lại Task 4.")

    return chunks


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not query or not query.strip():
        return []

    chunks = load_index()
    query_embedding = embed_text(query)

    results = []

    for chunk in chunks:
        chunk_embedding = chunk.get("embedding", [])
        score = cosine_similarity(query_embedding, chunk_embedding)

        results.append({
            "content": chunk.get("content", ""),
            "score": float(score),
            "metadata": chunk.get("metadata", {}),
            # Giữ lại embedding để Task 7/MMR có thể dùng nếu cần
            "embedding": chunk_embedding,
        })

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "danh mục chất ma túy và tiền chất",
        "địa bàn trọng điểm phức tạp về ma túy",
        "ca sĩ bị bắt vì sử dụng ma túy",
    ]

    for query in test_queries:
        print("=" * 80)
        print(f"Query: {query}")
        print("-" * 80)

        results = semantic_search(query, top_k=5)

        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "unknown")
            doc_type = result["metadata"].get("type", "unknown")
            score = result["score"]
            preview = result["content"][:200].replace("\n", " ")

            print(f"{i}. score={score:.4f} | type={doc_type} | source={source}")
            print(f"   {preview}...")
            print()