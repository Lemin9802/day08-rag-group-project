"""
Task 6 — Lexical Search Module (BM25).

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có content, score, metadata
    - Kết quả sorted descending theo BM25 score

Cách làm:
    - Đọc corpus từ data/index/chunks_with_embeddings.json do Task 4 tạo
    - Tokenize text
    - Build BM25 index
    - Search query bằng BM25
"""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


INDEX_PATH = Path(__file__).parent.parent / "data" / "index" / "chunks_with_embeddings.json"


def tokenize(text: str) -> list[str]:
    """
    Tokenize đơn giản cho tiếng Việt:
    - lowercase
    - giữ chữ tiếng Việt và số
    - phù hợp cho keyword search, số văn bản, tên riêng, điều khoản
    """
    text = text.lower()
    return re.findall(r"[0-9a-zA-ZÀ-ỹ]+", text, flags=re.UNICODE)


def load_corpus() -> list[dict]:
    """
    Load corpus từ local index của Task 4.

    Returns:
        List of {
            'content': str,
            'metadata': dict,
            'embedding': list | None
        }
    """
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy index: {INDEX_PATH}. "
            "Hãy chạy Task 4 trước: python -m src.task4_chunking_indexing"
        )

    chunks = json.loads(INDEX_PATH.read_text(encoding="utf-8"))

    corpus = []
    for chunk in chunks:
        content = chunk.get("content", "")
        if not content.strip():
            continue

        corpus.append({
            "content": content,
            "metadata": chunk.get("metadata", {}),
            "embedding": chunk.get("embedding"),
        })

    if not corpus:
        raise RuntimeError("Corpus rỗng. Hãy kiểm tra lại data/index/chunks_with_embeddings.json")

    return corpus


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

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

    corpus = load_corpus()
    bm25 = build_bm25_index(corpus)

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda idx: scores[idx],
        reverse=True,
    )

    results = []

    for idx in ranked_indices[:top_k]:
        score = float(scores[idx])

        # BM25 score <= 0 thường là không match keyword nào đáng kể
        if score <= 0:
            continue

        results.append({
            "content": corpus[idx]["content"],
            "score": score,
            "metadata": corpus[idx]["metadata"],
            # Giữ embedding để Task 7/9 có thể dùng nếu cần
            "embedding": corpus[idx].get("embedding"),
        })

    return results


if __name__ == "__main__":
    test_queries = [
        "Điều 4 địa bàn trọng điểm phức tạp về ma túy",
        "Nghị định 28 2026 danh mục chất ma túy tiền chất",
        "ca sĩ Miu Lê bị bắt dùng ma túy",
        "rapper Bình Gold dương tính ma túy",
    ]

    for query in test_queries:
        print("=" * 80)
        print(f"Query: {query}")
        print("-" * 80)

        results = lexical_search(query, top_k=5)

        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "unknown")
            doc_type = result["metadata"].get("type", "unknown")
            score = result["score"]
            preview = result["content"][:200].replace("\n", " ")

            print(f"{i}. score={score:.4f} | type={doc_type} | source={source}")
            print(f"   {preview}...")
            print()