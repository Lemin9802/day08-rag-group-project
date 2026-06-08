"""
Task 8 — PageIndex Vectorless RAG.

Yêu cầu:
    1. Có hàm upload_documents()
    2. Có hàm pageindex_search(query, top_k)
    3. Dùng làm fallback khi hybrid search không đủ tốt

"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
MANIFEST_PATH = INDEX_DIR / "pageindex_manifest.json"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[0-9a-zA-ZÀ-ỹ]+", text, flags=re.UNICODE)


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_markdown(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = clean_text(text)

    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            cut_points = [
                text.rfind("\n## ", start, end),
                text.rfind("\n# ", start, end),
                text.rfind("\n\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            ]
            best_cut = max(cut_points)

            if best_cut > start + int(chunk_size * 0.5):
                end = best_cut + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def load_markdown_documents() -> list[dict]:
    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {STANDARDIZED_DIR}. Hãy chạy Task 3 trước."
        )

    documents = []

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        if not md_file.is_file():
            continue

        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"

        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": str(relative_path).replace("\\", "/"),
                "type": doc_type,
            }
        })

    if not documents:
        raise RuntimeError("Không tìm thấy Markdown documents. Hãy chạy Task 3 trước.")

    return documents


def build_vectorless_candidates() -> list[dict]:
    documents = load_markdown_documents()
    candidates = []

    for doc_id, doc in enumerate(documents, start=1):
        chunks = split_markdown(doc["content"])

        for chunk_index, chunk_text in enumerate(chunks):
            first_line = chunk_text.splitlines()[0] if chunk_text.splitlines() else ""
            section_title = first_line[:120]

            candidates.append({
                "content": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "chunk_id": f"pageindex-local::{doc['metadata']['source']}::{chunk_index:04d}",
                    "section_title": section_title,
                }
            })

    return candidates


def score_candidate(query: str, candidate: dict) -> float:
    query_lower = query.lower()
    content = candidate.get("content", "")
    metadata = candidate.get("metadata", {})

    content_lower = content.lower()
    source_lower = metadata.get("source", "").lower()
    section_lower = metadata.get("section_title", "").lower()

    query_tokens = set(tokenize(query))
    content_tokens = set(tokenize(content))
    source_tokens = set(tokenize(source_lower))
    section_tokens = set(tokenize(section_lower))

    if not query_tokens:
        return 0.0

    content_overlap = len(query_tokens.intersection(content_tokens)) / len(query_tokens)
    section_overlap = len(query_tokens.intersection(section_tokens)) / len(query_tokens)
    source_overlap = len(query_tokens.intersection(source_tokens)) / len(query_tokens)

    phrase_bonus = 0.0
    if query_lower in content_lower:
        phrase_bonus += 0.5

    important_terms = [
        "điều",
        "nghị định",
        "quyết định",
        "thông tư",
        "ma túy",
        "tiền chất",
        "cai nghiện",
    ]

    important_bonus = 0.0
    for term in important_terms:
        if term in query_lower and term in content_lower:
            important_bonus += 0.05

    score = (
        0.70 * content_overlap
        + 0.15 * section_overlap
        + 0.10 * source_overlap
        + phrase_bonus
        + important_bonus
    )

    return float(score)


def upload_documents():
    """
    Chuẩn bị documents cho PageIndex/vectorless retrieval.

    Bản hiện tại:
        - Tạo manifest local.
        - Không làm crash pipeline nếu PageIndex SDK/API chưa dùng được.
    """
    documents = load_markdown_documents()
    candidates = build_vectorless_candidates()

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "mode": "local_vectorless_fallback",
        "pageindex_api_configured": bool(PAGEINDEX_API_KEY),
        "gemini_api_configured": bool(GEMINI_API_KEY),
        "num_documents": len(documents),
        "num_candidates": len(candidates),
        "documents": [
            {
                "source": doc["metadata"]["source"],
                "path": doc["metadata"]["path"],
                "type": doc["metadata"]["type"],
                "num_chars": len(doc["content"]),
            }
            for doc in documents
        ],
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"✓ Prepared {len(documents)} documents for vectorless retrieval")
    print(f"✓ Built {len(candidates)} local vectorless candidates")
    print(f"✓ Saved manifest: {MANIFEST_PATH}")

    if PAGEINDEX_API_KEY:
        print("✓ PAGEINDEX_API_KEY đã được cấu hình.")
    else:
        print("ℹ PAGEINDEX_API_KEY chưa có, đang dùng local vectorless fallback.")

    if GEMINI_API_KEY:
        print("✓ GEMINI_API_KEY đã được cấu hình. Gemini sẽ dùng ở Task 10.")

    return manifest


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval interface.

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex_local'
        }
    """
    if not query or not query.strip():
        return []

    candidates = build_vectorless_candidates()
    scored_results = []

    for candidate in candidates:
        score = score_candidate(query, candidate)

        if score <= 0:
            continue

        scored_results.append({
            "content": candidate["content"],
            "score": score,
            "metadata": candidate["metadata"],
            "source": "pageindex_local",
        })

    scored_results.sort(key=lambda item: item["score"], reverse=True)

    return scored_results[:top_k]


if __name__ == "__main__":
    print("=" * 60)
    print("Task 8: PageIndex Vectorless RAG")
    print("=" * 60)

    upload_documents()

    test_queries = [
        "danh mục chất ma túy và tiền chất theo Nghị định 28/2026",
        "tiêu chí xác định địa bàn trọng điểm phức tạp về ma túy",
        "cơ sở cai nghiện bắt buộc",
        "ca sĩ Miu Lê bị bắt sử dụng ma túy",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print("-" * 80)

        results = pageindex_search(query, top_k=3)

        for i, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            source = metadata.get("source", "unknown")
            doc_type = metadata.get("type", "unknown")
            score = result.get("score", 0.0)
            preview = result.get("content", "")[:220].replace("\n", " ")

            print(f"{i}. score={score:.4f} | type={doc_type} | source={source}")
            print(f"   {preview}...")