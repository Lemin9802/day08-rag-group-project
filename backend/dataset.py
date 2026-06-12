from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from backend.config import DATASET_PATH


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ỹ0-9]+", str(text).lower())


@lru_cache(maxsize=1)
def load_chunks() -> list[dict]:
    path = Path(DATASET_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("rag_chunks.json must be a list")

    clean = []
    for i, item in enumerate(data):
        content = str(item.get("content", "")).strip()
        metadata = item.get("metadata", {}) or {}
        if not content:
            continue
        clean.append(
            {
                "content": content,
                "metadata": metadata,
                "chunk_id": metadata.get("chunk_id") or f"chunk_{i:04d}",
            }
        )
    return clean


@lru_cache(maxsize=1)
def _bm25() -> BM25Okapi:
    chunks = load_chunks()
    corpus = [_tokens(c["content"] + " " + " ".join(str(v) for v in c["metadata"].values())) for c in chunks]
    return BM25Okapi(corpus)


def dataset_summary() -> dict:
    chunks = load_chunks()
    legal = sum(1 for c in chunks if c["metadata"].get("source_type") == "legal")
    news = sum(1 for c in chunks if c["metadata"].get("source_type") == "news")
    docs = sorted({c["metadata"].get("doc_id") or c["metadata"].get("source") for c in chunks})
    return {
        "dataset_path": str(DATASET_PATH),
        "chunks": len(chunks),
        "legal_chunks": legal,
        "news_chunks": news,
        "documents": len([d for d in docs if d]),
    }


LEGAL_INTENT_TERMS = [
    "quy định", "điều", "khoản", "luật", "nghị định", "thông tư", "pháp lệnh",
    "bị phạt", "xử phạt", "trách nhiệm hình sự", "tội", "khung hình phạt",
    "tàng trữ", "vận chuyển", "mua bán", "sản xuất", "tổ chức sử dụng",
    "cai nghiện bắt buộc", "hồ sơ", "thủ tục",
]

NEWS_INTENT_TERMS = [
    "tin", "tin tức", "mới nhất", "gần đây", "vụ", "chuyên án", "bắt", "khởi tố",
    "xét xử", "đường dây", "học đường", "thuốc lá điện tử", "xu hướng",
]


def _query_intent(query: str) -> str:
    q = str(query).lower()
    legal_hits = sum(1 for term in LEGAL_INTENT_TERMS if term in q)
    news_hits = sum(1 for term in NEWS_INTENT_TERMS if term in q)

    if news_hits > legal_hits and news_hits > 0:
        return "news"
    if legal_hits > 0:
        return "legal"
    return "general"


def _source_boost(intent: str, source_type: str) -> float:
    source_type = (source_type or "unknown").lower()

    if intent == "legal":
        if source_type == "legal":
            return 1.45
        if source_type == "news":
            return 0.72

    if intent == "news":
        if source_type == "news":
            return 1.35
        if source_type == "legal":
            return 0.92

    return 1.0


def retrieve(query: str, top_k: int = 6) -> list[dict]:
    chunks = load_chunks()
    bm25 = _bm25()
    raw_scores = bm25.get_scores(_tokens(query))
    intent = _query_intent(query)

    scored = []
    for idx, raw_score in enumerate(raw_scores):
        item = chunks[idx]
        meta = item.get("metadata", {}) or {}
        source_type = meta.get("source_type") or meta.get("type") or "unknown"
        adjusted_score = float(raw_score) * _source_boost(intent, source_type)

        # Small title/path boost when the document title/source contains query terms.
        title_blob = " ".join(
            str(meta.get(k, "")) for k in ["title", "source", "doc_id", "path", "news_group"]
        ).lower()
        query_terms = [t for t in _tokens(query) if len(t) >= 4]
        overlap = sum(1 for t in query_terms if t in title_blob)
        adjusted_score += overlap * 0.15

        scored.append((idx, adjusted_score))

    ranked = sorted(scored, key=lambda x: float(x[1]), reverse=True)[: max(top_k, 1)]
    max_score = max([float(s) for _, s in ranked], default=1.0) or 1.0

    results = []
    for idx, score in ranked:
        item = dict(chunks[idx])
        item["score"] = round(float(score) / max_score, 4)
        item["retrieval_intent"] = intent
        results.append(item)
    return results
