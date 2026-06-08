from __future__ import annotations

import re
from pathlib import Path


YEAR_PATTERN = re.compile(r"(20\d{2}|19\d{2})")


def infer_year(metadata: dict) -> str:
    text = " ".join(str(metadata.get(key, "")) for key in ("source", "path", "title", "url"))
    match = YEAR_PATTERN.search(text)
    return match.group(1) if match else "n.d."


def normalize_source_type(metadata: dict) -> str:
    source_type = metadata.get("source_type") or metadata.get("type") or metadata.get("doc_type")
    if source_type in {"legal", "news"}:
        return source_type
    path = str(metadata.get("path") or metadata.get("source") or "").lower()
    if "legal" in path:
        return "legal"
    if "news" in path:
        return "news"
    return "unknown"


def normalize_result(item: dict, index: int) -> dict:
    metadata = dict(item.get("metadata") or {})
    if "source" not in metadata:
        metadata["source"] = item.get("source") or f"source_{index}.md"
    metadata["source_type"] = normalize_source_type(metadata)
    metadata["type"] = metadata["source_type"]
    metadata["year"] = metadata.get("year") or infer_year(metadata)
    metadata["citation_id"] = metadata.get("citation_id") or f"S{index}"
    metadata["path"] = metadata.get("path") or str(metadata.get("source") or "")
    metadata["title"] = metadata.get("title") or Path(str(metadata["source"])).stem.replace("-", " ")

    score = item.get("score", 0.0)
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0

    return {
        "content": str(item.get("content") or "").strip(),
        "score": score,
        "metadata": metadata,
        "source": item.get("source") or "local",
    }


def add_citation_ids(results: list[dict]) -> list[dict]:
    normalized = [normalize_result(item, index) for index, item in enumerate(results, start=1)]
    for index, item in enumerate(normalized, start=1):
        item["metadata"]["citation_id"] = f"S{index}"
    return normalized


def citation_label(source: dict) -> str:
    metadata = source.get("metadata", {}) or {}
    return f"[{metadata.get('citation_id', 'S?')}]"


def source_reference(source: dict) -> str:
    metadata = source.get("metadata", {}) or {}
    return f"{citation_label(source)} {metadata.get('title') or metadata.get('source')} ({metadata.get('year', 'n.d.')})"
