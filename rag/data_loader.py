from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    RecursiveCharacterTextSplitter = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"


@dataclass(frozen=True)
class LoadedDocument:
    content: str
    metadata: dict


def _source_type(path: Path) -> str:
    lowered = {part.lower() for part in path.parts}
    if "legal" in lowered:
        return "legal"
    if "news" in lowered:
        return "news"
    return "unknown"


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


@lru_cache(maxsize=1)
def load_markdown_documents() -> list[LoadedDocument]:
    docs: list[LoadedDocument] = []
    if not STANDARDIZED_DIR.exists():
        return docs

    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        try:
            content = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            content = path.read_text(encoding="utf-8-sig", errors="ignore").strip()
        if not content:
            continue
        docs.append(
            LoadedDocument(
                content=content,
                metadata={
                    "source": path.name,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "source_type": _source_type(path),
                    "type": _source_type(path),
                },
            )
        )
    return docs


@lru_cache(maxsize=1)
def load_chunks() -> list[dict]:
    docs = load_markdown_documents()
    if not docs:
        return []

    chunks: list[dict] = []
    if RecursiveCharacterTextSplitter:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=650,
            chunk_overlap=90,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )
        for doc_index, doc in enumerate(docs, start=1):
            for chunk_index, text in enumerate(splitter.split_text(doc.content), start=1):
                text = text.strip()
                if not text:
                    continue
                source_type = doc.metadata["source_type"]
                chunks.append(
                    {
                        "content": text,
                        "score": 0.0,
                        "metadata": {
                            **doc.metadata,
                            "chunk_id": f"{source_type}_{doc_index:03d}_{chunk_index:03d}",
                            "chunk_index": chunk_index,
                        },
                    }
                )
        return chunks

    for doc_index, doc in enumerate(docs, start=1):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.content) if p.strip()]
        buffer = ""
        chunk_index = 1
        for paragraph in paragraphs:
            if len(buffer) + len(paragraph) > 750 and buffer:
                source_type = doc.metadata["source_type"]
                chunks.append(
                    {
                        "content": buffer.strip(),
                        "score": 0.0,
                        "metadata": {
                            **doc.metadata,
                            "chunk_id": f"{source_type}_{doc_index:03d}_{chunk_index:03d}",
                            "chunk_index": chunk_index,
                        },
                    }
                )
                chunk_index += 1
                buffer = ""
            buffer += "\n\n" + paragraph
        if buffer.strip():
            source_type = doc.metadata["source_type"]
            chunks.append(
                {
                    "content": buffer.strip(),
                    "score": 0.0,
                    "metadata": {
                        **doc.metadata,
                        "chunk_id": f"{source_type}_{doc_index:03d}_{chunk_index:03d}",
                        "chunk_index": chunk_index,
                    },
                }
            )
    return chunks


def dataset_summary() -> dict:
    docs = load_markdown_documents()
    chunks = load_chunks()
    return {
        "legal_docs": sum(1 for doc in docs if doc.metadata.get("source_type") == "legal"),
        "news_docs": sum(1 for doc in docs if doc.metadata.get("source_type") == "news"),
        "documents": len(docs),
        "chunks": len(chunks),
        "data_dir": str(STANDARDIZED_DIR),
    }


def clean_snippet(text: str, max_chars: int = 420) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."
