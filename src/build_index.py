"""
Build a shared curated local index for the group project.

Run:
    python -m src.build_index
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from .config import INDEX_DIR, STANDARDIZED_DIR
from .utils import hash_embedding


CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
EMBED_DIM = 512


def clean_markdown(text: str) -> str:
    """Remove metadata/header noise before chunking."""
    text = str(text or "")

    # YAML front matter metadata.
    text = re.sub(r"(?s)^---\s*\n.*?\n---\s*\n", "", text)

    # Markdown metadata lines from crawled files.
    text = re.sub(r"(?im)^\s*\*\*source file:\*\*.*$", "", text)
    text = re.sub(r"(?im)^\s*source file:.*$", "", text)
    text = re.sub(r"(?im)^\s*\*\*source:\*\*.*$", "", text)
    text = re.sub(r"(?im)^\s*\*\*url:\*\*.*$", "", text)
    text = re.sub(r"(?im)^\s*\*\*crawled:\*\*.*$", "", text)
    text = re.sub(r"(?im)^\s*crawled:.*$", "", text)

    # Markdown images and very noisy links.
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]{1,80})\]\([^)]*\)", r"\1", text)

    # Table/separator noise from legal PDFs.
    text = text.replace("| | |", " ")
    text = re.sub(r"\|{2,}", " ", text)
    text = re.sub(r"-{4,}", " ", text)
    text = re.sub(r"_{4,}", " ", text)

    # Remove duplicate filename slug headings.
    text = re.sub(r"(?im)^\s*#{1,6}\s*[a-z0-9]+(?:-[a-z0-9]+){4,}\s*$", "", text)
    text = re.sub(r"(?im)^\s*[a-z0-9]+(?:-[a-z0-9]+){5,}\s*$", "", text)

    # Keep headings as text, but not markdown headings.
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = clean_markdown(text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def infer_type(path: Path) -> str:
    path_str = str(path).lower()

    if "news" in path_str:
        return "news"

    if "legal" in path_str:
        return "legal"

    return "unknown"


def load_markdown_files() -> list[Path]:
    return sorted(STANDARDIZED_DIR.rglob("*.md"))


def build_index() -> list[dict]:
    all_chunks = []

    files = load_markdown_files()

    for file_path in files:
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
        doc_type = infer_type(file_path)
        source_name = file_path.name

        chunks = chunk_text(raw_text)

        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "content": chunk,
                    "embedding": hash_embedding(chunk, dim=EMBED_DIM),
                    "metadata": {
                        "source": source_name,
                        "path": str(file_path),
                        "type": doc_type,
                        "source_type": doc_type,
                        "chunk_id": f"{file_path.stem}_{chunk_index:04d}",
                        "chunk_index": chunk_index,
                    },
                }
            )

    return all_chunks


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = build_index()

    chunks_path = INDEX_DIR / "chunks_with_embeddings.json"
    config_path = INDEX_DIR / "index_config.json"
    manifest_path = INDEX_DIR / "pageindex_manifest.json"

    with chunks_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    config = {
        "created_at": datetime.now().isoformat(),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding": "local-hashing-word-char-ngram-v1",
        "embedding_dim": EMBED_DIM,
        "vector_store": "local_json",
        "num_documents": len(load_markdown_files()),
        "num_chunks": len(chunks),
        "dataset_policy": "curated_clean_group_dataset",
    }

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "num_documents": len(load_markdown_files()),
        "num_chunks": len(chunks),
        "mode": "curated_local_vectorless_compatible",
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("Built group index")
    print(f"Documents: {len(load_markdown_files())}")
    print(f"Chunks: {len(chunks)}")
    print(f"Saved: {chunks_path}")
    print(f"Saved: {config_path}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
