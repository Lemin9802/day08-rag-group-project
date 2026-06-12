from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

SRC = Path("data/MaiThuyLaw_vietnam_2025plus_dataset/data/index/rag_chunks.json")
OUT_DIR = Path("data/maithuylaw_dataset/data/index")
OUT = OUT_DIR / "rag_chunks.json"
REPORT = OUT_DIR / "neutral_dataset_report.json"


def strip_personal_prefix(value: str) -> str:
    value = str(value or "")
    value = re.sub(r"^(huy|nhi|nghia|nghĩa|nhi_|huy_|nghia_)[-_]+", "", value, flags=re.I)
    value = re.sub(r"\b(huy|nhi|nghia|nghĩa)[-_]+", "", value, flags=re.I)
    return value.strip()


def title_case_clean(value: str) -> str:
    value = strip_personal_prefix(value)
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value).strip()

    replacements = {
        "bo luat hinh su 2015": "Bộ luật Hình sự 2015",
        "luat phong chong ma tuy 2021": "Luật Phòng, chống ma túy 2021",
        "nghi dinh 28 2026 nd cp danh muc chat ma tuy tien chat": "Nghị định 28/2026/NĐ-CP - Danh mục chất ma túy và tiền chất",
        "quyet dinh 28 2025 qd ttg dia ban trong diem ma tuy": "Quyết định 28/2025/QĐ-TTg - Địa bàn trọng điểm về ma túy",
        "thong tu lien tich 03 2025 bca vksndtc tandtc co so cai nghien bat buoc": "Thông tư liên tịch 03/2025 - Cơ sở cai nghiện bắt buộc",
    }

    key = value.lower()
    return replacements.get(key, value)


def slugify(value: str) -> str:
    value = strip_personal_prefix(value)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = value.replace("đ", "d")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value[:90] or "document"


def get_doc_key(chunk: dict) -> str:
    meta = chunk.get("metadata", {}) or {}
    return (
        meta.get("doc_id")
        or meta.get("title")
        or meta.get("source")
        or meta.get("path")
        or chunk.get("chunk_id")
        or "unknown"
    )


def get_source_type(chunk: dict) -> str:
    meta = chunk.get("metadata", {}) or {}
    raw = (meta.get("source_type") or meta.get("type") or "").lower()
    if raw in {"legal", "news"}:
        return raw
    text = json.dumps(chunk, ensure_ascii=False).lower()
    if any(x in text for x in ["luật", "nghị định", "thông tư", "pháp lệnh", "quyết định"]):
        return "legal"
    return "news"


def get_title(chunks: list[dict]) -> str:
    meta = chunks[0].get("metadata", {}) or {}
    title = meta.get("title") or meta.get("source") or meta.get("doc_id") or "unknown"
    return title_case_clean(title)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing source dataset: {SRC}")

    chunks = json.loads(SRC.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise SystemExit("Source rag_chunks.json must be a list.")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        grouped[get_doc_key(chunk)].append(chunk)

    final_chunks = []
    docs_report = []
    used_doc_ids = set()

    sorted_docs = sorted(grouped.items(), key=lambda kv: (get_source_type(kv[1][0]), get_title(kv[1]).lower()))

    for doc_index, (old_key, doc_chunks) in enumerate(sorted_docs, start=1):
        source_type = get_source_type(doc_chunks[0])
        title = get_title(doc_chunks)

        base_slug = slugify(title)
        doc_id = f"{source_type}-{base_slug}"

        if doc_id in used_doc_ids:
            suffix = 2
            while f"{doc_id}-{suffix}" in used_doc_ids:
                suffix += 1
            doc_id = f"{doc_id}-{suffix}"

        used_doc_ids.add(doc_id)

        neutral_path = f"data/maithuylaw_dataset/{source_type}/{doc_id}.md"

        for chunk_index, chunk in enumerate(doc_chunks, start=1):
            old_meta = chunk.get("metadata", {}) or {}
            new_meta = dict(old_meta)

            # Remove old personal/group-ish naming from metadata.
            new_meta["doc_id"] = doc_id
            new_meta["chunk_id"] = f"{doc_id}_{chunk_index:04d}"
            new_meta["title"] = title
            new_meta["source"] = title
            new_meta["path"] = neutral_path
            new_meta["source_type"] = source_type
            new_meta["type"] = source_type
            new_meta["dataset"] = "maithuylaw_dataset"
            new_meta["scope"] = "vietnam_2025_plus"
            new_meta["normalized"] = True

            new_chunk = dict(chunk)
            new_chunk["chunk_id"] = f"{doc_id}_{chunk_index:04d}"
            new_chunk["metadata"] = new_meta
            final_chunks.append(new_chunk)

        docs_report.append({
            "old_key": old_key,
            "doc_id": doc_id,
            "title": title,
            "source_type": source_type,
            "chunks": len(doc_chunks),
            "path": neutral_path,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(final_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "source_dataset": str(SRC),
        "output_dataset": str(OUT),
        "chunks": len(final_chunks),
        "documents": len(docs_report),
        "legal_documents": sum(1 for d in docs_report if d["source_type"] == "legal"),
        "news_documents": sum(1 for d in docs_report if d["source_type"] == "news"),
        "docs": docs_report,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== NEUTRAL DATASET DONE =====")
    print("Output:", OUT)
    print("Chunks:", report["chunks"])
    print("Documents:", report["documents"])
    print("Legal docs:", report["legal_documents"])
    print("News docs:", report["news_documents"])

    print("\n===== Docs =====")
    for d in docs_report:
        print(f"- [{d['source_type']}] {d['doc_id']} | {d['title']} | chunks={d['chunks']}")


if __name__ == "__main__":
    main()
