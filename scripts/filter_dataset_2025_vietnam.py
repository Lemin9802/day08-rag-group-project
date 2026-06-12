from __future__ import annotations

import json
from pathlib import Path
import re
from collections import Counter, defaultdict

SRC = Path("data/MaiThuyLaw_clean_context_dataset/data/index/rag_chunks.json")
OUT_DIR = Path("data/MaiThuyLaw_vietnam_2025plus_dataset/data/index")
OUT = OUT_DIR / "rag_chunks.json"
REPORT = OUT_DIR / "filter_report.json"

VIETNAM_TERMS = [
    "việt nam", "viet nam", "cộng hòa xã hội chủ nghĩa việt nam",
    "quốc hội", "chính phủ", "thủ tướng", "bộ công an", "bộ y tế",
    "tòa án", "toà án", "viện kiểm sát", "ubtvqh", "ủy ban thường vụ quốc hội",
    "nghị định", "thông tư", "pháp lệnh", "luật", "quyết định",
    "bocongan.gov.vn", "pcmatuy.bocongan.gov.vn", "baochinhphu.vn",
    "chinhphu.vn", "tiengchuong.chinhphu.vn", "cand.vn",
    "nhandan.vn", "vietnamplus.vn", "tapchitoaan.vn", "vbpl.vn",
]

DRUG_TERMS = [
    "ma túy", "ma tuý", "ma tuy", "chất ma túy", "chất ma tuý",
    "narcotic", "drug", "tiền chất", "cai nghiện", "sau cai",
    "phòng chống ma túy", "phòng, chống ma túy",
    "tàng trữ", "vận chuyển", "mua bán", "sản xuất trái phép",
    "sử dụng trái phép", "người nghiện", "người sử dụng trái phép",
    "thuốc lá điện tử", "bóng cười", "n2o",
]

DATE_PATTERNS = [
    re.compile(r"\b(20[2-9][5-9]|20[3-9][0-9])\b"),
    re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-](20[2-9][5-9]|20[3-9][0-9])\b"),
    re.compile(r"\b(20[2-9][5-9]|20[3-9][0-9])/(?:QH|TT|NĐ|ND|NQ|UBTVQH|BTC|BCA)", re.I),
]


def norm(x: object) -> str:
    return str(x or "").lower()


def chunk_text(chunk: dict) -> str:
    meta = chunk.get("metadata", {}) or {}
    parts = [
        chunk.get("content", ""),
        json.dumps(meta, ensure_ascii=False),
    ]
    return "\n".join(str(p or "") for p in parts).lower()


def doc_key(chunk: dict) -> str:
    meta = chunk.get("metadata", {}) or {}
    return (
        meta.get("doc_id")
        or meta.get("title")
        or meta.get("source")
        or meta.get("path")
        or chunk.get("chunk_id")
        or "unknown"
    )


def has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def has_2025_plus(text: str) -> bool:
    return any(p.search(text) for p in DATE_PATTERNS)


def source_type(chunk: dict) -> str:
    meta = chunk.get("metadata", {}) or {}
    return meta.get("source_type") or meta.get("type") or "unknown"


def title_of(chunks: list[dict]) -> str:
    meta = chunks[0].get("metadata", {}) or {}
    return meta.get("title") or meta.get("source") or meta.get("doc_id") or "unknown"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing source dataset: {SRC}")

    chunks = json.loads(SRC.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise SystemExit("rag_chunks.json must be a list")

    docs = defaultdict(list)
    for c in chunks:
        docs[doc_key(c)].append(c)

    kept = []
    doc_rows = []
    reason_counter = Counter()

    for key, doc_chunks in docs.items():
        full_text = "\n".join(chunk_text(c) for c in doc_chunks)

        vietnam = has_any(full_text, VIETNAM_TERMS)
        drug = has_any(full_text, DRUG_TERMS)
        recent = has_2025_plus(full_text)

        if vietnam and drug and recent:
            decision = "kept"
            kept.extend(doc_chunks)
        else:
            missing = []
            if not vietnam:
                missing.append("not_vietnam")
            if not drug:
                missing.append("not_drug_related")
            if not recent:
                missing.append("not_2025_plus")
            decision = "+".join(missing)

        reason_counter[decision] += 1

        doc_rows.append({
            "doc_key": key,
            "title": title_of(doc_chunks),
            "source_type": source_type(doc_chunks[0]),
            "chunks": len(doc_chunks),
            "vietnam": vietnam,
            "drug_related": drug,
            "has_2025_plus": recent,
            "decision": decision,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "source_dataset": str(SRC),
        "output_dataset": str(OUT),
        "total_chunks_before": len(chunks),
        "total_docs_before": len(docs),
        "total_chunks_after": len(kept),
        "total_docs_after": sum(1 for r in doc_rows if r["decision"] == "kept"),
        "decisions": dict(reason_counter),
        "kept_docs": [r for r in doc_rows if r["decision"] == "kept"],
        "excluded_docs": [r for r in doc_rows if r["decision"] != "kept"],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== FILTER DONE =====")
    print("Before chunks:", len(chunks))
    print("Before docs:", len(docs))
    print("After chunks:", len(kept))
    print("After docs:", report["total_docs_after"])
    print("Output:", OUT)
    print("Report:", REPORT)

    print("\n===== Decisions =====")
    for k, v in reason_counter.most_common():
        print(f"{k}: {v}")

    print("\n===== Kept docs =====")
    for r in report["kept_docs"]:
        print(f"- [{r['source_type']}] {r['title']} | chunks={r['chunks']}")

    print("\n===== Excluded docs preview =====")
    for r in report["excluded_docs"][:20]:
        print(f"- {r['decision']} | [{r['source_type']}] {r['title']} | chunks={r['chunks']}")


if __name__ == "__main__":
    main()
