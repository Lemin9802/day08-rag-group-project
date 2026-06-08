import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval_pipeline import retrieve
from src.utils import metadata, source_name, write_json

GOLDEN = ROOT / "evaluation" / "golden_dataset.json"
DETAILS = ROOT / "evaluation" / "ab_test_details.json"
REPORT = ROOT / "evaluation" / "ab_test_results.md"

CONFIGS = {
    "A_hybrid": {"mode": "hybrid", "top_k": 8},
    "B_lexical": {"mode": "lexical", "top_k": 8},
    "C_vectorless": {"mode": "vectorless", "top_k": 8},
}


def load_golden():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def source_match(src, expected):
    src = str(src).lower()
    expected = str(expected).lower()
    return expected in src or src in expected


def context_recall(retrieved, expected_context):
    if not expected_context:
        return 0.0
    sources = [str(metadata(item).get("source") or source_name(item)) for item in retrieved]
    hits = sum(1 for expected in expected_context if any(source_match(src, expected) for src in sources))
    return hits / len(expected_context)


def main():
    golden = load_golden()
    details = []
    summary = {}
    for name, config in CONFIGS.items():
        scores = []
        for item in golden:
            retrieved = retrieve(item["question"], top_k=config["top_k"], mode=config["mode"])
            score = context_recall(retrieved, item.get("expected_context", []))
            scores.append(score)
            details.append({
                "config": name,
                "id": item["id"],
                "question": item["question"],
                "context_recall": score,
                "retrieved_sources": [metadata(row).get("source") or source_name(row) for row in retrieved],
            })
        summary[name] = sum(scores) / len(scores) if scores else 0.0
    write_json(DETAILS, details)
    lines = ["# A/B Test Results", "", "| Config | Avg Context Recall |", "|---|---:|"]
    for name, score in summary.items():
        lines.append(f"| {name} | {score:.3f} |")
    if summary:
        lines += ["", f"Best config: **{max(summary, key=summary.get)}**."]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("Saved", DETAILS)
    print("Saved", REPORT)


if __name__ == "__main__":
    main()
