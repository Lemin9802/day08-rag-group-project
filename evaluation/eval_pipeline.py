import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation import generate_with_citation
from src.retrieval_pipeline import retrieve
from src.utils import metadata, source_name, overlap, write_json

GOLDEN = ROOT / "evaluation" / "golden_dataset.json"
DETAILS = ROOT / "evaluation" / "eval_details.json"
REPORT = ROOT / "evaluation" / "results.md"


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


def context_precision(retrieved, expected_context):
    if not retrieved:
        return 0.0
    sources = [str(metadata(item).get("source") or source_name(item)) for item in retrieved]
    hits = sum(1 for src in sources if any(source_match(src, expected) for expected in expected_context))
    return hits / len(sources)


def citation_coverage(answer):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。])\s+", answer) if s.strip()]
    factual = [s for s in sentences if len(s) > 20]
    if not factual:
        return 0.0
    return sum(1 for s in factual if re.search(r"\[Document\s+\d+\]", s)) / len(factual)


def eval_one(item):
    retrieved = retrieve(item["question"], top_k=8)
    generated = generate_with_citation(item["question"], top_k=8)
    answer = generated.get("answer", "")
    return {
        "id": item["id"],
        "category": item.get("category"),
        "contributed_by": item.get("contributed_by"),
        "question": item["question"],
        "context_recall": context_recall(retrieved, item.get("expected_context", [])),
        "context_precision": context_precision(retrieved, item.get("expected_context", [])),
        "citation_coverage": citation_coverage(answer),
        "faithfulness_proxy": citation_coverage(answer),
        "answer_relevance": overlap(item.get("expected_answer", ""), answer),
        "generation_mode": generated.get("generation_mode"),
        "retrieved_sources": [metadata(item).get("source") or source_name(item) for item in retrieved],
        "answer": answer,
    }


def average(rows, key):
    return sum(float(row.get(key, 0)) for row in rows) / len(rows) if rows else 0.0


def write_report(rows):
    metrics = ["context_recall", "context_precision", "faithfulness_proxy", "citation_coverage", "answer_relevance"]
    lines = ["# Evaluation Results", "", "## Overall Scores", "", "| Metric | Score |", "|---|---:|"]
    for metric in metrics:
        lines.append(f"| {metric} | {average(rows, metric):.3f} |")
    lines += ["", "## Per Question", "", "| ID | Category | By | Recall | Precision | Faithfulness | Citation | Relevance |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['id']} | {row['category']} | {row['contributed_by']} | {row['context_recall']:.3f} | {row['context_precision']:.3f} | {row['faithfulness_proxy']:.3f} | {row['citation_coverage']:.3f} | {row['answer_relevance']:.3f} |")
    worst = sorted(rows, key=lambda row: row["context_recall"] + row["citation_coverage"] + row["answer_relevance"])[:5]
    lines += ["", "## Worst Performers", "", "| ID | Problem | Proposed Fix |", "|---|---|---|"]
    for row in worst:
        lines.append(f"| {row['id']} | Low combined score | Improve query expansion, reranking, metadata, or source chunks |")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    rows = []
    for item in load_golden():
        print("Evaluating", item["id"])
        rows.append(eval_one(item))
    write_json(DETAILS, rows)
    write_report(rows)
    print("Saved", DETAILS)
    print("Saved", REPORT)


if __name__ == "__main__":
    main()
