# Evaluation Results

## Overall Scores

| Metric | Score |
|---|---:|
| context_recall | 0.933 |
| context_precision | 0.300 |
| faithfulness_proxy | 0.359 |
| citation_coverage | 0.359 |
| answer_relevance | 0.798 |

## Per Question

| ID | Category | By | Recall | Precision | Faithfulness | Citation | Relevance |
|---|---|---|---:|---:|---:|---:|---:|
| nhi_q001 | legal | Nhi | 1.000 | 0.375 | 0.333 | 0.333 | 1.000 |
| nhi_q002 | legal | Nhi | 1.000 | 0.500 | 0.286 | 0.286 | 0.714 |
| nhi_q003 | legal | Nhi | 1.000 | 0.375 | 0.429 | 0.429 | 0.886 |
| nhi_q004 | news | Nhi | 1.000 | 0.125 | 1.000 | 1.000 | 0.444 |
| nhi_q005 | news | Nhi | 1.000 | 0.125 | 0.250 | 0.250 | 0.867 |
| huy_q001 | legal | Huy | 1.000 | 0.375 | 0.333 | 0.333 | 0.848 |
| huy_q002 | legal | Huy | 1.000 | 0.250 | 0.750 | 0.750 | 0.818 |
| huy_q003 | news | Huy | 0.000 | 0.000 | 0.231 | 0.231 | 0.810 |
| huy_q004 | legal | Huy | 1.000 | 0.375 | 0.167 | 0.167 | 0.703 |
| huy_q005 | legal | Huy | 1.000 | 0.375 | 0.429 | 0.429 | 1.000 |
| nghia_q001 | legal | Nghia | 1.000 | 0.375 | 0.200 | 0.200 | 1.000 |
| nghia_q002 | news | Nghia | 1.000 | 0.375 | 0.250 | 0.250 | 0.824 |
| nghia_q003 | news | Nghia | 1.000 | 0.125 | 0.125 | 0.125 | 0.533 |
| nghia_q004 | news | Nghia | 1.000 | 0.375 | 0.300 | 0.300 | 0.750 |
| nghia_q005 | legal | Nghia | 1.000 | 0.375 | 0.300 | 0.300 | 0.774 |

## Worst Performers

| ID | Problem | Proposed Fix |
|---|---|---|
| huy_q003 | Low combined score | Improve query expansion, reranking, metadata, or source chunks |
| nghia_q003 | Low combined score | Improve query expansion, reranking, metadata, or source chunks |
| huy_q004 | Low combined score | Improve query expansion, reranking, metadata, or source chunks |
| nhi_q002 | Low combined score | Improve query expansion, reranking, metadata, or source chunks |
| nghia_q004 | Low combined score | Improve query expansion, reranking, metadata, or source chunks |