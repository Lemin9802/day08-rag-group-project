# Architecture

```text
User Query
-> Retrieval Pipeline
-> Dense Search + BM25 + RRF + Query-aware Reranking
-> Vectorless Fallback
-> Context Formatter
-> Gemini / Extractive Fallback
-> Cited Answer
```

## Components

- `app.py`: Streamlit UI.
- `src/retrieval_pipeline.py`: retrieval backend.
- `src/generation.py`: generation with citations.
- `evaluation/`: golden dataset, metrics, and A/B testing.
