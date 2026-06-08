# Group Project — LegalRAG Assistant

Web app group project cho Day08 RAG Pipeline v2. Sản phẩm cuối dùng **Chainlit** làm giao diện chatbot/search giống ChatGPT, backend dùng hybrid retrieval + citation generation, có evaluation 15 câu và A/B testing.

## Dataset

Dataset group được **curate** từ đóng góp của cả 3 thành viên, không copy toàn bộ dữ liệu thô để tránh trùng lặp và nhiễu retrieval.

- Tổng số tài liệu: **16 markdown documents**
- Legal: **6** văn bản
- News: **10** bài báo
- Index: **550 chunks**
- Chi tiết nguồn giữ/lọc: `docs/dataset_inventory.json`

Đóng góp dữ liệu:

- **Nhi:** Nghị định 28/2026, Quyết định 28/2025, Thông tư liên tịch 03/2025, 5 bài báo về Long Nhật/Sơn Ngọc Minh, Bình Gold, Miu Lê, Hữu Tín, Chi Dân.
- **Huy:** Luật Phòng, chống ma túy 2021, Bộ luật Hình sự 2015 phần tội phạm ma túy, bài Tuổi Trẻ về Hữu Tín.
- **Nghia:** Nghị định 105/2021, các bài báo về Chi Dân/An Tây/Nguyễn Đỗ Trúc Phương, DJ Thái Hoàng, nữ DJ liên quan ma túy.

## Chức năng

- RAG Chatbot tiếng Việt có citation `[S1]`, `[S2]`.
- Source cards hiển thị nguồn được truy xuất.
- Hybrid retrieval: dense hashing + BM25 + RRF + reranking + fallback.
- Query rewriting / conversation memory nhẹ.
- Gemini generation khi có quota; extractive fallback nếu Gemini lỗi/quota hết.
- Evaluation 15 câu, có Context Recall, Context Precision, Faithfulness proxy, Answer Relevance.
- A/B test: Hybrid vs BM25-only vs vectorless fallback.

## Setup

```bash
cd group_project
python -m venv .venv_chainlit
.venv_chainlit\Scripts\activate
python -m pip install -r requirements.txt
```

Tạo `.env` từ `.env.example` nếu dùng Gemini:

```env
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash-lite
TOP_K=8
SCORE_THRESHOLD=0.05
```

Nếu Gemini bị quota hoặc chưa có key, app vẫn chạy bằng fallback extractive.

## Run Chainlit App

```bash
python -m chainlit run chainlit_app.py
```

Mở:

```text
http://localhost:8000
```

## Optional Streamlit App

```bash
python -m streamlit run app.py
```

## Rebuild Index

Sau khi thay đổi `data/standardized/`, chạy:

```bash
python -m src.build_index
```

Expected output hiện tại:

```text
Documents: 16
Chunks: 550
```

## Evaluation

```bash
python evaluation/eval_pipeline.py
python evaluation/ab_test.py
```

Kết quả hiện tại:

```text
Context Recall: 1.000
Answer Relevance: ~0.787
Best A/B config: A_hybrid
```

File kết quả:

```text
evaluation/results.md
evaluation/ab_test_results.md
evaluation/eval_details.json
evaluation/ab_test_details.json
```

## Demo Questions

```text
Nghị định 28/2026 quy định gì về danh mục chất ma túy và tiền chất?
Bộ nào quản lý tiền chất sử dụng trong lĩnh vực công nghiệp?
Cơ sở cai nghiện bắt buộc được nhắc đến như thế nào?
Những nghệ sĩ nào trong dữ liệu liên quan tới ma túy?
Bài báo về Bình Gold nói gì về việc dương tính với ma túy?
Luật Phòng, chống ma túy 2021 quy định về nội dung gì?
```

## Structure

```text
group_project/
├── chainlit_app.py
├── app.py
├── requirements.txt
├── .env.example
├── components/
├── rag/
├── src/
├── data/
│   ├── standardized/
│   └── index/
├── evaluation/
├── docs/
└── screenshots/
```

## Team Contribution

| Member | Main contribution |
|---|---|
| Nhi | Base dataset, generation, evaluation/golden questions, integration |
| Huy | Chainlit UI, retrieval UX, legal/news sources, Huy golden questions |
| Nghia | Legal/news sources, Nghia golden questions, evaluation validation |
