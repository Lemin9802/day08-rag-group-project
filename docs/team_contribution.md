# Team Contribution

| Thành viên | Đóng góp chính | Files liên quan |
|---|---|---|
| Nhi | Base dataset, backend generation, evaluation/golden dataset, integration | `src/generation.py`, `evaluation/`, `data/standardized/nhi_*` |
| Huy | Chainlit UI, chatbot UX, retrieval settings, Huy legal/news data | `chainlit_app.py`, `components/`, `rag/`, `data/standardized/huy_*` |
| Nghia | Additional legal/news data, evaluation questions, validation | `data/standardized/nghia_*`, `evaluation/golden_dataset.json` |

## Dataset Policy

Group dataset không copy toàn bộ dữ liệu thô của 3 người. Nhóm dùng cách **curated dataset**:

1. Giữ tài liệu sạch, liên quan trực tiếp đến chủ đề ma túy.
2. Loại file trùng nội dung hoặc crawl quá nhiễu.
3. Đảm bảo mỗi thành viên có nguồn đóng góp trong dataset cuối.
4. Rebuild một index chung sau khi lọc.

Chi tiết xem `docs/dataset_inventory.json`.
