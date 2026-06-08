from __future__ import annotations

APP_TITLE = "LegalRAG Assistant"
APP_SUBTITLE = "Hỏi đáp pháp luật ma túy có trích dẫn nguồn"

DEMO_QUESTIONS = [
    "Nghị định 28/2026 quy định gì về danh mục chất ma túy và tiền chất?",
    "Bộ nào quản lý tiền chất sử dụng trong lĩnh vực công nghiệp?",
    "Cơ sở cai nghiện bắt buộc được nhắc đến như thế nào?",
    "Những nghệ sĩ nào trong dữ liệu liên quan tới ma túy?",
    "Bài báo về Bình Gold nói gì về việc dương tính với ma túy?",
    "Luật Phòng, chống ma túy 2021 quy định về nội dung gì?",
]

LEGAL_NOTICE = (
    "Thông tin chỉ phục vụ học tập và tra cứu từ tài liệu đã cung cấp, "
    "không thay thế tư vấn pháp lý chính thức."
)


def badge_row() -> str:
    badges = ["Hybrid Retrieval", "Có trích dẫn", "Ghi nhớ hội thoại", "Sẵn sàng đánh giá"]
    return "  ".join(f"`{label}`" for label in badges)


def welcome_markdown(summary: dict) -> str:
    legal = summary.get("legal_docs", 0)
    news = summary.get("news_docs", 0)
    chunks = summary.get("chunks", 0)
    questions = "\n".join(f"- {q}" for q in DEMO_QUESTIONS[:4])
    return f"""
# LegalRAG Assistant

**Hỏi đáp pháp luật ma túy có trích dẫn nguồn**

{badge_row()}

**Bộ dữ liệu:** `{legal}` văn bản pháp luật, `{news}` bài báo, `{chunks}` chunks.

### Chức năng
| Tra cứu pháp luật | Hỏi đáp theo nguồn | Phân tích tin tức thận trọng |
|---|---|---|
| Tìm quy định trong văn bản đã nạp. | Mỗi kết luận quan trọng đều kèm citation. | Tin tức là nguồn báo chí, không thay thế kết luận pháp lý. |

### Câu hỏi gợi ý
{questions}

> {LEGAL_NOTICE}
""".strip()


def settings_markdown(summary: dict, top_k: int, threshold: float, mode: str, rerank: bool) -> str:
    return f"""
### Tóm tắt dữ liệu
- Văn bản pháp luật: **{summary.get("legal_docs", 0)}**
- Bài báo: **{summary.get("news_docs", 0)}**
- Chunks: **{summary.get("chunks", 0)}**

### Cài đặt retrieval
- `top_k`: **{top_k}**
- `score_threshold`: **{threshold:.2f}**
- `mode`: **{mode}**
- `rerank`: **{"on" if rerank else "off"}**
""".strip()


def friendly_error(message: str) -> str:
    return f"""
## Có lỗi nhỏ xảy ra
{message}

Ứng dụng vẫn đang chạy. Hãy thử đặt câu hỏi cụ thể hơn hoặc kiểm tra lại dữ liệu trong `data/standardized`.
""".strip()
