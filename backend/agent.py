from __future__ import annotations

import os
import re
from typing import Any

LEGAL_NOTICE = (
    "Thông tin chỉ phục vụ tra cứu từ nguồn đã thu thập, không thay thế tư vấn pháp lý chính thức. "
    "Với vụ việc cụ thể, cần đối chiếu văn bản gốc hoặc hỏi cơ quan/chuyên gia có thẩm quyền."
)

NO_EVIDENCE = """## Trả lời
Mình chưa tìm thấy nguồn đủ mạnh trong dataset hoặc nội dung đính kèm để trả lời chắc chắn.

## Gợi ý
Bạn có thể hỏi cụ thể hơn về văn bản pháp luật, hành vi, chính sách, tin tức hoặc gửi nguồn chính thống hơn.

## Lưu ý pháp lý
{notice}
""".format(notice=LEGAL_NOTICE)


def _compact(text: str, max_chars: int = 900) -> str:
    cleaned = str(text or "")

    # Remove markdown/source-card noise.
    cleaned = re.sub(r"(?i)^\s*URL:\s*", " ", cleaned)
    cleaned = re.sub(r"(?i)\bURL:\s*", " ", cleaned)
    cleaned = re.sub(r"(?i)\*\*source:\*\*.*?(?=\*\*|$)", " ", cleaned)
    cleaned = re.sub(r"(?i)\*\*date:\*\*.*?(?=\*\*|$)", " ", cleaned)
    cleaned = re.sub(r"(?i)\*\*url:\*\*.*?(?=\*\*|$)", " ", cleaned)
    cleaned = re.sub(r"(?i)\*\*group:\*\*.*?(?=\*\*|$)", " ", cleaned)
    cleaned = re.sub(r"(?i)\*\*content level:\*\*.*?(?=\*\*|$)", " ", cleaned)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)

    # Avoid isolated domain-tail artifacts such as "vn Bài viết..."
    cleaned = re.sub(r"(?i)\b(com|vn|gov|org|net)\b\s+(?=[A-ZÀ-Ỹ])", " ", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _citation_sources(dataset_results: list[dict], attachments: list[dict]) -> str:
    lines = []

    for i, att in enumerate(attachments[:4], start=1):
        lines.append(
            f"[A{i}] {att.get('name') or att.get('url') or 'attachment'} "
            f"(verdict={att.get('verdict')}, domain={att.get('domain_score')}, official={att.get('official_score')})"
        )

    for i, item in enumerate(dataset_results[:6], start=1):
        meta = item.get("metadata", {}) or {}
        title = meta.get("title") or meta.get("source") or meta.get("doc_id") or "unknown"
        source_type = meta.get("source_type") or meta.get("type") or "unknown"
        score = float(item.get("score", 0.0))
        lines.append(f"[S{i}] {title} ({source_type}, score={score:.3f})")

    return "\n".join(lines)


def _evidence_context(dataset_results: list[dict], attachments: list[dict]) -> str:
    blocks = []

    for i, att in enumerate(attachments[:4], start=1):
        if att.get("verdict") not in {"accepted", "needs_review"}:
            blocks.append(
                f"[A{i}] REJECTED ATTACHMENT\n"
                f"Name: {att.get('name')}\n"
                f"Reason: {att.get('reason')}"
            )
            continue

        blocks.append(
            f"[A{i}] ATTACHMENT\n"
            f"Name: {att.get('name')}\n"
            f"Verdict: {att.get('verdict')}\n"
            f"Reason: {att.get('reason')}\n"
            f"Text: {_compact(att.get('text', ''), 1800)}"
        )

    for i, item in enumerate(dataset_results[:6], start=1):
        meta = item.get("metadata", {}) or {}
        title = meta.get("title") or meta.get("source") or meta.get("doc_id") or "unknown"
        source_type = meta.get("source_type") or meta.get("type") or "unknown"
        blocks.append(
            f"[S{i}] DATASET SOURCE\n"
            f"Title: {title}\n"
            f"Type: {source_type}\n"
            f"Score: {float(item.get('score', 0.0)):.3f}\n"
            f"Text: {_compact(item.get('content', ''), 1600)}"
        )

    return "\n\n---\n\n".join(blocks)


def _wants_source_check(message: str) -> bool:
    q = message.lower()
    return any(x in q for x in ["chính thống", "chinh thong", "đáng tin", "dang tin", "kiểm tra nguồn", "kiem tra nguon", "nguồn này"])


def _wants_summary(message: str) -> bool:
    q = message.lower()
    return any(x in q for x in ["tóm tắt", "tom tat", "ý chính", "y chinh", "nội dung chính", "noi dung chinh"])


def _wants_compare(message: str) -> bool:
    q = message.lower()
    return any(x in q for x in ["so sánh", "so sanh", "đối chiếu", "doi chieu", "theo luật", "theo luat", "quy định"])


def _fallback_answer(message: str, dataset_results: list[dict], attachments: list[dict]) -> str:
    usable_attachments = [a for a in attachments if a.get("verdict") in {"accepted", "needs_review"}]
    rejected_attachments = [a for a in attachments if a.get("verdict") == "rejected"]

    if not usable_attachments and not dataset_results:
        return NO_EVIDENCE

    lines = []

    if usable_attachments:
        lines.append("## Trả lời dựa trên nội dung đính kèm")
        lines.append("")

        if _wants_source_check(message):
            lines.append("### Đánh giá nguồn")
            for i, att in enumerate(usable_attachments[:4], start=1):
                lines.append(
                    f"- **[A{i}] {att.get('name')}**: `{att.get('verdict')}`. "
                    f"{att.get('reason')} "
                    f"(domain={att.get('domain_score')}, official={att.get('official_score')}, dataset_match={att.get('dataset_match_score')})."
                )
            lines.append("")

        if _wants_summary(message) or not (_wants_source_check(message) or _wants_compare(message)):
            lines.append("### Tóm tắt")
            for i, att in enumerate(usable_attachments[:3], start=1):
                lines.append(f"**[A{i}] {att.get('name')}**")
                text = _compact(att.get("text", ""), 1200)
                parts = re.split(r"(?<=[.!?])\s+", text)
                bullets = [p.strip() for p in parts if len(p.strip()) >= 35][:4]
                if not bullets and text:
                    bullets = [text]
                for bullet in bullets:
                    lines.append(f"- {bullet} [A{i}]")
                lines.append("")

    else:
        lines.append("## Trả lời từ dataset MaiThuyLaw")
        lines.append("")

    if rejected_attachments:
        lines.append("### File/link không được dùng")
        for att in rejected_attachments:
            lines.append(f"- **{att.get('name')}**: {att.get('reason')}")
        lines.append("")

    if dataset_results:
        lines.append("### Đối chiếu với nguồn MaiThuyLaw")
        for i, item in enumerate(dataset_results[:4], start=1):
            lines.append(f"- {_compact(item.get('content', ''), 450)} [S{i}]")
        lines.append("")

    lines.append("## Nguồn")
    src = _citation_sources(dataset_results, usable_attachments)
    lines.append(src if src else "- Chưa có nguồn đủ mạnh.")
    lines.append("")
    lines.append("## Lưu ý pháp lý")
    lines.append(LEGAL_NOTICE)

    return "\n".join(lines).strip()


def _postprocess_llm(text: str, dataset_results: list[dict], attachments: list[dict]) -> str:
    answer = str(text or "").strip()
    answer = re.sub(r"(?i)chain of thought:.*", "", answer)
    answer = re.sub(r"(?i)internal reasoning:.*", "", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()

    if not answer:
        return _fallback_answer("", dataset_results, attachments)

    has_citation = bool(re.search(r"\[(?:A|S)\d+\]", answer))
    if not has_citation and (dataset_results or attachments):
        answer += "\n\n## Nguồn\n" + _citation_sources(dataset_results, attachments)

    if "Lưu ý pháp lý" not in answer:
        answer += "\n\n## Lưu ý pháp lý\n" + LEGAL_NOTICE

    return answer.strip()


async def _gemini_answer(message: str, dataset_results: list[dict], attachments: list[dict]) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()

    if not api_key:
        return None

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        evidence = _evidence_context(dataset_results, attachments)

        prompt = f"""
Bạn là MaiThuyLaw AI, trợ lý tra cứu tiếng Việt về pháp luật, chính sách và tin tức chính thống liên quan đến ma túy.

NHIỆM VỤ:
- Trả lời đúng yêu cầu của người dùng.
- Nếu có file/link đính kèm, dùng nội dung đính kèm làm ngữ cảnh tạm thời.
- Đối chiếu với dataset MaiThuyLaw khi phù hợp.
- Mỗi nhận định quan trọng phải có citation dạng [A1] hoặc [S1].
- Phân biệt nguồn pháp luật và nguồn tin tức.
- Nếu nguồn tin tức chỉ nói "bị bắt/khởi tố", không kết luận người đó có tội nếu chưa có bản án/kết luận có thẩm quyền.
- Nếu thiếu bằng chứng, nói rõ chưa đủ căn cứ từ nguồn hiện có.
- Không bịa số điều, mức phạt, ngày tháng, cơ quan, hoặc tình tiết.
- Không hướng dẫn lách luật, né công an, che giấu hành vi, sản xuất, mua bán, vận chuyển, sử dụng trái phép chất ma túy.

FORMAT:
## Trả lời ngắn gọn
2-5 câu.

## Phân tích từ nguồn
3-6 bullet points, có citation.

## Nguồn
Liệt kê citation đã dùng.

## Lưu ý pháp lý
{LEGAL_NOTICE}

CÂU HỎI:
{message}

EVIDENCE:
{evidence}
""".strip()

        response = client.models.generate_content(model=model, contents=prompt)
        return _postprocess_llm(getattr(response, "text", ""), dataset_results, attachments)
    except Exception as exc:
        print(f"[MaiThuyLaw Agent] Gemini fallback because: {exc}")
        return None


async def generate_answer(
    *,
    message: str,
    dataset_results: list[dict],
    attachments: list[dict],
) -> str:
    llm = await _gemini_answer(message, dataset_results, attachments)
    if llm:
        return llm
    return _fallback_answer(message, dataset_results, attachments)
