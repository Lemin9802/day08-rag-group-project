from __future__ import annotations

import re

from .config import GEMINI_API_KEY, GEMINI_MODEL
from .retrieval_pipeline import retrieve
from .utils import clip, metadata, source_name

SYSTEM_PROMPT = """Bạn là trợ lý RAG tiếng Việt cho hệ thống tra cứu pháp luật và tin tức về ma túy.
Chỉ trả lời dựa trên CONTEXT được cung cấp. Mỗi ý quan trọng phải có citation dạng [Document i].
Nếu context không đủ bằng chứng, nói: Tôi không thể xác minh thông tin này từ nguồn hiện có."""


def reorder_context(chunks):
    return chunks if len(chunks) <= 2 else chunks[0::2] + chunks[1::2][::-1]


def format_context(chunks):
    parts = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = metadata(chunk)
        parts.append(
            f"[Document {idx}]\n"
            f"Source: {meta.get('source') or source_name(chunk)}\n"
            f"Type: {meta.get('type', 'unknown')}\n"
            f"Score: {float(chunk.get('score', 0)):.4f}\n"
            f"Content:\n{clip(chunk.get('content', ''), 1800)}"
        )
    return "\n\n---\n\n".join(parts)


def extractive_answer(query, chunks):
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    names = ["Long Nhật", "Sơn Ngọc Minh", "Miu Lê", "Hữu Tín", "Chi Dân", "Bình Gold"]
    if any(x in query.lower() for x in ["nghệ sĩ", "ca sĩ", "rapper", "diễn viên", "những ai"]):
        found = []
        for name in names:
            docs = [f"[Document {idx}]" for idx, chunk in enumerate(chunks, start=1) if name.lower() in chunk.get("content", "").lower()]
            if docs:
                found.append(f"{name} {' '.join(docs[:2])}")
        if found:
            return "Các nghệ sĩ/người nổi tiếng được nhắc đến trong dữ liệu gồm: " + "; ".join(found) + "."
    lines = ["Dựa trên các nguồn đã truy xuất, thông tin liên quan nhất là:"]
    for idx, chunk in enumerate(chunks[:4], start=1):
        lines.append(f"- {clip(chunk.get('content', ''), 420)} [Document {idx}]")
    return "\n".join(lines)


def _call_gemini(query, context):
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"{SYSTEM_PROMPT}\n\nQUESTION:\n{query}\n\nCONTEXT:\n{context}\n\nTrả lời bằng tiếng Việt, có citation dạng [Document i]."
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    if not getattr(response, "text", None):
        raise RuntimeError("Gemini returned empty response")
    return response.text.strip()


def citation_coverage(answer):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。])\s+", answer) if s.strip()]
    factual = [s for s in sentences if len(s) > 20]
    if not factual:
        return 0.0
    return sum(1 for s in factual if re.search(r"\[Document\s+\d+\]", s)) / len(factual)


def generate_with_citation(query, top_k=8, mode="hybrid"):
    query = (query or "").strip()
    if not query:
        return {"answer": "Vui lòng nhập câu hỏi.", "sources": [], "retrieval_source": "none", "generation_mode": "none"}
    sources = retrieve(query, top_k=top_k, mode=mode)
    ordered = reorder_context(sources)
    context = format_context(ordered)
    generation_mode = "extractive_fallback"
    if GEMINI_API_KEY:
        try:
            answer = _call_gemini(query, context)
            generation_mode = "gemini"
        except Exception:
            answer = extractive_answer(query, ordered) + "\n\n_Ghi chú: Gemini không khả dụng, hệ thống dùng fallback extractive._"
            generation_mode = "extractive_fallback_after_gemini_error"
    else:
        answer = extractive_answer(query, ordered)
    return {
        "answer": answer,
        "sources": ordered,
        "retrieval_source": sources[0].get("source", "none") if sources else "none",
        "generation_mode": generation_mode,
        "citation_coverage": citation_coverage(answer),
    }


if __name__ == "__main__":
    for q in ["Nghị định 28/2026 quy định gì?", "Những nghệ sĩ nào liên quan tới ma túy?"]:
        result = generate_with_citation(q)
        print("=" * 80)
        print(q)
        print(result["generation_mode"])
        print(result["answer"])
