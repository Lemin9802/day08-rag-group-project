from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

import chainlit as cl

from components.cards import render_sources
from components.ui_helpers import DEMO_QUESTIONS, friendly_error, settings_markdown, welcome_markdown
from rag.data_loader import dataset_summary
from rag.generator import generate_answer
from rag.memory import ConversationMemory
from rag.retriever import retrieve


DEFAULT_TOP_K = int(os.getenv("TOP_K", "8"))
DEFAULT_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.05"))
DEFAULT_DEBUG = os.getenv("SHOW_DEBUG", "false").lower() == "true"


def _session_memory() -> ConversationMemory:
    memory = cl.user_session.get("memory")
    if memory is None:
        memory = ConversationMemory()
        cl.user_session.set("memory", memory)
    return memory


async def _send_sidebar(summary: dict) -> None:
    top_k = cl.user_session.get("top_k", DEFAULT_TOP_K)
    threshold = cl.user_session.get("score_threshold", DEFAULT_THRESHOLD)
    mode = cl.user_session.get("retrieval_mode", "Hybrid")
    rerank = cl.user_session.get("rerank", True)
    await cl.Message(
        content=settings_markdown(summary, top_k, threshold, mode, rerank),
        author="Bảng retrieval",
    ).send()


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("top_k", DEFAULT_TOP_K)
    cl.user_session.set("score_threshold", DEFAULT_THRESHOLD)
    cl.user_session.set("retrieval_mode", "Hybrid")
    cl.user_session.set("rerank", True)
    cl.user_session.set("answer_mode", "Chi tiết")
    cl.user_session.set("show_debug", DEFAULT_DEBUG)
    cl.user_session.set("memory", ConversationMemory())

    try:
        settings = await cl.ChatSettings(
            [
                cl.input_widget.Slider("top_k", label="Số nguồn trả về", initial=DEFAULT_TOP_K, min=3, max=10, step=1),
                cl.input_widget.Slider(
                    "score_threshold",
                    label="Ngưỡng điểm",
                    initial=DEFAULT_THRESHOLD,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                ),
                cl.input_widget.Select(
                    "retrieval_mode",
                    label="Chế độ retrieval",
                    values=["Hybrid", "Lexical Only", "Semantic Only"],
                    initial_index=0,
                ),
                cl.input_widget.Switch("rerank", label="Rerank đơn giản", initial=True),
                cl.input_widget.Select(
                    "answer_mode",
                    label="Kiểu trả lời",
                    values=["Ngắn gọn", "Chi tiết", "Dạng bullet", "Dạng so sánh"],
                    initial_index=1,
                ),
                cl.input_widget.Switch("show_debug", label="Hiện debug retrieval", initial=DEFAULT_DEBUG),
            ]
        ).send()
        for key, value in settings.items():
            cl.user_session.set(key, value)
    except Exception:
        pass

    summary = dataset_summary()
    actions = [
        cl.Action(name="demo_question", payload={"question": q}, label=q[:64])
        for q in DEMO_QUESTIONS[:3]
    ]
    actions.append(cl.Action(name="clear_conversation", payload={}, label="Xóa hội thoại"))
    await cl.Message(content=welcome_markdown(summary), actions=actions).send()
    await _send_sidebar(summary)


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    for key, value in settings.items():
        cl.user_session.set(key, value)
    await cl.Message(content="Đã cập nhật cài đặt retrieval.", author="System").send()


@cl.action_callback("demo_question")
async def on_demo_question(action: cl.Action) -> None:
    await handle_user_text(str(action.payload.get("question", "")))


@cl.action_callback("clear_conversation")
async def on_clear(_: cl.Action) -> None:
    _session_memory().clear()
    await cl.Message(content="Đã xóa bộ nhớ hội thoại.", author="System").send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    await handle_user_text(message.content)


async def handle_user_text(raw_query: str) -> None:
    query = (raw_query or "").strip()
    if not query:
        await cl.Message(content="Hãy nhập một câu hỏi cụ thể hơn nhé.", author="LegalRAG").send()
        return

    memory = _session_memory()
    rewritten_query = memory.rewrite_query(query)
    top_k = int(cl.user_session.get("top_k", DEFAULT_TOP_K))
    threshold = float(cl.user_session.get("score_threshold", DEFAULT_THRESHOLD))
    mode = cl.user_session.get("retrieval_mode", "Hybrid")
    rerank = bool(cl.user_session.get("rerank", True))
    answer_mode = cl.user_session.get("answer_mode", "Chi tiết")
    show_debug = bool(cl.user_session.get("show_debug", DEFAULT_DEBUG))

    thinking = cl.Message(content="Đang truy xuất tài liệu và đối chiếu nguồn...", author="LegalRAG")
    await thinking.send()

    try:
        retrieval = retrieve(
            rewritten_query,
            top_k=top_k,
            score_threshold=threshold,
            mode=mode,
            rerank=rerank,
        )
        sources = retrieval["results"]
        generated = await generate_answer(query, sources, answer_mode=answer_mode, min_score=max(threshold * 0.4, 0.05))
        answer = generated["answer"]

        await thinking.remove()
        await cl.Message(content=answer, author="LegalRAG").send()
        await cl.Message(content=render_sources(sources), author="Nguồn tham khảo").send()

        if show_debug:
            debug = retrieval.get("debug", {})
            debug_md = f"""
### Debug Retrieval
- Câu hỏi gốc: `{debug.get("original_query", query)}`
- Câu hỏi sau rewrite: `{rewritten_query}`
- Mode: `{debug.get("mode", mode)}`
- Pipeline đã dùng: `{retrieval.get("used")}`
- Có dùng PageIndex: `{debug.get("pageindex_used", False)}`
- Lỗi pipeline cá nhân: `{debug.get("individual_pipeline_error") or "không có"}`
- Lý do fallback: `{debug.get("fallback_reason") or "không có"}`
- Kết quả lexical: `{len(debug.get("lexical_results", []))}`
- Kết quả semantic: `{len(debug.get("semantic_results", []))}`
- Kết quả sau rerank: `{len(debug.get("reranked_results", []))}`
""".strip()
            await cl.Message(content=debug_md, author="Debug").send()

        memory.add(query, answer)
    except Exception as exc:
        await thinking.remove()
        await cl.Message(content=friendly_error(str(exc)), author="LegalRAG").send()
