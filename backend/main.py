from __future__ import annotations

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.agent import generate_answer
from backend.attachments import (
    get_attachment,
    list_attachments_for_chat,
    resolve_attachments,
    save_attachment,
)
from backend.config import APP_NAME, MIN_SCORE, TOP_K
from backend.dataset import dataset_summary, retrieve
from backend.file_checker import evaluate_uploaded_text, extract_upload_text, fetch_link_text
from backend.guards import detect_safety_issue, is_in_domain
from backend.security import auth_and_quota
from backend.realtime import extract_urls, is_official_or_allowed_url, realtime_enabled, realtime_unavailable_answer, wants_realtime
from backend.schemas import (
    ChatDetail,
    ChatRequest,
    ChatResponse,
    ChatSummary,
    CreateChatRequest,
    LinkAttachmentRequest,
    RenameChatRequest,
    Source,
)
from backend.store import (
    add_message,
    create_chat,
    delete_chat,
    ensure_chat,
    get_chat,
    list_chats,
    rename_chat,
    rewrite_with_memory,
)

app = FastAPI(title=f"{APP_NAME} Backend", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _source_label(item: dict, index: int) -> Source:
    meta = item.get("metadata", {}) or {}
    return Source(
        source_id=f"S{index}",
        title=meta.get("title") or meta.get("source") or meta.get("doc_id") or "unknown",
        source_type=meta.get("source_type") or meta.get("type") or "unknown",
        url=meta.get("url") or None,
        score=float(item.get("score", 0.0)),
    )


def _attachment_source(att: dict, index: int) -> Source:
    return Source(
        source_id=f"A{index}",
        title=att.get("name") or att.get("url") or "attachment",
        source_type=f"attachment:{att.get('verdict', 'unknown')}",
        url=att.get("url"),
        score=float(att.get("domain_score") or 0.0),
    )


def _compact(text: str, max_chars: int = 700) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _summarize_attachment(att: dict) -> list[str]:
    text = att.get("text", "")
    sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if len(s.strip()) > 35]
    selected = sentences[:4]
    if not selected and text:
        selected = [_compact(text, 500)]
    return [f"- {s}." for s in selected]


def _wants_summary(message: str) -> bool:
    q = message.lower()
    return any(x in q for x in ["tóm tắt", "tom tat", "ý chính", "y chinh", "nội dung chính", "noi dung chinh"])


def _wants_source_check(message: str) -> bool:
    q = message.lower()
    return any(x in q for x in ["chính thống", "chinh thong", "đáng tin", "dang tin", "kiểm tra nguồn", "kiem tra nguon", "nguồn này"])


def _wants_compare_law(message: str) -> bool:
    q = message.lower()
    return any(x in q for x in ["so sánh", "so sanh", "đối chiếu", "doi chieu", "theo luật", "theo luat", "quy định"])


def _format_dataset_answer(results: list[dict]) -> str:
    if not results or float(results[0].get("score", 0.0)) < MIN_SCORE:
        return (
            "Mình chưa tìm thấy nguồn đủ mạnh trong dataset hiện có để trả lời chắc chắn. "
            "Bạn có thể hỏi cụ thể hơn về văn bản pháp luật, hành vi, chính sách hoặc tin tức liên quan đến ma túy."
        )

    lines = [
        "## Trả lời từ nguồn hiện có",
        "",
        "Mình tìm được các nguồn liên quan trong dataset MaiThuyLaw:",
        "",
    ]

    for i, item in enumerate(results[:3], start=1):
        lines.append(f"**[S{i}]** {_compact(item.get('content', ''), 520)}")
        lines.append("")

    lines += [
        "## Lưu ý",
        "Thông tin này chỉ phục vụ tra cứu từ nguồn đã thu thập, không thay thế tư vấn pháp lý chính thức.",
    ]
    return "\n".join(lines).strip()


def _format_attachment_answer(message: str, attachments: list[dict], dataset_results: list[dict]) -> str:
    usable = [a for a in attachments if a.get("verdict") in {"accepted", "needs_review"}]
    rejected = [a for a in attachments if a.get("verdict") == "rejected"]

    lines = ["## Trả lời dựa trên nội dung đính kèm", ""]

    if rejected:
        lines.append("Một số file/link bị từ chối và không được dùng làm nguồn trả lời:")
        for att in rejected:
            lines.append(f"- **{att.get('name')}**: {att.get('reason')}")
        lines.append("")

    if not usable:
        lines.append("Không có file/link đủ an toàn và phù hợp để dùng làm ngữ cảnh trả lời.")
        return "\n".join(lines).strip()

    if _wants_source_check(message):
        lines.append("### Đánh giá nguồn")
        for i, att in enumerate(usable, start=1):
            lines.append(
                f"- **[A{i}] {att.get('name')}**: `{att.get('verdict')}`. "
                f"{att.get('reason')} "
                f"(domain={att.get('domain_score')}, official={att.get('official_score')}, dataset_match={att.get('dataset_match_score')})"
            )
        lines.append("")

    if _wants_summary(message) or not (_wants_source_check(message) or _wants_compare_law(message)):
        lines.append("### Tóm tắt nội dung đính kèm")
        for i, att in enumerate(usable, start=1):
            lines.append(f"**[A{i}] {att.get('name')}**")
            lines.extend(_summarize_attachment(att))
            lines.append("")

    if _wants_compare_law(message) or dataset_results:
        lines.append("### Đối chiếu với dataset MaiThuyLaw")
        for i, item in enumerate(dataset_results[:3], start=1):
            lines.append(f"- **[S{i}]** {_compact(item.get('content', ''), 380)}")
        lines.append("")

    lines.append("## Lưu ý")
    lines.append(
        "Mình chỉ dùng nội dung đính kèm nếu nó an toàn và nằm trong phạm vi pháp luật/chính sách/tin tức chính thống về ma túy. "
        "Nếu nguồn chỉ ở mức `needs_review`, bạn nên kiểm tra lại nguồn gốc trước khi trích dẫn."
    )

    return "\n".join(lines).strip()


@app.get("/")
def root() -> dict:
    return {
        "app": APP_NAME,
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "app": APP_NAME,
        "dataset": dataset_summary(),
    }


@app.get("/api/dataset/summary")
def api_dataset_summary() -> dict:
    return dataset_summary()


@app.post("/api/chats", response_model=ChatDetail)
def api_create_chat(req: CreateChatRequest, request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> dict:
    auth_and_quota(request, user_id=req.user_id, x_api_key=x_api_key)
    return create_chat(user_id=req.user_id, title=req.title)


@app.get("/api/chats", response_model=list[ChatSummary])
def api_list_chats(request: Request, user_id: str = Query("demo-user"), x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> list[dict]:
    auth_and_quota(request, user_id=user_id, x_api_key=x_api_key)
    return list_chats(user_id=user_id)


@app.get("/api/chats/{chat_id}", response_model=ChatDetail)
def api_get_chat(chat_id: str, request: Request, user_id: str = Query("demo-user"), x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> dict:
    auth_and_quota(request, user_id=user_id, x_api_key=x_api_key)
    chat = get_chat(chat_id, user_id=user_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.patch("/api/chats/{chat_id}", response_model=ChatDetail)
def api_rename_chat(chat_id: str, req: RenameChatRequest, request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> dict:
    auth_and_quota(request, user_id=req.user_id, x_api_key=x_api_key)
    chat = rename_chat(chat_id, title=req.title, user_id=req.user_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.delete("/api/chats/{chat_id}")
def api_delete_chat(chat_id: str, request: Request, user_id: str = Query("demo-user"), x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> dict:
    auth_and_quota(request, user_id=user_id, x_api_key=x_api_key)
    ok = delete_chat(chat_id, user_id=user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"deleted": True, "chat_id": chat_id}


@app.post("/api/upload-check")
async def api_upload_check(request: Request, file: UploadFile = File(...), x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> dict:
    auth_and_quota(request, user_id=request.headers.get("X-User-ID"), x_api_key=x_api_key)
    extracted = await extract_upload_text(file)
    if not extracted.get("ok"):
        return {
            "filename": extracted.get("filename"),
            "size_bytes": extracted.get("size_bytes"),
            "verdict": "rejected",
            "reason": extracted.get("error"),
            "source_matches": [],
        }

    evaluation = evaluate_uploaded_text(extracted["text"])
    return {
        "filename": extracted["filename"],
        "size_bytes": extracted["size_bytes"],
        **evaluation,
    }


@app.post("/api/attachments/upload")
async def api_attachment_upload(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form("demo-user"),
    chat_id: str | None = Form(None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict:
    auth_and_quota(request, user_id=user_id, x_api_key=x_api_key)
    extracted = await extract_upload_text(file)
    if not extracted.get("ok"):
        return {
            "id": None,
            "name": extracted.get("filename"),
            "kind": "file",
            "verdict": "rejected",
            "reason": extracted.get("error"),
            "source_matches": [],
        }

    return save_attachment(
        user_id=user_id,
        chat_id=chat_id,
        name=extracted["filename"],
        kind="file",
        text=extracted["text"],
        size_bytes=extracted["size_bytes"],
    )


@app.post("/api/attachments/link")
async def api_attachment_link(req: LinkAttachmentRequest, request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> dict:
    auth_and_quota(request, user_id=req.user_id, x_api_key=x_api_key)
    fetched = await fetch_link_text(req.url)
    if not fetched.get("ok"):
        return {
            "id": None,
            "name": req.title or req.url,
            "kind": "link",
            "url": req.url,
            "verdict": "rejected",
            "reason": fetched.get("error"),
            "source_matches": [],
        }

    return save_attachment(
        user_id=req.user_id,
        chat_id=req.chat_id,
        name=req.title or req.url,
        kind="link",
        url=req.url,
        text=f"URL: {req.url}\n\n{fetched['text']}",
        size_bytes=len(fetched["text"].encode("utf-8")),
    )


@app.get("/api/attachments/{attachment_id}")
def api_get_attachment(attachment_id: str, request: Request, user_id: str = Query("demo-user"), x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> dict:
    auth_and_quota(request, user_id=user_id, x_api_key=x_api_key)
    item = get_attachment(attachment_id, user_id=user_id, include_text=False)
    if not item:
        raise HTTPException(status_code=404, detail="Attachment not found")
    return item


@app.get("/api/chats/{chat_id}/attachments")
def api_list_chat_attachments(chat_id: str, request: Request, user_id: str = Query("demo-user"), x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> list[dict]:
    auth_and_quota(request, user_id=user_id, x_api_key=x_api_key)
    return list_attachments_for_chat(chat_id, user_id=user_id)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> ChatResponse:
    auth_and_quota(request, user_id=req.user_id, x_api_key=x_api_key)
    message = req.message.strip()
    active_chat = ensure_chat(req.chat_id, req.user_id, first_message=message)
    chat_id = active_chat["id"]

    message_urls = extract_urls(message)
    merged_links = []
    for url in list(req.links or []) + message_urls:
        if url not in merged_links:
            merged_links.append(url)

    saved_link_attachments = []
    failed_link_attachments = []

    for url in merged_links:
        fetched = await fetch_link_text(url)
        if fetched.get("ok"):
            saved = save_attachment(
                user_id=req.user_id,
                chat_id=chat_id,
                name=url,
                kind="link",
                url=url,
                text=f"URL: {url}\n\n{fetched['text']}",
                size_bytes=len(fetched["text"].encode("utf-8")),
            )
            saved_link_attachments.append(saved)
        else:
            failed_link_attachments.append(
                {
                    "id": None,
                    "user_id": req.user_id,
                    "chat_id": chat_id,
                    "name": url,
                    "kind": "link",
                    "url": url,
                    "verdict": "rejected",
                    "reason": fetched.get("error") or "Không đọc được link.",
                    "safety_reason": None,
                    "domain_score": 0.0,
                    "official_score": 0.0,
                    "dataset_match_score": 0.0,
                    "source_matches": [],
                    "preview": "",
                    "text": "",
                }
            )

    all_attachment_ids = list(req.attachment_ids or []) + [
        a["id"] for a in saved_link_attachments if a.get("id")
    ]
    attachments = resolve_attachments(all_attachment_ids, user_id=req.user_id)
    attachments.extend(failed_link_attachments)

    attachment_text = "\n\n".join(a.get("text", "")[:2500] for a in attachments)
    safety_reason = detect_safety_issue(message + "\n" + attachment_text[:3000])

    if safety_reason:
        answer = (
            "Mình không thể hỗ trợ nội dung có thể giúp lách luật, né tránh xử lý, che giấu hành vi "
            "hoặc thực hiện hành vi liên quan đến ma túy. Mình có thể giúp giải thích quy định pháp luật, "
            "hậu quả pháp lý, chính sách phòng chống ma túy hoặc nguồn tin chính thống."
        )
        add_message(chat_id, "user", message, req.user_id)
        add_message(chat_id, "assistant", answer, req.user_id)
        return ChatResponse(chat_id=chat_id, refused=True, reason=safety_reason, answer=answer, sources=[])

    attachment_allows_domain = any(
        a.get("verdict") in {"accepted", "needs_review"} and float(a.get("domain_score") or 0) >= 0.3
        for a in attachments
    )

    if not attachments and not is_in_domain(message):
        answer = (
            "Mình chỉ hỗ trợ tra cứu thông tin pháp luật, chính sách và tin tức chính thống liên quan đến ma túy. "
            "Bạn hãy đặt câu hỏi trong phạm vi này nhé."
        )
        add_message(chat_id, "user", message, req.user_id)
        add_message(chat_id, "assistant", answer, req.user_id)
        return ChatResponse(chat_id=chat_id, refused=True, reason="out_of_domain", answer=answer, sources=[])

    if not attachments and wants_realtime(message) and not realtime_enabled():
        # Still allow dataset retrieval below if the query has enough in-domain/legal context.
        # But the final answer must clearly avoid pretending to have live web access.
        pass

    if attachments and not attachment_allows_domain and not is_in_domain(message):
        answer = (
            "Nội dung đính kèm chưa đủ phù hợp với phạm vi pháp luật, chính sách hoặc tin tức chính thống về ma túy, "
            "nên mình không dùng nó để trả lời. Bạn có thể gửi nguồn chính thống hơn hoặc hỏi lại trong đúng phạm vi."
        )
        add_message(chat_id, "user", message, req.user_id)
        add_message(chat_id, "assistant", answer, req.user_id)
        return ChatResponse(chat_id=chat_id, refused=True, reason="attachment_out_of_domain", answer=answer, sources=[])

    retrieval_query = rewrite_with_memory(message, chat_id, req.user_id)
    if attachment_text:
        retrieval_query = retrieval_query + "\n\nNội dung đính kèm:\n" + attachment_text[:2500]

    dataset_results = retrieve(retrieval_query, top_k=TOP_K)
    dataset_sources = [_source_label(item, i) for i, item in enumerate(dataset_results, start=1)]
    attachment_sources = [_attachment_source(att, i) for i, att in enumerate(attachments, start=1)]

    answer = await generate_answer(
        message=message,
        dataset_results=dataset_results,
        attachments=attachments,
    )

    if not attachments and wants_realtime(message) and not realtime_enabled():
        answer = (
            "## Cần nguồn realtime để trả lời chắc chắn\n"
            + realtime_unavailable_answer()
            + "\n\n## Thông tin từ dataset hiện có\n"
            + answer
        )

    if retrieval_query != message and not attachment_text:
        answer += "\n\n_Ngữ cảnh hội thoại trước đó đã được dùng để hiểu câu hỏi tiếp theo._"

    add_message(chat_id, "user", message, req.user_id)
    add_message(chat_id, "assistant", answer, req.user_id)

    return ChatResponse(
        chat_id=chat_id,
        answer=answer,
        sources=attachment_sources + dataset_sources,
        refused=False,
    )
