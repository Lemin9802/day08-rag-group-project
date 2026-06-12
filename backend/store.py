from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.config import PROJECT_ROOT

RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"
STORE_PATH = RUNTIME_DIR / "chats.json"

_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_store() -> dict:
    return {"chats": {}}


def _load_store() -> dict:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if not STORE_PATH.exists():
        return _empty_store()
    try:
        data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "chats" not in data:
            return _empty_store()
        return data
    except Exception:
        return _empty_store()


def _save_store(data: dict) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STORE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STORE_PATH)


def _auto_title(message: str) -> str:
    text = " ".join(str(message or "").split())
    if not text:
        return "Đoạn chat mới"
    return text[:48] + ("..." if len(text) > 48 else "")


def create_chat(user_id: str = "demo-user", title: str | None = None) -> dict:
    with _LOCK:
        data = _load_store()
        chat_id = str(uuid.uuid4())
        now = _now()
        chat = {
            "id": chat_id,
            "user_id": user_id or "demo-user",
            "title": title or "Đoạn chat mới",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        data["chats"][chat_id] = chat
        _save_store(data)
        return chat


def ensure_chat(chat_id: str | None, user_id: str, first_message: str | None = None) -> dict:
    with _LOCK:
        data = _load_store()
        if chat_id and chat_id in data["chats"]:
            return data["chats"][chat_id]

        new_id = str(uuid.uuid4())
        now = _now()
        chat = {
            "id": new_id,
            "user_id": user_id or "demo-user",
            "title": _auto_title(first_message or ""),
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        data["chats"][new_id] = chat
        _save_store(data)
        return chat


def list_chats(user_id: str = "demo-user") -> list[dict]:
    data = _load_store()
    chats = [
        {
            "id": c["id"],
            "user_id": c.get("user_id", "demo-user"),
            "title": c.get("title", "Đoạn chat mới"),
            "created_at": c.get("created_at"),
            "updated_at": c.get("updated_at"),
            "message_count": len(c.get("messages", [])),
        }
        for c in data.get("chats", {}).values()
        if c.get("user_id", "demo-user") == (user_id or "demo-user")
    ]
    return sorted(chats, key=lambda c: c.get("updated_at") or "", reverse=True)


def get_chat(chat_id: str, user_id: str = "demo-user") -> dict | None:
    data = _load_store()
    chat = data.get("chats", {}).get(chat_id)
    if not chat:
        return None
    if chat.get("user_id", "demo-user") != (user_id or "demo-user"):
        return None
    return chat


def rename_chat(chat_id: str, title: str, user_id: str = "demo-user") -> dict | None:
    with _LOCK:
        data = _load_store()
        chat = data.get("chats", {}).get(chat_id)
        if not chat or chat.get("user_id", "demo-user") != (user_id or "demo-user"):
            return None
        chat["title"] = " ".join(str(title or "Đoạn chat mới").split())[:80]
        chat["updated_at"] = _now()
        _save_store(data)
        return chat


def delete_chat(chat_id: str, user_id: str = "demo-user") -> bool:
    with _LOCK:
        data = _load_store()
        chat = data.get("chats", {}).get(chat_id)
        if not chat or chat.get("user_id", "demo-user") != (user_id or "demo-user"):
            return False
        del data["chats"][chat_id]
        _save_store(data)
        return True


def add_message(chat_id: str, role: str, content: str, user_id: str = "demo-user") -> dict | None:
    with _LOCK:
        data = _load_store()
        chat = data.get("chats", {}).get(chat_id)
        if not chat or chat.get("user_id", "demo-user") != (user_id or "demo-user"):
            return None
        msg = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "created_at": _now(),
        }
        chat.setdefault("messages", []).append(msg)
        chat["updated_at"] = msg["created_at"]
        _save_store(data)
        return msg


def history_text(chat_id: str, user_id: str = "demo-user", limit: int = 4) -> str:
    chat = get_chat(chat_id, user_id)
    if not chat:
        return ""
    messages = chat.get("messages", [])[-limit * 2 :]
    lines = []
    for msg in messages:
        role = msg.get("role", "")
        content = " ".join(str(msg.get("content", "")).split())
        if len(content) > 420:
            content = content[:417] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


FOLLOWUP_MARKERS = [
    "vậy", "thế", "còn", "trường hợp đó", "người đó", "họ", "nó",
    "tiếp", "như trên", "ý đó", "cái đó", "thì sao",
]


def rewrite_with_memory(message: str, chat_id: str, user_id: str = "demo-user") -> str:
    q = str(message or "").strip()
    lower = q.lower()
    is_short = len(lower.split()) <= 8
    is_followup = any(m in lower for m in FOLLOWUP_MARKERS)
    if not (is_short or is_followup):
        return q

    chat = get_chat(chat_id, user_id)
    if not chat:
        return q

    previous_user_messages = [
        m.get("content", "")
        for m in chat.get("messages", [])
        if m.get("role") == "user"
    ]
    if not previous_user_messages:
        return q

    last_user = previous_user_messages[-1]
    return f"{last_user}. Câu hỏi tiếp theo: {q}"
