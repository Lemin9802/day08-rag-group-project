from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.config import PROJECT_ROOT
from backend.file_checker import clean_text, evaluate_uploaded_text

RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"
ATTACHMENTS_PATH = RUNTIME_DIR / "attachments.json"
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_store() -> dict:
    return {"attachments": {}}


def _load_store() -> dict:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if not ATTACHMENTS_PATH.exists():
        return _empty_store()
    try:
        data = json.loads(ATTACHMENTS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "attachments" not in data:
            return _empty_store()
        return data
    except Exception:
        return _empty_store()


def _save_store(data: dict) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ATTACHMENTS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ATTACHMENTS_PATH)


def save_attachment(
    *,
    user_id: str,
    chat_id: str | None,
    name: str,
    kind: str,
    text: str,
    size_bytes: int | None = None,
    url: str | None = None,
) -> dict:
    evaluation = evaluate_uploaded_text(text)
    attachment_id = str(uuid.uuid4())
    now = _now()

    item = {
        "id": attachment_id,
        "user_id": user_id or "demo-user",
        "chat_id": chat_id,
        "name": name,
        "kind": kind,
        "url": url,
        "size_bytes": size_bytes,
        "created_at": now,
        "text": clean_text(text)[:12000],
        **evaluation,
    }

    with _LOCK:
        data = _load_store()
        data["attachments"][attachment_id] = item
        _save_store(data)

    return public_attachment(item, include_text=False)


def get_attachment(attachment_id: str, user_id: str = "demo-user", include_text: bool = False) -> dict | None:
    data = _load_store()
    item = data.get("attachments", {}).get(attachment_id)
    if not item:
        return None
    if item.get("user_id", "demo-user") != (user_id or "demo-user"):
        return None
    return public_attachment(item, include_text=include_text)


def list_attachments_for_chat(chat_id: str, user_id: str = "demo-user") -> list[dict]:
    data = _load_store()
    items = []
    for item in data.get("attachments", {}).values():
        if item.get("user_id", "demo-user") == (user_id or "demo-user") and item.get("chat_id") == chat_id:
            items.append(public_attachment(item, include_text=False))
    return sorted(items, key=lambda x: x.get("created_at") or "", reverse=True)


def resolve_attachments(attachment_ids: list[str], user_id: str = "demo-user") -> list[dict]:
    resolved = []
    for attachment_id in attachment_ids or []:
        item = get_attachment(attachment_id, user_id=user_id, include_text=True)
        if item:
            resolved.append(item)
    return resolved


def public_attachment(item: dict, include_text: bool = False) -> dict:
    result = {
        "id": item.get("id"),
        "user_id": item.get("user_id"),
        "chat_id": item.get("chat_id"),
        "name": item.get("name"),
        "kind": item.get("kind"),
        "url": item.get("url"),
        "size_bytes": item.get("size_bytes"),
        "created_at": item.get("created_at"),
        "verdict": item.get("verdict"),
        "reason": item.get("reason"),
        "safety_reason": item.get("safety_reason"),
        "domain_score": item.get("domain_score"),
        "official_score": item.get("official_score"),
        "dataset_match_score": item.get("dataset_match_score"),
        "source_matches": item.get("source_matches", []),
        "preview": item.get("preview", ""),
    }
    if include_text:
        result["text"] = item.get("text", "")
    return result
