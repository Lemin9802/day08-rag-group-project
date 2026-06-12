from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import Header, HTTPException, Request

_LOCK = threading.Lock()

_MINUTE_BUCKETS: dict[str, list[float]] = defaultdict(list)
_DAILY_BUCKETS: dict[str, dict[str, int]] = defaultdict(dict)


def _api_key() -> str:
    return os.getenv("MAITHUYLAW_API_KEY", os.getenv("AGENT_API_KEY", "dev-maithuylaw-key")).strip()


def _rate_per_minute() -> int:
    return int(os.getenv("MAITHUYLAW_RATE_LIMIT_PER_MINUTE", "15"))


def _daily_limit() -> int:
    return int(os.getenv("MAITHUYLAW_DAILY_LIMIT", "500"))


def _today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _client_key(request: Request, user_id: str | None) -> str:
    user = user_id or request.headers.get("X-User-ID") or "anonymous"
    host = request.client.host if request.client else "unknown"
    return f"{user}:{host}"


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected = _api_key()
    if not expected:
        return

    if not x_api_key or x_api_key.strip() != expected:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-API-Key.",
        )


def enforce_quota(request: Request, user_id: str | None = None) -> None:
    now = time.time()
    minute_window_start = now - 60
    key = _client_key(request, user_id)
    today = _today_key()

    with _LOCK:
        recent = [ts for ts in _MINUTE_BUCKETS[key] if ts >= minute_window_start]
        if len(recent) >= _rate_per_minute():
            _MINUTE_BUCKETS[key] = recent
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {_rate_per_minute()} requests per minute.",
            )

        day_count = _DAILY_BUCKETS[key].get(today, 0)
        if day_count >= _daily_limit():
            raise HTTPException(
                status_code=429,
                detail=f"Daily quota exceeded. Max {_daily_limit()} requests per day.",
            )

        recent.append(now)
        _MINUTE_BUCKETS[key] = recent
        _DAILY_BUCKETS[key] = {today: day_count + 1}


def auth_and_quota(request: Request, user_id: str | None = None, x_api_key: str | None = None) -> None:
    expected = _api_key()
    if expected and (not x_api_key or x_api_key.strip() != expected):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")
    enforce_quota(request, user_id=user_id)
