from __future__ import annotations

import os
import re


REALTIME_TERMS = [
    "mới nhất",
    "gần đây",
    "hôm nay",
    "tuần này",
    "tháng này",
    "năm nay",
    "vừa xảy ra",
    "vừa được",
    "cập nhật",
    "hiện nay",
    "hiện tại",
    "2026",
    "2025",
]


OFFICIAL_DOMAINS = [
    "chinhphu.vn",
    "baochinhphu.vn",
    "tiengchuong.chinhphu.vn",
    "bocongan.gov.vn",
    "pcmatuy.bocongan.gov.vn",
    "cand.vn",
    "nhandan.vn",
    "vietnamplus.vn",
    "tapchitoaan.vn",
    "vbpl.vn",
]


def wants_realtime(message: str) -> bool:
    q = str(message or "").lower()
    return any(term in q for term in REALTIME_TERMS)


def realtime_enabled() -> bool:
    return bool(os.getenv("MAITHUYLAW_REALTIME_ENABLED", "").lower() in {"1", "true", "yes"})


def extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s)>\"]+", str(text or ""))


def is_official_or_allowed_url(url: str) -> bool:
    lower = str(url or "").lower()
    return any(domain in lower for domain in OFFICIAL_DOMAINS)


def realtime_unavailable_answer() -> str:
    return (
        "Mình nhận thấy câu hỏi cần thông tin realtime/mới nhất. "
        "Hiện backend chưa bật realtime search provider, nên mình không tự bịa hoặc suy đoán. "
        "Bạn có thể gửi link nguồn chính thống cần kiểm tra, hoặc bật cấu hình realtime search cho backend."
    )
