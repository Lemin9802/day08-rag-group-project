from __future__ import annotations

import re
from pathlib import Path

from fastapi import UploadFile

from backend.dataset import retrieve
from backend.guards import detect_safety_issue, is_in_domain

MAX_FILE_BYTES = 2_000_000
MAX_TEXT_CHARS = 12000

ALLOWED_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".pdf", ".docx"}

OFFICIAL_SOURCE_HINTS = [
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
    "thuvienphapluat.vn",
    "luật",
    "nghị định",
    "thông tư",
    "pháp lệnh",
    "quốc hội",
    "chính phủ",
    "bộ công an",
    "tòa án",
    "viện kiểm sát",
]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _read_pdf(data: bytes) -> str:
    try:
        from io import BytesIO
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(data))
        parts = []
        for page in reader.pages[:20]:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except Exception:
        return ""


def _read_docx(data: bytes) -> str:
    try:
        from io import BytesIO
        from docx import Document

        doc = Document(BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""


def _read_plain(data: bytes) -> str:
    for enc in ["utf-8", "utf-8-sig", "cp1258", "latin-1"]:
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore")


async def extract_upload_text(file: UploadFile) -> dict:
    filename = file.filename or "uploaded_file"
    ext = Path(filename).suffix.lower()

    data = await file.read()
    size = len(data)

    if size > MAX_FILE_BYTES:
        return {
            "ok": False,
            "filename": filename,
            "size_bytes": size,
            "error": "File quá lớn. Giới hạn hiện tại là 2MB.",
            "text": "",
        }

    if ext not in ALLOWED_EXTENSIONS:
        return {
            "ok": False,
            "filename": filename,
            "size_bytes": size,
            "error": f"Định dạng {ext or '(không rõ)'} chưa được hỗ trợ. Hỗ trợ: txt, md, json, csv, pdf, docx.",
            "text": "",
        }

    if ext == ".pdf":
        text = _read_pdf(data)
    elif ext == ".docx":
        text = _read_docx(data)
    else:
        text = _read_plain(data)

    text = clean_text(text)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]

    if not text:
        return {
            "ok": False,
            "filename": filename,
            "size_bytes": size,
            "error": "Không trích xuất được nội dung text từ file.",
            "text": "",
        }

    return {
        "ok": True,
        "filename": filename,
        "size_bytes": size,
        "error": None,
        "text": text,
    }


def extract_html_text(html: str) -> str:
    """Extract clean article-like text from a news/legal web page."""
    import html as html_lib

    raw = html_lib.unescape(str(html or ""))

    def bad_nav_text(value: str) -> bool:
        lower = value.lower()
        bad_terms = [
            "english 中文",
            "trang chủ chính trị đối ngoại",
            "góp ý hiến kế",
            "doanh nghiệp kiến quốc",
            "cổng ttđt chính phủ",
            "văn phòng chính phủ",
            "chỉ đạo, quyết định của chính phủ",
            "an giang bình dương bình phước",
            "bình thuận bình định bạc liêu",
            "hà nội hồ chí minh",
            "điện biên đà nẵng",
            "vĩnh long vĩnh phúc",
            "yên bái 0 aa",
        ]
        return any(term in lower for term in bad_terms)

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")

        for tag in soup(["script", "style", "noscript", "svg", "form", "header", "footer", "nav", "aside"]):
            tag.decompose()

        title = ""

        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            title = clean_text(html_lib.unescape(og.get("content", "")))

        if not title:
            h1 = soup.find("h1")
            if h1:
                title = clean_text(h1.get_text(" ", strip=True))

        paragraphs = []
        seen = set()

        for node in soup.find_all(["p", "h1", "h2", "h3"]):
            value = clean_text(html_lib.unescape(node.get_text(" ", strip=True)))

            if len(value) < 45:
                continue
            if bad_nav_text(value):
                continue
            if value.lower() in seen:
                continue

            lower = value.lower()

            # Skip menu/province/navigation blobs.
            province_hits = sum(
                city in lower
                for city in [
                    "an giang", "bình dương", "bình phước", "bình thuận", "bắc giang",
                    "cần thơ", "đà nẵng", "đồng nai", "hà nội", "hồ chí minh",
                    "khánh hòa", "kiên giang", "lâm đồng", "nghệ an", "quảng ninh",
                    "sơn la", "thanh hóa", "vĩnh long", "yên bái",
                ]
            )
            if province_hits >= 4:
                continue

            # Prefer text that looks like article content.
            article_signals = [
                "ma túy", "ma tuý", "chính phủ", "bộ", "ngành", "địa phương",
                "phòng, chống", "phòng chống", "cai nghiện", "ngăn cung",
                "giảm cầu", "giảm tác hại", "người nghiện", "tội phạm",
            ]
            if not any(signal in lower for signal in article_signals) and len(paragraphs) >= 1:
                continue

            seen.add(value.lower())
            paragraphs.append(value)

        if paragraphs:
            parts = []
            if title and title.lower() not in paragraphs[0].lower():
                parts.append(title)
            parts.extend(paragraphs)
            text = "\n".join(parts)
        else:
            body = soup.body or soup
            text = body.get_text(" ", strip=True)

    except Exception:
        text = re.sub(r"(?is)<script.*?</script>", " ", raw)
        text = re.sub(r"(?is)<style.*?</style>", " ", text)
        text = re.sub(r"(?is)<[^>]+>", " ", text)

    text = html_lib.unescape(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b(English|中文)\b", " ", text)
    text = re.sub(r"(?i)^\s*URL:\s*", " ", text)
    text = clean_text(text)

    return text[:MAX_TEXT_CHARS]


async def fetch_link_text(url: str) -> dict:
    try:
        import httpx

        async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": "MaiThuyLawAI/0.1 legal-news-checker",
                },
            )
        content_type = resp.headers.get("content-type", "")
        raw = resp.text

        if resp.status_code >= 400:
            return {
                "ok": False,
                "url": url,
                "error": f"Không đọc được link. HTTP {resp.status_code}",
                "text": "",
                "content_type": content_type,
            }

        if "html" in content_type.lower():
            text = extract_html_text(raw)
        else:
            text = clean_text(raw)[:MAX_TEXT_CHARS]

        if not text:
            return {
                "ok": False,
                "url": url,
                "error": "Không trích xuất được text từ link.",
                "text": "",
                "content_type": content_type,
            }

        return {
            "ok": True,
            "url": url,
            "error": None,
            "text": text,
            "content_type": content_type,
        }
    except Exception as exc:
        return {
            "ok": False,
            "url": url,
            "error": f"Lỗi khi đọc link: {exc}",
            "text": "",
            "content_type": "",
        }


def _official_score(text: str) -> float:
    lower = text.lower()
    hits = sum(1 for hint in OFFICIAL_SOURCE_HINTS if hint in lower)
    return min(1.0, hits / 4)


def _domain_score(text: str) -> float:
    lower = text.lower()
    terms = [
        "ma túy", "ma tuý", "chất ma túy", "tiền chất", "cai nghiện",
        "phòng chống ma túy", "phòng, chống ma túy", "tàng trữ",
        "vận chuyển", "mua bán", "sử dụng trái phép", "bộ luật hình sự",
        "luật phòng chống ma túy",
    ]
    hits = sum(1 for term in terms if term in lower)
    return min(1.0, hits / 5)


def _dataset_match_score(text: str) -> tuple[float, list[dict]]:
    query = text[:1000]
    results = retrieve(query, top_k=5)
    best = float(results[0].get("score", 0.0)) if results else 0.0
    return best, results


def _looks_explicitly_unrelated(text: str) -> bool:
    lower = text.lower()
    patterns = [
        r"không liên quan.{0,80}ma túy",
        r"không liên quan.{0,80}pháp luật",
        r"nấu ăn",
        r"du lịch cuối tuần",
        r"thời trang",
        r"bóng đá",
    ]
    return any(re.search(p, lower) for p in patterns)


def evaluate_uploaded_text(text: str) -> dict:
    cleaned = clean_text(text)
    safety_reason = detect_safety_issue(cleaned)
    in_domain = is_in_domain(cleaned)
    domain_score = _domain_score(cleaned)
    official_score = _official_score(cleaned)
    match_score, matches = _dataset_match_score(cleaned)

    if safety_reason:
        verdict = "rejected"
        reason = "File/link có dấu hiệu chứa nội dung hỗ trợ lách luật, né tránh xử lý hoặc hành vi nguy hiểm liên quan đến ma túy."
    elif _looks_explicitly_unrelated(cleaned):
        verdict = "rejected"
        reason = "File/link có dấu hiệu tự mô tả là không liên quan đến phạm vi pháp luật, chính sách hoặc tin tức chính thống về ma túy."
    elif domain_score < 0.3 and official_score < 0.5:
        verdict = "rejected"
        reason = "File/link không đủ tín hiệu thuộc phạm vi pháp luật, chính sách hoặc tin tức chính thống liên quan đến ma túy."
    elif not in_domain and domain_score < 0.35:
        verdict = "rejected"
        reason = "File/link không nằm trong phạm vi chủ đề của MaiThuyLaw AI."
    elif official_score >= 0.5 and domain_score >= 0.3:
        verdict = "accepted"
        reason = "File/link có dấu hiệu phù hợp với phạm vi MaiThuyLaw và có tín hiệu nguồn chính thống."
    elif domain_score >= 0.4:
        verdict = "needs_review"
        reason = "File/link có liên quan đến chủ đề ma túy/pháp luật nhưng cần kiểm tra thêm độ chính thống của nguồn."
    else:
        verdict = "rejected"
        reason = "File/link chưa đủ phù hợp để dùng làm nguồn trả lời."

    source_matches = []
    for i, item in enumerate(matches[:5], start=1):
        meta = item.get("metadata", {}) or {}
        source_matches.append(
            {
                "source_id": f"S{i}",
                "title": meta.get("title") or meta.get("source") or meta.get("doc_id") or "unknown",
                "source_type": meta.get("source_type") or meta.get("type") or "unknown",
                "url": meta.get("url") or None,
                "score": float(item.get("score", 0.0)),
            }
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "safety_reason": safety_reason,
        "domain_score": round(domain_score, 3),
        "official_score": round(official_score, 3),
        "dataset_match_score": round(match_score, 3),
        "source_matches": source_matches,
        "preview": cleaned[:700],
    }
