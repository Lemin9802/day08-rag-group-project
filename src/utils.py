import json
import math
import re
import unicodedata
from pathlib import Path

VI_STOPWORDS = {
    "và", "là", "của", "có", "cho", "các", "một", "những", "về", "ở", "trong",
    "được", "theo", "này", "đó", "thì", "để", "với", "từ", "khi", "như", "đến",
    "ra", "bị", "đã", "sẽ", "nào"
}


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    tokens = re.findall(r"[a-zA-ZÀ-ỹ0-9]+", normalize_text(text))
    return [token for token in tokens if token not in VI_STOPWORDS and len(token) > 1]


def clip(text, max_chars=1200):
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def cosine(vec_a, vec_b):
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def hash_embedding(text, dim=512):
    vec = [0.0] * dim
    for token in tokenize(text):
        vec[hash(("word", token)) % dim] += 1.0
        for i in range(max(0, len(token) - 2)):
            vec[hash(("tri", token[i:i+3])) % dim] += 0.35
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def content(item):
    return str(item.get("content") or item.get("text") or item.get("page_content") or "")


def metadata(item):
    value = item.get("metadata") or {}
    return value if isinstance(value, dict) else {}


def source_name(item):
    meta = metadata(item)
    return str(meta.get("source") or meta.get("filename") or meta.get("path") or item.get("source_file") or "unknown")


def doc_type(item):
    meta = metadata(item)
    value = meta.get("type") or meta.get("doc_type")
    if value:
        return str(value)
    src = source_name(item).lower()
    if "article" in src or "news" in src:
        return "news"
    if any(x in src for x in ["nghi-dinh", "quyet-dinh", "thong-tu", "luat"]):
        return "legal"
    return "unknown"


def overlap(a, b):
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, len(a_tokens))
