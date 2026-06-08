from __future__ import annotations

import re


def _clean_source_text(text: object, max_chars: int = 520) -> str:
    """Clean raw markdown/PDF-converted text before showing in source cards."""
    value = str(text or "")

    # YAML and markdown metadata.
    value = re.sub(r"(?s)^---\s*\n.*?\n---\s*\n", "", value)
    value = re.sub(r"(?m)^#{1,6}\s*", "", value)
    value = re.sub(r"(?im)^\s*\*\*source file:\*\*.*$", "", value)
    value = re.sub(r"(?im)^\s*source file:.*$", "", value)
    value = re.sub(r"(?im)^\s*\*\*source:\*\*.*$", "", value)
    value = re.sub(r"(?im)^\s*\*\*url:\*\*.*$", "", value)
    value = re.sub(r"(?im)^\s*\*\*crawled:\*\*.*$", "", value)
    value = re.sub(r"(?im)^\s*crawled:.*$", "", value)

    # Remove filename-like slugs.
    value = re.sub(r"(?im)^\s*[a-z0-9]+(?:-[a-z0-9]+){4,}\s*$", "", value)
    value = re.sub(r"(?i)\b[a-z0-9]+(?:-[a-z0-9]+){5,}\b", "", value)

    # Markdown image/link noise.
    value = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", value)
    value = re.sub(r"\[([^\]]{1,80})\]\([^)]*\)", r"\1", value)

    # Table/OCR separators.
    value = value.replace("| | |", " ")
    value = re.sub(r"\|{2,}", " ", value)
    value = re.sub(r"-{4,}", " ", value)
    value = re.sub(r"_{4,}", " ", value)

    value = re.sub(r"\s+", " ", value).strip()
    value = value.replace("```", "'''")

    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _md_escape_inline(text: object) -> str:
    value = str(text or "")
    return (
        value.replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("|", "\\|")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("`", "'")
    )


def _short_path(path: object) -> str:
    value = str(path or "").replace("\\", "/")
    if not value:
        return ""
    marker = "data/standardized/"
    if marker in value:
        return marker + value.split(marker, 1)[1]
    marker = "data/"
    if marker in value:
        return marker + value.split(marker, 1)[1]
    parts = value.split("/")
    return "/".join(parts[-4:])


def source_card(source: dict, index: int) -> str:
    metadata = source.get("metadata", {}) or {}
    title = metadata.get("source") or source.get("source") or f"Source {index}"
    source_type = metadata.get("source_type") or metadata.get("type") or "unknown"
    score = float(source.get("score") or 0.0)
    citation_id = metadata.get("citation_id") or f"S{index}"
    snippet = _clean_source_text(source.get("content") or "", 520)
    path = _short_path(metadata.get("path") or "")
    type_label = str(source_type).upper()

    path_line = f"\n`{_md_escape_inline(path)}`" if path else ""

    return f"""
**[{_md_escape_inline(type_label)}] [{_md_escape_inline(citation_id)}] {_md_escape_inline(title)}**

`Score: {score:.3f}`{path_line}

> {_md_escape_inline(snippet)}
""".strip()


def render_sources(sources: list[dict]) -> str:
    if not sources:
        return "### Nguồn tham khảo\nKhông có nguồn phù hợp."

    cards = "\n\n".join(
        source_card(source, i)
        for i, source in enumerate(sources, start=1)
    )
    return f"### Nguồn tham khảo\n\n{cards}"
