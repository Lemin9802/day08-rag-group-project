from __future__ import annotations

from dataclasses import dataclass, field


FOLLOW_UP_MARKERS = {
    "neu",
    "vay",
    "the",
    "con",
    "tai pham",
    "truong hop do",
    "nguoi do",
    "ho",
    "thi sao",
}


@dataclass
class ConversationMemory:
    max_turns: int = 6
    turns: list[dict] = field(default_factory=list)

    def add(self, user: str, assistant: str) -> None:
        self.turns.append({"user": user, "assistant": assistant})
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def clear(self) -> None:
        self.turns.clear()

    def history_text(self, limit: int = 3) -> str:
        recent = self.turns[-limit:]
        return "\n".join(f"User: {t['user']}\nAssistant: {t['assistant'][:500]}" for t in recent)

    def rewrite_query(self, query: str) -> str:
        normalized = query.lower().strip()
        is_short = len(normalized.split()) <= 8
        depends_on_context = any(marker in normalized for marker in FOLLOW_UP_MARKERS)
        if not self.turns or not (is_short or depends_on_context):
            return query
        previous_question = self.turns[-1]["user"]
        return f"{previous_question}. Câu hỏi tiếp theo: {query}"
