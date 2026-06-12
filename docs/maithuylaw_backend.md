# MaiThuyLaw AI Backend

MaiThuyLaw AI is a Vietnamese legal, policy, and official-news assistant focused on drug-related law and public policy.

## Final dataset

The backend uses the strict neutral dataset:

```text
data/maithuylaw_dataset/data/index/rag_chunks.json
```

Dataset scope:

```text
Vietnam-related only
2025 onward only
Neutral metadata, no personal/team prefixes
Legal + official/news context
```

Current dataset size:

```text
230 chunks
28 documents
188 legal chunks
42 news chunks
```

## Backend capabilities

- FastAPI API
- Legal/news-aware retrieval
- Domain guard
- Safety guard against evasion/crime-enabling requests
- Anti-hallucination fallback
- Per-chat memory
- Chat history
- Attachment upload
- Link ingestion
- Source checking
- Attachment-aware answers
- Realtime intent warning
- API key authentication
- Per-user quota guard

## Local run

```bash
export MAITHUYLAW_API_KEY="dev-maithuylaw-key"
export MAITHUYLAW_RATE_LIMIT_PER_MINUTE="15"
export MAITHUYLAW_DAILY_LIMIT="500"

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8010
```

Open:

```text
http://127.0.0.1:8010/health
http://127.0.0.1:8010/docs
```

Protected endpoints require:

```text
X-API-Key: dev-maithuylaw-key
X-User-ID: demo-user
```

## Main endpoints

```text
GET    /health
GET    /api/dataset/summary
POST   /api/chats
GET    /api/chats
GET    /api/chats/{chat_id}
PATCH  /api/chats/{chat_id}
DELETE /api/chats/{chat_id}
POST   /api/chat
POST   /api/attachments/upload
POST   /api/attachments/link
GET    /api/attachments/{attachment_id}
GET    /api/chats/{chat_id}/attachments
```
