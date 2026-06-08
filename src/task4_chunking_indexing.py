"""
Task 4 — Chunking & Indexing vào local vector store.

Yêu cầu:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chunk documents
    3. Tạo embedding/vector cho từng chunk
    4. Lưu index local để Task 5 semantic search sử dụng

Ghi chú:
    Repo khuyến cáo Weaviate, nhưng ở bản cá nhân này dùng local JSON vector store
    để tránh cài Docker/Weaviate và tránh kéo torch quá nặng.
"""

import hashlib
import json
import math
import re
from pathlib import Path


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# Recursive chunking: phù hợp cho markdown hỗn hợp legal/news.
# chunk_size=800 đủ dài để giữ ngữ cảnh pháp luật, nhưng không quá dài.
# overlap=120 giúp câu/điều khoản bị cắt vẫn có ngữ cảnh ở chunk kế tiếp.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
CHUNKING_METHOD = "recursive"

# Lightweight local embedding:
# Dùng hashing word/char n-gram để tạo vector cố định, không cần torch/API key.
# Ưu điểm: nhẹ, deterministic, chạy offline.
# Nhược điểm: không mạnh bằng sentence-transformers/OpenAI embedding.
EMBEDDING_MODEL = "local-hashing-word-char-ngram-v1"
EMBEDDING_DIM = 512

# Local vector store:
# Lưu chunks + vectors vào JSON để Task 5 đọc lại và search bằng cosine similarity.
VECTOR_STORE = "local_json"


# =============================================================================
# DOCUMENT LOADING
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {
            'content': str,
            'metadata': {
                'source': str,
                'path': str,
                'type': str
            }
        }
    """
    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {STANDARDIZED_DIR}")

    documents = []

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        if not md_file.is_file():
            continue

        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"

        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": str(relative_path).replace("\\", "/"),
                "type": doc_type,
            }
        })

    if not documents:
        raise RuntimeError("Không tìm thấy file .md nào trong data/standardized/")

    return documents


# =============================================================================
# CHUNKING
# =============================================================================

def simple_recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Fallback recursive splitter nếu chưa cài langchain-text-splitters.
    Ưu tiên cắt ở đoạn trắng, xuống dòng, dấu chấm, khoảng trắng.
    """
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            cut_points = [
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            ]
            best_cut = max(cut_points)

            if best_cut > start + int(chunk_size * 0.5):
                end = best_cut + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo recursive strategy.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    chunks = []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        use_langchain = True
        print("✓ Using RecursiveCharacterTextSplitter from langchain-text-splitters")

    except Exception:
        splitter = None
        use_langchain = False
        print("⚠ langchain-text-splitters chưa sẵn sàng. Dùng fallback splitter.")

    for doc_id, doc in enumerate(documents, start=1):
        if use_langchain:
            splits = splitter.split_text(doc["content"])
        else:
            splits = simple_recursive_split(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk_index, chunk_text in enumerate(splits):
            chunk_id = f"{doc['metadata']['source']}::chunk_{chunk_index:04d}"

            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                }
            })

    if not chunks:
        raise RuntimeError("Chunking không tạo ra chunk nào.")

    return chunks


# =============================================================================
# LOCAL HASHING EMBEDDING
# =============================================================================

def stable_hash(text: str) -> int:
    """Hash ổn định giữa các lần chạy."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def tokenize_for_embedding(text: str) -> list[str]:
    """
    Tokenize tiếng Việt đơn giản:
    - word tokens
    - char n-grams để tăng khả năng match tên riêng / cụm từ
    """
    text = text.lower()
    words = re.findall(r"[0-9a-zA-ZÀ-ỹ]+", text, flags=re.UNICODE)

    tokens = []

    for word in words:
        tokens.append(f"w:{word}")

        # char n-gram giúp bắt các biến thể từ và tên riêng
        if len(word) >= 4:
            for n in (3, 4):
                for i in range(0, len(word) - n + 1):
                    tokens.append(f"c{n}:{word[i:i+n]}")

    return tokens


def embed_text(text: str) -> list[float]:
    """
    Tạo vector hashing có dimension cố định.
    Vector được L2-normalize để dùng cosine similarity ở Task 5.
    """
    vector = [0.0] * EMBEDDING_DIM
    tokens = tokenize_for_embedding(text)

    if not tokens:
        return vector

    for token in tokens:
        h = stable_hash(token)
        idx = h % EMBEDDING_DIM
        sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
        vector[idx] += sign

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector

    return [round(v / norm, 6) for v in vector]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng local hashing embedding.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    embedded_chunks = []

    for i, chunk in enumerate(chunks, start=1):
        embedded = chunk.copy()
        embedded["embedding"] = embed_text(chunk["content"])
        embedded_chunks.append(embedded)

        if i % 50 == 0:
            print(f"  Embedded {i}/{len(chunks)} chunks")

    return embedded_chunks


# =============================================================================
# INDEXING
# =============================================================================

def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào local JSON vector store.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path = INDEX_DIR / "chunks_with_embeddings.json"
    config_path = INDEX_DIR / "index_config.json"

    index_path.write_text(
        json.dumps(chunks, ensure_ascii=False),
        encoding="utf-8",
    )

    config = {
        "chunking_method": CHUNKING_METHOD,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "vector_store": VECTOR_STORE,
        "num_chunks": len(chunks),
    }

    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"✓ Saved local vector index: {index_path}")
    print(f"✓ Saved index config: {config_path}")


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 60)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 60)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to local vector store")


if __name__ == "__main__":
    run_pipeline()