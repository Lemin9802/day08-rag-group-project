"""
Task 10 — Generation Có Citation.

Pipeline:
    1. Retrieve relevant chunks từ Task 9
    2. Reorder chunks để tránh "lost in the middle"
    3. Format context với source labels
    4. Gọi Gemini LLM
    5. Trả lời bằng tiếng Việt, có citation
    6. Nếu thiếu evidence → không đoán
"""

import os
from dotenv import load_dotenv

from google import genai
from google.genai import types

from .retrieval_pipeline import retrieve

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

# top_k = 8 vì câu hỏi tổng hợp cần đủ context từ nhiều nguồn,
# đặc biệt với câu hỏi liệt kê nhiều nghệ sĩ/bài báo.
TOP_K = 8

# RAG cần factual, ít sáng tạo.
TEMPERATURE = 0.2

# top_p 0.9 giữ output tự nhiên nhưng vẫn ổn định.
TOP_P = 0.9

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý RAG trả lời bằng tiếng Việt.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin trong phần Context được cung cấp.
2. Mỗi khẳng định factual phải có citation ngay sau câu.
3. Citation dùng đúng nhãn tài liệu trong context, ví dụ: [Document 1] hoặc [Document 2].
4. Nếu context không đủ bằng chứng, hãy trả lời:
   "Tôi không thể xác minh thông tin này từ nguồn hiện có."
5. Không dùng kiến thức bên ngoài.
6. Nếu nhiều tài liệu mâu thuẫn, hãy nêu rõ mâu thuẫn thay vì đoán.
7. Trả lời rõ ràng, ngắn gọn, có cấu trúc.
8. Nếu câu hỏi yêu cầu liệt kê hoặc tổng hợp nhiều đối tượng, hãy kiểm tra tất cả Documents trong context, không chỉ Document đầu tiên.
9. Với câu hỏi hỏi "những ai", "nghệ sĩ nào", "liệt kê", phải rà soát từng Document và liệt kê tất cả nhân vật/nghệ sĩ được nêu trong context; không được bỏ sót Document nào có liên quan.
"""


# =============================================================================
# DOCUMENT REORDERING
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để giảm lost-in-the-middle.

    Input theo score: [1, 2, 3, 4, 5]
    Output:          [1, 3, 5, 4, 2]

    Ý tưởng:
        - Chunk tốt nhất ở đầu.
        - Chunk tốt thứ 2 đưa về cuối.
        - Chunk kém hơn nằm giữa.
    """
    if len(chunks) <= 2:
        return chunks

    reordered = []

    # 0, 2, 4, ...
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])

    # ..., 3, 1
    start = len(chunks) - 1
    if start % 2 == 0:
        start -= 1

    for i in range(start, 0, -2):
        reordered.append(chunks[i])

    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho LLM.
    Mỗi chunk có nhãn [Document i] để LLM cite.
    """
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})

        source = metadata.get("source", f"source_{i}")
        doc_type = metadata.get("type", "unknown")
        chunk_id = metadata.get("chunk_id", f"chunk_{i}")
        score = float(chunk.get("score", 0.0))
        retrieval_source = chunk.get("source", "unknown")
        content = chunk.get("content", "").strip()

        context_parts.append(
            f"[Document {i}]\n"
            f"Source: {source}\n"
            f"Type: {doc_type}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Retrieval: {retrieval_source}\n"
            f"Score: {score:.4f}\n"
            f"Content:\n{content}\n"
        )

    return "\n---\n".join(context_parts)


def build_user_message(query: str, context: str) -> str:
    """
    Build user prompt cho Gemini.
    """
    return f"""Context:
{context}

---

Question:
{query}

Yêu cầu trả lời:
- Trả lời bằng tiếng Việt.
- Mỗi câu factual phải có citation dạng [Document 1], [Document 2].
- Nếu câu hỏi yêu cầu liệt kê nhiều đối tượng, bắt buộc rà soát lần lượt tất cả Documents trong context.
- Với mỗi Document có liên quan, hãy kiểm tra tiêu đề, Source, và Content để trích xuất tên nhân vật/nghệ sĩ.
- Không được bỏ qua Document chỉ vì nó nằm gần cuối context.
- Nếu một Document nhắc đến rapper, ca sĩ, diễn viên hoặc nghệ sĩ liên quan tới ma túy, phải đưa người đó vào câu trả lời.
- Không được dùng thông tin ngoài context.
"""


# =============================================================================
# GEMINI GENERATION
# =============================================================================

def call_gemini(user_message: str) -> str:
    """
    Gọi Gemini API.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return (
            "GEMINI_API_KEY chưa được cấu hình. "
            "Tôi không thể tạo câu trả lời bằng LLM."
        )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            SYSTEM_PROMPT,
            user_message,
        ],
        config=types.GenerateContentConfig(
            temperature=TEMPERATURE,
            top_p=TOP_P,
        ),
    )

    return response.text


# =============================================================================
# GENERATION PIPELINE
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Returns:
        {
            "answer": str,
            "sources": list[dict],
            "retrieval_source": str
        }
    """
    if not query or not query.strip():
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 2: Reorder để tránh lost-in-the-middle
    reordered_chunks = reorder_for_llm(chunks)

    # Step 3: Format context theo đúng thứ tự Document citation
    context = format_context(reordered_chunks)

    # Step 4: Build prompt
    user_message = build_user_message(query, context)

    # Step 5: Call Gemini
    try:
        answer = call_gemini(user_message)
    except Exception as exc:
        answer = (
            "Không thể gọi Gemini API ở thời điểm hiện tại. "
            f"Lỗi: {type(exc).__name__}: {exc}"
        )

    # Step 6: Return
    retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"

    # Quan trọng:
    # sources phải trả theo reordered_chunks để khớp với citation [Document 1], [Document 2]
    return {
        "answer": answer,
        "sources": reordered_chunks,
        "retrieval_source": retrieval_source,
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    test_queries = [
        "Nghị định 28/2026 quy định gì về danh mục chất ma túy và tiền chất?",
        "Tiêu chí xác định địa bàn cấp xã trọng điểm phức tạp về ma túy là gì?",
        "Cơ sở cai nghiện bắt buộc được nhắc đến như thế nào?",
        "Những nghệ sĩ nào trong dữ liệu bị bắt hoặc liên quan tới ma túy?",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"Q: {query}")
        print("=" * 80)

        result = generate_with_citation(query, top_k=TOP_K)

        print("\nA:")
        print(result["answer"])

        print("\nSources:")
        for i, source in enumerate(result["sources"], 1):
            metadata = source.get("metadata", {})
            print(
                f"  Document {i}. {metadata.get('source', 'unknown')} "
                f"| type={metadata.get('type', 'unknown')} "
                f"| score={source.get('score', 0):.4f}"
            )

        print(f"\nRetrieval source: {result['retrieval_source']}")