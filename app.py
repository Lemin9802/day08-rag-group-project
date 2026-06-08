from __future__ import annotations

from html import escape

import streamlit as st

from src.generation import generate_with_citation
from src.retrieval_pipeline import retrieve

st.set_page_config(page_title="Tra cứu pháp luật & tin tức ma túy", page_icon="⚖️", layout="wide")

SAMPLE_QUESTIONS = [
    "Nghị định 28/2026 quy định gì về danh mục chất ma túy và tiền chất?",
    "Tiêu chí xác định địa bàn cấp xã trọng điểm phức tạp về ma túy là gì?",
    "Cơ sở cai nghiện bắt buộc được nhắc đến như thế nào?",
    "Những nghệ sĩ nào trong dữ liệu bị bắt hoặc liên quan tới ma túy?",
    "Rapper Bình Gold liên quan tới ma túy như thế nào?",
]

for key, value in {"chat_messages": [], "latest_sources": [], "search_query": "", "theme_mode": "Light"}.items():
    st.session_state.setdefault(key, value)


def apply_theme(mode):
    dark = mode == "Dark"
    bg = "#0b1220" if dark else "#f6f8fc"
    card = "#111827" if dark else "#ffffff"
    card2 = "#0f172a" if dark else "#f8fafc"
    text = "#f8fafc" if dark else "#0f172a"
    muted = "#cbd5e1" if dark else "#64748b"
    border = "#243244" if dark else "#dbe2ea"
    chip_gray_bg = "#1f2937" if dark else "#f1f5f9"
    st.markdown(
        f"""
        <style>
        .stApp {{ background: radial-gradient(circle at top left, rgba(37,99,235,.18), transparent 28%), radial-gradient(circle at bottom right, rgba(14,165,233,.14), transparent 32%), {bg}; color: {text}; }}
        .block-container {{ max-width: 1240px; padding-top: 1.5rem; padding-bottom: 3rem; }}
        .hero {{ background: linear-gradient(135deg,#0f172a 0%,#1e3a8a 62%,#2563eb 100%); color:white; border-radius:28px; padding:2.25rem 2.35rem; box-shadow:0 22px 55px rgba(15,23,42,.24); margin-bottom:1.2rem; }}
        .hero-title {{ font-size:2.35rem; font-weight:900; letter-spacing:-.05em; line-height:1.08; margin-bottom:.7rem; }}
        .hero-desc {{ color:#dbeafe; line-height:1.7; max-width:880px; }}
        .hero-kicker {{ display:inline-block; background:rgba(255,255,255,.13); border:1px solid rgba(255,255,255,.22); border-radius:999px; padding:.35rem .75rem; color:#dbeafe; font-size:.76rem; font-weight:800; letter-spacing:.06em; text-transform:uppercase; margin-bottom:.75rem; }}
        .chip {{ display:inline-block; padding:.28rem .62rem; border-radius:999px; font-size:.74rem; font-weight:800; margin-right:.35rem; margin-bottom:.3rem; }}
        .chip-blue {{ background:#dbeafe; color:#1d4ed8; }} .chip-yellow {{ background:#fef3c7; color:#92400e; }} .chip-green {{ background:#dcfce7; color:#166534; }} .chip-gray {{ background:{chip_gray_bg}; color:{muted}; }}
        .card {{ background:{card}; border:1px solid {border}; border-radius:22px; padding:1.15rem 1.25rem; box-shadow:0 12px 32px rgba(15,23,42,.08); color:{text}; }}
        .result-card {{ background:{card}; border:1px solid {border}; border-left:5px solid #1d4ed8; border-radius:18px; padding:1rem 1.1rem; margin-bottom:.9rem; color:{text}; }}
        .source-card {{ background:{card2}; border:1px solid {border}; border-radius:16px; padding:.9rem 1rem; margin-bottom:.7rem; color:{text}; }}
        .muted {{ color:{muted}; font-size:.88rem; line-height:1.55; }}
        .section-title {{ color:{text}; font-size:1.35rem; font-weight:900; letter-spacing:-.035em; margin-bottom:.3rem; }}
        .result-title {{ color:{text}; font-size:1.02rem; font-weight:900; margin-bottom:.35rem; }}
        div[data-testid='stMetric'] {{ background:{card}; border:1px solid {border}; border-radius:18px; padding:1rem; color:{text}; }}
        .stButton>button {{ border-radius:12px; font-weight:800; min-height:40px; }}
        .stButton>button[kind='primary'] {{ background:linear-gradient(135deg,#0f172a 0%,#1d4ed8 100%); border:none; }}
        .footer {{ color:{muted}; text-align:center; padding:1.5rem 0 .5rem; font-size:.86rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def meta(item):
    return item.get("metadata") or {}


def source_title(item):
    return str(meta(item).get("source") or meta(item).get("path") or "unknown")


def badge(doc_type, score=None, retrieval=None):
    if doc_type == "legal":
        label = '<span class="chip chip-blue">VĂN BẢN PHÁP LUẬT</span>'
    elif doc_type == "news":
        label = '<span class="chip chip-yellow">BÀI BÁO</span>'
    else:
        label = f'<span class="chip chip-gray">{escape(str(doc_type).upper())}</span>'
    if score is not None:
        label += f'<span class="chip chip-green">score={score:.4f}</span>'
    if retrieval:
        label += f'<span class="chip chip-gray">retrieval={escape(str(retrieval))}</span>'
    return label


def render_result(result, idx):
    m = meta(result)
    text = str(result.get("content", ""))
    st.markdown(
        f"""
        <div class='result-card'>
          <div class='result-title'>{idx}. {escape(source_title(result))}</div>
          {badge(m.get('type', 'unknown'), float(result.get('score', 0)), result.get('source', 'unknown'))}
          <div class='muted' style='margin-top:.35rem'>Path: <code>{escape(str(m.get('path', 'unknown')))}</code><br>Chunk ID: <code>{escape(str(m.get('chunk_id', 'unknown')))}</code></div>
          <div style='margin-top:.65rem; line-height:1.6'>{escape(text[:1100])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"Xem nội dung đầy đủ hơn — Result {idx}"):
        st.write(text[:4500])


def render_source(source, idx):
    m = meta(source)
    st.markdown(
        f"""
        <div class='source-card'>
          <div class='result-title'>Document {idx}: {escape(source_title(source))}</div>
          {badge(m.get('type', 'unknown'), float(source.get('score', 0)))}
          <div class='muted' style='margin-top:.35rem'>Path: <code>{escape(str(m.get('path', 'unknown')))}</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"Preview Document {idx}"):
        st.write(str(source.get("content", ""))[:3000])


def set_sample(question):
    st.session_state.search_query = question


with st.sidebar:
    st.session_state.theme_mode = st.radio("Theme", ["Light", "Dark"], horizontal=True)

apply_theme(st.session_state.theme_mode)

left, right = st.columns([0.72, 0.28], vertical_alignment="center")
with left:
    st.markdown("### ⚖️ Cổng tra cứu RAG")
    st.caption("Pháp luật ma túy & tin tức liên quan")
with right:
    st.session_state.theme_mode = st.segmented_control("Chế độ giao diện", options=["Light", "Dark"], default=st.session_state.theme_mode, label_visibility="collapsed")
    apply_theme(st.session_state.theme_mode)

st.markdown(
    """
    <div class='hero'>
      <div class='hero-kicker'>Hệ thống tra cứu thông tin pháp luật</div>
      <div class='hero-title'>Tra cứu pháp luật và tin tức liên quan đến ma túy</div>
      <div class='hero-desc'>Search Engine và RAG Chatbot hỗ trợ tìm kiếm, hỏi đáp tiếng Việt và trích dẫn nguồn từ dữ liệu văn bản pháp luật cùng bài báo đã chuẩn hóa.</div>
      <div style='margin-top:1rem'><span class='chip chip-gray'>Semantic Search</span><span class='chip chip-gray'>BM25</span><span class='chip chip-gray'>RRF Reranking</span><span class='chip chip-gray'>Gemini + Fallback</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

cols = st.columns(4)
cols[0].metric("Tài liệu", "16", "6 legal + 10 news")
cols[1].metric("Chunks", "550", "Curated group index")
cols[2].metric("Retrieval", "Hybrid", "Dense + BM25")
cols[3].metric("Generation", "Gemini", "Citation")

tab_search, tab_chat, tab_about = st.tabs(["🔎 Search Engine", "💬 RAG Chatbot", "ℹ️ About"])

with tab_search:
    st.markdown("<div class='card'><div class='section-title'>Tra cứu tài liệu</div><div class='muted'>Nhập truy vấn để tìm các đoạn tài liệu liên quan. Kết quả hiển thị theo ranking, score và source.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([0.78, 0.22])
    c1.text_input("Truy vấn", key="search_query", placeholder="Ví dụ: cơ sở cai nghiện bắt buộc", label_visibility="collapsed")
    top_k = c2.slider("Top K", 3, 20, 10, 1)
    st.markdown("**Truy vấn mẫu**")
    sample_cols = st.columns(5)
    for i, question in enumerate(SAMPLE_QUESTIONS):
        sample_cols[i].button(f"Mẫu {i+1}", key=f"sample_{i}", use_container_width=True, on_click=set_sample, args=(question,))
    run = st.button("Tra cứu", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if run:
        query = st.session_state.search_query.strip()
        if not query:
            st.warning("Vui lòng nhập truy vấn.")
        else:
            with st.spinner("Đang chạy retrieval pipeline..."):
                try:
                    results = retrieve(query, top_k=top_k)
                except Exception as exc:
                    st.error(f"Retrieval error: {exc}")
                    results = []
            st.markdown("### Kết quả tra cứu")
            st.caption(f"Truy vấn: {query} | Số kết quả: {len(results)}")
            for idx, result in enumerate(results, start=1):
                render_result(result, idx)

with tab_chat:
    left_col, right_col = st.columns([0.64, 0.36], gap="large")
    with left_col:
        st.markdown("<div class='card'><div class='section-title'>RAG Chatbot</div><div class='muted'>Đặt câu hỏi bằng tiếng Việt. Trợ lý chỉ dùng context đã retrieve và trả lời có citation [Document i].</div>", unsafe_allow_html=True)
        selected = st.selectbox("Câu hỏi mẫu", [""] + SAMPLE_QUESTIONS)
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if st.button("Xóa hội thoại"):
            st.session_state.chat_messages = []
            st.session_state.latest_sources = []
            st.rerun()
        prompt = st.chat_input("Nhập câu hỏi cho trợ lý RAG...")
        if selected and st.button("Gửi câu hỏi mẫu", type="primary", use_container_width=True):
            prompt = selected
        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.spinner("Đang truy xuất và sinh câu trả lời..."):
                try:
                    result = generate_with_citation(prompt, top_k=8)
                except Exception as exc:
                    result = {"answer": f"Lỗi khi sinh câu trả lời: {exc}", "sources": [], "generation_mode": "error"}
            answer = result.get("answer", "")
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            st.session_state.latest_sources = result.get("sources", [])
            with st.chat_message("assistant"):
                st.markdown(answer)
                st.caption(f"Generation mode: {result.get('generation_mode', 'unknown')}")
        st.markdown("</div>", unsafe_allow_html=True)
    with right_col:
        st.markdown("<div class='card'><div class='section-title'>Nguồn tham chiếu</div><div class='muted'>Các source documents dùng cho câu trả lời gần nhất.</div>", unsafe_allow_html=True)
        if not st.session_state.latest_sources:
            st.info("Chưa có source document. Hãy đặt câu hỏi trước.")
        else:
            for idx, source in enumerate(st.session_state.latest_sources, start=1):
                render_source(source, idx)
        st.markdown("</div>", unsafe_allow_html=True)

with tab_about:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Kiến trúc hệ thống")
    st.code("""User Query
→ Semantic Search
→ BM25 Lexical Search
→ RRF Merge
→ Query-aware Reranking
→ Vectorless Fallback
→ Gemini Generation / Extractive Fallback
→ Answer with Citation""", language="text")
    st.markdown("""
- **Dataset:** văn bản pháp luật Việt Nam và bài báo tiếng Việt về ma túy.
- **Search Engine:** ranked chunks, score, source, preview.
- **RAG Chatbot:** trả lời tiếng Việt có citation `[Document i]`.
- **Evaluation:** golden dataset 15 câu, retrieval metrics, citation coverage và A/B testing.
""")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Day08 Group Project · Search Engine + RAG Chatbot · Citation-grounded answers</div>", unsafe_allow_html=True)
