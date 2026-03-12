import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

from retrieve_v2 import HybridRetrieverV2
from rerank_v2 import TwoStageCalibratedReranker
from llm import GenerationClient
from html_renderer import build_answer_html
from query import expand_query
from config import (
    EXAMPLE_QUERIES,
    INGEST_EMBEDDING_MODEL,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    SPARSE_MODE,
    VECTOR_DB_BACKEND,
)
try:
    import voice
except Exception:
    voice = None

st.set_page_config(
    page_title="RAG for Climate Challenges",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# UI Styling
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    .stApp { background-color: #ffffff; }
    .block-container {
        max-width: 720px;
        padding-top: 3rem;
        padding-bottom: 2rem;
    }
    h1 { font-weight: 500; font-size: 1.6rem; color: #111; letter-spacing: -0.02em; }
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 12px 16px;
        font-size: 15px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #999;
        box-shadow: none;
    }
    .stButton > button {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        background: #fafafa;
        color: #333;
        font-size: 13px;
        padding: 8px 14px;
        font-weight: 400;
    }
    .stButton > button:hover {
        background: #f0f0f0;
        border-color: #ccc;
    }
    .stSpinner > div { color: #666; }
    .voice-status {
        font-size: 12px;
        color: #888;
        margin-top: 2px;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_retriever():
    sparse_mode = SPARSE_MODE if SPARSE_MODE in {"none", "bm42", "splade"} else "bm42"

    def _has_qdrant_collection(path: str, collection: str) -> bool:
        storage = Path(path) / "collection" / collection / "storage.sqlite"
        return storage.exists()

    candidates = [
        (QDRANT_PATH, QDRANT_COLLECTION),
        ("./qdrant_db_ci", "hvac_documents_qdrant_ci"),
        ("./qdrant_db", "hvac_documents_qdrant"),
    ]
    qdrant_path, qdrant_collection = next(
        ((path, collection) for path, collection in candidates if _has_qdrant_collection(path, collection)),
        (QDRANT_PATH, QDRANT_COLLECTION),
    )

    return HybridRetrieverV2(
        backend="qdrant",
        embedding_model=INGEST_EMBEDDING_MODEL,
        sparse_mode=sparse_mode,
        qdrant_path=qdrant_path,
        qdrant_collection=qdrant_collection,
    )


@st.cache_resource
def get_reranker_model():
    return TwoStageCalibratedReranker()


@st.cache_resource
def get_generator():
    return GenerationClient()


@st.cache_resource
def load_whisper_model():
    if voice is None:
        return None
    try:
        return voice.load_model()
    except Exception:
        return None


def _init_voice_state():
    defaults = {
        "voice_query": "",
        "voice_status": "",
        "show_recorder": False,
        "just_transcribed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_voice_recorder(whisper_model):
    """Render the mic button and audio recorder."""
    url_query = st.query_params.get("q", "")
    query_from_voice = None
    if url_query:
        st.session_state["query_input"] = url_query
        st.query_params.clear()
    elif st.session_state["voice_query"]:
        query_from_voice = st.session_state["voice_query"]
        st.session_state["query_input"] = query_from_voice
        st.session_state["voice_query"] = ""

    col_form, col_mic = st.columns([11, 1])
    with col_form:
        with st.form("query_form", clear_on_submit=False):
            query = st.text_input(
                "Ask a question",
                placeholder="e.g. What is India's cooling action plan?",
                label_visibility="collapsed",
                key="query_input",
            )
            form_submitted = st.form_submit_button("Search")

    with col_mic:
        mic_clicked = st.button(
            "Mic",
            key="mic_btn",
            help="Record a question. Supports English and Indian languages.",
            use_container_width=True,
            disabled=(whisper_model is None),
        )

    if mic_clicked:
        st.session_state["show_recorder"] = not st.session_state["show_recorder"]

    if st.session_state["show_recorder"]:
        audio_value = st.audio_input(
            "Speak your question",
            label_visibility="collapsed",
            key="voice_recorder",
        )
        if audio_value is not None:
            with st.spinner("Transcribing..."):
                try:
                    audio_np = voice.decode_audio(audio_value)
                    text, status = voice.transcribe(whisper_model, audio_np)
                    if text:
                        st.session_state["voice_query"] = text
                        st.session_state["voice_status"] = status
                    else:
                        st.session_state["voice_status"] = "No speech detected. Please try again."
                    st.session_state["show_recorder"] = False
                    st.session_state["just_transcribed"] = True
                except Exception as exc:
                    st.session_state["voice_status"] = f"Transcription error: {exc}"
            st.rerun()

    if st.session_state["voice_status"]:
        st.markdown(
            f'<p class="voice-status">{st.session_state["voice_status"]}</p>',
            unsafe_allow_html=True,
        )

    if query_from_voice is not None and not form_submitted:
        return query_from_voice
    return query


def _retrieve(query: str, retriever: HybridRetrieverV2, reranker: TwoStageCalibratedReranker, generator: GenerationClient) -> list:
    """Expansion, search, and reranking pipeline."""
    # Use generator's groq client for expansion (or just use dedicated client)
    queries = expand_query(query, generator.groq)

    seen_ids = set()
    candidates = []
    for q in queries:
        for result in retriever.search(q):
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                candidates.append(result)

    return reranker.rerank(query, candidates)


def _render_answer(query: str, retriever: HybridRetrieverV2, reranker: TwoStageCalibratedReranker, generator: GenerationClient):
    """Execute the full RAG pipeline and render the result."""
    with st.spinner("Searching..."):
        try:
            results = _retrieve(query, retriever, reranker, generator)
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            st.stop()

        if not results:
            st.info("No relevant documents found. Try a different query.")
            st.stop()

        # Selection of top 5 for generation
        top_results = results[:5]
        answer = generator.generate(query, top_results)

    answer_html = build_answer_html(answer, top_results)
    answer_lines = answer.count("\n") + 1
    estimated_height = 350 + (answer_lines * 22) + (len(top_results) * 55)
    estimated_height = min(max(estimated_height, 450), 1800)
    components.html(answer_html, height=estimated_height, scrolling=True)


def _render_example_queries():
    """Display clickable example query buttons."""
    st.markdown("")
    st.markdown("##### Try asking")
    cols = st.columns(2)
    for idx, example in enumerate(EXAMPLE_QUERIES):
        with cols[idx % 2]:
            if st.button(example, key=f"ex_{idx}", use_container_width=True):
                st.query_params.update({"q": example})
                st.rerun()


def main():
    st.title("Retrieval Augmented Generation for Climate Challenges")
    st.caption("Search across your document collection")

    retriever = get_retriever()
    reranker = get_reranker_model()
    generator = get_generator()
    whisper_model = load_whisper_model()

    _init_voice_state()
    query = _render_voice_recorder(whisper_model)

    if st.session_state["just_transcribed"]:
        st.session_state["just_transcribed"] = False

    if query:
        _render_answer(query, retriever, reranker, generator)
    else:
        _render_example_queries()


if __name__ == "__main__":
    main()
