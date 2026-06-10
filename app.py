import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="OrgMind",
    page_icon="O",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* base */
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #1f2937; }
    
    }
    /* metric cards */
    .metric-card {
        background: #161b27;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 16px 8px;
        text-align: center;
        min-width: 0;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #6ee7b7; margin: 0; line-height: 1.2; }
    .metric-label { font-size: 0.7rem; color: #6b7280; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: 0.08em; white-space: nowrap; }

    /* route badge */
    .badge-tree {
        display: inline-block;
        background: #064e3b;
        color: #6ee7b7;
        padding: 2px 10px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
            
    .badge-vector {
        display: inline-block;
        background: #1e3a5f;
        color: #93c5fd;
        padding: 2px 10px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    /* source tag */
    .source-tag {
        display: inline-block;
        background: #1f2937;
        color: #9ca3af;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        margin: 2px;
    }

    /* chat bubbles */
    .chat-answer {
        background: #161b27;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 8px;
    }

    /* doc item */
    .doc-item {
        background: #1a2235;
        border-left: 3px solid #6ee7b7;
        padding: 8px 12px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 6px;
        font-size: 0.85rem;
        color: #d1d5db;
    }
    [data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;

    /* hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }
    
</style>
""", unsafe_allow_html=True)

# ── session state ────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexes" not in st.session_state:
    st.session_state.indexes = None
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# ── load indexes once ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_indexes():
    from query import load_indexes
    return load_indexes()

storage = Path("storage")
doc_folders = [f.name for f in storage.iterdir() if f.is_dir()] if storage.exists() else []

if doc_folders and st.session_state.indexes is None:
    with st.spinner("Loading indexes..."):
        st.session_state.indexes = get_indexes()

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## OrgMind")
    st.caption("Intelligent document Q&A for your organisation")
    st.divider()

    # metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{len(doc_folders)}</p>
            <p class="metric-label">Documents</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{st.session_state.query_count}</p>
            <p class="metric-label">Queries</p>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # indexed docs
    st.markdown("#### Indexed Documents")
    if doc_folders:
        for doc in doc_folders:
            st.markdown(f'<div class="doc-item">· {doc.replace("_", " ")}</div>', unsafe_allow_html=True)
    else:
        st.caption("No documents indexed yet")

    st.divider()

    # upload
    st.markdown("#### Add Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded_files:
        if st.button("Index Documents", use_container_width=True, type="primary"):
            with st.spinner("Indexing..."):
                os.makedirs("docs", exist_ok=True)
                file_paths = []
                for f in uploaded_files:
                    path = f"docs/{f.name}"
                    with open(path, "wb") as out:
                        out.write(f.getbuffer())
                    file_paths.append(path)
                from ingest import index_document
                index_document(file_paths)
                st.cache_resource.clear()
                st.success(f"Indexed {len(file_paths)} document(s)")
                st.rerun()

    # clear chat
    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.query_count = 0
        st.rerun()

# ── main area ────────────────────────────────────────────────────────────────
st.markdown("## Ask your documents")
st.caption("Questions are routed intelligently specific queries use the tree index, broad queries use vector search.")
st.divider()

# display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(f'<div class="chat-answer">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not doc_folders:
        st.warning("Please upload and index documents first using the sidebar.")
    else:
        # user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                from query import ask
                stream, route, st.session_state.chat_history = ask(
                    prompt,
                    st.session_state.chat_history,
                    indexes=st.session_state.indexes
                )
                st.session_state.query_count += 1

            # route badge
            if route == "specific":
                st.markdown('<span class="badge-tree">TREE INDEX</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-vector">VECTOR SEARCH</span>', unsafe_allow_html=True)

            # stream answer
            answer = st.write_stream(stream)

            # update history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.messages.append({"role": "assistant", "content": answer})
