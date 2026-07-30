"""RAG app over user-uploaded PDFs.

Primary LLM: GLM-5.2 via Ollama Cloud (Ollama native API, bearer auth).
Fallback LLM: a local Hugging Face model (Qwen2-0.5B-Instruct via transformers),
loaded only when the cloud endpoint is unreachable or errors.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os

# Silence tqdm/transformers progress bars BEFORE importing any ML libs.
# Streamlit replaces sys.stderr with a proxy that rejects some writes,
# which surfaces as "OSError [Errno 22] Invalid argument" on Windows.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import logging
import time
import traceback
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# LangChain core
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# PDF + chunking
from pypdf import PdfReader

# Embeddings + vector store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Primary model: Ollama (local or Ollama Cloud) via native Ollama API
from langchain_ollama import ChatOllama

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("rag_app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("rag-app")

# ---------------------------------------------------------------------------
# Config (env vars, cached so we read once)
# ---------------------------------------------------------------------------

load_dotenv()

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "glm-5.2:cloud")

HF_FALLBACK_MODEL = os.getenv("HF_FALLBACK_MODEL", "Qwen/Qwen2-0.5B-Instruct")
HF_DEVICE = os.getenv("HF_DEVICE", "cuda")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "4"))

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------------
# Session-state init
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

for key, default in {
    "vector_store": None,
    "processing": False,
    "processed_file_name": None,
    "chat_history": [],
    "primary_ready": None,   # None = untested, True/False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract readable text from a PDF. Raises ValueError if no text found."""
    from io import BytesIO
    reader = PdfReader(BytesIO(file_bytes))
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as e:
            log.warning("Failed to read page %d: %s", i, e)
            continue
        if text:
            parts.append(text)
    if not parts:
        raise ValueError(
            "No readable text found in this PDF. It may be a scanned/image-only "
            "PDF (would need OCR) or empty."
        )
    return "\n\n".join(parts)


def split_text(text: str) -> list[Document]:
    """Split raw text into overlapping chunks."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=c, metadata={"chunk": idx}) for idx, c in enumerate(chunks)]


@st.cache_resource(show_spinner=False)
def get_embedder():
    import torch
    device = HF_DEVICE if (HF_DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    if device == "cpu" and HF_DEVICE == "cuda":
        log.warning("CUDA requested but torch has no CUDA support; using CPU for embeddings.")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks: list[Document]) -> FAISS:
    embedder = get_embedder()
    return FAISS.from_documents(chunks, embedder)


# ---------------------------------------------------------------------------
# LLM providers (primary = Ollama Cloud GLM-5.2, fallback = local HF)
# ---------------------------------------------------------------------------

def _try_primary_llm() -> Optional[ChatOllama]:
    """Build a ChatOllama client for Ollama (local or Ollama Cloud) if configured."""
    if not OLLAMA_BASE_URL:
        log.info("Primary LLM env var missing (OLLAMA_BASE_URL).")
        return None
    try:
        client_kwargs = {}
        if OLLAMA_API_KEY:
            client_kwargs["headers"] = {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
            client_kwargs=client_kwargs,
        )
    except Exception as e:
        log.warning("Failed to construct primary LLM: %s", e)
        return None


@st.cache_resource(show_spinner=False)
def get_fallback_llm():
    """Load a local Hugging Face causal LM via transformers + HuggingFacePipeline."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from langchain_community.llms import HuggingFacePipeline
        import torch
    except Exception as e:
        log.error("Missing deps for fallback LLM: %s", e)
        raise

    has_cuda = HF_DEVICE == "cuda" and torch.cuda.is_available()
    hf_device = "cuda" if has_cuda else "cpu"
    dtype = torch.float16 if has_cuda else torch.float32

    log.info("Loading HF fallback model %s on %s", HF_FALLBACK_MODEL, hf_device)
    tokenizer = AutoTokenizer.from_pretrained(HF_FALLBACK_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        HF_FALLBACK_MODEL,
        torch_dtype=dtype,
        device_map=hf_device,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
    )
    return HuggingFacePipeline(pipeline=pipe)


def get_llm():
    """Return a usable LLM. Tries primary first; falls back to local HF model."""
    primary = _try_primary_llm()
    if primary is not None:
        try:
            # cheap ping so failures here don't surface mid-answer
            primary.invoke("ping")
            st.session_state.primary_ready = True
            return primary, OLLAMA_MODEL
        except Exception as e:
            log.warning("Primary LLM ping failed, falling back: %s", e)
            st.session_state.primary_ready = False
    else:
        st.session_state.primary_ready = False

    log.info("Using local Hugging Face fallback model.")
    return get_fallback_llm(), HF_FALLBACK_MODEL


# ---------------------------------------------------------------------------
# Prompt + chain
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise assistant answering questions strictly based on the provided "
    "PDF excerpts. Use ONLY the context below to answer. If the answer is not present, "
    "say 'The PDF does not contain enough information to answer that.' "
    "Quote relevant lines when helpful, and be concise."
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context excerpts from the PDF:\n\n{context}\n\nQuestion: {question}"),
    ]
)


def _format_context(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[excerpt {d.metadata.get('chunk', i)+1}] {d.page_content}" for i, d in enumerate(docs)
    )


def build_chain(llm, retriever):
    rag = (
        {"context": retriever | RunnableLambda(_format_context), "question": RunnablePassthrough()}
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title(":books: PDF RAG Chat")
st.caption(
    f"Primary: **{OLLAMA_MODEL}** via Ollama · "
    f"Fallback: local **{HF_FALLBACK_MODEL}** · Embeddings: `{EMBEDDING_MODEL}`"
)

with st.sidebar:
    st.header("Configuration")
    st.markdown(
        f"- Chunk size: `{CHUNK_SIZE}` (overlap `{CHUNK_OVERLAP}`)\n"
        f"- Retrieval top-k: `{TOP_K}`\n"
        f"- Embedding device: `{HF_DEVICE}`"
    )
    st.divider()
    if st.session_state.primary_ready is False:
        st.warning("Primary LLM unavailable — using local fallback model.", icon="⚠️")
    elif st.session_state.primary_ready is True:
        st.success("Primary LLM (GLM-5.2) connected.", icon="✅")
    else:
        st.info("LLM status will be checked on first query.", icon="ℹ️")

    if st.button("Clear session", help="Forget the current PDF and history"):
        st.session_state.vector_store = None
        st.session_state.processed_file_name = None
        st.session_state.chat_history = []
        st.session_state.primary_ready = None
        st.rerun()

# --- Upload ---
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded is not None:
    if st.session_state.processed_file_name != uploaded.name:
        st.session_state.processed_file_name = uploaded.name
        st.session_state.vector_store = None
        st.session_state.chat_history = []

    if st.session_state.vector_store is None:
        with st.status("Processing PDF…", expanded=True) as status:
            try:
                st.write("Reading PDF…")
                file_bytes = uploaded.getvalue()
                text = extract_pdf_text(file_bytes)
                st.write(f"Extracted {len(text):,} characters.")

                st.write("Splitting into chunks…")
                chunks = split_text(text)
                st.write(f"Created {len(chunks)} chunks.")

                st.write("Building vector database (FAISS)…")
                vs = build_vector_store(chunks)
                st.session_state.vector_store = vs
                status.update(label="Vector database ready.", state="complete")
            except ValueError as ve:
                status.update(label="Could not read PDF", state="error")
                st.error(str(ve))
            except Exception as e:
                status.update(label="Error processing PDF", state="error")
                st.error(f"Unexpected error: {e}")
                log.error(traceback.format_exc())

# --- Query box (enabled only when DB ready) ---
db_ready = st.session_state.vector_store is not None
if db_ready:
    st.success(":white_check_mark: Ready — ask a question about your PDF.")
else:
    st.info("Upload a PDF to enable the query box.")

question = st.text_input(
    "Ask a question about the PDF",
    value="",
    disabled=not db_ready,
    placeholder="e.g. What are the key takeaways from the book?",
)

if db_ready and question.strip():
    try:
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": TOP_K}
        )
        llm, model_name = get_llm()
        chain = build_chain(llm, retriever)

        with st.spinner(f"Answering with {model_name}…"):
            t0 = time.time()
            answer = chain.invoke(question)
            dt = time.time() - t0

        st.session_state.chat_history.append((question, answer, model_name, dt))
    except Exception as e:
        st.error(f"Failed to answer: {e}")
        log.error(traceback.format_exc())

# --- Chat history ---
if st.session_state.chat_history:
    st.divider()
    st.subheader("Conversation")
    for q, a, model_name, dt in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
            st.caption(f"{model_name} · {dt:.1f}s")