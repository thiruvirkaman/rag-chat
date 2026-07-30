# PDF RAG Chat

A Retrieval-Augmented Generation app over any user-uploaded PDF — built with
**LangChain** + **Streamlit** + **FAISS**.

Upload a PDF, ask questions, get answers grounded in the document. Tested with
full books like *The Richest Man in Babylon*.

## Models

| Role        | Provider              | Model                                         |
|-------------|-----------------------|-----------------------------------------------|
| Primary LLM | Ollama Cloud (Ollama native API) | `glm-5.2:cloud`                                  |
| Fallback LLM| Local Hugging Face (transformers) | `Qwen/Qwen2-0.5B-Instruct` |
| Embeddings  | sentence-transformers | `all-MiniLM-L6-v2`                            |
| Vector DB   | FAISS (in-memory)     | —                                             |

The app tries GLM-5.2 first (via Ollama Cloud's native API, bearer auth). If the
endpoint is unreachable or errors, it automatically falls back to a local
Hugging Face model loaded via `transformers` (`HuggingFacePipeline`). Set
`HF_DEVICE=cuda` to run it on your NVIDIA GPU (requires a CUDA-enabled `torch`
build — see *Notes on GPU* below).

## Setup

```powershell
# 1. Create + activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1

# 2. Install deps
pip install -r requirements.txt

# 3. Configure environment variables
copy .env.example .env
#   then edit .env and fill in your Ollama Cloud key + endpoint

# 4. Run
streamlit run app.py
```

App opens at http://localhost:8501.

## Environment variables

See `.env.example`. The important ones:

| Variable              | Purpose                                              |
|-----------------------|------------------------------------------------------|
| `OLLAMA_API_KEY`      | Ollama API key (https://ollama.com/settings/keys)    |
| `OLLAMA_BASE_URL`     | Ollama host (default `https://ollama.com` for cloud) |
| `OLLAMA_MODEL`        | Primary model id (default `glm-5.2:cloud`)          |
| `HF_FALLBACK_MODEL`   | Hugging Face repo id for the fallback model         |
| `HF_DEVICE`           | `cuda` for NVIDIA GPU, else `cpu`                   |
| `EMBEDDING_MODEL`     | sentence-transformers model name                    |
| `TOP_K`               | Number of chunks retrieved per query                 |

> **Never commit your `.env`.** It is gitignored.

## How it works

1. User uploads a PDF in the Streamlit UI.
2. `pypdf` extracts text; if no text is found (e.g. scanned images), a clear
   error is shown.
3. Text is split into 1000-char chunks with 200-char overlap using
   `RecursiveCharacterTextSplitter`.
4. Chunks are embedded with `all-MiniLM-L6-v2` and stored in an in-memory
   FAISS index.
5. The query box is only enabled once the vector store is built.
6. On a question, top-k chunks are retrieved and passed to the LLM with a strict
   "answer only from context" system prompt.
7. GLM-5.2 is tried first; on failure the local Qwen2-0.5B fallback is used.

The vector store is cached per session and only rebuilt when a new PDF is
uploaded — no rebuild on every keystroke.

## Notes on GPU

The default `torch` install from PyPI is CPU-only on Windows. To run the
fallback model on your RTX 5050, install a CUDA build of torch, e.g.:

```powershell
venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu121
```

If CUDA is unavailable, the app automatically falls back to CPU (slower but
functional).

## Notes on large books

*The Richest Man in Babylon* is ~150 pages. Chunking produces a few hundred
chunks; embeddings run once (on GPU, a minute or two), then retrieval is fast.
If you hit memory limits on very large PDFs, lower `CHUNK_SIZE` in `app.py`.

## Assignment checklist

- [x] Streamlit UI with a PDF file uploader
- [x] Load → chunk → embed → store in vector DB
- [x] Query box (enabled only after DB is ready)
- [x] LangChain retrieval + LLM answer grounded in the PDF
- [x] Answer displayed clearly in the app
- [x] API key read from environment variable (never hard-coded)
- [x] All code in a single `app.py`
- [x] Clear processing state while the DB builds
- [x] Handles PDFs with no readable text gracefully