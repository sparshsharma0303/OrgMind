# OrgMind
### Intelligent Document Q&A for Organisations

> Ask questions across your organisation's documents in plain English. Get accurate, sourced answers in seconds.

Built because navigating 10+ subdocuments during a leadership role shouldn't take 20 minutes.

---

## What it does

OrgMind lets you upload your organisation's PDFs policies, SOPs, handbooks, meeting notes and ask questions naturally. It searches across all documents simultaneously, figures out the best way to retrieve the answer, and tells you exactly which document it came from.

Ask *"What is the penalty for an ethics violation?"* and get a precise answer with the source, not a list of 50 search results to manually read through.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   Query Router      │  ← LLM classifies: specific or broad?
└─────────────────────┘
    │               │
    ▼               ▼
┌─────────┐   ┌──────────────┐
│  Tree   │   │    FAISS     │
│  Index  │   │ Vector Index │
│(specific│   │  (broad/     │
│queries) │   │  semantic)   │
└─────────┘   └──────────────┘
    │               │
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │ Answer Fusion │  ← Groq LLaMA consolidates answers
    │  across docs  │     from all documents
    └───────────────┘
            │
            ▼
    Sourced Answer + Citations
```

**Why hybrid retrieval?**
- Tree index excels at structured, specific questions *"What is the exact fine for X?"*
- Vector search excels at broad, semantic questions *"Tell me about AIESEC values"*
- Routing between both gives better results than either alone

---

## Features

- **Multi-document search** indexes and queries across all uploaded PDFs simultaneously
- **Hybrid retrieval** LLM-based routing between tree index and FAISS vector search
- **Answer fusion** consolidates context from multiple documents into one coherent answer
- **Source citations** every answer tells you which document it came from
- **Conversation memory** ask follow-up questions, it remembers context
- **Response streaming** answers stream token by token, no waiting
- **Parallel retrieval** all documents queried simultaneously, not one by one
- **Dashboard UI** upload, index, and query from a clean Streamlit interface
- **Docker support** runs anywhere with one command

---

## Tech Stack

| Layer | Tool |
|---|---|
| Retrieval | LlamaIndex (tree index) + FAISS (vector search) |
| Embeddings | BAAI/bge-small-en-v1.5 via HuggingFace (local, free) |
| LLM | Groq API llama-3.1-8b-instant / llama-3.3-70b-versatile |
| PDF parsing | PyPDF2 + PyMuPDF |
| UI | Streamlit |
| Deployment | Docker + Docker Compose |

---

## Getting Started

### Prerequisites
- Python 3.11+
- Docker (for containerised deployment)
- Groq API key — free at [console.groq.com](https://console.groq.com)

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/sparshsharma0303/OrgMind.git
cd OrgMind

# 2. Clone PageIndex (required for tree indexing)
git clone https://github.com/VectifyAI/PageIndex.git

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set up environment variables
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### Index your documents

```bash
# Drop PDFs into the docs/ folder, then:
python ingest.py
```

### Run the app

```bash
streamlit run app.py
```

### Docker deployment

```bash
# Add API keys to .env, then:
docker-compose up --build
```

App runs at `http://localhost:8501`

---

## Project Structure

```
OrgMind/
├── ingest.py          # PDF loading, vector + tree index building
├── query.py           # Hybrid retrieval, routing, answer fusion
├── app.py             # Streamlit dashboard UI
├── exception.py       # Custom exception handling
├── src_logging/
│   └── logger.py      # Logging configuration
├── guardrails/        # NeMo Guardrails config (in progress)
│   ├── config.yml
│   └── rails.co
├── storage/           # Persisted indexes (auto-generated)
├── docs/              # Your PDF documents (gitignored)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Why I built this

During my time as CEO at AIESEC in Indore, managing 90+ members across marketing, sales, talent, and external relations, one thing consistently slowed everyone down finding information buried in subdocuments.

We had policies, SOPs, election procedures, ethics guidelines, HR policies each a separate PDF. Someone would ask a simple question and spend 20 minutes opening files, searching, and still not finding a definitive answer.

OrgMind is the tool I wish I had then. It's also a genuine exploration of where RAG is heading away from pure vector search toward hybrid retrieval systems that understand document structure.

---

## Roadmap

- [ ] Pre-filtering rank document relevance before querying to scale to 100+ docs
- [ ] NeMo Guardrails complete input/output safety rails integration
- [ ] PageIndex integration vectorless tree building for higher accuracy on structured docs
- [ ] Authentication multi-user support with role-based document access
- [ ] Open source contribution fix PageIndex compatibility with non-OpenAI LLMs

---

## Local Development Notes

- Indexes are saved to `storage/<doc_name>/` and persist across restarts
- Re-index only when documents change — no need to rebuild every run
- Groq free tier has token limits — use `llama-3.1-8b-instant` for retrieval, `llama-3.3-70b-versatile` for fusion only

---

*Built by [Sparsh Sharma](https://github.com/sparshsharma0303)*
