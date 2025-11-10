# 🧠 Agents

A fully containerized **Retrieval-Augmented Generation (RAG)** system built on **FastAPI**, **Ollama**, and **CrewAI agents** — with integrated observability, PostgreSQL + pgvector storage, and OpenWebUI interface.

______________________________________________________________________

## 🚀 1. Quick Start

### Clone and Setup

```bash
git clone https://github.com/ar-sotnikov/open-source-rag-demo
cd agents
```

### Create Environment File

```bash
cp .env.example .env
```

### Start All Services

```bash
# First run or rebuild
sudo docker compose up --build -d

# Regular start (after initial setup)
docker compose up -d
```

### Verify Containers

```bash
docker compose ps
```

______________________________________________________________________

## 🌐 Running Services

| Service                     | URL                                              | Description                |
| --------------------------- | ------------------------------------------------ | -------------------------- |
| **OpenWebUI**               | [http://localhost:3000](http://localhost:3000)   | Frontend chat interface    |
| **Backend API**             | [http://localhost:8000](http://localhost:8000)   | FastAPI backend            |
| **Phoenix (Observability)** | [http://localhost:6006](http://localhost:6006)   | Tracing & monitoring       |
| **PostgreSQL**              | localhost:5432                                   | Vector database (pgvector) |
| **Ollama**                  | [http://localhost:11434](http://localhost:11434) | Local LLM hosting          |

______________________________________________________________________

## 🤖 2. Agentic RAG Setup (CrewAI + Ollama)

Two specialized **CrewAI agents** handle queries for improved factuality and formatting:

1. **Researcher Agent** — validates facts from retrieved documents
1. **Finisher Agent** — refines responses and adds citations

### Pull Required Models

```bash
# Embedding model (for document indexing)
docker exec ollama ollama pull nomic-embed-text

# Generation model (for answering queries)
docker exec ollama ollama pull qwen3:0.6b
```

______________________________________________________________________

## 📄 3. Uploading and Indexing Documents

Upload documents to automatically:

1. Store the original file (`data/raw_docs/`)
1. Chunk it (`data/clear_docs/filename_chunks.json`)
1. Index the chunks in PostgreSQL with pgvector

### Example Requests

```bash
# Default chunk size (1000 chars)
curl -X POST -F "file=@test2.docx" http://localhost:8000/api/upload-and-chunk

# Custom chunk size
curl -X POST -F "file=@test2.docx" -F "chunk_size=500" http://localhost:8000/api/upload-and-chunk

# Multiple uploads
curl -X POST -F "file=@document1.pdf" http://localhost:8000/api/upload-and-chunk
curl -X POST -F "file=@document2.docx" -F "chunk_size=800" http://localhost:8000/api/upload-and-chunk
```

**Supported formats:** `.pdf`, `.docx`, `.doc`, `.md`
All documents share the same vector index and are searchable immediately after upload.

______________________________________________________________________

## 🔧 4. Useful Docker Commands

```bash
# Stream logs for a service
docker compose logs -f backend

# Restart backend after code changes
docker compose restart backend
```

______________________________________________________________________

## 🧩 5. Testing the System

### RAG Retrieval Test

```bash
curl -X POST http://localhost:8000/api/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What does Outside missions mean?"}'
```

**Expected fields:**

- `document_id` — unique doc ID
- `source` — filename
- `metadata` — full context info
- `score` — relevance ranking

______________________________________________________________________

### Agentic Chat Test

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Valkyria Chronicles?", "model": "qwen3:0.6b"}'
```

**Expected:** structured answer with `[1][2][3]` citations and a `sources` array.

______________________________________________________________________

### Debug and Verify

```bash
# Trace agent activity
docker compose logs --tail=100 backend | grep -E "\[AGENT\]|Retrieved.*documents|POST.*chat"

# Check retrieval
docker compose logs -f backend | grep "Retrieved.*documents"

# Inspect database vector count
docker exec postgres_db psql -U rag_user -d rag_db -c "SELECT COUNT(*) FROM data_rag_vectors;"
```

______________________________________________________________________

### Re-indexing Documents

To reset the index:

```bash
docker exec postgres_db psql -U rag_user -d rag_db \
  -c "DROP TABLE IF EXISTS data_rag_vectors CASCADE;"
```

Then re-upload documents using `/api/upload-and-chunk`.

______________________________________________________________________

## 📊 6. Observability (Phoenix)

Phoenix provides complete tracing for the RAG pipeline.
Access the UI at [http://localhost:6006](http://localhost:6006).

### Traced Operations

| Operation         | Description              | Key Attributes                     |
| ----------------- | ------------------------ | ---------------------------------- |
| `document.upload` | File upload & indexing   | filename, chunk_size, total_chunks |
| `rag.retrieve`    | Vector similarity search | query, documents_found             |
| `agents.process`  | CrewAI agent execution   | model, docs_count                  |
| `rag.chat`        | Full RAG pipeline        | model, query_length                |
| LLM calls         | Embeddings & generations | auto-instrumented                  |

**To view traces:**

1. Open Phoenix UI
1. Upload a document and make a query
1. Inspect spans showing all pipeline steps, timings, and metadata

______________________________________________________________________

## 🧹 7. Code Quality

```bash
# Check for linting issues
uv run ruff check

# Automatically fix issues
uv run ruff check --fix
```
