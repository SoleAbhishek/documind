# DocuMind — Project Context & Engineering Guide

**DocuMind** is an enterprise multi-tenant Hybrid RAG (Retrieval-Augmented Generation) platform built with FastAPI, PostgreSQL, Qdrant, MinIO/S3, Google Gemini, and React 19 + Vite.

---

## 0. Learning, Resume & Interview Guidelines (Follow After Every Implementation)

This project is built **for learning RAG, Hybrid Search, and production AI engineering**, and is also a **flagship resume/portfolio project**.

After **every feature or phase implementation**, always include a brief wrap-up covering:
1. **How It Was Done**: Walk through the exact files, data flow, and design choices made.
2. **Core Concepts Explained**: Clearly explain the underlying RAG / distributed-systems / ML concepts (and *why* this approach was chosen over alternatives).
3. **Resume & Interview Talking Points**: Summarize how to pitch this feature on a resume and in technical interviews—problem solved, architectural trade-offs, and measurable metrics/impact (e.g., eval deltas, latency/cost savings).

---

## 1. Architecture & Core Modules

* **API & Entrypoint**: [src/main.py](src/main.py), [src/config.py](src/config.py), [src/api/schemas.py](src/api/schemas.py), [src/api/routes/](src/api/routes/) (`tenants.py`, `documents.py`, `chat.py`).
* **Database (PostgreSQL + SQLModel)**: [src/db/models.py](src/db/models.py), [src/db/session.py](src/db/session.py) — `Tenant`, `Document`, `ChunkRecord`, `ChatSession`, `ChatMessage` (stores `retrieved_chunks` provenance JSON).
* **Object Storage**: [src/storage/service.py](src/storage/service.py) — S3/MinIO (`pgsty/silo`) with lazy bucket creation and local filesystem fallback.
* **Parsing & Table-Aware Chunking**:
  * [src/parsing/parser.py](src/parsing/parser.py) — LlamaParse Cloud OCR primary parser with local `pypdf` and plain-text fallbacks.
  * [src/chunking/chunker.py](src/chunking/chunker.py) — `MarkdownChunker` splits on `#`/`##`/`###`, injects `[Context: ...]` breadcrumbs, and uses `split_markdown_table()` / `split_section_blocks()` so split tables always inherit header + separator rows.
* **Hybrid Search & Reranking**:
  * [src/search/vector_store.py](src/search/vector_store.py) — Qdrant (`documind_hybrid_chunks`) with 768-dim Matryoshka Gemini dense vectors (`gemini-embedding-001`, fallback `BAAI/bge-small-en-v1.5`) + `Qdrant/bm25` sparse vectors fused server-side via `models.Fusion.RRF`.
  * [src/services/chat_service.py](src/services/chat_service.py) — Cross-Encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`), prompt grounding with `[Source N]` citations, and direct `google-genai` SDK streaming (`gemini-3.1-flash-lite`) over SSE.
* **Semantic Cache & Observability**:
  * [src/services/semantic_cache.py](src/services/semantic_cache.py) — Dedicated Qdrant collection (`documind_semantic_cache`, cosine threshold `0.95`), scoped by `tenant_id` and `document_id`, invalidated on document deletion, and skipping refusals/errors.
  * [src/services/cost_tracker.py](src/services/cost_tracker.py) — Token usage and cross-provider cost comparison (Gemini, OpenAI, Anthropic) + cache savings.
  * LangSmith `@traceable` instrumentation across `ChatService.chat`, `SemanticCacheService.lookup`/`store`, `QdrantVectorStore.search_hybrid`, `ChatService.rerank_chunks`, and `_generate_gemini_stream`.
* **Frontend**: [frontend/](frontend/) — React 19, Vite, Tailwind CSS consuming SSE frames (`session`, `status`, `citations`, `token`, `done`, `error`).

---

## 2. Architectural Invariants (Follow Always)

1. **Strict Multi-Tenant Isolation**: Every PostgreSQL query, Qdrant search (`FieldCondition` on `tenant_id` in both `documind_hybrid_chunks` and `documind_semantic_cache`), and MinIO object key (`tenants/{tenant_id}/docs/{document_id}/{filename}`) must enforce `tenant_id` isolation.
2. **Direct `google-genai` SDK for Generation**: Use `google-genai` directly rather than LangChain LCEL chains for LLM streaming and structured SSE events.
3. **Table-Header Inheritance**: Preserve `split_markdown_table()` and `split_section_blocks()` behavior so no Markdown table chunk loses its column headers.
4. **No Caching of Refusals/Errors**: Never store `"Error..."`, `"I could not find..."`, or `"I cannot find..."` responses in `SemanticCacheService`.
5. **Architecture Decision Records**: Every new architectural decision must be documented in [decisions.md](decisions.md).

---

## 3. Local Development Environment & Commands

> **Note on Docker Ports**: Ports in `docker-compose.yml` and `.env` are remapped from defaults to avoid colliding with other local containers:
> * **PostgreSQL 16**: `localhost:5434` (`documind_db`)
> * **Qdrant**: `localhost:6335` (REST/Dashboard) & `localhost:6336` (gRPC)
> * **MinIO / Silo**: `localhost:9003` (S3 API) & `localhost:9004` (Console)

```powershell
# Start infrastructure
docker compose up -d

# Run backend (uses ./venv)
.\venv\Scripts\python.exe -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Run frontend
npm --prefix frontend run dev
npm --prefix frontend run build
```

---

## 4. Tests & Evaluations

```powershell
# Unit & integration tests
.\venv\Scripts\python.exe tests/test_table_chunker.py
.\venv\Scripts\python.exe tests/test_semantic_cache.py
.\venv\Scripts\python.exe tests/test_langsmith_tracing.py
.\venv\Scripts\python.exe tests/test_chat_e2e.py
.\venv\Scripts\python.exe tests/test_api_e2e.py

# 49-cell 2D RAG evaluation matrix
.\venv\Scripts\python.exe scripts/run_evals.py
```

* **Baseline Scorecard** ([evals/results/latest_scorecard.md](evals/results/latest_scorecard.md)): Composite `0.91` | Faithfulness `1.00` | Answer Relevancy `0.95` | Recall@3 `0.78` | Precision@3 `0.60` | MRR `0.72` | NDCG@3 `0.72` | Refusal Accuracy `0.80`.

---

## 5. Key Reference Docs & Roadmap

* [PROJECT_REFERENCE.md](PROJECT_REFERENCE.md) — Completed phases and planned roadmap:
  * **Phase 3**: Conversational Memory (STM query rewriting + LTM `tenant_memory`).
  * **Phase 4**: Query Decomposition & Multi-Query Fan-Out (priority target for Multi-hop & `0.80` Refusal Accuracy).
  * **Phase 5**: Metadata Filtering & LLM Self-Querying (+ planned stratified `EnterpriseRAG-Bench` slice evaluation).
  * **Phase 6**: LLM Gateway, Model Tiering & `TokenUsageLog`.
  * **Phase 7**: Generative UI (interactive React widgets).
  * **Phase 8**: Enterprise Guardrails & Parent-Child Retrieval.
* [decisions.md](decisions.md) — Architecture Decision Records (ADR).
* [EMBEDDING_SIZE.MD](EMBEDDING_SIZE.MD) — 768-dim Matryoshka Representation Learning (MRL) trade-offs.
