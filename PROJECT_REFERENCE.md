# 📋 DocuMind Project Reference: Completed Work & Roadmap

This document tracks the current implementation status of **DocuMind**—what has been built and verified versus what remains on the enterprise roadmap.

---

## ✅ Part 1: What Is Done (Implemented & Verified)

### 1. Core Infrastructure & Multi-Tenant Isolation
- **Docker Stack** ([`docker-compose.yml`](docker-compose.yml)): PostgreSQL 16 (`5433`), Qdrant v1.11.3 (`6333`/`6334`), and MinIO S3 (`9000`/`9001`).
- **Relational Schema & Audit Trail** ([`src/db/models.py`](src/db/models.py)): `Tenant`, `Document`, `ChunkRecord`, `ChatSession`, and `ChatMessage` (persisting retrieved chunk provenance as JSON).
- **Storage Provider** ([`src/storage/service.py`](src/storage/service.py)): S3/MinIO and local storage backends with lazy bucket initialization (`_ensure_bucket`) so offline/sandbox imports never block.
- **FastAPI REST & SSE API** ([`src/api/routes/`](src/api/routes/)): Tenant management (`tenants.py`), document upload/deletion (`documents.py`), and sync/streaming chat (`chat.py`).

### 2. Document Parsing & Table-Aware Chunking
- **OCR & Parsing** ([`src/parsing/parser.py`](src/parsing/parser.py)): LlamaParse Cloud OCR primary parser with local PyPDF and plain-text fallbacks.
- **Table-Header Inheritance** ([`src/chunking/chunker.py`](src/chunking/chunker.py)):
  - `split_markdown_table()` and `split_section_blocks()` ensure oversized Markdown tables split across chunks **always inherit the header and separator rows**.
  - Verified by [`tests/test_table_chunker.py`](tests/test_table_chunker.py) and confirmed in the 49-cell evals (`Table-based` queries achieved `1.00` Faithfulness and `1.00` Answer Relevancy).

### 3. Hybrid Retrieval & Neural Reranking
- **Dual-Vector Store** ([`src/search/vector_store.py`](src/search/vector_store.py)):
  - Dense embeddings: Google Gemini `gemini-embedding-001` (768 dims) with local FastEmbed (`BAAI/bge-small-en-v1.5`) fallback.
  - Sparse embeddings: FastEmbed `Qdrant/bm25`.
  - Server-side Qdrant **Reciprocal Rank Fusion (RRF)** strictly filtered by `tenant_id` (and optional `document_id`).
- **Cross-Encoder Reranker** ([`src/services/chat_service.py`](src/services/chat_service.py)): `cross-encoder/ms-marco-MiniLM-L-6-v2` rescoring candidate chunks before prompt assembly.

### 4. 49-Cell 2D Evaluation Suite (Baseline Complete)
- **Dataset** ([`evals/datasets/golden_eval_dataset.json`](evals/datasets/golden_eval_dataset.json)): 49 test cases spanning **7 Document Categories** (`Text-heavy`, `Structured`, `Technical`, `Long-form`, `Visual`, `Complex`, `Mixed`) × **7 Question Types** (`Factual`, `Semantic`, `Numerical`, `Comparative`, `Multi-hop`, `Table-based`, `Unanswerable`).
- **Evaluators**:
  - Retrieval ([`evals/evaluators/retrieval.py`](evals/evaluators/retrieval.py)): `Recall@K`, `Precision@K`, `MRR`, `NDCG@K`.
  - Generation ([`evals/evaluators/generation.py`](evals/evaluators/generation.py)): Gemini LLM-as-a-Judge for `Faithfulness`, `Answer Relevancy`, and `Refusal Accuracy`.
  - Matrix & Runner ([`evals/matrix.py`](evals/matrix.py), [`evals/runner.py`](evals/runner.py), [`scripts/run_evals.py`](scripts/run_evals.py)): Retry/backoff handling for API rate limits and automated Markdown/JSON reporting.
- **Verified Baseline Results** ([`evals/results/latest_scorecard.md`](evals/results/latest_scorecard.md)):
  - **Composite Quality Score**: **`0.91`**
  - **Faithfulness**: **`1.00`** across all 49 cells (zero hallucination)
  - **Answer Relevancy**: **`0.95`**
  - **Recall@3**: **`0.78`** | **Precision@3**: **`0.60`** | **MRR**: **`0.72`** | **NDCG@3**: **`0.72`**
  - **Refusal Accuracy**: **`0.80`** *(Identified key improvement target: static `rerank_top_k=4` occasionally drops rank 5–6 evidence on multi-hop queries, causing conservative false refusals.)*

### 5. Multi-Tenant Semantic Caching
- **Cache Engine** ([`src/services/semantic_cache.py`](src/services/semantic_cache.py)): Dedicated `documind_semantic_cache` Qdrant collection with `tenant_id` and `document_id` payload indexes and configurable cosine similarity threshold ($\tau = 0.95$ default).
- **Pipeline Integration** ([`src/services/chat_service.py`](src/services/chat_service.py)): Pre-retrieval lookup (Step 1.5) and post-generation storage (Step 7), explicitly excluding refusals and error responses.
- **Lifecycle Invalidation** ([`src/api/routes/documents.py`](src/api/routes/documents.py)): Automatic cache purge when a document is deleted.
- **Frontend UI** ([`frontend/src/App.jsx`](frontend/src/App.jsx), [`frontend/src/components/ChatArea.jsx`](frontend/src/components/ChatArea.jsx)): `⚡ Semantic Cache Hit` status badge; Vite build verified.
- **Verified Benchmarks** ([`tests/test_semantic_cache.py`](tests/test_semantic_cache.py)): All 6 lifecycle tests passing—exact hit `~495–670 ms` vs `~13.2–16.4 s` live generation (**24.5x–26.7x speedup**), paraphrased match verified (`0.9277` similarity), cross-tenant isolation verified, and invalidation verified.

### 6. LangSmith `@traceable` Observability & Multi-Model Cost Tracking
- **Configuration** ([`src/config.py`](src/config.py)): Added `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`, and `LANGCHAIN_ENDPOINT` with `os.environ` sync and graceful no-op fallback when disabled.
- **Cost & Pricing Engine** ([`src/services/cost_tracker.py`](src/services/cost_tracker.py)):
  - Tracks exact input/output tokens (from Gemini `usage_metadata` on live runs, or `estimate_tokens()` fallback) and computes USD cost.
  - Compares live cost against **Gemini 3.1 Flash-Lite**, **Gemini 2.5 Pro**, **GPT-4o Mini**, **GPT-4o**, and **Claude 3.5 Sonnet**.
  - Reports **$0.00 actual cost** and **100% savings** on Semantic Cache hits.
- **Trace Instrumentation**:
  - `ChatService.chat()` (`chain`: `DocuMind RAG Pipeline`)
  - `SemanticCacheService.lookup()` (`retriever`: `Semantic Cache Lookup`)
  - `QdrantVectorStore.search_hybrid()` (`retriever`: `Qdrant Hybrid Search (Dense + BM25 RRF)`)
  - `ChatService.rerank_chunks()` (`tool`: `Cross-Encoder Reranker`)
  - `ChatService._generate_gemini_stream()` (`llm`: `Gemini LLM Generation`)
  - `SemanticCacheService.store()` (`tool`: `Semantic Cache Store`)
- **API Exposure** ([`src/api/schemas.py`](src/api/schemas.py), [`src/api/routes/chat.py`](src/api/routes/chat.py)): `cached` and `cost_summary` returned in `ChatResponse` and SSE `done` events.
- **Verified** ([`tests/test_langsmith_tracing.py`](tests/test_langsmith_tracing.py)): All unit calculations, cross-model comparisons, and live + cached pipeline runs passed.

---

## 🗺️ Part 2: What Is In The Roadmap (Planned)

| Phase | Initiative | Scope & Objective | Priority / Target Impact |
| :--- | :--- | :--- | :--- |
| **Phase 3** | **Conversational Memory (STM + LTM)** | **Short-Term Memory**: Rewrite pronoun-heavy follow-ups (e.g., *"What are its penalties?"*) using recent session turns before Qdrant search.<br>**Long-Term Memory**: Persist tenant-level glossaries, acronyms, and preferences in a PostgreSQL `tenant_memory` table. | High — enables multi-turn coherent chat |
| **Phase 4** | **Query Decomposition & Multi-Query Fan-Out** | Classify queries as `SINGLE_STEP` vs `COMPOUND`; split compound/multi-hop questions into $N$ atomic sub-queries run concurrently via `asyncio.gather`, then fuse via RRF and dynamic `top_k`. | **Critical** — directly fixes the `Multi-hop` & `Refusal Accuracy (0.80)` gap found in Phase 1 evals |
| **Phase 5** | **Metadata Filtering & LLM Self-Querying** | Convert natural-language constraints (e.g., *"contracts after 2024 with tables"*) into structured Qdrant `Filter` conditions (`has_table`, `content_type`, `year`) prior to HNSW search. | Medium — improves precision on filtered corpora |
| **Phase 6** | **LLM Gateway, Model Tiering & Billing Logs** | Dynamic complexity router between **Tier 1** (`gemini-3.1-flash-lite`) and **Tier 2** (`gemini-2.5-pro`), 429 fallback redundancy, and persisting per-tenant `TokenUsageLog` records in PostgreSQL. | Medium — optimizes cost vs. deep reasoning |
| **Phase 7** | **Generative UI (Interactive React Widgets)** | Structured output rendering dynamic frontend widgets (`ComparisonMatrix`, `MetricKpiGrid`, `StepTimeline`, `RiskAlertCard`) alongside streaming Markdown. | Medium — elevates UX beyond text walls |
| **Phase 8** | **Enterprise Guardrails & Parent-Child Retrieval** | **Guardrails**: Input/output PII detection & masking, prompt-injection defense, out-of-scope boundary checks.<br>**Parent-Child Retrieval**: Search ~300-char child chunks for precision, return ~1,500-char parent sections for full context. | High for regulated enterprise deployments |
| **Post-Impl** | **Regression Eval Diff (`--compare`)** | Re-run the 49-cell evaluation matrix after Phases 3–8 to measure delta against the `0.91` Composite / `0.78` Recall@3 / `0.80` Refusal Accuracy baseline. | Ongoing quality gate |
