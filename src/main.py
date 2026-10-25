from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.db.session import init_db
from src.storage.service import storage_service
from src.search.vector_store import vector_store
from src.api.routes import tenants_router, documents_router, chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle management:
    - Verifies and initializes PostgreSQL tables
    - Verifies Qdrant hybrid vector collection
    - Ensures MinIO object storage bucket exists
    """
    print("=" * 60)
    print("DocuMind Enterprise Hybrid RAG Server Starting Up...")
    print(f"Environment: {settings.APP_ENV} | Debug: {settings.DEBUG}")
    print(f"Dense Embedder Model: {settings.GEMINI_EMBEDDING_MODEL} (768 dims)")
    print(f"Chat Generator Model: {settings.GEMINI_CHAT_MODEL}")
    print(f"Cross-Encoder Reranker: {settings.RERANKER_MODEL}")
    print("=" * 60)

    # 1. Initialize PostgreSQL schemas
    await init_db()

    # 2. Ensure MinIO storage bucket exists
    try:
        if hasattr(storage_service.provider, "_ensure_bucket"):
            storage_service.provider._ensure_bucket()
    except Exception as e:
        print(f"[Startup Warning] Could not verify MinIO bucket: {e}")

    yield

    print("[DocuMind Server] Shutting down cleanly.")


app = FastAPI(
    title="DocuMind Enterprise Hybrid RAG API",
    description="""
Multi-tenant Document Intelligence & Hybrid RAG Platform.
- **Hybrid Search**: Dual named vectors (Dense 768-dim + BM25 Sparse) merged via Reciprocal Rank Fusion (RRF).
- **Reranking**: Deep Cross-Encoder rescoring for precision grounded answers.
- **Auditability**: Source citations and provenance tracked per token and persisted to PostgreSQL.
- **Multi-Tenancy**: Complete logical isolation per tenant across SQL and Vector databases.
""",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware (permits local Vite React dev server and production origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 Routes
app.include_router(tenants_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for liveness probes and uptime monitoring."""
    return {
        "status": "healthy",
        "service": "DocuMind Hybrid RAG API",
        "version": "1.0.0",
        "dense_model": settings.GEMINI_EMBEDDING_MODEL,
        "chat_model": settings.GEMINI_CHAT_MODEL,
        "reranker_model": settings.RERANKER_MODEL
    }


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint redirecting to interactive Swagger docs."""
    return {
        "message": "Welcome to DocuMind Enterprise Hybrid RAG API",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }
