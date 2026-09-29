from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # PostgreSQL Database
    DATABASE_URL: str = "postgresql+asyncpg://documind:documind_secret@localhost:5433/documind_db"
    SYNC_DATABASE_URL: str = "postgresql://documind:documind_secret@localhost:5433/documind_db"

    # Qdrant Hybrid Vector DB
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "documind_hybrid_chunks"
    DENSE_VECTOR_DIM: int = 768  # Standard for Gemini text-embedding-004 / text-embedding-005

    # Object / Blob Storage (S3 / MinIO)
    STORAGE_BACKEND: Literal["s3", "local"] = "s3"
    S3_ENDPOINT_URL: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin_secret"
    S3_BUCKET_NAME: str = "documind-documents"
    S3_REGION: str = "us-east-1"
    LOCAL_STORAGE_DIR: str = "./data/uploads"

    # Document OCR & Parsing (LlamaParse)
    LLAMA_CLOUD_API_KEY: Optional[str] = None

    # Google Gemini Models
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_EMBEDDING_MODEL: str = "text-embedding-005"
    GEMINI_CHAT_MODEL: str = "gemini-3.1-flash-lite"

    # Reranker Model
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"


# Global singleton instance
settings = Settings()
