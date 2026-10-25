from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
from datetime import datetime


# ==========================================
# 1. TENANT SCHEMAS
# ==========================================
class TenantCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=255, description="Tenant organization name")


class TenantResponse(BaseModel):
    id: str
    name: str
    created_at: datetime


# ==========================================
# 2. DOCUMENT SCHEMAS
# ==========================================
class DocumentResponse(BaseModel):
    id: str
    tenant_id: str
    filename: str
    content_type: str
    file_size_bytes: int
    status: str
    num_chunks: int
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str


# ==========================================
# 3. CHAT SCHEMAS
# ==========================================
class ChatRequest(BaseModel):
    tenant_id: str = Field(..., description="Target tenant ID for multi-tenant isolation")
    query: str = Field(..., min_length=1, description="User question")
    session_id: Optional[str] = Field(default=None, description="Existing chat session ID, or null to create new")
    document_id: Optional[str] = Field(default=None, description="Optional document filter")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of top reranked chunks to feed to LLM")


class CitationItem(BaseModel):
    source_id: str
    point_id: Optional[str] = None
    document_id: Optional[str] = None
    page_number: Optional[int] = None
    header_path: Optional[str] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    snippet: str


class ChatResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    citations: List[CitationItem]


class ChatSessionResponse(BaseModel):
    id: str
    tenant_id: str
    title: str
    created_at: datetime


class ChatMessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    created_at: datetime
