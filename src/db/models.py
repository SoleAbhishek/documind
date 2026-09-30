import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, JSON, BigInteger, Text


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    PARSING = "PARSING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


# ==========================================
# 1. TENANT ENTITY
# ==========================================
class Tenant(SQLModel, table=True):
    __tablename__ = "tenants"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True, max_length=64)
    name: str = Field(max_length=255)
    created_at: datetime = Field(default_factory=utc_now)

    # Relationships
    documents: List["Document"] = Relationship(back_populates="tenant", cascade_delete=True)
    chat_sessions: List["ChatSession"] = Relationship(back_populates="tenant", cascade_delete=True)


# ==========================================
# 2. DOCUMENT ENTITY
# ==========================================
class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True, max_length=64)
    tenant_id: str = Field(foreign_key="tenants.id", index=True, ondelete="CASCADE", max_length=64)
    filename: str = Field(max_length=255)
    content_type: str = Field(default="application/pdf", max_length=100)
    file_size_bytes: int = Field(default=0, sa_column=Column(BigInteger, default=0, nullable=False))
    storage_key: str = Field(max_length=512)

    status: str = Field(default=DocumentStatus.PENDING.value, index=True, max_length=32)
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    num_chunks: int = Field(default=0)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    # Relationships
    tenant: Optional[Tenant] = Relationship(back_populates="documents")
    chunks: List["ChunkRecord"] = Relationship(back_populates="document", cascade_delete=True)


# ==========================================
# 3. CHUNK RECORD ENTITY (Bridge to Qdrant)
# ==========================================
class ChunkRecord(SQLModel, table=True):
    __tablename__ = "chunks"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True, max_length=64)
    document_id: str = Field(foreign_key="documents.id", index=True, ondelete="CASCADE", max_length=64)
    tenant_id: str = Field(foreign_key="tenants.id", index=True, ondelete="CASCADE", max_length=64)

    chunk_index: int = Field(nullable=False)
    page_number: int = Field(default=1, nullable=False)
    header_path: Optional[str] = Field(default=None, max_length=512)
    token_count: int = Field(default=0)

    # Qdrant Point UUID
    qdrant_point_id: str = Field(unique=True, index=True, max_length=64)
    created_at: datetime = Field(default_factory=utc_now)

    # Relationships
    document: Optional[Document] = Relationship(back_populates="chunks")


# ==========================================
# 4. CHAT SESSION ENTITY
# ==========================================
class ChatSession(SQLModel, table=True):
    __tablename__ = "chat_sessions"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True, max_length=64)
    tenant_id: str = Field(foreign_key="tenants.id", index=True, ondelete="CASCADE", max_length=64)
    title: str = Field(default="New Chat", max_length=255)
    created_at: datetime = Field(default_factory=utc_now)

    # Relationships
    tenant: Optional[Tenant] = Relationship(back_populates="chat_sessions")
    messages: List["ChatMessage"] = Relationship(back_populates="session", cascade_delete=True)


# ==========================================
# 5. CHAT MESSAGE ENTITY (With Citations JSON)
# ==========================================
class ChatMessage(SQLModel, table=True):
    __tablename__ = "chat_messages"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True, max_length=64)
    session_id: str = Field(foreign_key="chat_sessions.id", index=True, ondelete="CASCADE", max_length=64)
    role: str = Field(max_length=20)  # "user" or "assistant"
    content: str = Field(sa_column=Column(Text, nullable=False))

    # JSON storage for retrieved chunk citations & scores
    retrieved_chunks: Optional[Any] = Field(default=None, sa_column=Column(JSON, nullable=True))
    created_at: datetime = Field(default_factory=utc_now)

    # Relationships
    session: Optional[ChatSession] = Relationship(back_populates="messages")
