import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from src.db.session import get_db
from src.db.models import ChatSession, ChatMessage
from src.services.chat_service import chat_service
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatSessionResponse,
    ChatMessageResponse
)

router = APIRouter(prefix="/chat", tags=["Chat & RAG"])


@router.post("/stream")
async def chat_stream_endpoint(payload: ChatRequest):
    """
    Streaming Server-Sent Events (SSE) endpoint for multi-tenant Hybrid RAG.
    Yields live tokens, search status, and citation provenance metadata.
    """
    async def event_generator():
        try:
            async for event in chat_service.stream_chat(
                tenant_id=payload.tenant_id,
                query=payload.query,
                session_id=payload.session_id,
                document_id=payload.document_id,
                rerank_top_k=payload.top_k
            ):
                # Standard SSE data frame format
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            error_payload = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("", response_model=ChatResponse)
async def chat_sync_endpoint(payload: ChatRequest):
    """
    Non-streaming JSON chat endpoint. Waits for full generation and returns
    answer along with retrieved citations.
    """
    try:
        result = await chat_service.chat(
            tenant_id=payload.tenant_id,
            query=payload.query,
            session_id=payload.session_id,
            document_id=payload.document_id
        )
        return ChatResponse(
            session_id=result["session_id"],
            query=result["query"],
            answer=result["answer"],
            citations=result["citations"]
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/sessions", response_model=List[ChatSessionResponse])
async def list_chat_sessions(
    tenant_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Lists past conversation sessions for a tenant."""
    stmt = (
        select(ChatSession)
        .where(ChatSession.tenant_id == tenant_id)
        .order_by(ChatSession.created_at.desc())
    )
    res = await db.execute(stmt)
    return res.scalars().all()


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageResponse])
async def get_session_messages(
    session_id: str,
    tenant_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Retrieves full conversation history with retrieved chunk audit trails."""
    session = await db.get(ChatSession, session_id)
    if not session or session.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found for tenant '{tenant_id}'."
        )

    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )
    res = await db.execute(stmt)
    return res.scalars().all()
