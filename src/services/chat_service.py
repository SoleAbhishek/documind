import os
import uuid
import asyncio
from typing import AsyncGenerator, Dict, Any, List, Optional
from datetime import datetime, timezone
from sqlmodel import select

from src.config import settings
from src.db.session import AsyncSessionLocal
from src.db.models import ChatSession, ChatMessage, Tenant
from src.search.vector_store import vector_store

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class ChatService:
    """
    Production-grade multi-tenant Hybrid RAG Chat Service.
    Retrieves candidate chunks via Qdrant RRF (Dense + BM25),
    applies Cross-Encoder reranking, constructs grounded citations,
    and streams responses using Google Gemini.
    """

    def __init__(self):
        self._reranker = None
        self._llm = None

    @property
    def reranker(self):
        """Lazy-load the cross-encoder reranker model."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            model_name = settings.RERANKER_MODEL or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            print(f"[ChatService] Loading Cross-Encoder reranker ({model_name})...")
            self._reranker = CrossEncoder(model_name)
        return self._reranker

    def _get_genai_client(self):
        """Initializes Google GenAI client if a valid API key is present."""
        api_key = settings.GEMINI_API_KEY
        if api_key and not api_key.startswith("AIzaSy-your") and len(api_key) > 20:
            try:
                from google import genai
                return genai.Client(api_key=api_key)
            except Exception as e:
                print(f"[ChatService] Warning: Could not initialize Google GenAI Client ({e}).")
        return None

    def rerank_chunks(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Scores candidate chunks against the query using Cross-Encoder attention.
        """
        if not candidates:
            return []

        # Prepare sentence pairs for cross-encoding
        pairs = [[query, c.get("text", "")] for c in candidates]
        scores = self.reranker.predict(pairs)

        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])

        # Sort descending by cross-encoder score
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def _build_system_prompt(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Constructs context-grounded prompt requiring explicit source citations.
        """
        context_str = ""
        for i, c in enumerate(context_chunks, 1):
            doc_id = c.get("document_id", "Unknown")
            page = c.get("page_number", 1)
            section = c.get("header_path", "General")
            text = c.get("text", "")
            context_str += f"\n--- [Source {i}] (Document: {doc_id}, Page: {page}, Section: {section}) ---\n{text}\n"

        prompt = f"""You are DocuMind, an enterprise AI assistant specializing in document intelligence.
Answer the user's question accurately and concisely based strictly on the provided context below.

Rules:
1. ONLY use information explicitly stated in the context sources. Do not extrapolate or hallucinate facts.
2. If the answer cannot be found in the provided sources, state clearly: "I cannot find sufficient information in the indexed documents to answer this question."
3. Whenever making a factual claim, cite the corresponding source index using bracket notation like [Source 1] or [Source 2].
4. If a source contains tabular data, synthesize it clearly and reference the table section.

Context Sources:
{context_str}
"""
        return prompt

    async def stream_chat(
        self,
        tenant_id: str,
        query: str,
        session_id: Optional[str] = None,
        document_id: Optional[str] = None,
        retrieval_limit: int = 10,
        rerank_top_k: int = 4
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Full streaming RAG pipeline:
        Yields status updates, citations, streamed tokens, and completion event.
        """
        # Step 1: Session Management & Tenant Isolation in Postgres
        async with AsyncSessionLocal() as db_session:
            # Verify Tenant exists
            tenant_stmt = select(Tenant).where(Tenant.id == tenant_id)
            tenant_res = await db_session.execute(tenant_stmt)
            tenant = tenant_res.scalars().first()
            if not tenant:
                raise ValueError(f"Tenant '{tenant_id}' does not exist.")

            # Load or create ChatSession
            if session_id:
                session_stmt = select(ChatSession).where(
                    ChatSession.id == session_id,
                    ChatSession.tenant_id == tenant_id
                )
                session_res = await db_session.execute(session_stmt)
                chat_session = session_res.scalars().first()
                if not chat_session:
                    raise ValueError(f"ChatSession '{session_id}' not found for tenant '{tenant_id}'.")
            else:
                chat_session = ChatSession(
                    id=str(uuid.uuid4()),
                    tenant_id=tenant_id,
                    title=query[:60]
                )
                db_session.add(chat_session)
                await db_session.commit()
                await db_session.refresh(chat_session)
                session_id = chat_session.id

            # Save User Message
            user_msg = ChatMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role="user",
                content=query
            )
            db_session.add(user_msg)
            await db_session.commit()

        yield {"type": "session", "session_id": session_id}
        yield {"type": "status", "stage": "retrieval", "message": "Executing Hybrid Search (Dense + BM25 RRF)..."}

        # Step 2: Hybrid Retrieval in Qdrant
        raw_candidates = vector_store.search_hybrid(
            tenant_id=tenant_id,
            query=query,
            limit=retrieval_limit,
            document_id=document_id
        )

        if not raw_candidates:
            empty_answer = "I could not find any relevant information in the documents for this tenant."
            async with AsyncSessionLocal() as db_session:
                asst_msg = ChatMessage(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    role="assistant",
                    content=empty_answer,
                    retrieved_chunks=[]
                )
                db_session.add(asst_msg)
                await db_session.commit()
            yield {"type": "token", "content": empty_answer}
            yield {"type": "done", "session_id": session_id, "citations": []}
            return

        yield {"type": "status", "stage": "reranking", "message": f"Reranking {len(raw_candidates)} candidates via Cross-Encoder..."}

        # Step 3: Cross-Encoder Reranking
        top_chunks = self.rerank_chunks(query, raw_candidates, top_k=rerank_top_k)

        # Build Citations Metadata for Auditability & UI
        citations = []
        for idx, c in enumerate(top_chunks, 1):
            citations.append({
                "source_id": f"Source {idx}",
                "point_id": c.get("point_id"),
                "document_id": c.get("document_id"),
                "page_number": c.get("page_number"),
                "header_path": c.get("header_path"),
                "rrf_score": c.get("rrf_score"),
                "rerank_score": round(c.get("rerank_score", 0.0), 4),
                "snippet": c.get("text", "")[:250] + ("..." if len(c.get("text", "")) > 250 else "")
            })

        yield {"type": "citations", "citations": citations}

        # Step 4: System Prompt & Context Assembly
        system_prompt = self._build_system_prompt(top_chunks)
        genai_client = self._get_genai_client()

        full_response_text = ""

        # Step 5: Streaming Generation via Google Gemini
        if genai_client:
            try:
                model_name = settings.GEMINI_CHAT_MODEL.replace("models/", "")
                combined_prompt = f"{system_prompt}\n\nUser Question:\n{query}"
                response_stream = genai_client.models.generate_content_stream(
                    model=model_name,
                    contents=combined_prompt
                )
                for chunk in response_stream:
                    token = chunk.text
                    if token:
                        full_response_text += token
                        yield {"type": "token", "content": token}
                        await asyncio.sleep(0.01)
            except Exception as e:
                print(f"[ChatService] Gemini streaming error: {e}. Falling back to local synthesizer.")
                genai_client = None

        if not genai_client:
            # High-fidelity local synthesizer when running offline or without Gemini key
            yield {"type": "token", "content": "Based on the retrieved documentation:\n\n"}
            for i, c in enumerate(top_chunks[:2], 1):
                clean_text = c.get("raw_text", c.get("text", "")).strip().replace("\n", " ")
                snippet = clean_text[:280]
                token = f"- **[Source {i}]** ({c.get('header_path', 'General')}): {snippet}...\n\n"
                full_response_text += token
                yield {"type": "token", "content": token}
                await asyncio.sleep(0.05)

        # Step 6: Record Assistant Message with Audit Trail in Postgres
        async with AsyncSessionLocal() as db_session:
            asst_msg = ChatMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role="assistant",
                content=full_response_text,
                retrieved_chunks=citations
            )
            db_session.add(asst_msg)
            await db_session.commit()

        yield {
            "type": "done",
            "session_id": session_id,
            "message_id": asst_msg.id,
            "citations": citations
        }

    async def chat(
        self,
        tenant_id: str,
        query: str,
        session_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Non-streaming helper convenience method.
        """
        full_text = ""
        citations = []
        final_session_id = session_id

        async for event in self.stream_chat(
            tenant_id=tenant_id,
            query=query,
            session_id=session_id,
            document_id=document_id
        ):
            if event["type"] == "token":
                full_text += event["content"]
            elif event["type"] == "citations":
                citations = event["citations"]
            elif event["type"] == "session":
                final_session_id = event["session_id"]

        return {
            "session_id": final_session_id,
            "query": query,
            "answer": full_text,
            "citations": citations
        }


# Global singleton instance
chat_service = ChatService()
