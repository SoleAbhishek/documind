import os
import uuid
import asyncio
from typing import AsyncGenerator, Dict, Any, List, Optional
from datetime import datetime, timezone
from sqlmodel import select

from src.config import settings
from src.db.session import AsyncSessionLocal
from src.db.models import ChatSession, ChatMessage, Tenant, TenantMemory
from src.search.vector_store import vector_store
from src.services.semantic_cache import semantic_cache
from src.services.cost_tracker import cost_tracker
from src.services.memory_service import memory_service

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class ChatService:
    """
    Production-grade multi-tenant Hybrid RAG Chat Service.
    Retrieves candidate chunks via Qdrant RRF (Dense + BM25),
    applies Cross-Encoder reranking, constructs grounded citations,
    and streams responses using Google Gemini.
    Instrumented with LangSmith @traceable for end-to-end tracing and cost analysis.
    """

    def __init__(self):
        self._reranker = None
        self._llm = None
        self._last_stream_usage = None

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

    @traceable(run_type="tool", name="Cross-Encoder Reranker")
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

    @traceable(run_type="llm", name="Gemini LLM Generation")
    def _generate_gemini_stream(self, system_prompt: str, query: str):
        """
        Streams content tokens directly from Google GenAI model and extracts token metrics.
        """
        genai_client = self._get_genai_client()
        if not genai_client:
            return
        model_name = settings.GEMINI_CHAT_MODEL.replace("models/", "")
        combined_prompt = f"{system_prompt}\n\nUser Question:\n{query}"
        stream = genai_client.models.generate_content_stream(
            model=model_name,
            contents=combined_prompt
        )
        last_chunk = None
        for chunk in stream:
            last_chunk = chunk
            if chunk.text:
                yield chunk.text

        # Extract token usage metadata from the final stream chunk
        if last_chunk and hasattr(last_chunk, "usage_metadata") and last_chunk.usage_metadata:
            self._last_stream_usage = {
                "prompt_tokens": getattr(last_chunk.usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(last_chunk.usage_metadata, "candidates_token_count", 0),
                "total_tokens": getattr(last_chunk.usage_metadata, "total_token_count", 0),
            }
        else:
            self._last_stream_usage = None

    def _build_system_prompt(
        self,
        context_chunks: List[Dict[str, Any]],
        stm_context: Optional[Dict[str, Any]] = None,
        ltm_memories: Optional[List[TenantMemory]] = None,
    ) -> str:
        """
        Constructs context-grounded prompt requiring explicit source citations,
        while injecting tenant LTM preferences/glossary and STM conversation continuity.
        """
        context_str = ""
        for i, c in enumerate(context_chunks, 1):
            doc_id = c.get("document_id", "Unknown")
            page = c.get("page_number", 1)
            section = c.get("header_path", "General")
            text = c.get("text", "")
            context_str += f"\n--- [Source {i}] (Document: {doc_id}, Page: {page}, Section: {section}) ---\n{text}\n"

        ltm_block = ""
        if ltm_memories:
            formatted_ltm = memory_service.format_ltm_for_prompt(ltm_memories)
            if formatted_ltm:
                ltm_block = f"\nTenant Long-Term Memory (Preferences, Glossary & Org Context):\n{formatted_ltm}\n"

        stm_block = ""
        if stm_context:
            formatted_stm = memory_service.format_stm_for_prompt(
                stm_context.get("summary"),
                stm_context.get("recent_messages", []),
            )
            if formatted_stm:
                stm_block = f"\nConversation History (For Continuity Only — Do NOT cite as document evidence):\n{formatted_stm}\n"

        prompt = f"""You are DocuMind, an enterprise AI assistant specializing in document intelligence.
Answer the user's question accurately and concisely based strictly on the provided Context Sources below.

Rules:
1. ONLY use factual information explicitly stated in the Context Sources. Do not extrapolate or hallucinate facts.
2. If the answer cannot be found in the provided Context Sources, state clearly: "I cannot find sufficient information in the indexed documents to answer this question."
3. Whenever making a factual claim, cite the corresponding source index using bracket notation like [Source 1] or [Source 2].
4. If a source contains tabular data, synthesize it clearly and reference the table section.
5. Respect any formatting/style preferences or glossary definitions in Tenant Long-Term Memory when present.
{ltm_block}{stm_block}
Context Sources:
{context_str}
"""
        return prompt

    def _schedule_memory_maintenance(
        self,
        session_id: str,
        tenant_id: str,
        user_query: str,
        assistant_answer: str,
    ) -> None:
        """Runs STM summarization and LTM extraction in the background with zero added latency."""
        async def _run():
            try:
                await memory_service.maybe_update_session_summary(session_id, tenant_id)
            except Exception as e:
                print(f"[ChatService] Warning: Background STM summary update failed: {e}")
            try:
                await memory_service.extract_and_persist_ltm(tenant_id, user_query, assistant_answer)
            except Exception as e:
                print(f"[ChatService] Warning: Background LTM extraction failed: {e}")

        try:
            asyncio.create_task(_run())
        except RuntimeError:
            pass

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

        # Load STM (summary + last 6 messages) & LTM (tenant glossary/preferences) before saving current turn
        stm_context = await memory_service.get_stm_context(session_id=session_id, tenant_id=tenant_id, recent_limit=6)
        ltm_memories = await memory_service.get_tenant_memories(tenant_id=tenant_id)

        # Save User Message
        async with AsyncSessionLocal() as db_session:
            user_msg = ChatMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role="user",
                content=query
            )
            db_session.add(user_msg)
            await db_session.commit()

        yield {"type": "session", "session_id": session_id}

        # Step 1.2: STM Query Rewriting (contextualizes pronouns/acronyms before Cache & Qdrant)
        standalone_query = await asyncio.to_thread(
            memory_service.rewrite_query,
            query,
            stm_context.get("summary"),
            stm_context.get("recent_messages"),
            ltm_memories,
        )
        was_rewritten = standalone_query.strip().lower() != query.strip().lower()
        rewritten_query_out = standalone_query if was_rewritten else None
        if was_rewritten:
            yield {
                "type": "status",
                "stage": "memory_rewrite",
                "message": f"Contextualized query: '{standalone_query}'",
            }

        # Step 1.5: Semantic Cache Lookup (uses standalone_query to avoid pronoun collisions)
        yield {"type": "status", "stage": "cache_lookup", "message": "Checking Semantic Cache..."}
        cached_result = semantic_cache.lookup(
            tenant_id=tenant_id,
            query=standalone_query,
            document_id=document_id
        )

        if cached_result:
            cached_answer = cached_result["answer"]
            cached_citations = cached_result.get("citations", [])
            sim = cached_result.get("similarity", 1.0)

            yield {
                "type": "status",
                "stage": "cache_hit",
                "message": f"⚡ Semantic Cache Hit (similarity: {sim:.3f}). Zero LLM latency."
            }
            yield {
                "type": "citations",
                "citations": cached_citations,
                "cached": True
            }

            # Stream cached tokens rapidly to the frontend
            words = cached_answer.split(" ")
            for i, word in enumerate(words):
                yield {"type": "token", "content": word + (" " if i < len(words) - 1 else "")}
                await asyncio.sleep(0.005)

            # Calculate cost and savings for cached response (Zero live LLM cost)
            p_tokens = cost_tracker.estimate_tokens(query)
            c_tokens = cost_tracker.estimate_tokens(cached_answer)
            cost_summary = cost_tracker.compare_all_models(
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                current_model=settings.GEMINI_CHAT_MODEL,
                is_cache_hit=True
            )

            # Record Assistant Message with Audit Trail in Postgres
            async with AsyncSessionLocal() as db_session:
                asst_msg = ChatMessage(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    role="assistant",
                    content=cached_answer,
                    retrieved_chunks=cached_citations
                )
                db_session.add(asst_msg)
                await db_session.commit()

            self._schedule_memory_maintenance(session_id, tenant_id, query, cached_answer)

            yield {
                "type": "done",
                "session_id": session_id,
                "message_id": asst_msg.id,
                "rewritten_query": rewritten_query_out,
                "citations": cached_citations,
                "cached": True,
                "similarity": sim,
                "cost_summary": cost_summary
            }
            return

        yield {"type": "status", "stage": "retrieval", "message": "Executing Hybrid Search (Dense + BM25 RRF)..."}

        # Step 2: Hybrid Retrieval in Qdrant
        raw_candidates = vector_store.search_hybrid(
            tenant_id=tenant_id,
            query=standalone_query,
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
        top_chunks = self.rerank_chunks(standalone_query, raw_candidates, top_k=rerank_top_k)

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

        # Step 4: System Prompt & Context Assembly (with STM history + LTM tenant memory)
        system_prompt = self._build_system_prompt(
            top_chunks,
            stm_context=stm_context,
            ltm_memories=ltm_memories,
        )
        genai_client = self._get_genai_client()

        full_response_text = ""

        # Step 5: Streaming Generation via Google Gemini
        self._last_stream_usage = None
        if genai_client:
            try:
                for token in self._generate_gemini_stream(system_prompt, query):
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

        # Calculate exact or estimated token counts and cross-model cost comparison
        if self._last_stream_usage and self._last_stream_usage.get("total_tokens", 0) > 0:
            p_tokens = self._last_stream_usage["prompt_tokens"]
            c_tokens = self._last_stream_usage["completion_tokens"]
        else:
            combined_prompt = f"{system_prompt}\n\nUser Question:\n{query}"
            p_tokens = cost_tracker.estimate_tokens(combined_prompt)
            c_tokens = cost_tracker.estimate_tokens(full_response_text)

        cost_summary = cost_tracker.compare_all_models(
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            current_model=settings.GEMINI_CHAT_MODEL,
            is_cache_hit=False
        )

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

        # Step 7: Populate Semantic Cache using standalone_query
        if (
            full_response_text
            and not full_response_text.startswith("Error")
            and not full_response_text.startswith("I could not find")
            and not full_response_text.startswith("I cannot find")
        ):
            try:
                semantic_cache.store(
                    tenant_id=tenant_id,
                    query=standalone_query,
                    answer=full_response_text,
                    citations=citations,
                    document_id=document_id
                )
            except Exception as e:
                print(f"[ChatService] Warning: Failed to populate semantic cache: {e}")

        # Step 8: Background STM Summarization & LTM Extraction (zero added latency)
        self._schedule_memory_maintenance(session_id, tenant_id, query, full_response_text)

        yield {
            "type": "done",
            "session_id": session_id,
            "message_id": asst_msg.id,
            "rewritten_query": rewritten_query_out,
            "citations": citations,
            "cached": False,
            "cost_summary": cost_summary
        }

    @traceable(run_type="chain", name="DocuMind RAG Pipeline")
    async def chat(
        self,
        tenant_id: str,
        query: str,
        session_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Non-streaming helper convenience method.
        Fully instrumented for LangSmith tracing with cost analysis.
        """
        full_text = ""
        citations = []
        final_session_id = session_id
        rewritten_query = None
        is_cached = False
        cost_summary = {}

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
            elif event["type"] == "done":
                is_cached = event.get("cached", False)
                cost_summary = event.get("cost_summary", {})
                rewritten_query = event.get("rewritten_query")

        return {
            "session_id": final_session_id,
            "query": query,
            "rewritten_query": rewritten_query,
            "answer": full_text,
            "citations": citations,
            "cached": is_cached,
            "cost_summary": cost_summary
        }


# Global singleton instance
chat_service = ChatService()
