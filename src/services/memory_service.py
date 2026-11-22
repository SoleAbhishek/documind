import json
import uuid
from typing import Any, Dict, List, Optional
from sqlmodel import select

from src.config import settings
from src.db.session import AsyncSessionLocal
from src.db.models import ChatSession, ChatMessage, TenantMemory, utc_now
from src.services.semantic_cache import semantic_cache

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


VALID_MEMORY_TYPES = {"glossary", "preference", "fact"}


class MemoryService:
    """
    Manages Short-Term Memory (STM) and Long-Term Memory (LTM) for multi-tenant RAG.

    - STM: Running session summary + last N raw messages, plus standalone query rewriting
      so Qdrant Hybrid Search, Cross-Encoder Reranking, and Semantic Cache see full context.
    - LTM: Tenant-scoped PostgreSQL `TenantMemory` store (glossaries, preferences, org facts)
      with CRUD operations and background LLM extraction/reconciliation.
    """

    def _get_genai_client(self):
        api_key = settings.GEMINI_API_KEY
        if api_key and not api_key.startswith("AIzaSy-your") and len(api_key) > 20:
            try:
                from google import genai
                return genai.Client(api_key=api_key)
            except Exception as e:
                print(f"[MemoryService] Warning: Could not initialize Google GenAI client ({e}).")
        return None

    def _generate_text(self, prompt: str) -> Optional[str]:
        client = self._get_genai_client()
        if not client:
            return None
        model_name = settings.GEMINI_CHAT_MODEL.replace("models/", "")
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"temperature": 0.0},
        )
        return response.text.strip() if response and response.text else None

    # ==========================================
    # SHORT-TERM MEMORY (STM)
    # ==========================================

    async def get_stm_context(
        self,
        session_id: str,
        tenant_id: str,
        recent_limit: int = 6,
    ) -> Dict[str, Any]:
        """Loads running session summary and the last `recent_limit` messages for a session."""
        async with AsyncSessionLocal() as db:
            sess_stmt = select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.tenant_id == tenant_id,
            )
            sess_res = await db.execute(sess_stmt)
            chat_session = sess_res.scalars().first()
            if not chat_session:
                return {"summary": None, "recent_messages": [], "summarized_count": 0}

            msg_stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(recent_limit)
            )
            msg_res = await db.execute(msg_stmt)
            rows = list(reversed(msg_res.scalars().all()))

            return {
                "summary": chat_session.summary,
                "recent_messages": [{"role": m.role, "content": m.content} for m in rows],
                "summarized_count": chat_session.summarized_count,
            }

    @traceable(run_type="llm", name="STM Query Rewriter")
    def rewrite_query(
        self,
        query: str,
        summary: Optional[str] = None,
        recent_messages: Optional[List[Dict[str, str]]] = None,
        ltm_memories: Optional[List[TenantMemory]] = None,
    ) -> str:
        """
        Rewrites a follow-up query into a standalone, self-contained search query
        using STM history and LTM glossary terms. Skips the LLM call on Turn 1
        when no history or glossary context exists.
        """
        recent_messages = recent_messages or []
        glossary_items = [m for m in (ltm_memories or []) if m.memory_type == "glossary"]

        # Fast-path: Turn 1 with no history and no glossary -> zero latency overhead
        if not summary and not recent_messages and not glossary_items:
            return query

        history_text = self.format_stm_for_prompt(summary, recent_messages)
        glossary_text = "\n".join(f"- {m.key}: {m.value}" for m in glossary_items) if glossary_items else "None"

        prompt = f"""Rewrite the user's latest question into a standalone, self-contained search query for document retrieval.
Rules:
1. Resolve any pronouns (it, they, that, those, etc.) or implicit references using the conversation history.
2. Expand any matching organization glossary acronyms if helpful for search.
3. If the question is already completely standalone, return it unchanged.
4. Output ONLY the rewritten query text—no quotes, no explanation, no preamble.

Organization Glossary:
{glossary_text}

Conversation History:
{history_text or "None"}

Latest Question:
{query}

Standalone Search Query:"""

        try:
            rewritten = self._generate_text(prompt)
            if rewritten:
                return rewritten.strip().strip('"').strip("'")
        except Exception as e:
            print(f"[MemoryService] STM query rewrite failed ({e}); using original query.")
        return query

    async def maybe_update_session_summary(
        self,
        session_id: str,
        tenant_id: str,
        recent_window: int = 6,
    ) -> Optional[str]:
        """
        Incrementally folds older turns beyond `recent_window` into `ChatSession.summary`.
        Designed to run in the background after streaming completes.
        """
        async with AsyncSessionLocal() as db:
            sess_stmt = select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.tenant_id == tenant_id,
            )
            chat_session = (await db.execute(sess_stmt)).scalars().first()
            if not chat_session:
                return None

            msg_stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
            )
            all_messages = (await db.execute(msg_stmt)).scalars().all()

            cutoff = len(all_messages) - recent_window
            if cutoff <= chat_session.summarized_count:
                return None

            delta_messages = all_messages[chat_session.summarized_count : cutoff]
            delta_text = "\n".join(f"{m.role.upper()}: {m.content}" for m in delta_messages)
            existing_summary = chat_session.summary or "None yet."

            prompt = f"""Update the concise running summary of this conversation by folding in the new messages below.
Preserve key entities, document topics, numbers, and user intents. Keep it under 150 words.

Previous Summary:
{existing_summary}

New Messages to Fold In:
{delta_text}

Updated Summary:"""

            new_summary = self._generate_text(prompt)
            if not new_summary:
                # Deterministic fallback if offline / no API key
                combined = f"{chat_session.summary}\n{delta_text}" if chat_session.summary else delta_text
                new_summary = combined[-1000:]

            chat_session.summary = new_summary
            chat_session.summarized_count = cutoff
            db.add(chat_session)
            await db.commit()
            return new_summary

    # ==========================================
    # LONG-TERM MEMORY (LTM)
    # ==========================================

    async def get_tenant_memories(
        self,
        tenant_id: str,
        memory_type: Optional[str] = None,
    ) -> List[TenantMemory]:
        """Lists all LTM records strictly scoped to `tenant_id`."""
        async with AsyncSessionLocal() as db:
            stmt = select(TenantMemory).where(TenantMemory.tenant_id == tenant_id)
            if memory_type:
                stmt = stmt.where(TenantMemory.memory_type == memory_type)
            stmt = stmt.order_by(TenantMemory.updated_at.desc())
            res = await db.execute(stmt)
            return list(res.scalars().all())

    async def upsert_memory(
        self,
        tenant_id: str,
        memory_type: str,
        key: str,
        value: str,
    ) -> TenantMemory:
        """Creates or updates a tenant memory by `(tenant_id, memory_type, key)` and invalidates cache."""
        norm_type = memory_type if memory_type in VALID_MEMORY_TYPES else "fact"
        clean_key = key.strip()
        clean_value = value.strip()

        async with AsyncSessionLocal() as db:
            stmt = select(TenantMemory).where(
                TenantMemory.tenant_id == tenant_id,
                TenantMemory.memory_type == norm_type,
                TenantMemory.key == clean_key,
            )
            existing = (await db.execute(stmt)).scalars().first()
            if existing:
                existing.value = clean_value
                existing.updated_at = utc_now()
                db.add(existing)
                await db.commit()
                await db.refresh(existing)
                record = existing
            else:
                record = TenantMemory(
                    id=str(uuid.uuid4()),
                    tenant_id=tenant_id,
                    memory_type=norm_type,
                    key=clean_key,
                    value=clean_value,
                )
                db.add(record)
                await db.commit()
                await db.refresh(record)

        try:
            semantic_cache.invalidate(tenant_id=tenant_id)
        except Exception as e:
            print(f"[MemoryService] Warning: Failed to invalidate semantic cache on LTM upsert: {e}")

        return record

    async def delete_memory(self, tenant_id: str, memory_id: str) -> bool:
        """Deletes an LTM record if it belongs to `tenant_id`."""
        async with AsyncSessionLocal() as db:
            record = await db.get(TenantMemory, memory_id)
            if not record or record.tenant_id != tenant_id:
                return False
            await db.delete(record)
            await db.commit()

        try:
            semantic_cache.invalidate(tenant_id=tenant_id)
        except Exception as e:
            print(f"[MemoryService] Warning: Failed to invalidate semantic cache on LTM delete: {e}")
        return True

    async def delete_memory_by_key(self, tenant_id: str, key: str) -> bool:
        """Deletes any LTM record matching `(tenant_id, key)`."""
        async with AsyncSessionLocal() as db:
            stmt = select(TenantMemory).where(
                TenantMemory.tenant_id == tenant_id,
                TenantMemory.key == key.strip(),
            )
            records = (await db.execute(stmt)).scalars().all()
            if not records:
                return False
            for rec in records:
                await db.delete(rec)
            await db.commit()

        try:
            semantic_cache.invalidate(tenant_id=tenant_id)
        except Exception:
            pass
        return True

    @traceable(run_type="llm", name="LTM Background Extractor")
    async def extract_and_persist_ltm(
        self,
        tenant_id: str,
        user_message: str,
        assistant_message: str = "",
    ) -> List[Dict[str, str]]:
        """
        Lightweight LLM check that extracts durable tenant preferences, glossaries/acronyms,
        or org facts from the user's message and reconciles them against existing LTM.
        Defaults to [] (no-op) for standard document questions.
        """
        existing = await self.get_tenant_memories(tenant_id)
        existing_str = (
            "\n".join(f"- [{m.memory_type}] {m.key}: {m.value}" for m in existing)
            if existing
            else "None"
        )

        prompt = f"""You are a memory extractor for an enterprise document assistant.
Inspect the user message and decide if it states any DURABLE long-term memory worth persisting across sessions:
- "preference": user formatting/style/language preferences (e.g., "Always answer in bullet points", "Use concise tables")
- "glossary": internal acronyms or domain definitions (e.g., "KDP means Key Decision Point")
- "fact": persistent org-level context (e.g., "Our fiscal year starts in April")

Rules:
1. Do NOT store one-off questions or facts that belong to uploaded documents.
2. For normal questions (e.g., "What is the Severity 1 SLA?"), return [].
3. If an existing memory is updated or contradicted, emit UPSERT with the same key.
4. If the user asks to forget a preference/definition, emit DELETE.
5. Respond ONLY with a valid JSON array (no markdown fences). Example:
[{{"action": "UPSERT", "memory_type": "preference", "key": "response_format", "value": "Always use bullet points"}}]

Existing Memories:
{existing_str}

User Message:
{user_message}

JSON Actions:"""

        raw = self._generate_text(prompt)
        if not raw:
            return []

        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            actions = json.loads(cleaned)
        except json.JSONDecodeError:
            return []

        if not isinstance(actions, list):
            return []

        applied: List[Dict[str, str]] = []
        for item in actions:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", "")).upper()
            key = str(item.get("key", "")).strip()
            if not key:
                continue

            if action == "UPSERT":
                value = str(item.get("value", "")).strip()
                m_type = str(item.get("memory_type", "fact")).lower()
                if value:
                    await self.upsert_memory(tenant_id, m_type, key, value)
                    applied.append({"action": "UPSERT", "memory_type": m_type, "key": key, "value": value})
            elif action == "DELETE":
                if await self.delete_memory_by_key(tenant_id, key):
                    applied.append({"action": "DELETE", "key": key})

        return applied

    # ==========================================
    # PROMPT FORMATTING HELPERS
    # ==========================================

    def format_stm_for_prompt(
        self,
        summary: Optional[str],
        recent_messages: List[Dict[str, str]],
    ) -> str:
        parts: List[str] = []
        if summary:
            parts.append(f"Earlier Conversation Summary:\n{summary}")
        if recent_messages:
            turns = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent_messages)
            parts.append(f"Recent Turns:\n{turns}")
        return "\n\n".join(parts)

    def format_ltm_for_prompt(self, memories: List[TenantMemory]) -> str:
        if not memories:
            return ""
        lines = [f"- [{m.memory_type.upper()}] {m.key}: {m.value}" for m in memories]
        return "\n".join(lines)


# Global singleton instance
memory_service = MemoryService()
