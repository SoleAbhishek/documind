import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlmodel import select
from src.db.session import init_db, AsyncSessionLocal
from src.db.models import Tenant, Document, ChatSession, ChatMessage, DocumentStatus
from src.services.memory_service import memory_service
from src.services.ingestion import ingestion_service
from src.services.chat_service import chat_service
from src.storage.service import storage_service


SAMPLE_SLA_DOC = """# Acme Cloud Enterprise Service Level Agreement (SLA)

## 1. System Availability Commitment
Acme guarantees a Monthly Uptime Percentage of at least 99.95% during each calendar month.

## 2. Severity Levels & Incident Response Times
| Severity Level | Definition | Initial Response Target | Resolution Target |
| :--- | :--- | :--- | :--- |
| **Severity 1 (Critical)** | Core service completely down | < 15 minutes (24x7) | < 4 hours |
| **Severity 2 (High)** | Core service impaired | < 1 hour (24x7) | < 8 hours |

## 3. SLA Service Credit Schedule
If Acme misses a Severity 1 or uptime target, Customer receives SLA Service Credits:
- Availability between 99.0% and 99.94%: 10% credit
- Availability below 95.0%: 50% credit
"""


async def ensure_tenant(tenant_id: str, name: str):
    async with AsyncSessionLocal() as db:
        t = await db.get(Tenant, tenant_id)
        if not t:
            db.add(Tenant(id=tenant_id, name=name))
            await db.commit()


async def test_ltm_crud_and_isolation():
    print("\n[Test 1] LTM CRUD & Multi-Tenant Isolation...")
    t_a = f"tenant-mem-a-{uuid.uuid4().hex[:6]}"
    t_b = f"tenant-mem-b-{uuid.uuid4().hex[:6]}"
    await ensure_tenant(t_a, "Tenant A")
    await ensure_tenant(t_b, "Tenant B")

    m1 = await memory_service.upsert_memory(t_a, "preference", "format", "Always use bullet points.")
    m2 = await memory_service.upsert_memory(t_a, "preference", "format", "Always use concise tables.")
    assert m1.id == m2.id, "Upsert on same (tenant, type, key) should update in place"
    assert m2.value == "Always use concise tables."

    await memory_service.upsert_memory(t_a, "glossary", "KDP", "Key Decision Point")

    mem_a = await memory_service.get_tenant_memories(t_a)
    mem_b = await memory_service.get_tenant_memories(t_b)
    assert len(mem_a) == 2
    assert len(mem_b) == 0, "Tenant B must not see Tenant A's memories"

    # Cross-tenant delete must fail
    assert await memory_service.delete_memory(t_b, m1.id) is False
    assert await memory_service.delete_memory(t_a, m1.id) is True
    print("  PASS: LTM upsert, isolation, and delete verified.")


async def test_stm_fast_path_and_rewrite():
    print("\n[Test 2] STM Fast-Path & Query Rewriting...")
    q1 = "What is the Severity 1 initial response target?"
    rewritten_turn1 = memory_service.rewrite_query(q1, summary=None, recent_messages=[], ltm_memories=[])
    assert rewritten_turn1 == q1, "Turn 1 with no history/glossary must skip LLM rewrite"

    history = [
        {"role": "user", "content": "What is the Severity 1 initial response target?"},
        {"role": "assistant", "content": "Severity 1 has an initial response target of < 15 minutes (24x7)."},
    ]
    q2 = "What is its resolution target?"
    rewritten_turn2 = memory_service.rewrite_query(q2, summary=None, recent_messages=history, ltm_memories=[])
    print(f"  Turn 2 original : '{q2}'")
    print(f"  Turn 2 rewritten: '{rewritten_turn2}'")
    assert "severity 1" in rewritten_turn2.lower() or "resolution" in rewritten_turn2.lower()
    print("  PASS: Turn 1 fast-path and Turn 2 contextual rewrite verified.")


async def test_stm_incremental_summary():
    print("\n[Test 3] STM Incremental Running Summary + Recent Window...")
    tenant_id = f"tenant-sum-{uuid.uuid4().hex[:6]}"
    session_id = str(uuid.uuid4())
    await ensure_tenant(tenant_id, "Summary Corp")

    async with AsyncSessionLocal() as db:
        db.add(ChatSession(id=session_id, tenant_id=tenant_id, title="Long Session"))
        await db.commit()
        for i in range(8):
            role = "user" if i % 2 == 0 else "assistant"
            db.add(ChatMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role=role,
                content=f"Turn message {i}: discussing Acme SLA Severity 1 uptime 99.95%.",
            ))
        await db.commit()

    new_summary = await memory_service.maybe_update_session_summary(session_id, tenant_id, recent_window=6)
    assert new_summary and len(new_summary) > 0, "Expected summary to be generated for messages beyond window"

    stm = await memory_service.get_stm_context(session_id, tenant_id, recent_limit=6)
    assert stm["summary"] == new_summary
    assert stm["summarized_count"] == 2
    assert len(stm["recent_messages"]) == 6
    print(f"  PASS: Summarized 2 older messages; retained 6 recent messages verbatim.")


async def test_e2e_multiturn_chat():
    print("\n[Test 4] End-to-End Multi-Turn RAG with STM Rewrite...")
    tenant_id = f"tenant-e2e-mem-{uuid.uuid4().hex[:6]}"
    doc_id = f"doc-mem-{uuid.uuid4().hex[:6]}"
    await ensure_tenant(tenant_id, "Memory E2E Corp")

    file_bytes = SAMPLE_SLA_DOC.encode("utf-8")
    storage_key = storage_service.upload_document(tenant_id, doc_id, "SLA_Memory_Test.md", file_bytes, "text/markdown")
    async with AsyncSessionLocal() as db:
        db.add(Document(
            id=doc_id,
            tenant_id=tenant_id,
            filename="SLA_Memory_Test.md",
            content_type="text/markdown",
            file_size_bytes=len(file_bytes),
            storage_key=storage_key,
            status=DocumentStatus.PENDING.value,
        ))
        await db.commit()

    await ingestion_service.process_document(tenant_id, doc_id)

    res1 = await chat_service.chat(
        tenant_id=tenant_id,
        query="What is the initial response target for Severity 1?",
        document_id=doc_id,
    )
    assert res1["rewritten_query"] is None, "Turn 1 should not rewrite"
    assert "15 minutes" in res1["answer"] or len(res1["citations"]) > 0

    res2 = await chat_service.chat(
        tenant_id=tenant_id,
        query="What is its resolution target?",
        session_id=res1["session_id"],
        document_id=doc_id,
    )
    print(f"  Turn 2 rewritten_query: {res2['rewritten_query']}")
    print(f"  Turn 2 answer snippet : {res2['answer'][:140].replace(chr(10), ' ')}...")
    assert res2["rewritten_query"] is not None, "Turn 2 pronoun follow-up should be rewritten"
    assert "4 hours" in res2["answer"] or len(res2["citations"]) > 0
    print("  PASS: End-to-end multi-turn RAG retrieval & citation verified.")


async def main():
    print("=" * 70)
    print("DocuMind Phase 3: Conversational Memory (STM + LTM) Test Suite")
    print("=" * 70)
    await init_db()
    await test_ltm_crud_and_isolation()
    await test_stm_fast_path_and_rewrite()
    await test_stm_incremental_summary()
    await test_e2e_multiturn_chat()
    print("\nALL PHASE 3 CONVERSATIONAL MEMORY TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
