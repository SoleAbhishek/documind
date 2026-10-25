import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlmodel import select
from src.config import settings
from src.db.session import init_db, AsyncSessionLocal
from src.db.models import Tenant, Document, ChatSession, ChatMessage
from src.services.ingestion import ingestion_service
from src.services.chat_service import chat_service

SAMPLE_SLA_MARKDOWN = """# Acme Cloud Enterprise Service Level Agreement (SLA)

## 1. System Availability Commitment
Acme guarantees a Monthly Uptime Percentage of at least 99.95% during each calendar month.
If Acme does not achieve 99.95% uptime, Customer is entitled to SLA Service Credits.

## 2. Severity Levels & Incident Response Times
The following response times are guaranteed based on incident severity:

| Severity Level | Definition | Initial Response Target | Resolution Target |
| :--- | :--- | :--- | :--- |
| **Severity 1 (Critical)** | Core service completely down, catastrophic business impact | < 15 minutes (24x7) | < 4 hours |
| **Severity 2 (High)** | Core service impaired, business operations degraded | < 1 hour (24x7) | < 8 hours |
| **Severity 3 (Medium)** | Non-critical functionality impacted, business workaround available | < 4 business hours | Next release |
| **Severity 4 (Low)** | General question, cosmetic UI bug, feature request | < 1 business day | Scheduled roadmap |

## 3. SLA Service Credit Schedule
Service credits are calculated as a percentage of the monthly subscription fee:
- Availability between 99.0% and 99.94%: 10% credit
- Availability between 95.0% and 98.99%: 25% credit
- Availability below 95.0%: 50% credit
"""


async def main():
    print("=" * 70)
    print("DocuMind Hybrid RAG Chat Service E2E Test")
    print("=" * 70)

    # 1. Initialize DB tables
    await init_db()

    tenant_id = "tenant-e2e-chat"
    doc_id = "doc-sla-chat-test"

    # 2. Upload to MinIO and Seed Tenant and Document in Postgres
    from src.storage.service import storage_service
    from src.db.models import DocumentStatus

    file_bytes = SAMPLE_SLA_MARKDOWN.encode("utf-8")
    filename = "Acme_SLA_Policy.md"
    storage_key = storage_service.upload_document(tenant_id, doc_id, filename, file_bytes, "text/markdown")
    print(f"[Test] Uploaded to MinIO: {storage_key}")

    async with AsyncSessionLocal() as session:
        t_stmt = select(Tenant).where(Tenant.id == tenant_id)
        t_res = await session.execute(t_stmt)
        tenant = t_res.scalars().first()
        if not tenant:
            tenant = Tenant(id=tenant_id, name="Chat Test Corp")
            session.add(tenant)
            await session.commit()
            print(f"[1/5] Created Tenant: {tenant_id}")
        else:
            print(f"[1/5] Tenant {tenant_id} already exists.")

        d_stmt = select(Document).where(Document.id == doc_id)
        d_res = await session.execute(d_stmt)
        doc = d_res.scalars().first()
        if doc:
            await session.delete(doc)
            await session.commit()

        doc = Document(
            id=doc_id,
            tenant_id=tenant_id,
            filename=filename,
            content_type="text/markdown",
            file_size_bytes=len(file_bytes),
            storage_key=storage_key,
            status=DocumentStatus.PENDING.value
        )
        session.add(doc)
        await session.commit()
        print(f"[2/5] Created Document record: {doc_id} (status=PENDING)")

    # 3. Ingest document via IngestionService
    print("[3/5] Ingesting document into MinIO, Qdrant & Postgres...")
    await ingestion_service.process_document(tenant_id, doc_id)

    # 4. Stream Chat Query
    query = "What is the guaranteed response time and resolution target for Severity 1 critical incidents?"
    print(f"\n[4/5] Executing Chat Query: '{query}'")
    print("-" * 70)

    streamed_text = ""
    citations = []
    session_id = None

    async for event in chat_service.stream_chat(
        tenant_id=tenant_id,
        query=query,
        retrieval_limit=5,
        rerank_top_k=3
    ):
        evt_type = event.get("type")
        if evt_type == "session":
            session_id = event["session_id"]
            print(f"[Session Created]: {session_id}")
        elif evt_type == "status":
            print(f"[Status]: {event['stage'].upper()} -> {event['message']}")
        elif evt_type == "citations":
            citations = event["citations"]
            print(f"[Citations Received]: {len(citations)} sources")
            for c in citations:
                print(f"   * {c['source_id']}: {c['header_path']} (RRF: {c['rrf_score']:.4f}, Rerank: {c['rerank_score']:.4f})")
        elif evt_type == "token":
            token = event["content"]
            streamed_text += token
            print(token, end="", flush=True)
        elif evt_type == "done":
            print("\n" + "-" * 70)
            print(f"[Done]: Completed answer generation. Message ID: {event.get('message_id')}")

    # 5. Verify PostgreSQL Persistence & Audit Trail
    print("\n[5/5] Verifying PostgreSQL Session & Audit Trail...")
    async with AsyncSessionLocal() as session:
        # Check ChatSession
        sess_stmt = select(ChatSession).where(ChatSession.id == session_id)
        sess_res = await session.execute(sess_stmt)
        saved_session = sess_res.scalars().first()
        assert saved_session is not None, "ChatSession was not saved in Postgres!"

        # Check ChatMessages
        msg_stmt = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at)
        msg_res = await session.execute(msg_stmt)
        messages = msg_res.scalars().all()
        assert len(messages) == 2, f"Expected 2 messages (user + assistant), found {len(messages)}"
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[1].retrieved_chunks is not None, "Assistant message retrieved_chunks audit JSON is empty!"
        assert len(messages[1].retrieved_chunks) > 0, "No citations stored in retrieved_chunks JSON!"

        print(f"Verified ChatSession: {saved_session.id} ('{saved_session.title}')")
        print(f"Verified User Message: '{messages[0].content}'")
        print(f"Verified Assistant Message ({len(messages[1].content)} chars)")
        print(f"Verified Audit Provenance: {len(messages[1].retrieved_chunks)} citations logged with scores.")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! CHAT SERVICE IS FULLY OPERATIONAL!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
