import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from sqlmodel import select
from src.db.session import AsyncSessionLocal, init_db
from src.db.models import Tenant, Document, ChunkRecord, DocumentStatus
from src.storage import storage_service
from src.services.ingestion import ingestion_service
from src.search.vector_store import vector_store


async def test_full_pipeline():
    await init_db()

    # 1. Create a Tenant
    async with AsyncSessionLocal() as session:
        tenant = Tenant(name="Acme Corp (Test Tenant)")
        session.add(tenant)
        await session.commit()
        await session.refresh(tenant)
        tenant_id = tenant.id
        print(f"[Test] Created Tenant: {tenant_id}")

    # 2. Upload file to MinIO
    sample_content = b"""# Acme Cloud SLA Policy

## Section 1: Uptime Commitments
Acme Cloud guarantees 99.95% monthly service uptime for all enterprise tier accounts.
If uptime drops below 99.95%, customer receives a 25% credit on their monthly bill.

## Section 2: Enterprise Pricing Table
The following matrix defines the pricing tiers:

| Tier | Monthly Fee | Max Users | Support SLA |
|---|---|---|---|
| Starter | $49 | 5 | 24 Hours |
| Business | $199 | 25 | 4 Hours |
| Enterprise | $999 | Unlimited | 1 Hour Dedicated |
"""
    doc_id = "test_doc_001"
    filename = "Acme_SLA_Policy.md"
    storage_key = storage_service.upload_document(tenant_id, doc_id, filename, sample_content, "text/markdown")
    print(f"[Test] Uploaded to MinIO: {storage_key}")

    # 3. Create Document record in PostgreSQL
    async with AsyncSessionLocal() as session:
        # Check if already exists from prior run
        existing = await session.get(Document, doc_id)
        if existing:
            await session.delete(existing)
            await session.commit()

        doc = Document(
            id=doc_id,
            tenant_id=tenant_id,
            filename=filename,
            content_type="text/markdown",
            file_size_bytes=len(sample_content),
            storage_key=storage_key,
            status=DocumentStatus.PENDING.value
        )
        session.add(doc)
        await session.commit()
        print("[Test] Created Document record in PostgreSQL (status=PENDING)")

    # 4. Run Ingestion Pipeline!
    print("[Test] Triggering ingestion_service.process_document()...")
    processed_doc = await ingestion_service.process_document(tenant_id, doc_id)
    print(f"[Test] Ingestion complete! Status: {processed_doc.status}, Chunks: {processed_doc.num_chunks}")

    # 5. Verify PostgreSQL ChunkRecords
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(ChunkRecord).where(ChunkRecord.document_id == doc_id))
        chunks = result.scalars().all()
        print(f"[Test] Verified {len(chunks)} ChunkRecords in PostgreSQL.")
        for c in chunks:
            print(f"   -> Chunk #{c.chunk_index}: HeaderPath='{c.header_path}' | QdrantPointId={c.qdrant_point_id}")

    # 6. Verify Qdrant Hybrid Retrieval
    print("\n[Test] Testing Qdrant Hybrid Search for: 'What is the monthly fee for Enterprise tier?'")
    search_res = vector_store.search_hybrid(
        tenant_id=tenant_id,
        query="What is the monthly fee for Enterprise tier?",
        limit=2
    )
    print(f"[Test] Retrieved {len(search_res)} chunks from Qdrant:")
    for r in search_res:
        snippet = r["text"].replace("\n", " ")[:90]
        print(f"   RRF Score: {r['rrf_score']} | Path: {r['header_path']} | Snippet: {snippet}...")


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
