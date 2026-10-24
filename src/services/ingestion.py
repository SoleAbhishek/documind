import traceback
from datetime import datetime, timezone
from typing import Optional
from sqlmodel import select

from src.db.session import AsyncSessionLocal
from src.db.models import Document, ChunkRecord, DocumentStatus, utc_now
from src.storage import storage_service
from src.parsing import document_parser
from src.chunking import markdown_chunker
from src.search.vector_store import vector_store


class IngestionService:
    """
    Orchestrates the complete document ingestion pipeline:
    1. Updates PostgreSQL document status to 'PARSING'.
    2. Downloads raw binary from MinIO object storage.
    3. Runs OCR & layout analysis via DocumentParser (LlamaParse).
    4. Splits structured Markdown into context-enriched chunks via MarkdownChunker.
    5. Computes Dense + Sparse BM25 vectors and upserts to Qdrant.
    6. Saves ChunkRecords in PostgreSQL with two-way point IDs.
    7. Updates document status to 'INDEXED'.
    """

    async def process_document(self, tenant_id: str, document_id: str) -> Document:
        """
        Executes end-to-end ingestion for a single document.
        """
        print(f"[Ingestion] Starting pipeline for Document ID: {document_id} (Tenant: {tenant_id})")

        # -------------------------------------------------------------
        # Step 1: Transition Document status to 'PARSING'
        # -------------------------------------------------------------
        async with AsyncSessionLocal() as session:
            doc = await session.get(Document, document_id)
            if not doc:
                raise ValueError(f"Document {document_id} not found in database.")
            if doc.tenant_id != tenant_id:
                raise PermissionError("Tenant ID mismatch.")

            doc.status = DocumentStatus.PARSING.value
            doc.updated_at = utc_now()
            session.add(doc)
            await session.commit()
            await session.refresh(doc)
            storage_key = doc.storage_key
            filename = doc.filename

        try:
            # -------------------------------------------------------------
            # Step 2: Download raw file bytes from MinIO / S3
            # -------------------------------------------------------------
            print(f"[Ingestion] Downloading file from MinIO key: '{storage_key}'...")
            file_bytes = storage_service.download_document(storage_key)

            # -------------------------------------------------------------
            # Step 3: Parse Document (OCR + Markdown extraction)
            # -------------------------------------------------------------
            print(f"[Ingestion] Parsing document '{filename}'...")
            parsed_doc = document_parser.parse_bytes(file_bytes, filename)
            print(f"[Ingestion] Parsed {parsed_doc.total_pages} page(s) using {parsed_doc.parser_used}.")

            # -------------------------------------------------------------
            # Step 4: Markdown-Aware Chunking (Breadcrumbs + Tables)
            # -------------------------------------------------------------
            print("[Ingestion] Generating context-enriched chunks...")
            chunks = markdown_chunker.chunk_document(parsed_doc)
            print(f"[Ingestion] Generated {len(chunks)} chunk(s).")

            if not chunks:
                raise ValueError(f"No text content could be extracted from '{filename}'.")

            # -------------------------------------------------------------
            # Step 5: Embed & Upsert Dual Vectors to Qdrant
            # -------------------------------------------------------------
            print("[Ingestion] Computing Dense & Sparse BM25 vectors and indexing into Qdrant...")
            point_ids = vector_store.upsert_chunks(
                tenant_id=tenant_id,
                document_id=document_id,
                chunks=chunks
            )

            # -------------------------------------------------------------
            # Step 6: Save ChunkRecords & Update Document to 'INDEXED'
            # -------------------------------------------------------------
            async with AsyncSessionLocal() as session:
                for chunk, point_id in zip(chunks, point_ids):
                    chunk_record = ChunkRecord(
                        document_id=document_id,
                        tenant_id=tenant_id,
                        chunk_index=chunk.chunk_index,
                        page_number=chunk.page_number,
                        header_path=chunk.header_path,
                        token_count=chunk.token_count,
                        qdrant_point_id=point_id
                    )
                    session.add(chunk_record)

                doc = await session.get(Document, document_id)
                doc.status = DocumentStatus.INDEXED.value
                doc.num_chunks = len(chunks)
                doc.error_message = None
                doc.updated_at = utc_now()
                session.add(doc)
                await session.commit()
                await session.refresh(doc)

            print(f"[Ingestion] SUCCESS: Document {document_id} fully indexed with {len(chunks)} chunks!")
            return doc

        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"[Ingestion] ERROR: Ingestion failed for document {document_id}:\n{error_trace}")

            # Record failure in PostgreSQL
            async with AsyncSessionLocal() as session:
                doc = await session.get(Document, document_id)
                if doc:
                    doc.status = DocumentStatus.FAILED.value
                    doc.error_message = str(e)
                    doc.updated_at = utc_now()
                    session.add(doc)
                    await session.commit()

            raise e


# Global singleton instance
ingestion_service = IngestionService()
