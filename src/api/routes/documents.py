import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from src.db.session import get_db
from src.db.models import Document, Tenant, DocumentStatus
from src.storage.service import storage_service
from src.services.ingestion import ingestion_service
from src.search.vector_store import vector_store
from src.api.schemas import DocumentResponse, DocumentUploadResponse

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    tenant_id: str = Form(..., description="Target tenant ID"),
    file: UploadFile = File(..., description="Document file (.pdf, .md, .txt)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Accepts document upload, saves raw file into MinIO object storage,
    creates PostgreSQL document record (PENDING), and dispatches
    asynchronous background ingestion (Parsing -> Chunking -> Hybrid Qdrant Indexing).
    """
    # 1. Verify tenant exists
    tenant = await db.get(Tenant, tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' does not exist."
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    document_id = str(uuid.uuid4())
    filename = file.filename or "uploaded_document"
    content_type = file.content_type or "application/octet-stream"

    # 2. Upload to MinIO / S3 Object Storage
    try:
        storage_key = storage_service.upload_document(
            tenant_id=tenant_id,
            doc_id=document_id,
            filename=filename,
            data=file_bytes,
            content_type=content_type
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document to storage backend: {e}"
        )

    # 3. Create initial Document entry in PostgreSQL
    doc_record = Document(
        id=document_id,
        tenant_id=tenant_id,
        filename=filename,
        content_type=content_type,
        file_size_bytes=len(file_bytes),
        storage_key=storage_key,
        status=DocumentStatus.PENDING.value
    )
    db.add(doc_record)
    await db.commit()
    await db.refresh(doc_record)

    # 4. Enqueue background ingestion pipeline
    background_tasks.add_task(
        ingestion_service.process_document,
        tenant_id=tenant_id,
        document_id=document_id
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        status=DocumentStatus.PENDING.value,
        message="Document uploaded successfully. Background OCR and hybrid vector indexing initiated."
    )


@router.get("", response_model=List[DocumentResponse])
async def list_documents(
    tenant_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Lists all indexed documents and their statuses for a given tenant."""
    stmt = (
        select(Document)
        .where(Document.tenant_id == tenant_id)
        .order_by(Document.created_at.desc())
    )
    res = await db.execute(stmt)
    return res.scalars().all()


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    tenant_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Gets details and indexing status of a single document."""
    doc = await db.get(Document, document_id)
    if not doc or doc.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found for tenant '{tenant_id}'."
        )
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document(
    document_id: str,
    tenant_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Deletes document metadata from PostgreSQL, removes raw binary from MinIO,
    and removes all dense/sparse vectors from Qdrant.
    """
    doc = await db.get(Document, document_id)
    if not doc or doc.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found for tenant '{tenant_id}'."
        )

    # 1. Delete vectors from Qdrant
    try:
        vector_store.delete_document_points(tenant_id, document_id)
    except Exception as e:
        print(f"[Warning] Failed to delete Qdrant points: {e}")

    # 2. Delete raw object from MinIO
    try:
        storage_service.delete_document(doc.storage_key)
    except Exception as e:
        print(f"[Warning] Failed to delete storage object: {e}")

    # 3. Delete from PostgreSQL (cascades to chunks)
    await db.delete(doc)
    await db.commit()

    return {"status": "success", "message": f"Document '{document_id}' deleted successfully."}
