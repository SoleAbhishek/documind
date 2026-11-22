from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.db.models import Tenant
from src.services.memory_service import memory_service
from src.api.schemas import TenantMemoryCreate, TenantMemoryResponse

router = APIRouter(prefix="/memory", tags=["Memory (LTM)"])


@router.post("", response_model=TenantMemoryResponse, status_code=status.HTTP_201_CREATED)
async def upsert_tenant_memory(
    payload: TenantMemoryCreate,
    db: AsyncSession = Depends(get_db),
):
    """Creates or updates a tenant-scoped Long-Term Memory entry (glossary, preference, or fact)."""
    tenant = await db.get(Tenant, payload.tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{payload.tenant_id}' does not exist.",
        )

    record = await memory_service.upsert_memory(
        tenant_id=payload.tenant_id,
        memory_type=payload.memory_type,
        key=payload.key,
        value=payload.value,
    )
    return record


@router.get("", response_model=List[TenantMemoryResponse])
async def list_tenant_memories(
    tenant_id: str,
    memory_type: Optional[str] = None,
):
    """Lists all Long-Term Memory records for a tenant."""
    return await memory_service.get_tenant_memories(
        tenant_id=tenant_id,
        memory_type=memory_type,
    )


@router.delete("/{memory_id}", status_code=status.HTTP_200_OK)
async def delete_tenant_memory(
    memory_id: str,
    tenant_id: str,
):
    """Deletes a Long-Term Memory record belonging to the specified tenant."""
    deleted = await memory_service.delete_memory(tenant_id=tenant_id, memory_id=memory_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory '{memory_id}' not found for tenant '{tenant_id}'.",
        )
    return {"status": "success", "message": f"Memory '{memory_id}' deleted successfully."}
