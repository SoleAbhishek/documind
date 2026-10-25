from typing import List
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from src.db.session import get_db
from src.db.models import Tenant
from src.api.schemas import TenantCreate, TenantResponse

router = APIRouter(prefix="/tenants", tags=["Tenants"])


@router.post("", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
async def create_tenant(payload: TenantCreate, db: AsyncSession = Depends(get_db)):
    """Creates a new multi-tenant isolation space."""
    tenant = Tenant(
        id=str(uuid.uuid4()),
        name=payload.name
    )
    db.add(tenant)
    await db.commit()
    await db.refresh(tenant)
    return tenant


@router.get("", response_model=List[TenantResponse])
async def list_tenants(db: AsyncSession = Depends(get_db)):
    """Lists all active tenants."""
    stmt = select(Tenant).order_by(Tenant.created_at.desc())
    res = await db.execute(stmt)
    return res.scalars().all()
