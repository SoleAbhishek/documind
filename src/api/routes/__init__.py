from .tenants import router as tenants_router
from .documents import router as documents_router
from .chat import router as chat_router

__all__ = ["tenants_router", "documents_router", "chat_router"]
