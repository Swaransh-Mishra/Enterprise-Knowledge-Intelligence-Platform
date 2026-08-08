from .health import router as health_router
from .system import router as system_router
from .documents import router as documents_router
from .search import router as search_router
from .chat import router as chat_router
from .analytics import router as analytics_router

__all__ = [
    "health_router",
    "system_router",
    "documents_router",
    "search_router",
    "chat_router",
    "analytics_router",
]