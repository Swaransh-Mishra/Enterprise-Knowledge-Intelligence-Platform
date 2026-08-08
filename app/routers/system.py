"""
System Routes.
"""

from fastapi import APIRouter

from app.config import settings

router = APIRouter(
    prefix="/system",
    tags=["System"]
)


@router.get("/info")
async def system_info():

    return {
        "application": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": "Development"
    }