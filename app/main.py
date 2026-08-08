"""
FastAPI Application Entry Point.
"""
from fastapi import FastAPI

from app.config import settings
from app.utils.logger import logger

from app.routers import (
    health_router,
    system_router,
    documents_router,
    search_router,
    chat_router,
    analytics_router,
)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    description="Enterprise Knowledge Intelligence Platform API"
)


@app.get("/", tags=["Root"])
async def root():

    logger.info("Root endpoint accessed.")

    return {
        "application": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "Running Successfully"
    }


app.include_router(health_router)
app.include_router(system_router)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(analytics_router)