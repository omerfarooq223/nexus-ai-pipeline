"""FastAPI application entrypoint."""

from __future__ import annotations
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import logging

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from src.db.mysql_client import MySQLClient
from src.pipeline.multimodal_service import MultiModalService

logger = logging.getLogger(__name__)


def _get_service(request: Request) -> MultiModalService:
    """Return the shared multimodal service instance."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Service not fully initialized yet.")
    return service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize shared resources at application startup."""
    logger.info("Starting up: Initializing connection pool and pre-warming models...")
    try:
        MySQLClient.init_pool()
        app.state.service = MultiModalService()
        service = app.state.service
        service.image_classifier._ensure_model_loaded()
        logger.info("Startup complete: Models loaded and DB pooled.")
    except Exception as e:
        logger.error("Startup failed: %s", e)
        raise
    yield

app = FastAPI(title="Multi-Modal AI Pipeline", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Simple liveness and database connectivity check."""
    db = MySQLClient()
    return {"api": "ok", "db": db.healthcheck()}


@app.post("/infer")
async def infer(request: Request, image: UploadFile = File(...), query: str = Form(...)) -> dict:
    """Accept image and text query for combined inference."""
    service = _get_service(request)

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    try:
        return service.run_inference(
            image_name=image.filename or "unknown",
            image_bytes=image_bytes,
            query_text=query,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
