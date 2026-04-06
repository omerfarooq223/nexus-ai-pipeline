"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.db.mysql_client import MySQLClient
from src.pipeline.multimodal_service import MultiModalService

app = FastAPI(title="Multi-Modal AI Pipeline", version="1.0.0")
service = MultiModalService()


@app.get("/health")
def health() -> dict:
    """Simple liveness and database connectivity check."""
    db = MySQLClient()
    return {"api": "ok", "db": db.healthcheck()}


@app.post("/infer")
async def infer(image: UploadFile = File(...), query: str = Form(...)) -> dict:
    """Accept image and text query for combined inference."""
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
