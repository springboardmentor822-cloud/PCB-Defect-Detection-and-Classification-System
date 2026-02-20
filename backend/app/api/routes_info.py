from __future__ import annotations

from fastapi import APIRouter

from app.services.model_manager import ModelManager

router = APIRouter()


@router.get("/health")
def health() -> dict:
    mm = ModelManager.get()
    return {
        "status": "ok",
        "model_loaded": mm.is_loaded,
        "device": mm.device,
    }


@router.get("/model")
def model_info() -> dict:
    mm = ModelManager.get()
    return mm.model_details()
