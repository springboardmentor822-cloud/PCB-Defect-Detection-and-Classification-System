from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.services.model_manager import ModelManager


configure_logging()
log = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="PCB Defect Detection System",
        version="1.0.0",
        description="FastAPI backend for YOLOv5 PCB defect detection + template comparison.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")

    # Expose outputs for browser downloads/preview
    outputs_dir = Path(settings.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

    @app.on_event("startup")
    def _startup() -> None:
        t0 = time.perf_counter()
        ModelManager.get().warmup()
        log.info("Startup complete in %.2fs", time.perf_counter() - t0)

    return app


app = create_app()
