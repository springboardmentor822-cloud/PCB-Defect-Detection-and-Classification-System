from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.core.config import settings


ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def new_run_id() -> str:
    return secrets.token_hex(8)


def _validate_ext(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Invalid file type '{ext}'. Use jpg/png/bmp/tiff.")
    return ext


async def save_upload(upload: UploadFile, run_id: str, role: str) -> Path:
    _validate_ext(upload.filename or "")
    uploads_dir = Path(settings.uploads_dir) / run_id
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dst = uploads_dir / f"{role}{Path(upload.filename).suffix.lower()}"

    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    dst.write_bytes(content)
    return dst


def run_output_dir(run_id: str) -> Path:
    d = Path(settings.outputs_dir) / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d
