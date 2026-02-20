from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.core.config import settings

router = APIRouter()


@router.get("/results/{run_id}/manifest")
def result_manifest(run_id: str) -> dict:
    run_dir = Path(settings.outputs_dir) / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found.")

    json_report = run_dir / "report.json"
    csv_report = run_dir / "report.csv"

    return {
        "run_id": run_id,
        "paths": {
            "run_dir": str(run_dir),
            "annotated_image": str(run_dir / "annotated.jpg"),
            "diff_image": str(run_dir / "diff.jpg"),
            "overlay_image": str(run_dir / "overlay.jpg"),
            "report_json": str(json_report) if json_report.exists() else None,
            "report_csv": str(csv_report) if csv_report.exists() else None,
            "rois_dir": str(run_dir / "rois"),
        },
    }


@router.get("/results/{run_id}/download")
def download_result(
    run_id: str,
    kind: str = Query("all", description="annotated|csv|json|zip|diff|overlay|all"),
):
    run_dir = Path(settings.outputs_dir) / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found.")

    kind = (kind or "all").lower()
    mapping = {
        "annotated": run_dir / "annotated.jpg",
        "diff": run_dir / "diff.jpg",
        "overlay": run_dir / "overlay.jpg",
        "csv": run_dir / "report.csv",
        "json": run_dir / "report.json",
    }

    if kind in mapping:
        f = mapping[kind]
        if not f.exists():
            raise HTTPException(status_code=404, detail=f"{kind} not available for this run.")
        return FileResponse(path=str(f), filename=f.name)

    # Zip entire run folder
    zip_path = Path(settings.outputs_dir) / f"{run_id}.zip"
    if kind in {"zip", "all"}:
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=str(run_dir))
        return FileResponse(path=str(zip_path), filename=zip_path.name)

    raise HTTPException(status_code=400, detail="Invalid kind.")
