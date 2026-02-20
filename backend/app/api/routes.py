from __future__ import annotations

from fastapi import APIRouter

from app.api.routes_detect import router as detect_router
from app.api.routes_info import router as info_router
from app.api.routes_results import router as results_router

router = APIRouter()
router.include_router(info_router, tags=["info"])
router.include_router(detect_router, tags=["detection"])
router.include_router(results_router, tags=["results"])
