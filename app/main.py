"""Glove detection API (FastAPI): the CLI's detector behind an HTTP endpoint.

The model loads at startup in a background thread: /health answers at once, /ready is
503 until it is loaded. Uploads are decoded in memory and not stored.

Run: uv run python run.py
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

import numpy as np
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import MODEL_SHA256, Settings, get_settings
from services.images import ImageRejected, decode_upload

log = logging.getLogger("glove")


class DetectorLike(Protocol):
    def detect(self, image: np.ndarray, conf_threshold: float | None = None) -> list[dict]: ...


class Detection(BaseModel):
    label: str = Field(description="gloved_hand or bare_hand")
    confidence: float = Field(ge=0, le=1)
    bbox: tuple[int, int, int, int] = Field(
        description="[left, top, right, bottom] in pixels of the upright image"
    )


class DetectResponse(BaseModel):
    request_id: str
    detections: list[Detection] = Field(description="Most confident first")
    counts: dict[str, int]
    image: dict[str, int | str]
    timings: dict[str, float]
    model: str
    confidence_threshold: float


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str


class ApiError(Exception):
    def __init__(self, status: int, code: str, detail: str, headers: dict | None = None) -> None:
        self.status, self.code, self.detail, self.headers = status, code, detail, headers


def _error(request: Request, status: int, code: str, detail: str, headers=None) -> JSONResponse:
    rid = getattr(request.state, "request_id", "-")
    body = ErrorResponse(error=code, detail=detail, request_id=rid).model_dump()
    return JSONResponse(body, status_code=status, headers=headers)


def load_detector(settings: Settings) -> DetectorLike:
    import torch

    from detector import HandDetector

    torch.set_num_threads(settings.torch_threads)
    return HandDetector(settings.confidence, settings.model_path)


def create_app(settings: Settings | None = None, detector: DetectorLike | None = None) -> FastAPI:
    settings = settings or get_settings()
    state = {"det": detector, "loading": "ready" if detector else "starting", "failed": False}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["det"] is None:

            def load() -> None:
                state["loading"] = "loading the detector"
                try:
                    state["det"] = load_detector(settings)
                    state["loading"] = "ready"
                except Exception:
                    state["failed"] = True
                    state["loading"] = "model loading failed; see the server log"
                    log.exception("model loading failed")

            threading.Thread(target=load, name="model-loader", daemon=True).start()
        yield

    app = FastAPI(
        title="Glove detection",
        version="0.2.0",
        summary="Finds hands in a work photo and labels each gloved_hand or bare_hand "
        "(YOLOv8s fine-tuned on factory CCTV images).",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:16]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return _error(request, exc.status, exc.code, exc.detail, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", []) if p not in ("body",))
        msg = first.get("msg", "invalid request")
        return _error(request, 422, "invalid_request", f"{where}: {msg}" if where else msg)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error(request, exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        log.exception("unhandled error")
        return _error(request, 500, "internal_error", "Something went wrong; it was logged.")

    demo_dir = Path(settings.demo_dir)
    if (demo_dir / "samples").is_dir():
        app.mount("/samples", StaticFiles(directory=demo_dir / "samples"), name="samples")

    @app.get("/", include_in_schema=False)
    def demo():
        page = demo_dir / "index.html"
        return FileResponse(page) if page.is_file() else RedirectResponse("/docs")

    @app.get("/health", tags=["ops"])
    def health():
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready(request: Request):
        """Readiness: the model is loaded."""
        if state["det"] is None:
            code = "startup_failed" if state["failed"] else "not_ready"
            return _error(request, 503, code, f"The service is starting ({state['loading']}).")
        return {"status": "ready", "model_sha256": MODEL_SHA256}

    responses = {code: {"model": ErrorResponse} for code in (413, 415, 422, 503)}

    @app.post("/v1/detect", response_model=DetectResponse, responses=responses, tags=["detect"])
    def detect(
        request: Request,
        file: UploadFile = File(..., description="Work photo"),  # noqa: B008
        confidence: float | None = Query(  # noqa: B008
            None, ge=0.0, le=1.0, description="Override the confidence threshold"
        ),
    ):
        """Find hands and label each gloved_hand or bare_hand."""
        t0 = time.perf_counter()
        det = state["det"]
        if det is None:
            raise ApiError(503, "not_ready", f"The service is starting ({state['loading']}).")
        limit = int(settings.max_upload_mb * 1024 * 1024)
        data = file.file.read(limit + 1)
        if len(data) > limit:
            raise ApiError(
                413, "too_large", f"Uploads are limited to {settings.max_upload_mb:g} MB."
            )
        try:
            decoded = decode_upload(data, settings.max_pixels, settings.min_side_px)
        except ImageRejected as exc:
            raise ApiError(exc.status, exc.code, exc.detail) from None
        bgr = np.ascontiguousarray(
            np.asarray(decoded.image)[:, :, ::-1]
        )  # RGB -> BGR (OpenCV order)
        t1 = time.perf_counter()
        if not lock.acquire(timeout=settings.busy_timeout_s):
            raise ApiError(
                503,
                "busy",
                "Another image is being processed; try again shortly.",
                {"Retry-After": "5"},
            )
        try:
            dets = det.detect(bgr, confidence)
        finally:
            lock.release()
        t2 = time.perf_counter()
        counts = Counter(d["label"] for d in dets)
        return DetectResponse(
            request_id=request.state.request_id,
            detections=dets,
            counts={"gloved_hand": counts["gloved_hand"], "bare_hand": counts["bare_hand"]},
            image={
                "format": decoded.format,
                "width": decoded.width,
                "height": decoded.height,
                "exif_orientation": decoded.exif_orientation,
            },
            timings={
                "decode_ms": round((t1 - t0) * 1000, 1),
                "model_ms": round((t2 - t1) * 1000, 1),
                "total_ms": round((time.perf_counter() - t0) * 1000, 1),
            },
            model=f"yolov8s-glove best_50epc.pt (sha256 {MODEL_SHA256[:12]})",
            confidence_threshold=settings.confidence if confidence is None else confidence,
        )

    return app
