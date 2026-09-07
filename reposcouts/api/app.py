from __future__ import annotations

import asyncio
import json
import queue
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from reposcouts.api.models import (
    RunCancelResponse,
    RunCreateRequest,
    RunCreateResponse,
)
from reposcouts.api.run_manager import (
    RunConflictError,
    RunNotFoundError,
    TERMINAL_EVENT_TYPES,
    run_manager,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


def _get_run_or_404(run_id: str):
    try:
        return run_manager.get_run(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc


async def _event_stream(run_id: str) -> AsyncIterator[str]:
    record = _get_run_or_404(run_id)

    while True:
        try:
            event = await asyncio.to_thread(record.events.get, True, 15)
        except queue.Empty:
            yield ": keep-alive\n\n"
            continue

        payload = json.dumps(
            event.model_dump(mode="json"),
            ensure_ascii=False,
        )
        yield f"event: {event.type}\ndata: {payload}\n\n"

        if event.type in TERMINAL_EVENT_TYPES:
            break


def create_app() -> FastAPI:
    app = FastAPI(
        title="RepoScouts API",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/api/runs",
        response_model=RunCreateResponse,
        status_code=202,
    )
    def create_run(request: RunCreateRequest) -> RunCreateResponse:
        try:
            record = run_manager.create_run(request.user_request)
        except RunConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RunCreateResponse(run_id=record.run_id)

    @app.get("/api/runs/{run_id}/events")
    def stream_run(run_id: str) -> StreamingResponse:
        _get_run_or_404(run_id)
        return StreamingResponse(
            _event_stream(run_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post(
        "/api/runs/{run_id}/cancel",
        response_model=RunCancelResponse,
    )
    def cancel_run(run_id: str) -> RunCancelResponse:
        _get_run_or_404(run_id)
        record = run_manager.cancel_run(run_id)
        return RunCancelResponse(
            run_id=record.run_id,
            status=record.status,
        )

    if FRONTEND_DIST.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=FRONTEND_DIST, html=True),
            name="frontend",
        )

    return app


app = create_app()
