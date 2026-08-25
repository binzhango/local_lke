"""Stable HTTP API routes and error contracts."""

import json
from collections.abc import AsyncIterator
from urllib.parse import quote

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from local_lke.errors import LKEError, ProviderUnavailableError
from local_lke.models import (
    AnswerResponse,
    ComponentHealth,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
)
from local_lke.rag import RAGPipeline


def create_router(pipeline: RAGPipeline) -> APIRouter:
    router = APIRouter()

    @router.get("/healthz", response_model=HealthResponse)
    def health() -> HealthResponse:
        components: dict[str, ComponentHealth] = {}
        checks = {
            "chat": pipeline.chat.check_models,
            "embeddings": pipeline.embeddings.check_initialization,
        }
        for name, check in checks.items():
            try:
                components[name] = ComponentHealth(status="ok", detail=check())
            except ProviderUnavailableError as exc:
                components[name] = ComponentHealth(status="unavailable", detail=str(exc))
        overall = "ok" if all(item.status == "ok" for item in components.values()) else "degraded"
        return HealthResponse(status=overall, components=components)

    @router.post(
        "/api/v1/query",
        response_model=AnswerResponse,
        responses={503: {"model": ErrorResponse}},
    )
    def query(payload: QueryRequest) -> AnswerResponse:
        return pipeline.query(payload.question, payload.top_k)

    @router.post("/api/v1/query/stream")
    def stream_query(payload: QueryRequest) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            yield _sse("start", {"question": payload.question})
            try:
                for event_type, data in pipeline.stream_query(payload.question, payload.top_k):
                    if hasattr(data, "model_dump"):
                        data = data.model_dump(mode="json")
                    yield _sse(event_type, data)
            except LKEError as exc:
                yield _sse(
                    "error",
                    ErrorDetail(
                        code=exc.code,
                        message=str(exc),
                        component=exc.component,
                    ).model_dump(mode="json"),
                )

        return StreamingResponse(events(), media_type="text/event-stream")

    @router.get("/api/v1/sources/{source_id}", response_class=PlainTextResponse)
    def source(source_id: str) -> PlainTextResponse:
        for document in pipeline.documents:
            if document.source_id == source_id:
                return PlainTextResponse(document.content)
        return PlainTextResponse("Source not found", status_code=404)

    return router


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(LKEError)
    async def handle_lke_error(_request: Request, exc: LKEError) -> JSONResponse:
        payload = ErrorResponse(
            error=ErrorDetail(code=exc.code, message=str(exc), component=exc.component)
        )
        return JSONResponse(status_code=503, content=payload.model_dump(mode="json"))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first_error = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first_error.get("loc", []))
        message = first_error.get("msg", "Invalid request")
        payload = ErrorResponse(
            error=ErrorDetail(
                code="validation_error",
                message=f"{location}: {message}" if location else str(message),
                component="request",
            )
        )
        return JSONResponse(status_code=422, content=payload.model_dump(mode="json"))


def citation_url(source_id: str) -> str:
    return f"/api/v1/sources/{quote(source_id, safe='')}"


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
