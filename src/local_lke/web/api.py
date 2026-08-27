"""Stable HTTP API routes and error contracts."""

import json
from collections.abc import AsyncIterator
from typing import Annotated
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from local_lke.errors import IngestionError, LKEError, NotFoundError, RetrievalError
from local_lke.ingestion import IngestionService
from local_lke.ingestion.safety import validate_upload
from local_lke.models import (
    AnswerResponse,
    ChunkStrategy,
    CollectionCreate,
    CollectionResponse,
    ComponentHealth,
    DocumentResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    IngestionJobResponse,
    ParserPreviewResponse,
    ParserStrategy,
    QueryRequest,
    StructuredQueryRequest,
    StructuredQueryResponse,
    StructuredTableResponse,
)
from local_lke.rag import RAGPipeline
from local_lke.retrieval import AdvancedRetrievalService, StructuredDataService


def create_router(
    pipeline: RAGPipeline,
    ingestion: IngestionService,
    retrieval: AdvancedRetrievalService,
    structured: StructuredDataService,
) -> APIRouter:
    router = APIRouter()

    @router.get("/healthz", response_model=HealthResponse)
    def health() -> HealthResponse:
        components: dict[str, ComponentHealth] = {}
        checks = {
            "database": ingestion.check_health,
            "chat": pipeline.chat.check_models,
            "embeddings": pipeline.embeddings.check_initialization,
        }
        for name, check in checks.items():
            try:
                components[name] = ComponentHealth(status="ok", detail=check())
            except Exception as exc:
                detail = str(exc)
                if name == "database":
                    detail = (
                        "Database unavailable; run 'make init-postgres' and verify "
                        "the redacted LKE_DATABASE_URL configuration."
                    )
                components[name] = ComponentHealth(status="unavailable", detail=detail)
        overall = "ok" if all(item.status == "ok" for item in components.values()) else "degraded"
        return HealthResponse(status=overall, components=components)

    @router.post(
        "/api/v1/query",
        response_model=AnswerResponse,
        responses={503: {"model": ErrorResponse}},
    )
    def query(payload: QueryRequest) -> AnswerResponse:
        if payload.collection_id is None and payload.strategy.value == "dense":
            return pipeline.query(payload.question, payload.top_k)
        return retrieval.query(payload)

    @router.post("/api/v1/query/stream")
    def stream_query(payload: QueryRequest) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            yield _sse("start", {"question": payload.question})
            try:
                if payload.collection_id is not None:
                    response = retrieval.query(payload)
                    if response.trace.retrieval is not None:
                        yield _sse(
                            "retrieval",
                            response.trace.retrieval.model_dump(mode="json"),
                        )
                    yield _sse("completion", response.model_dump(mode="json"))
                    return
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
        if source_id.startswith("version:"):
            try:
                preview = ingestion.preview(UUID(source_id.removeprefix("version:")))
            except (ValueError, LKEError):
                return PlainTextResponse("Source not found", status_code=404)
            return PlainTextResponse("\n\n".join(chunk.text for chunk in preview.chunks))
        for document in pipeline.documents:
            if document.source_id == source_id:
                return PlainTextResponse(document.content)
        return PlainTextResponse("Source not found", status_code=404)

    @router.post(
        "/api/v1/collections",
        response_model=CollectionResponse,
        status_code=201,
    )
    def create_collection(payload: CollectionCreate) -> CollectionResponse:
        return ingestion.create_collection(payload.name)

    @router.get("/api/v1/collections", response_model=list[CollectionResponse])
    def list_collections() -> list[CollectionResponse]:
        return ingestion.list_collections()

    @router.post(
        "/api/v1/collections/{collection_id}/documents",
        response_model=list[IngestionJobResponse],
        status_code=202,
    )
    async def upload_documents(
        collection_id: UUID,
        background_tasks: BackgroundTasks,
        files: Annotated[list[UploadFile], File()],
        parser_strategy: Annotated[ParserStrategy, Form()] = ParserStrategy.FAST,
        chunk_strategy: Annotated[ChunkStrategy, Form()] = ChunkStrategy.MARKDOWN,
        chunk_size: Annotated[int | None, Form(ge=100, le=100_000)] = None,
        chunk_overlap: Annotated[int | None, Form(ge=0, le=10_000)] = None,
    ) -> list[IngestionJobResponse]:
        buffered: list[tuple[str, str | None, bytes]] = []
        batch_size = 0
        for upload in files:
            content = await upload.read(ingestion.settings.max_upload_bytes + 1)
            batch_size += len(content)
            if batch_size > ingestion.settings.max_batch_bytes:
                raise IngestionError(
                    "Upload batch exceeds the configured total-size limit",
                    code="batch_too_large",
                )
            buffered.append((upload.filename or "", upload.content_type, content))
        for filename, content_type, content in buffered:
            validate_upload(
                filename,
                content_type,
                content,
                max_bytes=ingestion.settings.max_upload_bytes,
            )
        jobs: list[IngestionJobResponse] = []
        for filename, content_type, content in buffered:
            job = ingestion.enqueue(
                collection_id=collection_id,
                filename=filename,
                content_type=content_type,
                content=content,
                parser_strategy=parser_strategy,
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            jobs.append(job)
            background_tasks.add_task(ingestion.process, job.id)
        return jobs

    @router.get("/api/v1/jobs/{job_id}", response_model=IngestionJobResponse)
    def get_job(job_id: UUID) -> IngestionJobResponse:
        return ingestion.get_job(job_id)

    @router.post("/api/v1/jobs/{job_id}/retry", response_model=IngestionJobResponse)
    def retry_job(job_id: UUID) -> IngestionJobResponse:
        return ingestion.retry(job_id)

    @router.get(
        "/api/v1/collections/{collection_id}/documents",
        response_model=list[DocumentResponse],
    )
    def list_documents(collection_id: UUID) -> list[DocumentResponse]:
        return ingestion.list_documents(collection_id)

    @router.get(
        "/api/v1/document-versions/{version_id}/preview",
        response_model=ParserPreviewResponse,
    )
    def preview_version(version_id: UUID) -> ParserPreviewResponse:
        return ingestion.preview(version_id)

    @router.delete("/api/v1/documents/{document_id}", response_model=DocumentResponse)
    def delete_document(document_id: UUID, reason: str = "deleted by user") -> DocumentResponse:
        return ingestion.delete_document(document_id, reason)

    @router.post(
        "/api/v1/collections/{collection_id}/structured-tables",
        response_model=StructuredTableResponse,
        status_code=201,
    )
    async def upload_structured_table(
        collection_id: UUID,
        file: Annotated[UploadFile, File()],
    ) -> StructuredTableResponse:
        content = await file.read(ingestion.settings.max_upload_bytes + 1)
        return structured.ingest_csv(
            collection_id=collection_id,
            filename=file.filename or "",
            content=content,
        )

    @router.get(
        "/api/v1/collections/{collection_id}/structured-tables",
        response_model=list[StructuredTableResponse],
    )
    def list_structured_tables(collection_id: UUID) -> list[StructuredTableResponse]:
        return structured.list_tables(collection_id)

    @router.post(
        "/api/v1/structured/query",
        response_model=StructuredQueryResponse,
    )
    def structured_query(payload: StructuredQueryRequest) -> StructuredQueryResponse:
        return structured.query(payload)

    return router


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(LKEError)
    async def handle_lke_error(_request: Request, exc: LKEError) -> JSONResponse:
        payload = ErrorResponse(
            error=ErrorDetail(code=exc.code, message=str(exc), component=exc.component)
        )
        status_code = 503
        if isinstance(exc, NotFoundError):
            status_code = 404
        elif isinstance(exc, IngestionError):
            status_code = 413 if exc.code in {"file_too_large", "batch_too_large"} else 422
        elif isinstance(exc, RetrievalError):
            status_code = 404 if exc.code.endswith("not_found") else 422
        return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))

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
