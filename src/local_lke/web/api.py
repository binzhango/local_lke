"""Stable HTTP API routes and error contracts."""

import json
from collections.abc import AsyncIterator
from typing import Annotated
from urllib.parse import quote
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

from local_lke.errors import (
    AuthenticationError,
    AuthorizationError,
    EvaluationError,
    GenerationError,
    IndexingError,
    IngestionError,
    LKEError,
    NotFoundError,
    RetrievalError,
)
from local_lke.evaluation import (
    EvaluationComparison,
    EvaluationDatasetCreate,
    EvaluationDatasetResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
    EvaluationService,
    ProviderCapabilityProfile,
)
from local_lke.indexing import IndexingService, MultimodalIndexingService
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
    ImageAssetResponse,
    ImageSearchResponse,
    IndexingJobResponse,
    IndexStateResponse,
    IngestionJobResponse,
    ParserPreviewResponse,
    ParserStrategy,
    QueryRequest,
    StructuredQueryRequest,
    StructuredQueryResponse,
    StructuredTableResponse,
    VectorSearchRequest,
    VectorSearchResponse,
)
from local_lke.rag import RAGPipeline
from local_lke.retrieval import AdvancedRetrievalService, StructuredDataService
from local_lke.security import (
    AuditEventResponse,
    CollectionAccessResponse,
    CollectionGrantRequest,
    Permission,
    SecurityService,
)


def create_router(
    pipeline: RAGPipeline,
    ingestion: IngestionService,
    retrieval: AdvancedRetrievalService,
    structured: StructuredDataService,
    indexing: IndexingService,
    multimodal: MultimodalIndexingService,
    evaluation: EvaluationService,
    security: SecurityService,
) -> APIRouter:
    router = APIRouter()

    @router.get("/healthz", response_model=HealthResponse)
    def health() -> HealthResponse:
        components: dict[str, ComponentHealth] = {}
        checks = {
            "database": ingestion.check_health,
            "chat": pipeline.chat.check_models,
            "embeddings": pipeline.embeddings.check_initialization,
            "vector_index": indexing.check_health,
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

    api = APIRouter(
        prefix="/api/v1",
        dependencies=[Depends(security.authenticate)],
    )

    @api.post(
        "/query",
        response_model=AnswerResponse,
        responses={503: {"model": ErrorResponse}},
    )
    def query(payload: QueryRequest, request: Request) -> AnswerResponse:
        principal = security.principal(request)
        if payload.collection_id is not None:
            security.authorize_collection(
                principal,
                payload.collection_id,
                Permission.READ,
                action="query.execute",
            )
        else:
            security.audit(principal, "query.execute", "fixture", None, "allowed")
        if payload.collection_id is None and payload.strategy.value == "dense":
            return pipeline.query(
                payload.question,
                payload.top_k,
                output_mode=payload.output_mode,
                schema_name=payload.schema_name,
            )
        return retrieval.query(payload)

    @api.post("/query/stream")
    def stream_query(payload: QueryRequest, request: Request) -> StreamingResponse:
        principal = security.principal(request)
        if payload.collection_id is not None:
            security.authorize_collection(
                principal,
                payload.collection_id,
                Permission.READ,
                action="query.stream",
            )
        else:
            security.audit(principal, "query.stream", "fixture", None, "allowed")
        async def events() -> AsyncIterator[str]:
            yield _sse(
                "start",
                {
                    "question": payload.question,
                    "output_mode": payload.output_mode.value,
                    "schema_name": (
                        payload.schema_name.value if payload.schema_name is not None else None
                    ),
                },
            )
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
                for event_type, data in pipeline.stream_query(
                    payload.question,
                    payload.top_k,
                    output_mode=payload.output_mode,
                    schema_name=payload.schema_name,
                ):
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

    @api.get("/sources/{source_id}", response_class=PlainTextResponse)
    def source(source_id: str, request: Request) -> PlainTextResponse:
        if source_id.startswith("version:"):
            try:
                version_id = UUID(source_id.removeprefix("version:"))
                collection_id = security.collection_for_version(version_id)
                security.authorize_collection(
                    security.principal(request),
                    collection_id,
                    Permission.READ,
                    action="source.read",
                )
                preview = ingestion.preview(version_id)
            except (ValueError, LKEError):
                return PlainTextResponse("Source not found", status_code=404)
            return PlainTextResponse("\n\n".join(chunk.text for chunk in preview.chunks))
        for document in pipeline.documents:
            if document.source_id == source_id:
                return PlainTextResponse(document.content)
        return PlainTextResponse("Source not found", status_code=404)

    @api.post(
        "/collections",
        response_model=CollectionResponse,
        status_code=201,
    )
    def create_collection(payload: CollectionCreate, request: Request) -> CollectionResponse:
        principal = security.principal(request)
        result = ingestion.create_collection(
            payload.name,
            principal.principal_id if security.enabled else None,
        )
        security.audit(
            principal,
            "collection.create",
            "collection",
            str(result.id),
            "allowed",
        )
        return result

    @api.get("/collections", response_model=list[CollectionResponse])
    def list_collections(request: Request) -> list[CollectionResponse]:
        principal = security.principal(request)
        allowed_ids = security.accessible_collection_ids(principal)
        collections = ingestion.list_collections()
        if allowed_ids is not None:
            collections = [item for item in collections if item.id in allowed_ids]
        security.audit(principal, "collection.list", "collection", None, "allowed")
        return collections

    @api.post(
        "/collections/{collection_id}/documents",
        response_model=list[IngestionJobResponse],
        status_code=202,
    )
    async def upload_documents(
        collection_id: UUID,
        request: Request,
        background_tasks: BackgroundTasks,
        files: Annotated[list[UploadFile], File()],
        parser_strategy: Annotated[ParserStrategy, Form()] = ParserStrategy.FAST,
        chunk_strategy: Annotated[ChunkStrategy, Form()] = ChunkStrategy.MARKDOWN,
        chunk_size: Annotated[int | None, Form(ge=100, le=100_000)] = None,
        chunk_overlap: Annotated[int | None, Form(ge=0, le=10_000)] = None,
    ) -> list[IngestionJobResponse]:
        security.authorize_collection(
            security.principal(request),
            collection_id,
            Permission.WRITE,
            action="document.upload",
        )
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
            background_tasks.add_task(_process_and_index, ingestion, indexing, job.id)
        return jobs

    @api.get("/jobs/{job_id}", response_model=IngestionJobResponse)
    def get_job(job_id: UUID, request: Request) -> IngestionJobResponse:
        security.authorize_collection(
            security.principal(request),
            security.collection_for_job(job_id),
            Permission.READ,
            action="ingestion_job.read",
        )
        return ingestion.get_job(job_id)

    @api.post("/jobs/{job_id}/retry", response_model=IngestionJobResponse)
    def retry_job(job_id: UUID, request: Request) -> IngestionJobResponse:
        security.authorize_collection(
            security.principal(request),
            security.collection_for_job(job_id),
            Permission.WRITE,
            action="ingestion_job.retry",
        )
        job = ingestion.retry(job_id)
        if job.status.value == "completed" and job.version_id is not None:
            indexing.index_version(job.version_id)
        return job

    @api.get(
        "/collections/{collection_id}/documents",
        response_model=list[DocumentResponse],
    )
    def list_documents(collection_id: UUID, request: Request) -> list[DocumentResponse]:
        security.authorize_collection(
            security.principal(request),
            collection_id,
            Permission.READ,
            action="document.list",
        )
        return ingestion.list_documents(collection_id)

    @api.get(
        "/document-versions/{version_id}/preview",
        response_model=ParserPreviewResponse,
    )
    def preview_version(version_id: UUID, request: Request) -> ParserPreviewResponse:
        security.authorize_collection(
            security.principal(request),
            security.collection_for_version(version_id),
            Permission.READ,
            action="document_version.preview",
        )
        return ingestion.preview(version_id)

    @api.delete("/documents/{document_id}", response_model=DocumentResponse)
    def delete_document(
        document_id: UUID, request: Request, reason: str = "deleted by user"
    ) -> DocumentResponse:
        security.authorize_collection(
            security.principal(request),
            security.collection_for_document(document_id),
            Permission.WRITE,
            action="document.delete",
        )
        return ingestion.delete_document(document_id, reason)

    @api.post(
        "/document-versions/{version_id}/index",
        response_model=IndexingJobResponse,
    )
    def index_version(
        version_id: UUID, request: Request, force: bool = False
    ) -> IndexingJobResponse:
        security.authorize_collection(
            security.principal(request),
            security.collection_for_version(version_id),
            Permission.WRITE,
            action="index.version",
        )
        return indexing.index_version(version_id, force=force)

    @api.post(
        "/collections/{collection_id}/index",
        response_model=list[IndexingJobResponse],
    )
    def index_collection(collection_id: UUID, request: Request) -> list[IndexingJobResponse]:
        security.authorize_collection(
            security.principal(request),
            collection_id,
            Permission.WRITE,
            action="index.collection",
        )
        return indexing.index_collection(collection_id)

    @api.get(
        "/collections/{collection_id}/index-state",
        response_model=IndexStateResponse,
    )
    def index_state(collection_id: UUID, request: Request) -> IndexStateResponse:
        security.authorize_collection(
            security.principal(request),
            collection_id,
            Permission.READ,
            action="index.state",
        )
        return indexing.state(collection_id)

    @api.get("/indexing-jobs/{job_id}", response_model=IndexingJobResponse)
    def indexing_job(job_id: UUID, request: Request) -> IndexingJobResponse:
        security.authorize_collection(
            security.principal(request),
            security.collection_for_job(job_id, indexing=True),
            Permission.READ,
            action="indexing_job.read",
        )
        return indexing.repository.get_job(job_id)

    @api.post("/retrieval-lab", response_model=VectorSearchResponse)
    def retrieval_lab(payload: VectorSearchRequest, request: Request) -> VectorSearchResponse:
        security.authorize_collection(
            security.principal(request),
            payload.collection_id,
            Permission.READ,
            action="retrieval_lab.search",
        )
        return indexing.search(payload)

    @api.post(
        "/collections/{collection_id}/images",
        response_model=ImageAssetResponse,
        status_code=201,
    )
    async def upload_image(
        collection_id: UUID,
        file: Annotated[UploadFile, File()],
        request: Request,
    ) -> ImageAssetResponse:
        security.authorize_collection(
            security.principal(request),
            collection_id,
            Permission.WRITE,
            action="image.upload",
        )
        content = await file.read(ingestion.settings.max_upload_bytes + 1)
        return multimodal.ingest(
            collection_id=collection_id,
            filename=file.filename or "",
            content_type=file.content_type,
            content=content,
        )

    @api.post(
        "/collections/{collection_id}/images/search/text",
        response_model=ImageSearchResponse,
    )
    def text_to_image_search(
        collection_id: UUID,
        request: Request,
        query: Annotated[str, Form(min_length=1)],
        top_k: Annotated[int, Form(ge=1, le=50)] = 5,
    ) -> ImageSearchResponse:
        security.authorize_collection(
            security.principal(request),
            collection_id,
            Permission.READ,
            action="image.search.text",
        )
        return multimodal.search_text(collection_id, query, top_k)

    @api.post(
        "/collections/{collection_id}/images/search/image",
        response_model=ImageSearchResponse,
    )
    async def image_to_image_search(
        collection_id: UUID,
        request: Request,
        file: Annotated[UploadFile, File()],
        top_k: Annotated[int, Form(ge=1, le=50)] = 5,
    ) -> ImageSearchResponse:
        security.authorize_collection(
            security.principal(request),
            collection_id,
            Permission.READ,
            action="image.search.image",
        )
        content = await file.read(ingestion.settings.max_upload_bytes + 1)
        return multimodal.search_image(
            collection_id,
            file.filename or "",
            file.content_type,
            content,
            top_k,
        )

    @api.get("/images/{image_id}/content", response_class=FileResponse)
    def image_content(image_id: UUID, request: Request) -> FileResponse:
        security.authorize_collection(
            security.principal(request),
            security.collection_for_image(image_id),
            Permission.READ,
            action="image.content.read",
        )
        path = multimodal.get_content_path(image_id)
        return FileResponse(path, filename=path.name)

    @api.post(
        "/collections/{collection_id}/structured-tables",
        response_model=StructuredTableResponse,
        status_code=201,
    )
    async def upload_structured_table(
        collection_id: UUID,
        file: Annotated[UploadFile, File()],
        request: Request,
    ) -> StructuredTableResponse:
        security.authorize_collection(
            security.principal(request),
            collection_id,
            Permission.WRITE,
            action="structured_table.upload",
        )
        content = await file.read(ingestion.settings.max_upload_bytes + 1)
        return structured.ingest_csv(
            collection_id=collection_id,
            filename=file.filename or "",
            content=content,
        )

    @api.get(
        "/collections/{collection_id}/structured-tables",
        response_model=list[StructuredTableResponse],
    )
    def list_structured_tables(
        collection_id: UUID, request: Request
    ) -> list[StructuredTableResponse]:
        security.authorize_collection(
            security.principal(request),
            collection_id,
            Permission.READ,
            action="structured_table.list",
        )
        return structured.list_tables(collection_id)

    @api.post(
        "/structured/query",
        response_model=StructuredQueryResponse,
    )
    def structured_query(
        payload: StructuredQueryRequest, request: Request
    ) -> StructuredQueryResponse:
        security.authorize_collection(
            security.principal(request),
            security.collection_for_table(payload.table_id),
            Permission.READ,
            action="structured_query.execute",
        )
        return structured.query(payload)

    @api.post(
        "/evaluations/datasets",
        response_model=EvaluationDatasetResponse,
        status_code=201,
    )
    def create_evaluation_dataset(
        payload: EvaluationDatasetCreate,
        request: Request,
    ) -> EvaluationDatasetResponse:
        security.require_admin(
            security.principal(request), action="evaluation_dataset.create"
        )
        return evaluation.create_dataset(payload)

    @api.get(
        "/evaluations/datasets",
        response_model=list[EvaluationDatasetResponse],
    )
    def list_evaluation_datasets(request: Request) -> list[EvaluationDatasetResponse]:
        security.require_admin(
            security.principal(request), action="evaluation_dataset.list"
        )
        return evaluation.list_datasets()

    @api.get(
        "/evaluations/datasets/{dataset_id}",
        response_model=EvaluationDatasetResponse,
    )
    def get_evaluation_dataset(
        dataset_id: UUID, request: Request
    ) -> EvaluationDatasetResponse:
        security.require_admin(
            security.principal(request), action="evaluation_dataset.read"
        )
        return evaluation.get_dataset(dataset_id)

    @api.post(
        "/evaluations/runs",
        response_model=EvaluationRunResponse,
        status_code=201,
    )
    def run_evaluation(
        payload: EvaluationRunRequest, request: Request
    ) -> EvaluationRunResponse:
        security.require_admin(security.principal(request), action="evaluation_run.execute")
        return evaluation.run(payload)

    @api.get(
        "/evaluations/runs",
        response_model=list[EvaluationRunResponse],
    )
    def list_evaluation_runs(
        request: Request,
        dataset_id: UUID | None = None,
    ) -> list[EvaluationRunResponse]:
        security.require_admin(security.principal(request), action="evaluation_run.list")
        return evaluation.list_runs(dataset_id)

    @api.get(
        "/evaluations/runs/{run_id}",
        response_model=EvaluationRunResponse,
    )
    def get_evaluation_run(run_id: UUID, request: Request) -> EvaluationRunResponse:
        security.require_admin(security.principal(request), action="evaluation_run.read")
        return evaluation.get_run(run_id)

    @api.get(
        "/evaluations/compare",
        response_model=EvaluationComparison,
    )
    def compare_evaluation_runs(
        baseline_run_id: UUID, candidate_run_id: UUID, request: Request
    ) -> EvaluationComparison:
        security.require_admin(security.principal(request), action="evaluation_run.compare")
        return evaluation.compare(baseline_run_id, candidate_run_id)

    @api.get(
        "/evaluations/provider-profile",
        response_model=ProviderCapabilityProfile,
    )
    def evaluation_provider_profile(request: Request) -> ProviderCapabilityProfile:
        security.require_admin(
            security.principal(request), action="evaluation_provider_profile.read"
        )
        return evaluation.provider_profile()

    @api.get(
        "/collections/{collection_id}/access",
        response_model=list[CollectionAccessResponse],
    )
    def list_collection_access(
        collection_id: UUID, request: Request
    ) -> list[CollectionAccessResponse]:
        security.authorize_collection(
            security.principal(request),
            collection_id,
            Permission.MANAGE,
            action="collection.access.list",
        )
        return security.list_access(collection_id)

    @api.put(
        "/collections/{collection_id}/access",
        response_model=CollectionAccessResponse,
    )
    def grant_collection_access(
        collection_id: UUID, payload: CollectionGrantRequest, request: Request
    ) -> CollectionAccessResponse:
        principal = security.principal(request)
        security.authorize_collection(
            principal,
            collection_id,
            Permission.MANAGE,
            action="collection.access.manage",
        )
        return security.grant(principal, collection_id, payload.principal_id, payload.role)

    @api.delete(
        "/collections/{collection_id}/access/{principal_id}",
        status_code=204,
    )
    def revoke_collection_access(
        collection_id: UUID, principal_id: str, request: Request
    ) -> Response:
        principal = security.principal(request)
        security.authorize_collection(
            principal,
            collection_id,
            Permission.MANAGE,
            action="collection.access.manage",
        )
        security.revoke(principal, collection_id, principal_id)
        return Response(status_code=204)

    @api.get("/audit-events", response_model=list[AuditEventResponse])
    def list_audit_events(
        request: Request, limit: Annotated[int, Query(ge=1, le=1000)] = 100
    ) -> list[AuditEventResponse]:
        security.require_admin(security.principal(request), action="audit_event.list")
        return security.list_audit_events(limit)

    router.include_router(api)
    return router


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(LKEError)
    async def handle_lke_error(_request: Request, exc: LKEError) -> JSONResponse:
        payload = ErrorResponse(
            error=ErrorDetail(code=exc.code, message=str(exc), component=exc.component)
        )
        status_code = 503
        headers: dict[str, str] | None = None
        if isinstance(exc, AuthenticationError):
            status_code = 401
            headers = {"WWW-Authenticate": "Bearer"}
        elif isinstance(exc, AuthorizationError):
            status_code = 403
        elif isinstance(exc, NotFoundError):
            status_code = 404
        elif isinstance(exc, IngestionError):
            status_code = 413 if exc.code in {"file_too_large", "batch_too_large"} else 422
        elif isinstance(exc, RetrievalError):
            status_code = 404 if exc.code.endswith("not_found") else 422
        elif isinstance(exc, IndexingError):
            status_code = 413 if exc.code in {"file_too_large", "image_too_large"} else 422
        elif isinstance(exc, GenerationError):
            status_code = 422
        elif isinstance(exc, EvaluationError):
            status_code = 404 if exc.code.endswith("not_found") else 422
        return JSONResponse(
            status_code=status_code,
            content=payload.model_dump(mode="json"),
            headers=headers,
        )

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


def _process_and_index(
    ingestion: IngestionService,
    indexing: IndexingService,
    job_id: UUID,
) -> None:
    job = ingestion.process(job_id)
    if job.status.value == "completed" and job.version_id is not None:
        indexing.index_version(job.version_id)
