"""FastAPI application factory with the mounted Gradio workbench."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import cast

import gradio as gr
from fastapi import FastAPI

from local_lke.factory import (
    create_indexing_services,
    create_ingestion_service,
    create_pipeline,
    create_retrieval_services,
)
from local_lke.indexing import IndexingService, MultimodalIndexingService
from local_lke.ingestion import IngestionService
from local_lke.rag import RAGPipeline
from local_lke.retrieval import AdvancedRetrievalService, StructuredDataService
from local_lke.settings import Settings, get_settings
from local_lke.web.api import create_router, install_error_handlers
from local_lke.web.workbench import build_workbench


def create_app(
    settings: Settings | None = None,
    pipeline: RAGPipeline | None = None,
    ingestion: IngestionService | None = None,
    retrieval: AdvancedRetrievalService | None = None,
    structured: StructuredDataService | None = None,
    indexing: IndexingService | None = None,
    multimodal: MultimodalIndexingService | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_pipeline = pipeline or create_pipeline(resolved_settings)
    resolved_ingestion = ingestion or create_ingestion_service(resolved_settings)
    default_indexing, default_multimodal = create_indexing_services(
        resolved_settings, resolved_pipeline, resolved_ingestion
    )
    resolved_indexing = indexing or default_indexing
    resolved_multimodal = multimodal or default_multimodal
    default_retrieval, default_structured = create_retrieval_services(
        resolved_settings, resolved_pipeline, resolved_ingestion, resolved_indexing
    )
    resolved_retrieval = retrieval or default_retrieval
    resolved_structured = structured or default_structured

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        with suppress(Exception):
            resolved_ingestion.startup()
        # A missing database leaves the UI and /healthz available so users receive
        # an actionable degraded status instead of a startup crash.
        yield

    app = FastAPI(
        title="Local LKE RAG API",
        version="0.5.0",
        description="Validated Chapter 5 generation over persistent advanced retrieval.",
        lifespan=lifespan,
    )
    app.state.pipeline = resolved_pipeline
    app.state.ingestion = resolved_ingestion
    app.state.retrieval = resolved_retrieval
    app.state.structured = resolved_structured
    app.state.indexing = resolved_indexing
    app.state.multimodal = resolved_multimodal
    app.include_router(
        create_router(
            resolved_pipeline,
            resolved_ingestion,
            resolved_retrieval,
            resolved_structured,
            resolved_indexing,
            resolved_multimodal,
        )
    )
    install_error_handlers(app)
    workbench = build_workbench(
        resolved_pipeline,
        resolved_settings,
        resolved_ingestion,
        resolved_retrieval,
        resolved_structured,
        resolved_indexing,
        resolved_multimodal,
    )
    return cast(FastAPI, gr.mount_gradio_app(app, workbench, path="/app"))
