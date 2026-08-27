"""FastAPI application factory with the mounted Gradio workbench."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import cast

import gradio as gr
from fastapi import FastAPI

from local_lke.factory import create_ingestion_service, create_pipeline
from local_lke.ingestion import IngestionService
from local_lke.rag import RAGPipeline
from local_lke.settings import Settings, get_settings
from local_lke.web.api import create_router, install_error_handlers
from local_lke.web.workbench import build_workbench


def create_app(
    settings: Settings | None = None,
    pipeline: RAGPipeline | None = None,
    ingestion: IngestionService | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_pipeline = pipeline or create_pipeline(resolved_settings)
    resolved_ingestion = ingestion or create_ingestion_service(resolved_settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        with suppress(Exception):
            resolved_ingestion.startup()
        # A missing database leaves the UI and /healthz available so users receive
        # an actionable degraded status instead of a startup crash.
        yield

    app = FastAPI(
        title="Local LKE RAG API",
        version="0.2.0",
        description="Chapter 2 safe, versioned ingestion plus the cited RAG baseline.",
        lifespan=lifespan,
    )
    app.state.pipeline = resolved_pipeline
    app.state.ingestion = resolved_ingestion
    app.include_router(create_router(resolved_pipeline, resolved_ingestion))
    install_error_handlers(app)
    workbench = build_workbench(resolved_pipeline, resolved_settings, resolved_ingestion)
    return cast(FastAPI, gr.mount_gradio_app(app, workbench, path="/app"))
