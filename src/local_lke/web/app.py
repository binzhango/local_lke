"""FastAPI application factory with the mounted Gradio workbench."""

from typing import cast

import gradio as gr
from fastapi import FastAPI

from local_lke.factory import create_pipeline
from local_lke.rag import RAGPipeline
from local_lke.settings import Settings, get_settings
from local_lke.web.api import create_router, install_error_handlers
from local_lke.web.workbench import build_workbench


def create_app(
    settings: Settings | None = None,
    pipeline: RAGPipeline | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_pipeline = pipeline or create_pipeline(resolved_settings)
    app = FastAPI(
        title="Local LKE RAG API",
        version="0.1.0",
        description="Chapter 1 in-memory, cited RAG baseline.",
    )
    app.state.pipeline = resolved_pipeline
    app.include_router(create_router(resolved_pipeline))
    install_error_handlers(app)
    workbench = build_workbench(resolved_pipeline, resolved_settings)
    return cast(FastAPI, gr.mount_gradio_app(app, workbench, path="/app"))
