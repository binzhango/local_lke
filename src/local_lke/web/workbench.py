"""Gradio workbench mounted inside the FastAPI process."""

import mimetypes
import os
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from uuid import UUID

os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import gradio as gr

from local_lke.errors import LKEError
from local_lke.ingestion import IngestionService
from local_lke.models import AnswerResponse, ChunkStrategy, ParserStrategy
from local_lke.rag import RAGPipeline
from local_lke.settings import Settings
from local_lke.web.api import citation_url


def build_workbench(
    pipeline: RAGPipeline,
    settings: Settings,
    ingestion: IngestionService | None = None,
) -> gr.Blocks:
    with gr.Blocks(title="Local LKE RAG Workbench") as workbench:
        gr.Markdown("# Local LKE · Chapter 2 Ingestion Workbench")

        with gr.Tab("Setup"):
            gr.Markdown(
                "The workbench uses a local OpenAI-compatible chat server and a local "
                "Hugging Face embedding model. Secrets are never displayed."
            )
            configuration = gr.JSON(value=settings.redacted_summary, label="Configuration")
            health_output = gr.JSON(label="System health")
            health_button = gr.Button("Check system health")
            health_button.click(
                fn=lambda: provider_health(pipeline, ingestion),
                outputs=health_output,
            )

        with gr.Tab("Documents"):
            gr.Markdown(
                "Create a collection, then safely ingest Markdown, text, or PDF files. "
                "The Chapter 1 fixtures remain available to the chat baseline."
            )
            documents = gr.JSON(value=document_summary(pipeline), label="Indexed sources")
            with gr.Row():
                collection_name = gr.Textbox(label="New collection name")
                create_collection_button = gr.Button("Create collection")
                refresh_collections_button = gr.Button("Refresh collections")
            collection_result = gr.JSON(label="Collection result")
            collection = gr.Dropdown(label="Collection", choices=[])
            files = gr.File(
                label="Upload .md, .txt, or .pdf",
                file_count="multiple",
                type="filepath",
            )
            with gr.Row():
                parser_strategy = gr.Dropdown(
                    [item.value for item in ParserStrategy],
                    value=settings.default_parser_strategy,
                    label="PDF parser strategy",
                )
                chunk_strategy = gr.Dropdown(
                    [item.value for item in ChunkStrategy],
                    value=settings.default_chunk_strategy,
                    label="Chunk strategy",
                )
                chunk_size = gr.Number(value=settings.chunk_size, label="Chunk size")
                chunk_overlap = gr.Number(value=settings.chunk_overlap, label="Chunk overlap")
            upload_button = gr.Button("Ingest files", variant="primary")
            job_result = gr.JSON(label="Ingestion jobs and progress")
            job_id = gr.Textbox(label="Job ID to poll or retry")
            with gr.Row():
                poll_job_button = gr.Button("Poll job status")
                retry_job_button = gr.Button("Retry failed/interrupted job")
            polled_job = gr.JSON(label="Polled job status")
            refresh_documents_button = gr.Button("Refresh version history")
            version_history = gr.JSON(label="Document version history")
            version_id = gr.Textbox(label="Version ID to inspect")
            inspect_button = gr.Button("Inspect parser preview and chunks")
            parser_preview = gr.JSON(label="Parser preview and chunk inspection")

            if ingestion is not None:
                create_collection_button.click(
                    fn=lambda name: create_collection_callback(ingestion, name),
                    inputs=collection_name,
                    outputs=[collection_result, collection],
                )
                refresh_collections_button.click(
                    fn=lambda: gr.update(choices=collection_choices(ingestion)),
                    outputs=collection,
                )
                upload_button.click(
                    fn=lambda collection_id, selected_files, parser, chunker, size, overlap: (
                        upload_callback(
                            ingestion,
                            collection_id,
                            selected_files,
                            parser,
                            chunker,
                            int(size),
                            int(overlap),
                        )
                    ),
                    inputs=[
                        collection,
                        files,
                        parser_strategy,
                        chunk_strategy,
                        chunk_size,
                        chunk_overlap,
                    ],
                    outputs=job_result,
                )
                poll_job_button.click(
                    fn=lambda selected_job: job_callback(ingestion, selected_job, retry=False),
                    inputs=job_id,
                    outputs=polled_job,
                )
                retry_job_button.click(
                    fn=lambda selected_job: job_callback(ingestion, selected_job, retry=True),
                    inputs=job_id,
                    outputs=polled_job,
                )
                refresh_documents_button.click(
                    fn=lambda collection_id: document_history_callback(ingestion, collection_id),
                    inputs=collection,
                    outputs=version_history,
                )
                inspect_button.click(
                    fn=lambda selected_version: preview_callback(ingestion, selected_version),
                    inputs=version_id,
                    outputs=parser_preview,
                )

        with gr.Tab("Chat"):
            question = gr.Textbox(
                label="Question",
                value="How quickly does Atlas acknowledge a priority-one incident?",
            )
            top_k = gr.Slider(1, 5, value=settings.default_top_k, step=1, label="Top K")
            ask = gr.Button("Ask")
            answer = gr.Markdown(label="Grounded answer")
            citations = gr.Markdown(label="Citations")

        with gr.Tab("Trace"):
            trace = gr.JSON(label="Retrieval order and timings")

        ask.click(
            fn=lambda user_question, k: chat_callback(pipeline, user_question, int(k)),
            inputs=[question, top_k],
            outputs=[answer, citations, trace],
        )

        # Keep component references alive and visible to Gradio's configuration builder.
        _ = configuration, documents
    return cast(gr.Blocks, workbench)


def provider_health(
    pipeline: RAGPipeline, ingestion: IngestionService | None = None
) -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    checks = {
        "models": pipeline.chat.check_models,
        "completion": pipeline.chat.check_completion,
        "embeddings": pipeline.embeddings.check_initialization,
    }
    if ingestion is not None:
        checks["database"] = ingestion.check_health
    for name, check in checks.items():
        try:
            results[name] = {"status": "ok", "detail": check()}
        except Exception as exc:
            detail = str(exc)
            if name == "database":
                detail = "Database unavailable; run 'make init-postgres'."
            results[name] = {"status": "unavailable", "detail": detail}
    return results


def document_summary(pipeline: RAGPipeline) -> list[dict[str, str]]:
    return [
        {
            "source_id": document.source_id,
            "title": document.title,
            "locator": document.locator,
            "preview": " ".join(document.content.split())[:220],
        }
        for document in pipeline.documents
    ]


def chat_callback(
    pipeline: RAGPipeline, question: str, top_k: int
) -> Iterator[tuple[str, str, dict[str, object]]]:
    answer_text = ""
    trace: dict[str, object] = {}
    try:
        for event_type, data in pipeline.stream_query(question, top_k):
            if event_type == "retrieval":
                trace = data
                yield "Retrieving evidence…", "", trace
            elif event_type == "delta":
                answer_text += str(data)
                yield answer_text, "", trace
            elif event_type == "completion" and isinstance(data, AnswerResponse):
                yield data.answer, format_citations(data), data.trace.model_dump(mode="json")
    except LKEError as exc:
        yield f"Provider error: {exc}", "", trace


def format_citations(response: AnswerResponse) -> str:
    if not response.citations:
        return "No supporting citation was available."
    lines = ["### Sources"]
    for index, citation in enumerate(response.citations, start=1):
        lines.append(
            f"{index}. [{citation.source_id}]({citation_url(citation.source_id)}) — "
            f"`{citation.chunk_id}`: {citation.excerpt}"
        )
    return "\n\n".join(lines)


def collection_choices(ingestion: IngestionService) -> list[tuple[str, str]]:
    return [(item.name, str(item.id)) for item in ingestion.list_collections()]


def create_collection_callback(
    ingestion: IngestionService, name: str
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        collection = ingestion.create_collection(name)
        return (
            collection.model_dump(mode="json"),
            gr.update(choices=collection_choices(ingestion), value=str(collection.id)),
        )
    except LKEError as exc:
        return ({"error": str(exc), "code": exc.code}, gr.update())


def upload_callback(
    ingestion: IngestionService,
    collection_id: str | None,
    files: list[str] | str | None,
    parser_strategy: str,
    chunk_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, object]] | dict[str, str]:
    if not collection_id:
        return {"error": "Choose a collection before uploading."}
    if not files:
        return {"error": "Choose at least one file."}
    paths = [files] if isinstance(files, str) else files
    results: list[dict[str, object]] = []
    total = sum(Path(item).stat().st_size for item in paths)
    if total > ingestion.settings.max_batch_bytes:
        return {"error": "The selected files exceed the configured batch-size limit."}
    for item in paths:
        path = Path(item)
        content_type, _ = mimetypes.guess_type(path.name)
        try:
            job = ingestion.ingest(
                collection_id=UUID(collection_id),
                filename=path.name,
                content_type=content_type,
                content=path.read_bytes(),
                parser_strategy=ParserStrategy(parser_strategy),
                chunk_strategy=ChunkStrategy(chunk_strategy),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            results.append(job.model_dump(mode="json"))
        except LKEError as exc:
            results.append({"filename": path.name, "error": str(exc), "code": exc.code})
    return results


def document_history_callback(
    ingestion: IngestionService, collection_id: str | None
) -> list[dict[str, object]] | dict[str, str]:
    if not collection_id:
        return {"error": "Choose a collection."}
    try:
        return [
            item.model_dump(mode="json") for item in ingestion.list_documents(UUID(collection_id))
        ]
    except (LKEError, ValueError) as exc:
        return {"error": str(exc)}


def preview_callback(
    ingestion: IngestionService, version_id: str
) -> dict[str, object] | dict[str, str]:
    try:
        return ingestion.preview(UUID(version_id)).model_dump(mode="json")
    except (LKEError, ValueError) as exc:
        return {"error": str(exc)}


def job_callback(
    ingestion: IngestionService, job_id: str, *, retry: bool
) -> dict[str, object] | dict[str, str]:
    try:
        resolved_id = UUID(job_id)
        job = ingestion.retry(resolved_id) if retry else ingestion.get_job(resolved_id)
        return job.model_dump(mode="json")
    except (LKEError, ValueError) as exc:
        return {"error": str(exc)}
