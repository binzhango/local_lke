"""Gradio workbench mounted inside the FastAPI process."""

import html
import json
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
from local_lke.indexing import IndexingService, MultimodalIndexingService
from local_lke.ingestion import IngestionService
from local_lke.models import (
    AnswerResponse,
    ChunkStrategy,
    ExpansionStrategy,
    MetadataFilterPlan,
    OutputMode,
    ParserStrategy,
    QueryRequest,
    RetrievalStrategy,
    RewriteStrategy,
    StructuredQueryPlan,
    StructuredQueryRequest,
    StructuredSchemaName,
    VectorSearchRequest,
)
from local_lke.rag import RAGPipeline
from local_lke.retrieval import AdvancedRetrievalService, StructuredDataService
from local_lke.settings import Settings
from local_lke.web.api import citation_url


def build_workbench(
    pipeline: RAGPipeline,
    settings: Settings,
    ingestion: IngestionService | None = None,
    retrieval: AdvancedRetrievalService | None = None,
    structured: StructuredDataService | None = None,
    indexing: IndexingService | None = None,
    multimodal: MultimodalIndexingService | None = None,
) -> gr.Blocks:
    with gr.Blocks(title="Local LKE RAG Workbench") as workbench:
        gr.Markdown("# Local LKE · Chapters 1-5 RAG Workbench")

        with gr.Tab("Setup"):
            gr.Markdown(
                "The workbench uses a local OpenAI-compatible chat server and a local "
                "Hugging Face embedding model. Secrets are never displayed."
            )
            configuration = gr.JSON(value=settings.redacted_summary, label="Configuration")
            health_output = gr.JSON(label="System health")
            health_button = gr.Button("Check system health")
            health_button.click(
                fn=lambda: provider_health(pipeline, ingestion, indexing),
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
                            indexing,
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
            with gr.Row():
                output_mode = gr.Dropdown(
                    [item.value for item in OutputMode],
                    value=OutputMode.CONVERSATIONAL.value,
                    label="Output mode",
                )
                output_schema = gr.Dropdown(
                    [item.value for item in StructuredSchemaName],
                    value=StructuredSchemaName.FACT_LIST.value,
                    label="Structured schema",
                )
            ask = gr.Button("Ask")
            answer = gr.Markdown(label="Grounded answer")
            citations = gr.Markdown(label="Citations")

        with gr.Tab("Trace"):
            trace = gr.JSON(label="Retrieval order and timings")

        with gr.Tab("Retrieval Lab"):
            gr.Markdown(
                "Compare dense, lexical, fused, reranked, and final-context decisions "
                "against active persisted chunks."
            )
            retrieval_collection = gr.Dropdown(label="Collection", choices=[])
            refresh_retrieval_collections = gr.Button("Refresh retrieval collections")
            with gr.Row():
                build_index = gr.Button("Index active versions")
                inspect_index = gr.Button("Inspect index state")
            index_state_output = gr.JSON(
                label="Embedding profile, build progress, active nodes, and missing vectors"
            )
            vector_question = gr.Textbox(label="Persistent vector-lab question")
            with gr.Row():
                expansion = gr.Dropdown(
                    [item.value for item in ExpansionStrategy],
                    value=ExpansionStrategy.NONE.value,
                    label="Context expansion",
                )
                vector_top_k = gr.Slider(1, 20, value=5, step=1, label="Vector Top K")
                token_budget = gr.Number(
                    value=settings.retrieval_context_tokens,
                    label="Context token budget",
                )
            run_vector_lab = gr.Button("Run persistent vector lab")
            vector_candidates = gr.JSON(
                label="Child scores, parent/window expansion, duplicates, and final context"
            )
            retrieval_question = gr.Textbox(label="Persisted-data question")
            with gr.Row():
                retrieval_strategy = gr.Dropdown(
                    [RetrievalStrategy.DENSE.value, RetrievalStrategy.HYBRID.value],
                    value=RetrievalStrategy.HYBRID.value,
                    label="Strategy",
                )
                rewrite_strategy = gr.Dropdown(
                    [item.value for item in RewriteStrategy],
                    value=RewriteStrategy.NONE.value,
                    label="Query rewrite",
                )
                retrieval_top_k = gr.Slider(1, 20, value=3, step=1, label="Final Top K")
            metadata_plan = gr.JSON(
                value={"conditions": [], "allow_unfiltered_fallback": False},
                label="Allowlisted metadata filter plan",
            )
            infer_metadata = gr.Checkbox(
                value=False,
                label="Ask local model for a validated metadata plan",
            )
            with gr.Row():
                retrieval_output_mode = gr.Dropdown(
                    [item.value for item in OutputMode],
                    value=OutputMode.CONVERSATIONAL.value,
                    label="Generation output mode",
                )
                retrieval_output_schema = gr.Dropdown(
                    [item.value for item in StructuredSchemaName],
                    value=StructuredSchemaName.FACT_LIST.value,
                    label="Generation structured schema",
                )
            run_retrieval = gr.Button("Run retrieval", variant="primary")
            retrieval_answer = gr.Markdown(label="Answer or abstention")
            retrieval_citations = gr.Markdown(label="Versioned citations")
            stage_comparison = gr.JSON(
                label="Dense / lexical / fused / reranked candidate comparison"
            )
            context_manifest = gr.JSON(label="Final context manifest and answerability")

            if ingestion is not None and retrieval is not None:
                refresh_retrieval_collections.click(
                    fn=lambda: gr.update(choices=collection_choices(ingestion)),
                    outputs=retrieval_collection,
                )
                run_retrieval.click(
                    fn=lambda collection_id, user_question, strategy, rewrite, k,
                    filters, infer, mode, schema: (
                        retrieval_callback(
                            retrieval,
                            collection_id,
                            user_question,
                            strategy,
                            rewrite,
                            int(k),
                            filters,
                            infer,
                            mode,
                            schema,
                        )
                    ),
                    inputs=[
                        retrieval_collection,
                        retrieval_question,
                        retrieval_strategy,
                        rewrite_strategy,
                        retrieval_top_k,
                        metadata_plan,
                        infer_metadata,
                        retrieval_output_mode,
                        retrieval_output_schema,
                    ],
                    outputs=[
                        retrieval_answer,
                        retrieval_citations,
                        stage_comparison,
                        context_manifest,
                    ],
                )
                if indexing is not None:
                    build_index.click(
                        fn=lambda collection_id: index_collection_callback(
                            indexing, collection_id
                        ),
                        inputs=retrieval_collection,
                        outputs=index_state_output,
                    )
                    inspect_index.click(
                        fn=lambda collection_id: index_state_callback(indexing, collection_id),
                        inputs=retrieval_collection,
                        outputs=index_state_output,
                    )
                    run_vector_lab.click(
                        fn=lambda collection_id, query, mode, k, budget: vector_lab_callback(
                            indexing,
                            collection_id,
                            query,
                            mode,
                            int(k),
                            int(budget),
                        ),
                        inputs=[
                            retrieval_collection,
                            vector_question,
                            expansion,
                            vector_top_k,
                            token_budget,
                        ],
                        outputs=vector_candidates,
                    )

        with gr.Tab("Structured Data"):
            gr.Markdown(
                "Upload a bounded UTF-8 CSV, inspect inferred schema and provenance, "
                "then execute a validated plan—never raw model-generated SQL."
            )
            structured_collection = gr.Dropdown(label="Collection", choices=[])
            refresh_structured_collections = gr.Button("Refresh structured collections")
            csv_file = gr.File(label="Upload .csv", file_count="single", type="filepath")
            upload_csv = gr.Button("Ingest CSV")
            structured_table_result = gr.JSON(label="Inferred schema and provenance")
            table_id = gr.Textbox(label="Structured table ID")
            structured_question = gr.Textbox(label="Natural-language question")
            structured_plan = gr.JSON(
                value={
                    "projections": [],
                    "filters": [],
                    "group_by": [],
                    "aggregations": [],
                    "order_by": [],
                    "limit": 50,
                },
                label="Optional validated plan (leave null to ask the local model)",
            )
            run_structured = gr.Button("Run structured query", variant="primary")
            structured_result = gr.JSON(label="Rows, safe SQL preview, and provenance")

            if ingestion is not None and structured is not None:
                refresh_structured_collections.click(
                    fn=lambda: gr.update(choices=collection_choices(ingestion)),
                    outputs=structured_collection,
                )
                upload_csv.click(
                    fn=lambda collection_id, path: structured_upload_callback(
                        structured, collection_id, path
                    ),
                    inputs=[structured_collection, csv_file],
                    outputs=structured_table_result,
                )
                run_structured.click(
                    fn=lambda selected_table, user_question, plan: structured_query_callback(
                        structured, selected_table, user_question, plan
                    ),
                    inputs=[table_id, structured_question, structured_plan],
                    outputs=structured_result,
                )

        with gr.Tab("Multimodal Search"):
            gr.Markdown(
                "Images are decoded and embedded locally. Results expose image provenance; "
                "the text-only chat model is never described as having inspected them."
            )
            image_collection = gr.Dropdown(label="Collection", choices=[])
            refresh_image_collections = gr.Button("Refresh image collections")
            image_upload = gr.File(label="Upload image", file_count="single", type="filepath")
            upload_image_button = gr.Button("Validate and index image")
            image_upload_result = gr.JSON(label="Image metadata and provenance")
            image_text_query = gr.Textbox(label="Text-to-image query")
            image_query_file = gr.File(
                label="Optional image-to-image query", file_count="single", type="filepath"
            )
            image_top_k = gr.Slider(1, 20, value=5, step=1, label="Image Top K")
            with gr.Row():
                text_image_search = gr.Button("Search images by text")
                image_image_search = gr.Button("Search images by image")
            image_search_result = gr.JSON(label="Ranked image content URLs and scores")
            if ingestion is not None and multimodal is not None:
                refresh_image_collections.click(
                    fn=lambda: gr.update(choices=collection_choices(ingestion)),
                    outputs=image_collection,
                )
                upload_image_button.click(
                    fn=lambda collection_id, path: image_upload_callback(
                        multimodal, collection_id, path
                    ),
                    inputs=[image_collection, image_upload],
                    outputs=image_upload_result,
                )
                text_image_search.click(
                    fn=lambda collection_id, query, k: image_text_callback(
                        multimodal, collection_id, query, int(k)
                    ),
                    inputs=[image_collection, image_text_query, image_top_k],
                    outputs=image_search_result,
                )
                image_image_search.click(
                    fn=lambda collection_id, path, k: image_query_callback(
                        multimodal, collection_id, path, int(k)
                    ),
                    inputs=[image_collection, image_query_file, image_top_k],
                    outputs=image_search_result,
                )

        ask.click(
            fn=lambda user_question, k, mode, schema: chat_callback(
                pipeline, user_question, int(k), mode, schema
            ),
            inputs=[question, top_k, output_mode, output_schema],
            outputs=[answer, citations, trace],
        )

        # Keep component references alive and visible to Gradio's configuration builder.
        _ = configuration, documents
    return cast(gr.Blocks, workbench)


def provider_health(
    pipeline: RAGPipeline,
    ingestion: IngestionService | None = None,
    indexing: IndexingService | None = None,
) -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    checks = {
        "models": pipeline.chat.check_models,
        "completion": pipeline.chat.check_completion,
        "embeddings": pipeline.embeddings.check_initialization,
    }
    if ingestion is not None:
        checks["database"] = ingestion.check_health
    if indexing is not None:
        checks["vector_index"] = indexing.check_health
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
    pipeline: RAGPipeline,
    question: str,
    top_k: int,
    output_mode: str = OutputMode.CONVERSATIONAL.value,
    schema_name: str | None = None,
) -> Iterator[tuple[str, str, dict[str, object]]]:
    answer_text = ""
    trace: dict[str, object] = {}
    try:
        mode = OutputMode(output_mode)
        selected_schema = (
            StructuredSchemaName(schema_name or StructuredSchemaName.FACT_LIST.value)
            if mode is OutputMode.STRUCTURED
            else None
        )
        for event_type, data in pipeline.stream_query(
            question,
            top_k,
            output_mode=mode,
            schema_name=selected_schema,
        ):
            if event_type == "retrieval":
                trace = data
                yield "Retrieving evidence…", "", trace
            elif event_type == "delta":
                answer_text += str(data)
                yield sanitize_markdown(answer_text), "", trace
            elif event_type == "completion" and isinstance(data, AnswerResponse):
                yield (
                    render_answer(data),
                    format_citations(data),
                    data.trace.model_dump(mode="json"),
                )
    except LKEError as exc:
        yield f"Provider error: {exc}", "", trace


def format_citations(response: AnswerResponse) -> str:
    if not response.citations:
        return "No supporting citation was available."
    lines = ["### Sources"]
    for citation in response.citations:
        source_id = sanitize_markdown(citation.source_id)
        chunk_id = sanitize_markdown(citation.chunk_id)
        excerpt = sanitize_markdown(citation.excerpt)
        lines.append(
            f"- **{citation.citation_id}** [{source_id}]({citation_url(citation.source_id)}) — "
            f"`{chunk_id}`: {excerpt}"
        )
    return "\n\n".join(lines)


def render_answer(response: AnswerResponse) -> str:
    status = sanitize_markdown(response.status.value.upper())
    if response.structured_result is not None:
        body = json.dumps(
            response.structured_result.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    else:
        body = response.answer
    warning = ""
    if response.warnings:
        warning = "\n\nWarnings: " + "; ".join(response.warnings)
    return f"**Status: {status}**\n\n{sanitize_markdown(body + warning)}"


def sanitize_markdown(value: str) -> str:
    """Render model/source text literally in Gradio Markdown, including hostile HTML."""

    escaped = html.escape(value, quote=True)
    for character in "\\`*_{}[]()#+!|":
        escaped = escaped.replace(character, f"\\{character}")
    return escaped


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
    indexing: IndexingService | None = None,
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
            if indexing is not None and job.version_id is not None:
                indexed = indexing.index_version(job.version_id)
                results[-1]["indexing"] = indexed.model_dump(mode="json")
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


def retrieval_callback(
    retrieval: AdvancedRetrievalService,
    collection_id: str | None,
    question: str,
    strategy: str,
    rewrite: str,
    top_k: int,
    filters: dict[str, object] | None,
    infer_metadata: bool,
    output_mode: str = OutputMode.CONVERSATIONAL.value,
    schema_name: str | None = None,
) -> tuple[str, str, object, object]:
    if not collection_id:
        return "Choose a collection.", "", {}, {}
    try:
        mode = OutputMode(output_mode)
        response = retrieval.query(
            QueryRequest(
                collection_id=UUID(collection_id),
                question=question,
                strategy=RetrievalStrategy(strategy),
                rewrite=RewriteStrategy(rewrite),
                top_k=top_k,
                metadata_filter=MetadataFilterPlan.model_validate(filters or {}),
                infer_metadata_filter=infer_metadata,
                output_mode=mode,
                schema_name=(
                    StructuredSchemaName(
                        schema_name or StructuredSchemaName.FACT_LIST.value
                    )
                    if mode is OutputMode.STRUCTURED
                    else None
                ),
            )
        )
        advanced = response.trace.retrieval
        candidates = (
            [item.model_dump(mode="json") for item in advanced.candidates]
            if advanced is not None
            else []
        )
        decision: dict[str, object] = (
            {
                "context_manifest": [
                    item.model_dump(mode="json") for item in advanced.context_manifest
                ],
                "answerability": advanced.answerability.model_dump(mode="json"),
                "transform": advanced.transform.model_dump(mode="json"),
                "metadata_filter": advanced.metadata_filter.model_dump(mode="json"),
            }
            if advanced is not None
            else {}
        )
        decision["generation"] = (
            response.trace.generation.model_dump(mode="json")
            if response.trace.generation is not None
            else None
        )
        decision["warnings"] = response.warnings
        return render_answer(response), format_citations(response), candidates, decision
    except (LKEError, ValueError) as exc:
        return f"Retrieval error: {exc}", "", {}, {}


def structured_upload_callback(
    structured: StructuredDataService,
    collection_id: str | None,
    path: str | None,
) -> dict[str, object]:
    if not collection_id or not path:
        return {"error": "Choose a collection and CSV file."}
    try:
        source = Path(path)
        return structured.ingest_csv(
            collection_id=UUID(collection_id),
            filename=source.name,
            content=source.read_bytes(),
        ).model_dump(mode="json")
    except (LKEError, ValueError) as exc:
        return {"error": str(exc)}


def structured_query_callback(
    structured: StructuredDataService,
    table_id: str,
    question: str,
    plan: dict[str, object] | None,
) -> dict[str, object]:
    try:
        payload = StructuredQueryRequest(
            table_id=UUID(table_id),
            question=question,
            plan=StructuredQueryPlan.model_validate(plan) if plan else None,
        )
        return structured.query(payload).model_dump(mode="json")
    except (LKEError, ValueError) as exc:
        return {"error": str(exc)}


def index_collection_callback(
    indexing: IndexingService, collection_id: str | None
) -> dict[str, object]:
    if not collection_id:
        return {"error": "Choose a collection."}
    try:
        jobs = indexing.index_collection(UUID(collection_id))
        return {
            "jobs": [item.model_dump(mode="json") for item in jobs],
            "state": indexing.state(UUID(collection_id)).model_dump(mode="json"),
        }
    except (LKEError, ValueError) as exc:
        return {"error": str(exc)}


def index_state_callback(
    indexing: IndexingService, collection_id: str | None
) -> dict[str, object]:
    if not collection_id:
        return {"error": "Choose a collection."}
    try:
        return indexing.state(UUID(collection_id)).model_dump(mode="json")
    except (LKEError, ValueError) as exc:
        return {"error": str(exc)}


def vector_lab_callback(
    indexing: IndexingService,
    collection_id: str | None,
    question: str,
    expansion: str,
    top_k: int,
    token_budget: int,
) -> dict[str, object]:
    if not collection_id:
        return {"error": "Choose a collection."}
    try:
        return indexing.search(
            VectorSearchRequest(
                collection_id=UUID(collection_id),
                question=question,
                expansion=ExpansionStrategy(expansion),
                top_k=top_k,
                token_budget=token_budget,
            )
        ).model_dump(mode="json")
    except (LKEError, ValueError) as exc:
        return {"error": str(exc)}


def image_upload_callback(
    multimodal: MultimodalIndexingService,
    collection_id: str | None,
    path: str | None,
) -> dict[str, object]:
    if not collection_id or not path:
        return {"error": "Choose a collection and image."}
    try:
        source = Path(path)
        content_type, _ = mimetypes.guess_type(source.name)
        return multimodal.ingest(
            collection_id=UUID(collection_id),
            filename=source.name,
            content_type=content_type,
            content=source.read_bytes(),
        ).model_dump(mode="json")
    except (LKEError, ValueError) as exc:
        return {"error": str(exc)}


def image_text_callback(
    multimodal: MultimodalIndexingService,
    collection_id: str | None,
    query: str,
    top_k: int,
) -> dict[str, object]:
    if not collection_id:
        return {"error": "Choose a collection."}
    try:
        return multimodal.search_text(
            UUID(collection_id), query, top_k
        ).model_dump(mode="json")
    except (LKEError, ValueError) as exc:
        return {"error": str(exc)}


def image_query_callback(
    multimodal: MultimodalIndexingService,
    collection_id: str | None,
    path: str | None,
    top_k: int,
) -> dict[str, object]:
    if not collection_id or not path:
        return {"error": "Choose a collection and query image."}
    try:
        source = Path(path)
        content_type, _ = mimetypes.guess_type(source.name)
        return multimodal.search_image(
            UUID(collection_id),
            source.name,
            content_type,
            source.read_bytes(),
            top_k,
        ).model_dump(mode="json")
    except (LKEError, ValueError) as exc:
        return {"error": str(exc)}
