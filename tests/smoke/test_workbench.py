from pathlib import Path

import gradio as gr

from local_lke.ingestion import IngestionService
from local_lke.rag import RAGPipeline
from local_lke.settings import Settings
from local_lke.web.workbench import (
    build_workbench,
    chat_callback,
    create_collection_callback,
    document_summary,
    upload_callback,
)


def test_gradio_blocks_constructs_with_expected_tabs(
    settings: Settings, pipeline: RAGPipeline, ingestion: IngestionService
) -> None:
    workbench = build_workbench(pipeline, settings, ingestion)
    config = workbench.get_config_file()
    labels = {component.get("props", {}).get("label") for component in config["components"]}

    assert isinstance(workbench, gr.Blocks)
    assert {
        "Configuration",
        "System health",
        "Indexed sources",
        "Collection",
        "Ingestion jobs and progress",
        "Document version history",
        "Parser preview and chunk inspection",
        "Question",
        "Strategy",
        "Query rewrite",
        "Final context manifest and answerability",
        "Rows, safe SQL preview, and provenance",
    } <= labels


def test_primary_chat_callback_streams_answer_and_trace(pipeline: RAGPipeline) -> None:
    outputs = list(
        chat_callback(
            pipeline,
            "How quickly does Atlas acknowledge a priority-one incident?",
            1,
        )
    )

    final_answer, citations, trace = outputs[-1]
    assert "15 minutes" in final_answer
    assert "fixture:atlas-support" in citations
    assert trace["retrieved"][0]["rank"] == 1
    assert document_summary(pipeline)[0]["source_id"].startswith("fixture:")


def test_ingestion_callbacks_create_upload_and_report_progress(
    tmp_path: Path, ingestion: IngestionService
) -> None:
    collection, dropdown = create_collection_callback(ingestion, "Workbench")
    collection_id = str(collection["id"])
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\nSafe upload through Gradio.", encoding="utf-8")

    jobs = upload_callback(
        ingestion,
        collection_id,
        [str(path)],
        "fast",
        "markdown",
        200,
        20,
    )

    assert dropdown["value"] == collection_id
    assert isinstance(jobs, list)
    assert jobs[0]["status"] == "completed"
    assert jobs[0]["progress"] == 100
