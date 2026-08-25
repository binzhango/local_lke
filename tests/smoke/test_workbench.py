import gradio as gr

from local_lke.rag import RAGPipeline
from local_lke.settings import Settings
from local_lke.web.workbench import build_workbench, chat_callback, document_summary


def test_gradio_blocks_constructs_with_expected_tabs(
    settings: Settings, pipeline: RAGPipeline
) -> None:
    workbench = build_workbench(pipeline, settings)
    config = workbench.get_config_file()
    labels = {component.get("props", {}).get("label") for component in config["components"]}

    assert isinstance(workbench, gr.Blocks)
    assert {"Configuration", "Provider health", "Indexed sources", "Question"} <= labels


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

