"""Gradio workbench mounted inside the FastAPI process."""

from collections.abc import Iterator
from typing import cast

import gradio as gr

from local_lke.errors import LKEError, ProviderUnavailableError
from local_lke.models import AnswerResponse
from local_lke.rag import RAGPipeline
from local_lke.settings import Settings
from local_lke.web.api import citation_url


def build_workbench(pipeline: RAGPipeline, settings: Settings) -> gr.Blocks:
    with gr.Blocks(title="Local LKE RAG Workbench") as workbench:
        gr.Markdown("# Local LKE · Chapter 1 RAG Workbench")

        with gr.Tab("Setup"):
            gr.Markdown(
                "The workbench uses a local OpenAI-compatible chat server and a local "
                "Hugging Face embedding model. Secrets are never displayed."
            )
            configuration = gr.JSON(value=settings.redacted_summary, label="Configuration")
            health_output = gr.JSON(label="Provider health")
            health_button = gr.Button("Check providers")
            health_button.click(
                fn=lambda: provider_health(pipeline),
                outputs=health_output,
            )

        with gr.Tab("Documents"):
            gr.Markdown("Bundled English fixtures used by this in-memory Chapter 1 baseline.")
            documents = gr.JSON(value=document_summary(pipeline), label="Indexed sources")

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


def provider_health(pipeline: RAGPipeline) -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    checks = {
        "models": pipeline.chat.check_models,
        "completion": pipeline.chat.check_completion,
        "embeddings": pipeline.embeddings.check_initialization,
    }
    for name, check in checks.items():
        try:
            results[name] = {"status": "ok", "detail": check()}
        except ProviderUnavailableError as exc:
            results[name] = {"status": "unavailable", "detail": str(exc)}
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
