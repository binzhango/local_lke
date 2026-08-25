from local_lke.models import AnswerResponse, AnswerStatus
from local_lke.rag import RAGPipeline

QUESTION = "How quickly does Atlas acknowledge a priority-one incident?"


def test_fake_pipeline_returns_expected_cited_source(pipeline: RAGPipeline) -> None:
    response = pipeline.query(QUESTION, top_k=1)

    assert isinstance(response, AnswerResponse)
    assert response.status is AnswerStatus.ANSWERED
    assert response.citations[0].source_id == "fixture:atlas-support"
    assert "15 minutes" in response.answer
    assert set(response.trace.timings_ms) == {"load", "split", "embed", "retrieve", "generate"}


def test_stream_orders_retrieval_deltas_and_completion(pipeline: RAGPipeline) -> None:
    events = list(pipeline.stream_query(QUESTION, top_k=1))
    types = [event_type for event_type, _ in events]

    assert types[0] == "retrieval"
    assert "delta" in types
    assert types[-1] == "completion"

