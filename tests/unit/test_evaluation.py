from local_lke.evaluation.models import (
    EvaluationCaseResult,
    EvaluationFault,
    EvaluationMetrics,
    EvaluationThresholds,
)
from local_lke.evaluation.service import _aggregate, _gate, _ranking_metrics
from local_lke.models import AnswerStatus


def test_ranking_metrics_use_labelled_relevance() -> None:
    reciprocal_rank, recall, ndcg = _ranking_metrics(
        ["chunk:relevant-a", "chunk:relevant-b"],
        ["chunk:noise", "chunk:relevant-a", "chunk:relevant-b"],
    )

    assert reciprocal_rank == 0.5
    assert recall == 1.0
    assert ndcg is not None and 0 < ndcg < 1


def test_aggregate_and_absolute_gate_are_deterministic() -> None:
    result = EvaluationCaseResult(
        case_id="case-1",
        status=AnswerStatus.ANSWERED,
        fault=EvaluationFault.NONE,
        latency_ms=12.0,
        retrieved_source_ids=["source"],
        retrieved_chunk_ids=["chunk"],
        reciprocal_rank=1.0,
        recall_at_k=1.0,
        ndcg_at_k=1.0,
        citation_precision=1.0,
        citation_recall=1.0,
        answer_match=True,
        status_match=True,
        passed=True,
        trace_id="00000000-0000-0000-0000-000000000001",
    )

    metrics = _aggregate([result])
    gate = _gate(
        metrics,
        EvaluationThresholds(
            min_case_pass_rate=1,
            min_recall_at_k=1,
            min_answer_match_rate=1,
            max_p95_latency_ms=20,
        ),
        None,
    )

    assert metrics == EvaluationMetrics(
        case_count=1,
        case_pass_rate=1,
        recall_at_k=1,
        mean_reciprocal_rank=1,
        ndcg_at_k=1,
        citation_precision=1,
        citation_recall=1,
        answer_match_rate=1,
        status_match_rate=1,
        p50_latency_ms=12,
        p95_latency_ms=12,
    )
    assert gate.passed is True
