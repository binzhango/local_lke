import pytest
from pydantic import ValidationError

from local_lke.models import (
    AnswerResponse,
    AnswerStatus,
    Citation,
    CitationLocator,
    CitationLocatorKind,
    ConfidenceExplanation,
    ConfidenceLevel,
    GenerationTrace,
    OutputMode,
    QueryRequest,
    StructuredSchemaName,
    TraceSummary,
)


def test_answered_response_requires_a_citation() -> None:
    with pytest.raises(ValidationError, match="require at least one citation"):
        AnswerResponse(
            status=AnswerStatus.ANSWERED,
            answer="Unsupported answer",
            citations=[],
            confidence=ConfidenceExplanation(
                level=ConfidenceLevel.LOW,
                rationale="No supporting evidence.",
            ),
            trace=TraceSummary(),
        )


def test_abstention_can_have_no_citations() -> None:
    response = AnswerResponse(
        status=AnswerStatus.ABSTAINED,
        answer="I do not know.",
        citations=[],
        confidence=ConfidenceExplanation(
            level=ConfidenceLevel.LOW,
            rationale="Evidence is insufficient.",
        ),
        trace=TraceSummary(),
    )
    assert response.status is AnswerStatus.ABSTAINED


def test_degraded_response_requires_cited_evidence() -> None:
    with pytest.raises(ValidationError, match="answered and degraded"):
        AnswerResponse(
            status=AnswerStatus.DEGRADED,
            answer="Extractive fallback",
            citations=[],
            confidence=ConfidenceExplanation(
                level=ConfidenceLevel.LOW,
                rationale="Generation failed.",
            ),
            trace=TraceSummary(),
        )


def test_structured_mode_requires_an_allowlisted_schema_name() -> None:
    with pytest.raises(ValidationError, match="schema_name is required"):
        QueryRequest(question="Structured answer", output_mode=OutputMode.STRUCTURED)
    request = QueryRequest(
        question="Structured answer",
        output_mode=OutputMode.STRUCTURED,
        schema_name=StructuredSchemaName.FACT_LIST,
    )
    assert request.schema_name is StructuredSchemaName.FACT_LIST


def test_typed_locator_rejects_reversed_text_ranges() -> None:
    with pytest.raises(ValidationError, match="end_line"):
        CitationLocator(
            kind=CitationLocatorKind.TEXT,
            label="lines 9-2",
            start_line=9,
            end_line=2,
        )


def test_answered_citations_require_unique_stable_ids() -> None:
    citation = Citation(
        source_id="source",
        chunk_id="chunk",
        locator="line 1",
        excerpt="Evidence",
    )
    with pytest.raises(ValidationError, match="require citation_id"):
        AnswerResponse(
            status=AnswerStatus.ANSWERED,
            answer="Answer",
            citations=[citation],
            confidence=ConfidenceExplanation(
                level=ConfidenceLevel.MEDIUM,
                rationale="One evidence item.",
            ),
            trace=TraceSummary(
                generation=GenerationTrace(
                    prompt_version="test",
                    output_mode=OutputMode.CONVERSATIONAL,
                )
            ),
        )
