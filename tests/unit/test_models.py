import pytest
from pydantic import ValidationError

from local_lke.models import AnswerResponse, AnswerStatus, TraceSummary


def test_answered_response_requires_a_citation() -> None:
    with pytest.raises(ValidationError, match="require at least one citation"):
        AnswerResponse(
            status=AnswerStatus.ANSWERED,
            answer="Unsupported answer",
            citations=[],
            trace=TraceSummary(),
        )


def test_abstention_can_have_no_citations() -> None:
    response = AnswerResponse(
        status=AnswerStatus.ABSTAINED,
        answer="I do not know.",
        citations=[],
        trace=TraceSummary(),
    )
    assert response.status is AnswerStatus.ABSTAINED

