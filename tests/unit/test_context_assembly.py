from datetime import UTC, datetime
from uuid import UUID, uuid4

from local_lke.models import ActiveChunk
from local_lke.retrieval.service import Candidate, _assemble_context


def _candidate(
    chunk_id: str,
    filename: str,
    document_id: UUID,
    text: str,
    subquery: str,
) -> Candidate:
    candidate = Candidate(
        chunk=ActiveChunk(
            chunk_id=chunk_id,
            collection_id=uuid4(),
            document_id=document_id,
            version_id=uuid4(),
            filename=filename,
            media_type="text/plain",
            parser_strategy="fast",
            chunk_strategy="recursive",
            ordinal=0,
            text=text,
            locator="lines:1-1",
            token_count=80,
            created_at=datetime.now(UTC),
        ),
        dense_score=0.9,
    )
    candidate.matched_subqueries.add(subquery)
    return candidate


def test_context_deduplicates_preserves_source_order_and_enforces_budgets() -> None:
    first_document = uuid4()
    duplicate_text = " ".join(f"alpha{i}" for i in range(80))
    first = _candidate("a" * 64, "a.txt", first_document, duplicate_text, "alpha")
    duplicate = _candidate("b" * 64, "z.txt", uuid4(), duplicate_text, "alpha")
    second = _candidate(
        "c" * 64,
        "b.txt",
        uuid4(),
        " ".join(f"beta{i}" for i in range(80)),
        "beta",
    )

    context, manifest = _assemble_context(
        [first, duplicate, second],
        ["alpha", "beta"],
        total_budget=64,
        source_budget=32,
        top_k=3,
    )

    assert [item.candidate.chunk.filename for item in context] == ["a.txt", "b.txt"]
    assert sum(item.token_count for item in context) == 64
    assert all(item.token_count == 32 for item in context)
    decisions = {item.chunk_id: (item.decision, item.reason) for item in manifest}
    assert decisions[duplicate.chunk.chunk_id] == (
        "excluded",
        "exact or near-duplicate evidence",
    )
    assert {item.chunk_id for item in manifest} == {
        first.chunk.chunk_id,
        duplicate.chunk.chunk_id,
        second.chunk.chunk_id,
    }
