from uuid import uuid4

import pytest

from local_lke.ingestion.chunking import chunk_elements
from local_lke.models import ChunkStrategy, DocumentElement


@pytest.mark.parametrize("strategy", list(ChunkStrategy))
def test_every_chunk_strategy_preserves_provenance_and_stable_ids(
    strategy: ChunkStrategy,
) -> None:
    version_id = uuid4()
    element = DocumentElement(
        element_id=uuid4(),
        document_version_id=version_id,
        ordinal=0,
        category="NarrativeText",
        text=("A grounded sentence explains recovery. " * 12).strip(),
        locator="lines:3-8",
        heading_path=("Recovery",),
    )

    first, _ = chunk_elements(
        [element],
        version_id=version_id,
        strategy=strategy,
        chunk_size=120,
        chunk_overlap=20,
    )
    second, _ = chunk_elements(
        [element],
        version_id=version_id,
        strategy=strategy,
        chunk_size=120,
        chunk_overlap=20,
    )

    assert first
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert all(item.parent_element_id == element.element_id for item in first)
    assert all(item.heading_path == ("Recovery",) for item in first)
    assert all(item.character_count and item.token_count for item in first)


def test_repeated_chunks_are_omitted_with_a_warning() -> None:
    version_id = uuid4()
    repeated = "Repeated navigation footer text long enough to retain."
    elements = [
        DocumentElement(
            element_id=uuid4(),
            document_version_id=version_id,
            ordinal=index,
            category="Footer",
            text=repeated,
            locator=f"page:{index + 1}",
        )
        for index in range(2)
    ]

    chunks, warnings = chunk_elements(
        elements,
        version_id=version_id,
        strategy=ChunkStrategy.RECURSIVE,
        chunk_size=200,
        chunk_overlap=20,
    )

    assert len(chunks) == 1
    assert "boilerplate" in chunks[0].flags
    assert warnings == ("repeated chunk omitted at page:2",)
