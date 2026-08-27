from local_lke.rag.documents import load_fixture_documents
from local_lke.rag.splitting import split_documents


def test_fixture_source_ids_and_chunks_are_stable() -> None:
    documents = load_fixture_documents()
    chunks = split_documents(documents, chunk_size=240, chunk_overlap=30)

    assert [document.source_id for document in documents] == [
        "fixture:atlas-retention",
        "fixture:atlas-support",
    ]
    assert chunks[0].chunk_id == "fixture:atlas-retention:chunk:0000"
    assert all(chunk.locator.startswith("fixtures/") for chunk in chunks)


def test_normalized_fixtures_have_no_windows_line_endings() -> None:
    assert all("\r" not in document.content for document in load_fixture_documents())
