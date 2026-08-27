from uuid import UUID

from local_lke.errors import IngestionError
from local_lke.ingestion import IngestionService
from local_lke.ingestion.parsers import parse_document as real_parse_document
from local_lke.models import ChunkStrategy, JobStatus, ParserStrategy


def _ingest(
    service: IngestionService,
    collection_id: UUID,
    content: bytes,
    *,
    chunk_size: int = 200,
):
    return service.ingest(
        collection_id=collection_id,
        filename="guide.md",
        content_type="text/markdown",
        content=content,
        parser_strategy=ParserStrategy.FAST,
        chunk_strategy=ChunkStrategy.MARKDOWN,
        chunk_size=chunk_size,
        chunk_overlap=20,
    )


def test_identical_ingestion_writes_no_duplicate_version_or_chunks(
    ingestion: IngestionService,
) -> None:
    collection = ingestion.create_collection("Knowledge")
    content = b"# Runbook\n\nRestart the service safely after draining work."

    first = _ingest(ingestion, collection.id, content)
    second = _ingest(ingestion, collection.id, content)
    documents = ingestion.list_documents(collection.id)

    assert first.status is JobStatus.COMPLETED
    assert second.status is JobStatus.COMPLETED
    assert second.skipped is True
    assert second.version_id == first.version_id
    assert len(documents) == 1
    assert len(documents[0].versions) == 1


def test_pipeline_change_creates_one_new_active_immutable_version(
    ingestion: IngestionService,
) -> None:
    collection = ingestion.create_collection("Versions")
    content = b"# Runbook\n\nRestart the service safely after draining work."

    first = _ingest(ingestion, collection.id, content, chunk_size=200)
    second = _ingest(ingestion, collection.id, content, chunk_size=300)
    versions = ingestion.list_documents(collection.id)[0].versions

    assert first.version_id != second.version_id
    assert len(versions) == 2
    assert sum(version.active for version in versions) == 1
    assert versions[0].inactive_reason == "superseded by a newer version"


def test_soft_delete_retains_versions_but_deactivates_searchable_state(
    ingestion: IngestionService,
) -> None:
    collection = ingestion.create_collection("Deletion")
    job = _ingest(ingestion, collection.id, b"# Policy\n\nRetain provenance.")
    document = ingestion.list_documents(collection.id)[0]

    deleted = ingestion.delete_document(document.id, "retention request")

    assert job.version_id is not None
    assert deleted.deleted_at is not None
    assert deleted.versions[0].active is False
    assert deleted.versions[0].inactive_reason == "retention request"


def test_failed_job_remains_inspectable_and_can_be_explicitly_retried(
    ingestion: IngestionService, monkeypatch
) -> None:
    collection = ingestion.create_collection("Retry")

    def fail_parser(*args, **kwargs):
        del args, kwargs
        raise IngestionError("synthetic parser failure", code="parser_failed")

    monkeypatch.setattr("local_lke.ingestion.service.parse_document", fail_parser)
    failed = _ingest(ingestion, collection.id, b"# Retry\n\nTry this again.")
    monkeypatch.setattr("local_lke.ingestion.service.parse_document", real_parse_document)
    retried = ingestion.retry(failed.id)

    assert failed.status is JobStatus.FAILED
    assert failed.error_code == "parser_failed"
    assert ingestion.get_job(failed.id).id == failed.id
    assert retried.status is JobStatus.COMPLETED
    assert retried.version_id is not None


def test_startup_marks_abandoned_running_jobs_interrupted(
    ingestion: IngestionService,
) -> None:
    collection = ingestion.create_collection("Recovery")
    job = ingestion.repository.create_job(
        collection_id=str(collection.id),
        filename="interrupted.txt",
        display_filename="interrupted.txt",
        storage_path="/tmp/interrupted.txt",
        media_type="text/plain",
        parser_strategy="fast",
        chunk_strategy="recursive",
        chunk_size=200,
        chunk_overlap=20,
        status=JobStatus.RUNNING.value,
        progress=50,
    )

    recovered = ingestion.startup()
    current = ingestion.get_job(job.id)

    assert recovered == 1
    assert current.status is JobStatus.INTERRUPTED
    assert current.error_code == "process_interrupted"
