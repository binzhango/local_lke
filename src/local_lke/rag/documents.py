"""Fixture loading and LangChain conversion at the indexing boundary."""

from pathlib import Path

from langchain_core.documents import Document

from local_lke.models import Chunk, SourceDocument

SUPPORTED_SUFFIXES = {".md": "text/markdown", ".txt": "text/plain"}


def default_fixture_directory() -> Path:
    return Path(__file__).resolve().parents[3] / "fixtures"


def load_fixture_documents(directory: Path | None = None) -> list[SourceDocument]:
    fixture_directory = directory or default_fixture_directory()
    documents: list[SourceDocument] = []
    for path in sorted(fixture_directory.iterdir()):
        media_type = SUPPORTED_SUFFIXES.get(path.suffix.lower())
        if media_type is None:
            continue
        content = normalize_text(path.read_text(encoding="utf-8"))
        documents.append(
            SourceDocument(
                source_id=f"fixture:{path.stem}",
                title=path.stem.replace("-", " ").title(),
                locator=f"fixtures/{path.name}",
                content=content,
                media_type=media_type,
            )
        )
    if not documents:
        raise ValueError(f"No Markdown or text fixtures found in {fixture_directory}")
    return documents


def normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


def chunk_to_langchain_document(chunk: Chunk) -> Document:
    """Keep LangChain's schema private to the vector-store boundary."""
    return Document(
        page_content=chunk.text,
        metadata={
            "chunk_id": chunk.chunk_id,
            "source_id": chunk.source_id,
            "locator": chunk.locator,
            "ordinal": chunk.ordinal,
        },
    )


def langchain_document_to_chunk(document: Document) -> Chunk:
    metadata = document.metadata
    return Chunk(
        chunk_id=str(metadata["chunk_id"]),
        source_id=str(metadata["source_id"]),
        locator=str(metadata["locator"]),
        ordinal=int(metadata["ordinal"]),
        text=document.page_content,
    )
