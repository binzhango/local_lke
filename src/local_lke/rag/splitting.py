"""Stable, metadata-preserving recursive text splitting."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_lke.models import Chunk, SourceDocument


def split_documents(
    documents: list[SourceDocument],
    *,
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n\n", "\n", ". ", " "],
    )
    chunks: list[Chunk] = []
    for document in documents:
        for ordinal, text in enumerate(splitter.split_text(document.content)):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.source_id}:chunk:{ordinal:04d}",
                    source_id=document.source_id,
                    locator=document.locator,
                    text=text.strip(),
                    ordinal=ordinal,
                )
            )
    return chunks
