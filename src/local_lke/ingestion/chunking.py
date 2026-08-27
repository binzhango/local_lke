"""Inspectable chunking strategies preserving source structure."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Sequence
from uuid import UUID

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

from local_lke.models import ChunkStrategy, DocumentElement, IngestedChunk

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def chunk_elements(
    elements: Sequence[DocumentElement],
    *,
    version_id: UUID,
    strategy: ChunkStrategy,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[IngestedChunk], tuple[str, ...]]:
    """Chunk normalized elements and return persisted chunks plus warnings."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    candidates: list[tuple[DocumentElement, str, str]] = []
    for element in elements:
        if strategy is ChunkStrategy.SEMANTIC:
            texts = _semantic_segments(element.text, chunk_size)
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            texts = splitter.split_text(element.text)
        for part_index, text in enumerate(texts, start=1):
            normalized = " ".join(text.split())
            if normalized:
                locator = element.locator
                if len(texts) > 1:
                    locator = f"{locator};part:{part_index}/{len(texts)}"
                candidates.append((element, normalized, locator))

    frequencies = Counter(text.casefold() for _, text, _ in candidates)
    chunks: list[IngestedChunk] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for element, text, locator in candidates:
        fingerprint = text.casefold()
        if fingerprint in seen:
            warnings.append(f"repeated chunk omitted at {locator}")
            continue
        seen.add(fingerprint)
        flags: list[str] = []
        if frequencies[fingerprint] > 1:
            flags.append("repeated")
        if element.category in {"Header", "Footer", "PageBreak"}:
            flags.append("boilerplate")
        if len(text) < 24 and element.category not in {"Title", "Header"}:
            flags.append("short")
        ordinal = len(chunks)
        chunk_id = hashlib.sha256(
            f"{version_id}:{strategy.value}:{ordinal}:{text}".encode()
        ).hexdigest()
        chunks.append(
            IngestedChunk(
                chunk_id=chunk_id,
                document_version_id=version_id,
                parent_element_id=element.element_id,
                ordinal=ordinal,
                strategy=strategy,
                text=text,
                locator=locator,
                page_number=element.page_number,
                heading_path=element.heading_path,
                character_count=len(text),
                token_count=max(1, len(re.findall(r"\w+|[^\w\s]", text))),
                flags=tuple(flags),
            )
        )
    return chunks, tuple(warnings)


def _semantic_segments(text: str, chunk_size: int) -> list[str]:
    """Experimental local TF-IDF topic-shift chunking without a model call."""
    sentences = [item.strip() for item in SENTENCE_BOUNDARY.split(text) if item.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences
    try:
        vectors = TfidfVectorizer(stop_words="english").fit_transform(sentences)
        similarities = cosine_similarity(vectors[:-1], vectors[1:]).diagonal().tolist()
    except ValueError:
        similarities = [1.0] * (len(sentences) - 1)
    groups: list[str] = []
    current = sentences[0]
    minimum_group_size = max(40, chunk_size // 3)
    for index, sentence in enumerate(sentences[1:]):
        candidate = f"{current} {sentence}".strip()
        topic_changed = similarities[index] < 0.12 and len(current) >= minimum_group_size
        if len(candidate) > chunk_size or topic_changed:
            groups.append(current)
            current = sentence
        else:
            current = candidate
    groups.append(current)
    return groups
