"""Normalized parsers for Chapter 2 supported document formats."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from local_lke.errors import IngestionError
from local_lke.models import DocumentElement, ParserStrategy

ELEMENT_NAMESPACE = UUID("16c39e57-5087-43a2-a14f-81e72d4ccb58")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class ParseResult:
    parser_name: str
    parser_version: str
    elements: list[DocumentElement]
    warnings: tuple[str, ...] = ()


def parse_document(
    path: Path,
    *,
    media_type: str,
    version_id: UUID,
    strategy: ParserStrategy,
) -> ParseResult:
    if media_type == "text/markdown":
        return parse_markdown(path, version_id)
    if media_type == "text/plain":
        return parse_text(path, version_id)
    if media_type == "application/pdf":
        return parse_pdf(path, version_id, strategy)
    raise IngestionError(f"Unsupported media type: {media_type}", code="unsupported_type")


def parse_text(path: Path, version_id: UUID) -> ParseResult:
    text = _read_utf8(path)
    elements = _paragraph_elements(text, version_id)
    if not elements:
        raise IngestionError("The text document contains no usable content", code="empty_document")
    return ParseResult(parser_name="utf8-text", parser_version="1.0", elements=elements)


def parse_markdown(path: Path, version_id: UUID) -> ParseResult:
    text = _read_utf8(path)
    lines = text.splitlines()
    heading_path: list[str] = []
    elements: list[DocumentElement] = []
    paragraph_start: int | None = None
    paragraph_lines: list[str] = []

    def flush_paragraph(end_line: int) -> None:
        nonlocal paragraph_start, paragraph_lines
        body = "\n".join(paragraph_lines).strip()
        if body and paragraph_start is not None:
            elements.append(
                _element(
                    version_id,
                    len(elements),
                    "NarrativeText",
                    body,
                    f"lines:{paragraph_start}-{end_line}",
                    heading_path=tuple(heading_path),
                    metadata={"source_line_start": paragraph_start, "source_line_end": end_line},
                )
            )
        paragraph_start = None
        paragraph_lines = []

    for line_number, line in enumerate(lines, start=1):
        match = HEADING.match(line)
        if match:
            flush_paragraph(line_number - 1)
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_path[:] = heading_path[: level - 1]
            heading_path.append(title)
            elements.append(
                _element(
                    version_id,
                    len(elements),
                    "Title",
                    title,
                    f"line:{line_number}",
                    heading_path=tuple(heading_path),
                    metadata={"heading_level": level, "source_line_start": line_number},
                )
            )
        elif not line.strip():
            flush_paragraph(line_number - 1)
        else:
            if paragraph_start is None:
                paragraph_start = line_number
            paragraph_lines.append(line)
    flush_paragraph(len(lines))
    if not elements:
        raise IngestionError(
            "The Markdown document contains no usable content", code="empty_document"
        )
    return ParseResult(
        parser_name="markdown-heading-parser", parser_version="1.0", elements=elements
    )


def parse_pdf(path: Path, version_id: UUID, strategy: ParserStrategy) -> ParseResult:
    try:
        raw_elements = _partition_pdf(path, strategy)
    except Exception as exc:
        hint = (
            " The hi_res strategy also requires local PDF/OCR system dependencies."
            if strategy is ParserStrategy.HI_RES
            else ""
        )
        raise IngestionError(
            f"PDF parsing failed: {type(exc).__name__}.{hint}", code="pdf_parse_failed"
        ) from exc

    elements: list[DocumentElement] = []
    heading_path: list[str] = []
    for raw in raw_elements:
        text = str(raw).strip()
        if not text:
            continue
        category = getattr(raw, "category", type(raw).__name__)
        metadata_object = getattr(raw, "metadata", None)
        metadata = _metadata_dict(metadata_object)
        page_number = _positive_int(metadata.get("page_number"))
        if category in {"Title", "Header"}:
            heading_path[:] = [text]
        locator = (
            f"page:{page_number}" if page_number is not None else f"element:{len(elements) + 1}"
        )
        elements.append(
            _element(
                version_id,
                len(elements),
                category,
                text,
                locator,
                page_number=page_number,
                heading_path=tuple(heading_path),
                metadata={
                    key: value
                    for key, value in metadata.items()
                    if isinstance(value, str | int | float | bool) or value is None
                },
            )
        )
    if not elements:
        raise IngestionError(
            "The PDF contains no extractable content; scanned files may require hi_res OCR",
            code="empty_document",
        )
    try:
        parser_version = version("unstructured")
    except PackageNotFoundError:  # pragma: no cover
        parser_version = "unknown"
    return ParseResult(
        parser_name="unstructured.partition_pdf",
        parser_version=parser_version,
        elements=elements,
    )


def _partition_pdf(path: Path, strategy: ParserStrategy) -> list[Any]:
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as exc:  # pragma: no cover - dependency installation error
        raise IngestionError(
            "PDF parsing dependencies are unavailable; run 'uv sync' again",
            code="parser_unavailable",
        ) from exc
    return list(partition_pdf(filename=str(path), strategy=strategy.value))


def _paragraph_elements(text: str, version_id: UUID) -> list[DocumentElement]:
    lines = text.splitlines()
    elements: list[DocumentElement] = []
    start: int | None = None
    content: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if line.strip():
            if start is None:
                start = line_number
            content.append(line)
        elif content and start is not None:
            elements.append(
                _element(
                    version_id,
                    len(elements),
                    "NarrativeText",
                    "\n".join(content).strip(),
                    f"lines:{start}-{line_number - 1}",
                    metadata={"source_line_start": start, "source_line_end": line_number - 1},
                )
            )
            start, content = None, []
    if content and start is not None:
        elements.append(
            _element(
                version_id,
                len(elements),
                "NarrativeText",
                "\n".join(content).strip(),
                f"lines:{start}-{len(lines)}",
                metadata={"source_line_start": start, "source_line_end": len(lines)},
            )
        )
    return elements


def _element(
    version_id: UUID,
    ordinal: int,
    category: str,
    text: str,
    locator: str,
    *,
    page_number: int | None = None,
    heading_path: tuple[str, ...] = (),
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> DocumentElement:
    return DocumentElement(
        element_id=uuid5(ELEMENT_NAMESPACE, f"{version_id}:{ordinal}"),
        document_version_id=version_id,
        ordinal=ordinal,
        category=category,
        text=text,
        locator=locator,
        page_number=page_number,
        heading_path=heading_path,
        metadata=metadata or {},
    )


def _read_utf8(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise IngestionError(
            "Text and Markdown uploads must use valid UTF-8 encoding",
            code="invalid_encoding",
        ) from exc


def _metadata_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        return dict(result) if isinstance(result, dict) else {}
    return dict(vars(value)) if hasattr(value, "__dict__") else {}


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 1 else None
