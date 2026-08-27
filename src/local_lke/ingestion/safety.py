"""Upload boundary validation and collision-safe storage."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from uuid import UUID, uuid4

from pypdf import PdfReader

from local_lke.errors import IngestionError

ALLOWED_TYPES = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
}
SAFE_CHARACTER = re.compile(r"[^A-Za-z0-9._-]+")


def normalize_filename(filename: str) -> str:
    normalized = unicodedata.normalize("NFKC", filename).strip()
    if (
        not normalized
        or Path(normalized).name != normalized
        or "/" in normalized
        or "\\" in normalized
    ):
        raise IngestionError("Filename must not contain a path", code="unsafe_filename")
    cleaned = SAFE_CHARACTER.sub("-", normalized).strip(".-")
    if not cleaned:
        raise IngestionError("Filename has no safe characters", code="unsafe_filename")
    suffix = Path(normalized).suffix.lower()
    if suffix and not cleaned.lower().endswith(suffix):
        cleaned = f"{cleaned}{suffix}"
    return cleaned[:255]


def validate_upload(
    filename: str,
    declared_content_type: str | None,
    content: bytes,
    *,
    max_bytes: int,
) -> tuple[str, str]:
    safe_name = normalize_filename(filename)
    suffix = Path(safe_name).suffix.lower()
    expected = ALLOWED_TYPES.get(suffix)
    if expected is None:
        raise IngestionError(
            "Only .md, .txt, and .pdf uploads are supported",
            code="unsupported_type",
        )
    if len(content) > max_bytes:
        raise IngestionError(
            f"File exceeds the configured {max_bytes}-byte limit",
            code="file_too_large",
        )
    if not content:
        raise IngestionError("Empty files cannot be ingested", code="empty_file")
    declared = (declared_content_type or "").split(";", maxsplit=1)[0].strip().lower()
    allowed_declared = {
        "text/plain": {"text/plain", "application/octet-stream", ""},
        "text/markdown": {"text/markdown", "text/plain", "application/octet-stream", ""},
        "application/pdf": {"application/pdf", "application/octet-stream", ""},
    }
    if declared not in allowed_declared[expected]:
        raise IngestionError(
            f"Declared MIME type '{declared}' does not match {suffix}",
            code="mime_mismatch",
        )
    if expected == "application/pdf":
        if not content.startswith(b"%PDF-"):
            raise IngestionError("The file is not a valid PDF", code="mime_mismatch")
        _validate_pdf(content)
    else:
        if content.startswith(b"%PDF-") or b"\x00" in content[:4096]:
            raise IngestionError(
                "File content does not match its text extension", code="mime_mismatch"
            )
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise IngestionError(
                "Text and Markdown uploads must use valid UTF-8 encoding",
                code="invalid_encoding",
            ) from exc
    return safe_name, expected


def store_upload(
    content: bytes,
    *,
    root: Path,
    collection_id: UUID,
    filename: str,
) -> Path:
    destination = root / str(collection_id) / str(uuid4()) / filename
    destination.parent.mkdir(parents=True, exist_ok=False)
    destination.write_bytes(content)
    return destination


def _validate_pdf(content: bytes) -> None:
    from io import BytesIO

    try:
        reader = PdfReader(BytesIO(content), strict=True)
        if reader.is_encrypted:
            raise IngestionError("Encrypted PDF files are not supported", code="encrypted_pdf")
        _ = len(reader.pages)
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError("The PDF is malformed", code="malformed_pdf") from exc
