from io import BytesIO

import pytest
from pypdf import PdfWriter

from local_lke.errors import IngestionError
from local_lke.ingestion.safety import normalize_filename, validate_upload


def test_filename_normalization_rejects_traversal() -> None:
    with pytest.raises(IngestionError, match="must not contain a path"):
        normalize_filename("../secrets.txt")


def test_mime_and_extension_are_both_validated() -> None:
    with pytest.raises(IngestionError) as captured:
        validate_upload(
            "notes.txt",
            "text/plain",
            b"%PDF-1.7\n",
            max_bytes=1000,
        )

    assert captured.value.code == "mime_mismatch"


def test_encrypted_pdf_is_rejected_before_parsing() -> None:
    output = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("secret")
    writer.write(output)

    with pytest.raises(IngestionError) as captured:
        validate_upload(
            "secret.pdf",
            "application/pdf",
            output.getvalue(),
            max_bytes=100_000,
        )

    assert captured.value.code == "encrypted_pdf"


def test_upload_size_is_enforced_before_parsing() -> None:
    with pytest.raises(IngestionError) as captured:
        validate_upload("large.txt", "text/plain", b"abcdef", max_bytes=5)

    assert captured.value.code == "file_too_large"
