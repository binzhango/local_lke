from pathlib import Path
from uuid import uuid4

from local_lke.ingestion.parsers import parse_markdown, parse_pdf, parse_text
from local_lke.models import ParserStrategy


def test_markdown_preserves_heading_hierarchy_and_source_lines(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text(
        "# Atlas\n\nOverview.\n\n## Recovery\n\nRestart the worker.\n",
        encoding="utf-8",
    )

    result = parse_markdown(path, uuid4())

    assert [item.category for item in result.elements] == [
        "Title",
        "NarrativeText",
        "Title",
        "NarrativeText",
    ]
    assert result.elements[-1].heading_path == ("Atlas", "Recovery")
    assert result.elements[-1].locator == "lines:7-7"


def test_plain_text_preserves_paragraph_line_ranges(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("First paragraph.\n\nSecond paragraph.\n", encoding="utf-8")

    result = parse_text(path, uuid4())

    assert [item.locator for item in result.elements] == ["lines:1-1", "lines:3-3"]


def test_pdf_normalization_preserves_pages_categories_and_table_boundaries(
    tmp_path: Path, monkeypatch
) -> None:
    class Metadata:
        def __init__(self, page_number: int, text_as_html: str | None = None) -> None:
            self.page_number = page_number
            self.text_as_html = text_as_html

        def to_dict(self) -> dict[str, object]:
            return {
                "page_number": self.page_number,
                "text_as_html": self.text_as_html,
            }

    class FakeElement:
        def __init__(self, text: str, category: str, metadata: Metadata) -> None:
            self.text = text
            self.category = category
            self.metadata = metadata

        def __str__(self) -> str:
            return self.text

    raw = [
        FakeElement("Quarterly Report", "Title", Metadata(1)),
        FakeElement("Page one body", "NarrativeText", Metadata(1)),
        FakeElement("A | B\n1 | 2", "Table", Metadata(2, "<table></table>")),
    ]
    monkeypatch.setattr("local_lke.ingestion.parsers._partition_pdf", lambda *_: raw)
    path = tmp_path / "report.pdf"
    path.write_bytes(b"unused by patched partitioner")

    result = parse_pdf(path, uuid4(), ParserStrategy.FAST)

    assert [item.page_number for item in result.elements] == [1, 1, 2]
    assert result.elements[2].category == "Table"
    assert result.elements[2].metadata["text_as_html"] == "<table></table>"
    assert result.elements[1].heading_path == ("Quarterly Report",)
