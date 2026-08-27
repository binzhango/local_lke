"""Typed citation locator construction for every supported evidence family."""

import re
from uuid import UUID

from local_lke.models import CitationLocator, CitationLocatorKind

LINE_RANGE = re.compile(r"(?:lines?|L)(\d+)(?:-(?:L)?(\d+))?", re.IGNORECASE)


def citation_locator(
    locator: str,
    *,
    media_type: str,
    heading_path: tuple[str, ...] = (),
    page_number: int | None = None,
    element_id: UUID | None = None,
    image_id: UUID | None = None,
    table_id: UUID | None = None,
    row_start: int | None = None,
    row_end: int | None = None,
) -> CitationLocator:
    if image_id is not None or media_type.startswith("image/"):
        return CitationLocator(kind=CitationLocatorKind.IMAGE, label=locator, image_id=image_id)
    if table_id is not None or media_type == "text/csv":
        return CitationLocator(
            kind=CitationLocatorKind.TABLE,
            label=locator,
            table_id=table_id,
            row_start=row_start,
            row_end=row_end,
        )
    if page_number is not None or media_type == "application/pdf":
        return CitationLocator(
            kind=CitationLocatorKind.PDF,
            label=locator,
            page_number=page_number,
            element_id=element_id,
        )
    if heading_path or media_type == "text/markdown":
        return CitationLocator(
            kind=CitationLocatorKind.MARKDOWN,
            label=locator,
            heading_path=heading_path,
        )
    match = LINE_RANGE.search(locator)
    if match:
        return CitationLocator(
            kind=CitationLocatorKind.TEXT,
            label=locator,
            start_line=int(match.group(1)),
            end_line=int(match.group(2) or match.group(1)),
        )
    return CitationLocator(kind=CitationLocatorKind.GENERIC, label=locator)
