from io import BytesIO

import pytest
from PIL import Image

from local_lke.errors import IndexingError
from local_lke.indexing import MultimodalIndexingService
from local_lke.indexing.images import validate_image


def _png(color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def test_text_and_image_queries_rank_expected_local_image(
    multimodal: MultimodalIndexingService,
    ingestion,
) -> None:
    collection = ingestion.create_collection("Images")
    red = multimodal.ingest(
        collection_id=collection.id,
        filename="red.png",
        content_type="image/png",
        content=_png((255, 0, 0)),
    )
    blue = multimodal.ingest(
        collection_id=collection.id,
        filename="blue.png",
        content_type="image/png",
        content=_png((0, 0, 255)),
    )

    text_result = multimodal.search_text(collection.id, "bright red image", 2)
    image_result = multimodal.search_image(
        collection.id,
        "query.png",
        "image/png",
        _png((250, 5, 5)),
        2,
    )

    assert text_result.hits[0].image.id == red.id
    assert image_result.hits[0].image.id == red.id
    assert {item.image.id for item in text_result.hits} == {red.id, blue.id}
    assert red.content_url.startswith("/api/v1/images/")


def test_duplicate_image_is_idempotent_and_invalid_content_is_rejected(
    multimodal: MultimodalIndexingService,
    ingestion,
) -> None:
    collection = ingestion.create_collection("Image safety")
    content = _png((0, 255, 0))
    first = multimodal.ingest(
        collection_id=collection.id,
        filename="green.png",
        content_type="image/png",
        content=content,
    )
    second = multimodal.ingest(
        collection_id=collection.id,
        filename="renamed.png",
        content_type="image/png",
        content=content,
    )

    assert first.id == second.id

    try:
        multimodal.ingest(
            collection_id=collection.id,
            filename="fake.png",
            content_type="image/png",
            content=b"not an image",
        )
    except Exception as exc:
        assert getattr(exc, "code", None) == "invalid_image"
    else:
        raise AssertionError("invalid image was accepted")


def test_image_pixel_limit_is_checked_before_full_decode() -> None:
    with pytest.raises(IndexingError) as caught:
        validate_image(
            "large.png",
            "image/png",
            _png((10, 20, 30), size=(11, 11)),
            max_bytes=10_000,
            max_pixels=100,
        )

    assert caught.value.code == "image_too_large"
