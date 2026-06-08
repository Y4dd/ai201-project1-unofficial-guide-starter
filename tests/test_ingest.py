from pathlib import Path
import pytest
from ingest import (
    DocumentChunk,
    load_document,
    clean_document,
    chunk_document,
    validate_chunks,
    load_into_chromadb,
)


def test_load_document_reads_file(tmp_path: Path) -> None:
    doc = tmp_path / "test.md"
    doc.write_text("## Hello\nRent is $900 × 1.5 = $1,350.", encoding="utf-8")
    assert load_document(doc) == "## Hello\nRent is $900 × 1.5 = $1,350."


def test_clean_document_strips_html_comments() -> None:
    text = "## Section\n<!-- hidden comment -->\nVisible content here."
    result = clean_document(text)
    assert "hidden comment" not in result
    assert "Visible content here" in result


def test_clean_document_strips_hr_lines() -> None:
    text = "## Section\nContent here.\n\n---\n\n## Next\nMore content."
    result = clean_document(text)
    assert result == "## Section\nContent here.\n\n## Next\nMore content."


def test_clean_document_collapses_blank_lines() -> None:
    text = "## Section\nContent.\n\n\n\n## Next\nMore."
    result = clean_document(text)
    assert result == "## Section\nContent.\n\n## Next\nMore."


def test_clean_document_raises_on_unclosed_comment() -> None:
    text = "## Section\n<!-- unclosed\nContent that looks fine."
    with pytest.raises(ValueError, match="Unclosed HTML comment"):
        clean_document(text)


def test_chunk_document_splits_on_h2() -> None:
    text = (
        "## Section One\n"
        "Content here, enough to meet the minimum character requirement for this test.\n\n"
        "## Section Two\n"
        "More content here, also enough to meet the minimum character requirement."
    )
    chunks = chunk_document(text, "test.md")
    assert len(chunks) == 2
    assert chunks[0].header == "## Section One"
    assert chunks[1].header == "## Section Two"


def test_chunk_document_splits_on_h3() -> None:
    text = (
        "## Overview\n"
        "Introductory content that is long enough to pass the minimum chunk length filter.\n\n"
        "### Sub Section\n"
        "Subsection content that is also long enough to pass the minimum chunk length filter."
    )
    chunks = chunk_document(text, "test.md")
    assert len(chunks) == 2
    assert chunks[0].header == "## Overview"
    assert chunks[1].header == "### Sub Section"


def test_chunk_document_keeps_table_rows_together() -> None:
    text = (
        "### Pricing Table\n"
        "| Plan | Cost |\n"
        "|------|------|\n"
        "| Gold | $500 |\n"
        "| Silver | $400 |\n"
        "Some trailing note about the table."
    )
    chunks = chunk_document(text, "test.md")
    assert len(chunks) == 1
    assert "| Gold | $500 |" in chunks[0].text
    assert "| Silver | $400 |" in chunks[0].text


def test_chunk_document_drops_short_chunks() -> None:
    text = (
        "## Too Short\n"
        "Hi.\n\n"
        "## Long Enough\n"
        "This section has enough content to exceed the eighty character minimum threshold."
    )
    chunks = chunk_document(text, "test.md")
    assert len(chunks) == 1
    assert chunks[0].header == "## Long Enough"


def test_chunk_document_includes_header_in_text() -> None:
    text = "## My Section\nContent for this section that is long enough to pass the minimum character filter."
    chunks = chunk_document(text, "test.md")
    assert len(chunks) == 1
    assert chunks[0].text.startswith("## My Section")


def test_chunk_document_chunk_id_format() -> None:
    text = (
        "## Section A\n"
        "Content long enough to pass the minimum character length requirement filter.\n\n"
        "## Section B\n"
        "More content also long enough to pass the minimum character length requirement filter."
    )
    chunks = chunk_document(text, "source1_wmu_residence_halls.md")
    assert chunks[0].chunk_id == "source1_wmu_residence_halls_0"
    assert chunks[1].chunk_id == "source1_wmu_residence_halls_1"
