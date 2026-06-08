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
