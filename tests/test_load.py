from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.chunk import DocumentChunk
from src.load import load_all_documents, load_document, load_into_chromadb


def test_load_document_reads_file(tmp_path: Path) -> None:
    doc = tmp_path / "test.md"
    doc.write_text("## Hello\nRent is $900 × 1.5 = $1,350.", encoding="utf-8")
    assert load_document(doc) == "## Hello\nRent is $900 × 1.5 = $1,350."


def test_load_all_documents_returns_chunks(tmp_path: Path) -> None:
    doc1 = tmp_path / "source1_test.md"
    doc2 = tmp_path / "source2_test.md"
    doc1.write_text(
        "## Section A\nContent long enough to pass the minimum character length filter here.\n\n"
        "## Section B\nMore content also long enough to pass the minimum character length filter.",
        encoding="utf-8",
    )
    doc2.write_text(
        "## Only Section\nContent long enough to pass the minimum character length filter here.",
        encoding="utf-8",
    )
    chunks = load_all_documents(tmp_path)
    assert len(chunks) == 3
    source_files = {c.source_file for c in chunks}
    assert "source1_test.md" in source_files
    assert "source2_test.md" in source_files


def test_load_all_documents_raises_on_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "source1_bad.md"
    bad.write_text("## Section\n<!-- unclosed comment\nContent.", encoding="utf-8")
    with pytest.raises(ValueError, match="source1_bad.md"):
        load_all_documents(tmp_path)


def test_load_all_documents_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    assert load_all_documents(tmp_path) == []


def test_load_into_chromadb_calls_upsert_with_correct_args() -> None:
    chunks = [
        DocumentChunk(
            chunk_id="source7_0",
            source_file="source7_michigan_tenant_law.md",
            header="## Maximum Deposit Limit",
            text="## Maximum Deposit Limit\nLandlords may charge no more than 1.5× monthly rent.",
            char_count=79,
        ),
        DocumentChunk(
            chunk_id="source7_1",
            source_file="source7_michigan_tenant_law.md",
            header="## The 30-Day Return Deadline",
            text="## The 30-Day Return Deadline\nLandlord has 30 days to return the deposit.",
            char_count=72,
        ),
    ]
    mock_collection = MagicMock()
    load_into_chromadb(chunks, mock_collection)
    mock_collection.upsert.assert_called_once_with(
        ids=["source7_0", "source7_1"],
        documents=[chunks[0].text, chunks[1].text],
        metadatas=[
            {"source_file": "source7_michigan_tenant_law.md", "header": "## Maximum Deposit Limit", "char_count": 79},
            {"source_file": "source7_michigan_tenant_law.md", "header": "## The 30-Day Return Deadline", "char_count": 72},
        ],
    )


def test_load_into_chromadb_skips_upsert_on_empty_list() -> None:
    mock_collection = MagicMock()
    load_into_chromadb([], mock_collection)
    mock_collection.upsert.assert_not_called()
