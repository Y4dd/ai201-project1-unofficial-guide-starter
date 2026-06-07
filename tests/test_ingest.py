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
    doc.write_text("## Hello\nWorld content.", encoding="utf-8")
    assert load_document(doc) == "## Hello\nWorld content."
