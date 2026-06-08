# Milestone 3 — Document Ingestion Pipeline (`ingest.py`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ingest.py` that reads the 8 markdown source files in `documents/`, splits each file into header-aligned chunks, validates the output (printing 5 representative chunks), and loads all chunks into a persistent ChromaDB collection with sentence-transformer embeddings.

**Architecture:** Loading → Cleaning → Chunking → Validation → ChromaDB upsert. Each stage is a typed function. The chunker splits only on `##` and `###` header lines — table rows (`|`) never trigger a split, so pricing tables land intact in one chunk. Chunks under 80 characters are dropped. The ChromaDB collection is created with a `SentenceTransformerEmbeddingFunction` so embeddings are computed at ingest time.

**Tech Stack:** Python 3.11+, `sentence-transformers==3.4.1`, `chromadb>=0.6.0`, `pytest`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `ingest.py` | All pipeline logic: load → clean → chunk → validate → store |
| Create | `tests/__init__.py` | Empty — needed for pytest discovery |
| Create | `tests/test_ingest.py` | Unit tests for every public function |
| Runtime output | `chroma_db/` | ChromaDB PersistentClient directory (git-ignored) |

---

## Representative Chunks (Expected Validation Output)

These are the 5 chunks the `validate_chunks` step should print. Use them as a manual correctness check after running `python ingest.py`.

**Chunk 1 — Short legal prose** (`source7_michigan_tenant_law_1`, ~165 chars)
```
## Maximum Deposit Limit

Landlords may charge a security deposit of **no more than 1.5 times the monthly rent**.

- Example: $900/month rent → max deposit is $1,350
- Any deposit above this limit is illegal
```

**Chunk 2 — Numbered list with critical note** (`source7_michigan_tenant_law_2`, ~445 chars)
```
## The 30-Day Return Deadline

After a tenant vacates, the landlord has **30 days** to:
1. Return the full deposit, OR
2. Return the remaining amount after deductions AND provide a written itemized damage list

**Critical detail:** The 30-day clock does not start until the landlord receives the tenant's **written forwarding address**. Always send your forwarding address in writing (certified mail is best).
```

**Chunk 3 — Bullet-list hall description** (`source1_wmu_residence_halls_1`, ~380 chars)
```
### Henry Hall
- Location: Central campus — 1–6 minute walk to Rec Center, Student Center, and dining
- Style: Traditional (community bathrooms)
- Eligibility: Any student
- Room types: Double and single rooms available
- Amenities: Renovated kitchen, updated game room, spacious lounges
- Price tier: $ (lower end)
- Meal plan: Required (see rates below)
```

**Chunk 4 — Pricing table** (`source1_wmu_residence_halls_5`, ~245 chars) ← table guard validation
```
### Valley and Henry Halls — Double Room
| Meal Plan | Per Semester | Annual |
|-----------|-------------|--------|
| Bronco Gold Plus | $6,395.50 | $12,791 |
| Bronco Gold | $6,330.50 | $12,661 |
| Bronco 14 | $6,074.00 | $12,148 |
```

**Chunk 5 — Long section with statutory blockquote** (`source7_michigan_tenant_law_3`, ~680 chars)
```
## What the Itemized Damage List Must Include

If the landlord withholds any amount, the written notice must contain:
- The **specific reason** for each deduction
- The **estimated or actual cost** of repair for each item
- A **check or money order** for the difference between the deposit and the damages claimed
- The following statement in **bold, 12-point font** (at least 4 points larger than surrounding text):

> "You must respond to this notice by mail within 7 days after receipt of same, otherwise you will forfeit the amount claimed for damages."

If this exact statement is missing or improperly formatted, the itemized list is defective under Michigan law.
```

---

## Task 1: Module Skeleton + `DocumentChunk` Type

**Files:**
- Create: `ingest.py`
- Create: `tests/__init__.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Create `tests/__init__.py`** (empty file)

```python
```

- [ ] **Step 2: Write the skeleton `ingest.py`**

```python
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DOCS_DIR = Path(__file__).parent / "documents"
CHROMA_DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "wmu_housing"
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"

_HEADER_RE = re.compile(r"^#{2,3} ")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HR_RE = re.compile(r"^---+$", re.MULTILINE)
MIN_CHUNK_CHARS = 80


@dataclass
class DocumentChunk:
    chunk_id: str       # e.g. "source7_michigan_tenant_law_2"
    source_file: str    # e.g. "source7_michigan_tenant_law.md"
    header: str         # e.g. "## The 30-Day Return Deadline"
    text: str           # full chunk text including the header line
    char_count: int


def load_document(path: Path) -> str: ...

def clean_document(text: str) -> str: ...

def chunk_document(text: str, source_file: str) -> list[DocumentChunk]: ...

def load_all_documents(docs_dir: Path = DOCS_DIR) -> list[DocumentChunk]: ...

def validate_chunks(chunks: list[DocumentChunk]) -> None: ...

def load_into_chromadb(chunks: list[DocumentChunk], collection: chromadb.Collection) -> None: ...

def main() -> None: ...


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write the skeleton `tests/test_ingest.py`**

```python
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
```

- [ ] **Step 4: Verify the skeleton imports without errors**

Run: `python -c "import ingest"`
Expected: no output, no errors

- [ ] **Step 5: Commit**

```bash
git add ingest.py tests/__init__.py tests/test_ingest.py
git commit -m "feat: add ingest.py skeleton with DocumentChunk type and empty stubs"
```

---

## Task 2: `load_document`

**Files:**
- Modify: `ingest.py` — implement `load_document`
- Modify: `tests/test_ingest.py` — add test

- [ ] **Step 1: Write the failing test**

```python
def test_load_document_reads_file(tmp_path: Path) -> None:
    doc = tmp_path / "test.md"
    doc.write_text("## Hello\nWorld content.", encoding="utf-8")
    assert load_document(doc) == "## Hello\nWorld content."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest.py::test_load_document_reads_file -v`
Expected: FAIL — `TypeError` because the stub returns `None`

- [ ] **Step 3: Implement `load_document`**

```python
def load_document(path: Path) -> str:
    return path.read_text(encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest.py::test_load_document_reads_file -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ingest.py tests/test_ingest.py
git commit -m "feat: implement load_document"
```

---

## Task 3: `clean_document`

**Files:**
- Modify: `ingest.py` — implement `clean_document`
- Modify: `tests/test_ingest.py` — add tests

- [ ] **Step 1: Write the failing tests**

```python
def test_clean_document_strips_html_comments() -> None:
    text = "## Section\n<!-- hidden comment -->\nVisible content here."
    result = clean_document(text)
    assert "hidden comment" not in result
    assert "Visible content here" in result


def test_clean_document_strips_hr_lines() -> None:
    text = "## Section\nContent here.\n\n---\n\n## Next\nMore content."
    result = clean_document(text)
    assert "---" not in result
    assert "## Section" in result
    assert "## Next" in result


def test_clean_document_collapses_blank_lines() -> None:
    text = "## Section\nContent.\n\n\n\n## Next\nMore."
    result = clean_document(text)
    assert "\n\n\n" not in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingest.py -k "clean" -v`
Expected: 3 FAILs

- [ ] **Step 3: Implement `clean_document`**

```python
def clean_document(text: str) -> str:
    text = _HTML_COMMENT_RE.sub("", text)
    text = _HR_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingest.py -k "clean" -v`
Expected: 3 PASSes

- [ ] **Step 5: Commit**

```bash
git add ingest.py tests/test_ingest.py
git commit -m "feat: implement clean_document — strips HR lines, HTML comments, extra blank lines"
```

---

## Task 4: `chunk_document`

This is the core of the pipeline. Six test cases cover all rules from the Chunking Strategy.

**Files:**
- Modify: `ingest.py` — implement `chunk_document`
- Modify: `tests/test_ingest.py` — add 6 tests

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingest.py -k "chunk_document" -v`
Expected: 6 FAILs

- [ ] **Step 3: Implement `chunk_document`**

```python
def chunk_document(text: str, source_file: str) -> list[DocumentChunk]:
    stem = Path(source_file).stem
    lines = text.splitlines()
    chunks: list[DocumentChunk] = []
    current_header = ""
    current_lines: list[str] = []
    chunk_index = 0

    def _flush(header: str, body_lines: list[str]) -> DocumentChunk | None:
        nonlocal chunk_index
        body = "\n".join(body_lines).strip()
        full_text = f"{header}\n{body}".strip() if header else body
        if len(full_text) < MIN_CHUNK_CHARS:
            return None
        chunk = DocumentChunk(
            chunk_id=f"{stem}_{chunk_index}",
            source_file=source_file,
            header=header,
            text=full_text,
            char_count=len(full_text),
        )
        chunk_index += 1
        return chunk

    for line in lines:
        if _HEADER_RE.match(line):
            chunk = _flush(current_header, current_lines)
            if chunk:
                chunks.append(chunk)
            current_header = line
            current_lines = []
        else:
            current_lines.append(line)

    chunk = _flush(current_header, current_lines)
    if chunk:
        chunks.append(chunk)

    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingest.py -k "chunk_document" -v`
Expected: 6 PASSes

- [ ] **Step 5: Commit**

```bash
git add ingest.py tests/test_ingest.py
git commit -m "feat: implement chunk_document — header-aware split, table guard, 80-char minimum"
```

---

## Task 5: `load_all_documents` + `validate_chunks`

**Files:**
- Modify: `ingest.py` — implement both functions
- Modify: `tests/test_ingest.py` — add test

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest.py::test_load_all_documents_returns_chunks -v`
Expected: FAIL

- [ ] **Step 3: Implement `load_all_documents`**

```python
def load_all_documents(docs_dir: Path = DOCS_DIR) -> list[DocumentChunk]:
    all_chunks: list[DocumentChunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        raw = load_document(path)
        cleaned = clean_document(raw)
        chunks = chunk_document(cleaned, path.name)
        all_chunks.extend(chunks)
    return all_chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest.py::test_load_all_documents_returns_chunks -v`
Expected: PASS

- [ ] **Step 5: Implement `validate_chunks`**

No unit test — this is a diagnostic print function. Verify it manually in Step 7.

```python
def validate_chunks(chunks: list[DocumentChunk]) -> None:
    assert len(chunks) > 0, "No chunks produced — check documents/ path"
    for chunk in chunks:
        assert chunk.char_count > 0
        assert len(chunk.text) == chunk.char_count, (
            f"char_count mismatch on {chunk.chunk_id}"
        )

    print(f"\n=== Validation: {len(chunks)} total chunks across all documents ===\n")
    samples = chunks[:5]
    for i, c in enumerate(samples, 1):
        print(f"{'─' * 60}")
        print(f"[{i}] {c.chunk_id}  |  {c.char_count} chars  |  {c.source_file}")
        print(f"Header: {c.header!r}")
        print()
        print(c.text[:400])
        if c.char_count > 400:
            print(f"  ... ({c.char_count - 400} more chars)")
        print()
    print(f"{'─' * 60}")
    print("Validation passed.\n")
```

- [ ] **Step 6: Commit**

```bash
git add ingest.py tests/test_ingest.py
git commit -m "feat: implement load_all_documents and validate_chunks"
```

---

## Task 6: `load_into_chromadb`

**Files:**
- Modify: `ingest.py` — implement `load_into_chromadb`
- Modify: `tests/test_ingest.py` — add test

- [ ] **Step 1: Write the failing test**

```python
from unittest.mock import MagicMock

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest.py::test_load_into_chromadb_calls_upsert_with_correct_args -v`
Expected: FAIL

- [ ] **Step 3: Implement `load_into_chromadb`**

```python
def load_into_chromadb(chunks: list[DocumentChunk], collection: chromadb.Collection) -> None:
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {"source_file": c.source_file, "header": c.header, "char_count": c.char_count}
            for c in chunks
        ],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest.py::test_load_into_chromadb_calls_upsert_with_correct_args -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ingest.py tests/test_ingest.py
git commit -m "feat: implement load_into_chromadb — upserts chunks with metadata to ChromaDB collection"
```

---

## Task 7: `main()` Entrypoint + Full Suite

**Files:**
- Modify: `ingest.py` — implement `main()`

- [ ] **Step 1: Implement `main()`**

```python
def main() -> None:
    print("Loading and chunking documents...")
    chunks = load_all_documents()
    validate_chunks(chunks)

    print(f"Loading {len(chunks)} chunks into ChromaDB at {CHROMA_DB_PATH}...")
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )
    load_into_chromadb(chunks, collection)
    print(f"Done. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")
```

- [ ] **Step 2: Run the full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 3: Run `ingest.py` end-to-end**

Run: `python ingest.py`

Expected output (approximate):
```
Loading and chunking documents...

=== Validation: ~120 total chunks across all documents ===

────────────────────────────────────────────────────────────
[1] source1_wmu_residence_halls_0  |  ...  |  source1_wmu_residence_halls.md
...
Validation passed.

Loading ~120 chunks into ChromaDB at .../chroma_db...
Done. Collection 'wmu_housing' now has ~120 chunks.
```

- [ ] **Step 4: Verify the 5 representative chunks appear in validation output**

Compare the printed chunks against the **Representative Chunks** section at the top of this plan:
- Chunk 3 (`### Henry Hall`) should appear — bullet list, ~380 chars
- Chunk 4 (`### Valley and Henry Halls — Double Room`) should show `|` table rows intact
- Chunk 5 (`## What the Itemized Damage List Must Include`) should show the statutory blockquote

If a table chunk appears split (table rows in separate chunks), the `_HEADER_RE` regex has a bug — verify it only matches lines starting with `##` or `###`.

- [ ] **Step 5: Commit**

```bash
git add ingest.py
git commit -m "feat: implement main() — full ingest pipeline wired end to end"
```

---

## Verification

Run these three checks before marking Milestone 3 complete:

1. **All unit tests pass:** `pytest tests/ -v` — every test green
2. **ChromaDB populated:** `python ingest.py` exits cleanly and prints chunk count > 0
3. **Representative chunks match:** validation output matches the 5 examples in this plan — especially confirm the pricing table (Chunk 4) is one chunk with all `|` rows intact
