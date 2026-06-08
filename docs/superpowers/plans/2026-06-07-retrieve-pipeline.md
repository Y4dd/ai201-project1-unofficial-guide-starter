# Milestone 4 — Retrieval Pipeline (`retrieve.py`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `retrieve.py` that embeds a query string using the same model as ingest and returns the top-k most semantically relevant chunks from the `wmu_housing` ChromaDB collection, then verify retrieval quality against the 5 evaluation questions in `planning.md`.

**Architecture:** `retrieve()` reuses `make_embedding_function()` from `src/load.py` — this contract guarantees the query and all stored chunks share the same embedding space. The function opens the persistent ChromaDB collection, calls `collection.query()`, and returns a flat list of dicts. An `eval_retrieve.py` script runs the 5 planning.md questions against the real database and prints retrieved chunks for manual inspection.

**Tech Stack:** Python 3.11+, `chromadb>=0.6.0`, `sentence-transformers==3.4.1`, `pytest`

---

## Context

**Project path:** `/home/amjadw/learning/codepath/project1_unofficial-guide`
**Venv:** `venv_actual/bin/python` (NOT `.venv`)
**Git:** always use `git -c commit.gpgsign=false`
**Existing M4 modules:**
- `src/load.py` — exports `make_embedding_function()`, `CHROMA_DB_PATH`, `COLLECTION_NAME`
- `src/chunk.py` — exports `DocumentChunk`
- `ingest.py` — thin entrypoint, already run; `chroma_db/` is populated with ~120 chunks

**Blocking issue:** `tests/test_ingest.py` imports symbols from the old monolithic `ingest.py` that no longer exist. All its tests have been superseded by `tests/test_chunk.py` and `tests/test_load.py`. It must be deleted before `pytest tests/` can pass. This is Step 1 of Task 1.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Delete | `tests/test_ingest.py` | Obsolete — imports removed symbols, blocks pytest |
| Create | `retrieve.py` | Query function: embed → ChromaDB.query → return top-k dicts |
| Create | `tests/test_retrieve.py` | Unit tests using `EphemeralClient` (no real DB dependency) |
| Create | `eval_retrieve.py` | Run 5 planning.md eval questions, print retrieved chunks for manual review |

---

## Task 1: `retrieve()` function + unit tests

**Files:**
- Delete: `tests/test_ingest.py`
- Create: `retrieve.py`
- Create: `tests/test_retrieve.py`

- [ ] **Step 1: Delete the obsolete test file**

```bash
rm tests/test_ingest.py
```

Verify it's gone:
```bash
ls tests/
```
Expected output: `__init__.py  __pycache__  test_chunk.py  test_load.py`

- [ ] **Step 2: Write the failing tests**

Create `tests/test_retrieve.py`:

```python
import chromadb
import pytest
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from retrieve import retrieve

_EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"


@pytest.fixture(scope="module")
def housing_collection():
    ef = SentenceTransformerEmbeddingFunction(model_name=_EMBEDDING_MODEL)
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection(name="test_housing", embedding_function=ef)
    col.upsert(
        ids=["c0", "c1", "c2", "c3", "c4", "c5"],
        documents=[
            "## Security Deposit\nLandlords may charge no more than 1.5× monthly rent as a security deposit under Michigan law.",
            "## Bus Pass\nWMU students ride all KMetro bus routes for free with their Bronco Card, including routes 3, 16, and 21.",
            "## Henry Hall Rates\nHenry Hall double room costs $6,395.50 per semester with the Bronco Gold Plus meal plan included.",
            "## Lease Cancellation Fees\nCancelling a WMU apartment contract after April 1 costs 100% of one full month's rent.",
            "## Utility Costs\nKalamazoo winter heating bills average $80–$120 per month for a typical one-bedroom apartment.",
            "## Neighborhood Safety\nThe Vine neighborhood near downtown Kalamazoo has higher crime rates than other student rental zones near WMU.",
        ],
        metadatas=[
            {"source_file": "source7_michigan_tenant_law.md", "header": "## Security Deposit", "char_count": 103},
            {"source_file": "source6_winter_commuting.md", "header": "## Bus Pass", "char_count": 102},
            {"source_file": "source1_wmu_residence_halls.md", "header": "## Henry Hall Rates", "char_count": 99},
            {"source_file": "source2_wmu_apartment_policies.md", "header": "## Lease Cancellation Fees", "char_count": 95},
            {"source_file": "source8_utility_costs.md", "header": "## Utility Costs", "char_count": 89},
            {"source_file": "source5_neighborhood_geography.md", "header": "## Neighborhood Safety", "char_count": 102},
        ],
    )
    return col


def test_retrieve_returns_k_results(housing_collection) -> None:
    results = retrieve("How much does on-campus housing cost?", k=3, collection=housing_collection)
    assert len(results) == 3


def test_retrieve_result_has_required_keys(housing_collection) -> None:
    results = retrieve("security deposit rules", k=1, collection=housing_collection)
    assert len(results) == 1
    r = results[0]
    assert "text" in r
    assert "source_file" in r
    assert "header" in r
    assert "distance" in r


def test_retrieve_distance_is_float(housing_collection) -> None:
    results = retrieve("tenant rights", k=2, collection=housing_collection)
    for r in results:
        assert isinstance(r["distance"], float)


def test_retrieve_default_k_is_5(housing_collection) -> None:
    results = retrieve("housing options at WMU", collection=housing_collection)
    assert len(results) == 5


def test_retrieve_returns_relevant_chunk_for_bus_query(housing_collection) -> None:
    results = retrieve("Is the KMetro bus free for WMU students?", k=3, collection=housing_collection)
    top_source_files = [r["source_file"] for r in results]
    assert "source6_winter_commuting.md" in top_source_files


def test_retrieve_k_exceeding_collection_size_returns_all(housing_collection) -> None:
    # Collection has 6 items; asking for more should return all 6, not raise
    results = retrieve("housing", k=20, collection=housing_collection)
    assert len(results) == 6
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
venv_actual/bin/python -m pytest tests/test_retrieve.py -v
```

Expected: `ModuleNotFoundError: No module named 'retrieve'` (retrieve.py doesn't exist yet)

- [ ] **Step 4: Implement `retrieve.py`**

Create `retrieve.py` in the project root:

```python
import chromadb
from src.load import CHROMA_DB_PATH, COLLECTION_NAME, make_embedding_function


def retrieve(
    query: str,
    k: int = 5,
    collection: chromadb.Collection | None = None,
) -> list[dict]:
    if collection is None:
        ef = make_embedding_function()
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    n = min(k, collection.count())
    if n == 0:
        return []
    results = collection.query(query_texts=[query], n_results=n)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return [
        {
            "text": doc,
            "source_file": meta["source_file"],
            "header": meta["header"],
            "distance": dist,
        }
        for doc, meta, dist in zip(docs, metas, dists)
    ]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
venv_actual/bin/python -m pytest tests/test_retrieve.py -v
```

Expected:
```
tests/test_retrieve.py::test_retrieve_returns_k_results PASSED
tests/test_retrieve.py::test_retrieve_result_has_required_keys PASSED
tests/test_retrieve.py::test_retrieve_distance_is_float PASSED
tests/test_retrieve.py::test_retrieve_default_k_is_5 PASSED
tests/test_retrieve.py::test_retrieve_returns_relevant_chunk_for_bus_query PASSED
tests/test_retrieve.py::test_retrieve_k_exceeding_collection_size_returns_all PASSED

6 passed
```

- [ ] **Step 6: Run the full test suite to confirm nothing is broken**

```bash
venv_actual/bin/python -m pytest tests/ -v
```

Expected: all tests in `test_chunk.py`, `test_load.py`, `test_retrieve.py` pass. `test_ingest.py` is gone.

- [ ] **Step 7: Commit**

```bash
git -c commit.gpgsign=false add retrieve.py tests/test_retrieve.py tests/test_ingest.py
git -c commit.gpgsign=false commit -m "Milestone 4: feat: implement retrieve() — top-k semantic search over wmu_housing collection"
```

> `git add tests/test_ingest.py` stages the deletion. Git tracks it as a removed file.

---

## Task 2: `eval_retrieve.py` — Accuracy Evaluation Script

This script runs the 5 evaluation questions from `planning.md § Evaluation Plan` against the real `chroma_db/` (which must be populated by `python ingest.py` before running). It prints the retrieved chunks so you can manually judge whether the expected answer is contained.

**Why manual review instead of automated pass/fail:** Retrieval quality can't be checked with simple string equality — a chunk about "off-campus being cheaper" won't literally contain the string "off-campus is cheaper." The eval script shows you exactly what the retrieval system found, so you can judge whether the right *information* is present, even if phrased differently.

**Files:**
- Create: `eval_retrieve.py`

- [ ] **Step 1: Verify the database is populated**

```bash
venv_actual/bin/python -c "
import chromadb
from src.load import CHROMA_DB_PATH, COLLECTION_NAME, make_embedding_function
ef = make_embedding_function()
client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
col = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
print(f'Collection has {col.count()} chunks')
"
```

Expected: `Collection has <N> chunks` where N > 0. If it prints 0 or errors, run `venv_actual/bin/python ingest.py` first.

- [ ] **Step 2: Create `eval_retrieve.py`**

```python
from retrieve import retrieve

EVAL_QUESTIONS = [
    {
        "id": 1,
        "question": "Which is cheaper, On-campus or Off-campus?",
        "expected": "Off-campus",
    },
    {
        "id": 2,
        "question": "What is the closest Off-Campus housing option to WMU?",
        "expected": "The Tate (check if this is in the documents — may not be covered)",
    },
    {
        "id": 3,
        "question": "How is Off-campus transportation for WMU students?",
        "expected": "Free with Bronco Card (KMetro buses)",
    },
    {
        "id": 4,
        "question": "Does Hunter's Ridge include utilities in the rent?",
        "expected": "Yes, except the electric bill",
    },
    {
        "id": 5,
        "question": "What is the most dangerous area to rent near WMU?",
        "expected": "Near downtown / Vine neighborhood",
    },
]

_SEP = "─" * 70


def run_eval(k: int = 5) -> None:
    print(f"\n{'=' * 70}")
    print(f"  RETRIEVAL EVALUATION — top-{k} chunks per question")
    print(f"{'=' * 70}\n")

    for q in EVAL_QUESTIONS:
        print(f"[Q{q['id']}] {q['question']}")
        print(f"  Expected answer hint: {q['expected']}")
        print()

        chunks = retrieve(q["question"], k=k)
        for i, chunk in enumerate(chunks, 1):
            dist_label = f"dist={chunk['distance']:.4f}"
            print(f"  [{i}] {chunk['source_file']} | {chunk['header']} | {dist_label}")
            preview = chunk["text"][:300].replace("\n", " ")
            if len(chunk["text"]) > 300:
                preview += f"  ... ({len(chunk['text']) - 300} more chars)"
            print(f"      {preview}")
            print()

        print(_SEP)
        print()


if __name__ == "__main__":
    run_eval()
```

- [ ] **Step 3: Run the evaluation**

```bash
venv_actual/bin/python eval_retrieve.py
```

For each question, check the printed chunks against the expected answer hint:
- **Q1** (cheaper): Look for chunks comparing on-campus rates vs off-campus rent figures
- **Q2** (closest): Look for a chunk naming the nearest complex and its distance to WMU — if no chunk mentions proximity, this question may not be answerable from the current documents
- **Q3** (transportation): Look for the KMetro / Bronco Card chunk from `source6_winter_commuting.md`
- **Q4** (Hunter's Ridge utilities): Look for a utilities-included breakdown — if Hunter's Ridge isn't in any document, note this as a coverage gap
- **Q5** (dangerous area): Look for a chunk from `source5_neighborhood_geography.md` about crime stats

Note any questions where the expected answer is NOT present in any of the top-5 chunks — these represent document coverage gaps, not retrieval failures.

- [ ] **Step 4: Commit**

```bash
git -c commit.gpgsign=false add eval_retrieve.py
git -c commit.gpgsign=false commit -m "Milestone 4: feat: add eval_retrieve.py — manual accuracy evaluation against planning.md questions"
```

---

## Verification

Run these checks before marking Milestone 4 complete:

1. **Full test suite passes:** `venv_actual/bin/python -m pytest tests/ -v` — all green, no test_ingest.py
2. **retrieve() works against real DB:** `venv_actual/bin/python -c "from retrieve import retrieve; print(retrieve('security deposit')[0]['header'])"`
3. **Eval script runs cleanly:** `venv_actual/bin/python eval_retrieve.py` — prints 5 question blocks with chunks

Milestone 4 is complete when `retrieve()` returns semantically relevant chunks for student housing questions. Milestone 5 (Generation) will call `retrieve()` to build the grounding context passed to Groq.
