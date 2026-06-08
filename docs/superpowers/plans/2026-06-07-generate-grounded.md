# Milestone 5.1 — Grounded Generation (`generate.py`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/generate.py` that takes a student query and a list of retrieved chunks (from `src/retrieve.py`) and returns a grounded answer from Groq's `llama-3.3-70b-versatile` that cites sources and refuses to answer from outside the provided documents.

**Architecture:** `generate(query, chunks)` formats the chunk texts into a numbered context block, injects it into a system prompt that explicitly forbids the LLM from drawing on outside knowledge, calls the Groq chat completions API, and returns the response string. The Groq client is initialized inside the function so it can be patched in tests. A `_format_context()` helper builds the context block; a `_system_prompt()` helper builds the full system instruction. The grounding rule lives only in the system prompt — the user turn carries only the raw question.

**Tech Stack:** Python 3.11+, `groq==0.15.0`, `python-dotenv==1.0.1`, `pytest`, `unittest.mock`

---

## Context

**Project path:** `/home/amjadw/learning/codepath/project1_unofficial-guide`
**Venv:** `venv_actual/bin/python` (NOT `.venv`)
**Git:** always use `git -c commit.gpgsign=false`

**What already exists:**
- `src/retrieve.py` — `retrieve(query, k, collection) -> list[dict]` where each dict has `text`, `source_file`, `header`, `distance`
- `src/load.py` — `CHROMA_DB_PATH`, `COLLECTION_NAME`, `make_embedding_function()`
- `requirements.txt` — `groq==0.15.0` and `python-dotenv==1.0.1` already listed
- `.env.example` — template for environment variables (root of project)
- `chroma_db/` — populated with 126 chunks from the 8 source documents

**What needs to be created:**
- `.env` — copy of `.env.example` with real `GROQ_API_KEY` value (user must supply their key)
- `src/generate.py` — the generation module
- `tests/test_generate.py` — unit tests using mocked Groq client
- `tests/smoke_generate.py` — end-to-end manual grounding verification (not a pytest file)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create (user action) | `.env` | `GROQ_API_KEY=<your_key>` — never committed |
| Create | `src/generate.py` | Format context, build system prompt, call Groq, return answer string |
| Create | `tests/test_generate.py` | Unit tests with mocked Groq client — no real API calls |
| Create | `tests/smoke_generate.py` | End-to-end grounding check — runs against real DB and real Groq API |

---

## Prerequisite: Create `.env`

Before any code runs, the `GROQ_API_KEY` must be available.

- [ ] **Step 1: Create `.env` from the example template**

```bash
cp .env.example .env
```

- [ ] **Step 2: Add your Groq API key to `.env`**

Open `.env` and ensure it contains:
```
GROQ_API_KEY=your_actual_key_here
```

Get a free key at console.groq.com → API Keys. The model `llama-3.3-70b-versatile` is on the free tier.

- [ ] **Step 3: Confirm `.env` is git-ignored**

```bash
grep -n "\.env" .gitignore
```

Expected: a line matching `*.env` or `.env` (without `.env.example`). If `.env` is not ignored, add it:
```bash
echo ".env" >> .gitignore
```

---

## Task 1: Write failing tests

**Files:**
- Create: `tests/test_generate.py`

- [ ] **Step 1: Create `tests/test_generate.py`**

```python
from unittest.mock import MagicMock, patch

import pytest

from src.generate import generate, _format_context, _system_prompt

_SAMPLE_CHUNKS = [
    {
        "text": "## KMetro Bus Pass\nWMU students ride all KMetro bus routes for free with their Bronco Card.",
        "source_file": "source6_winter_commuting.md",
        "header": "## KMetro Bus Pass",
        "distance": 0.21,
    },
    {
        "text": "## Henry Hall Rates\nHenry Hall double room costs $6,395.50 per semester.",
        "source_file": "source1_wmu_residence_halls.md",
        "header": "## Henry Hall Rates",
        "distance": 0.35,
    },
]


def _make_mock_client(answer: str) -> MagicMock:
    mock = MagicMock()
    mock.chat.completions.create.return_value.choices[0].message.content = answer
    return mock


def test_generate_returns_string() -> None:
    with patch("src.generate.Groq", return_value=_make_mock_client("The bus is free.")):
        result = generate("Is the KMetro bus free?", _SAMPLE_CHUNKS)
    assert isinstance(result, str)
    assert result == "The bus is free."


def test_generate_passes_query_as_user_message() -> None:
    mock_client = _make_mock_client("answer")
    with patch("src.generate.Groq", return_value=mock_client):
        generate("my test query", _SAMPLE_CHUNKS)
    call_kwargs = mock_client.chat.completions.create.call_args
    messages = call_kwargs.kwargs["messages"]
    user_turn = next(m for m in messages if m["role"] == "user")
    assert "my test query" in user_turn["content"]


def test_generate_system_prompt_contains_grounding_instruction() -> None:
    mock_client = _make_mock_client("answer")
    with patch("src.generate.Groq", return_value=mock_client):
        generate("test query", _SAMPLE_CHUNKS)
    call_kwargs = mock_client.chat.completions.create.call_args
    messages = call_kwargs.kwargs["messages"]
    system_turn = next(m for m in messages if m["role"] == "system")
    content = system_turn["content"].lower()
    assert "only" in content
    assert "context" in content


def test_generate_context_includes_chunk_text() -> None:
    mock_client = _make_mock_client("answer")
    with patch("src.generate.Groq", return_value=mock_client):
        generate("test query", _SAMPLE_CHUNKS)
    call_kwargs = mock_client.chat.completions.create.call_args
    messages = call_kwargs.kwargs["messages"]
    system_turn = next(m for m in messages if m["role"] == "system")
    assert "Bronco Card" in system_turn["content"]
    assert "Henry Hall" in system_turn["content"]


def test_generate_context_includes_source_filenames() -> None:
    mock_client = _make_mock_client("answer")
    with patch("src.generate.Groq", return_value=mock_client):
        generate("test query", _SAMPLE_CHUNKS)
    call_kwargs = mock_client.chat.completions.create.call_args
    messages = call_kwargs.kwargs["messages"]
    system_turn = next(m for m in messages if m["role"] == "system")
    assert "source6_winter_commuting.md" in system_turn["content"]
    assert "source1_wmu_residence_halls.md" in system_turn["content"]


def test_generate_empty_chunks_returns_no_information_message() -> None:
    mock_client = _make_mock_client("any answer")
    with patch("src.generate.Groq", return_value=mock_client):
        result = generate("test query", [])
    assert "don't have enough information" in result.lower()
    mock_client.chat.completions.create.assert_not_called()


def test_format_context_numbers_each_chunk() -> None:
    context = _format_context(_SAMPLE_CHUNKS)
    assert "[1]" in context
    assert "[2]" in context


def test_format_context_includes_source_and_text() -> None:
    context = _format_context(_SAMPLE_CHUNKS)
    assert "source6_winter_commuting.md" in context
    assert "Bronco Card" in context


def test_system_prompt_contains_citation_instruction() -> None:
    prompt = _system_prompt("some context text")
    lower = prompt.lower()
    assert "source" in lower or "cite" in lower or "citation" in lower


def test_system_prompt_contains_fallback_instruction() -> None:
    prompt = _system_prompt("some context text")
    assert "don't have enough information" in prompt.lower() or "not enough information" in prompt.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
venv_actual/bin/python -m pytest tests/test_generate.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.generate'` — `generate.py` doesn't exist yet.

---

## Task 2: Implement `src/generate.py`

**Files:**
- Create: `src/generate.py`

- [ ] **Step 1: Create `src/generate.py`**

```python
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] {chunk['source_file']}\n{chunk['text']}")
    return "\n\n".join(parts)


def _system_prompt(context: str) -> str:
    return f"""You are a housing advisor for Western Michigan University students.

Answer the student's question using ONLY the information provided in the context below.
Do not use any knowledge from outside these documents.
If the context does not contain enough information to answer the question, respond with exactly:
"I don't have enough information in my documents to answer that question."

At the end of every answer, add a "Sources:" line listing the source filenames you used
(e.g., "Sources: source1_wmu_residence_halls.md, source6_winter_commuting.md").

CONTEXT:
{context}"""


def generate(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I don't have enough information in my documents to answer that question."

    context = _format_context(chunks)
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _system_prompt(context)},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
venv_actual/bin/python -m pytest tests/test_generate.py -v
```

Expected:
```
tests/test_generate.py::test_generate_returns_string PASSED
tests/test_generate.py::test_generate_passes_query_as_user_message PASSED
tests/test_generate.py::test_generate_system_prompt_contains_grounding_instruction PASSED
tests/test_generate.py::test_generate_context_includes_chunk_text PASSED
tests/test_generate.py::test_generate_context_includes_source_filenames PASSED
tests/test_generate.py::test_generate_empty_chunks_returns_no_information_message PASSED
tests/test_generate.py::test_format_context_numbers_each_chunk PASSED
tests/test_generate.py::test_format_context_includes_source_and_text PASSED
tests/test_generate.py::test_system_prompt_contains_citation_instruction PASSED
tests/test_generate.py::test_system_prompt_contains_fallback_instruction PASSED

10 passed
```

- [ ] **Step 3: Run the full test suite to confirm nothing is broken**

```bash
venv_actual/bin/python -m pytest tests/ -v
```

Expected: all 26 existing tests (test_chunk.py, test_load.py, test_retrieve.py) plus the 10 new ones — 36 total, all passing. `eval_retrieve.py` and `smoke_generate.py` are not pytest files so they are skipped automatically.

- [ ] **Step 4: Commit**

```bash
git -c commit.gpgsign=false add src/generate.py tests/test_generate.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: implement generate() — grounded Groq generation with source citation

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: End-to-end grounding smoke test

This is a manual verification script that runs against the real ChromaDB and real Groq API. It is **not** a pytest file — run it directly with `python -m`.

**Files:**
- Create: `tests/smoke_generate.py`

- [ ] **Step 1: Create `tests/smoke_generate.py`**

```python
"""
End-to-end grounding smoke test.
Run with: venv_actual/bin/python -m tests.smoke_generate

For each query, ask yourself:
  Could this response have come from anywhere other than the retrieved chunks printed above it?
  If yes — the grounding prompt needs tightening.
  If no  — grounding is working.
"""

from src.retrieve import retrieve
from src.generate import generate

SMOKE_QUERIES = [
    "What do students say about noise and security at The Wyatt?",
    "Which KMetro bus routes stop at WMU, and is it free for students?",
    "If my landlord keeps my security deposit after 30 days, what are my rights under Michigan law?",
]

_SEP = "=" * 70


def run_smoke() -> None:
    print(f"\n{_SEP}")
    print("  GROUNDED GENERATION SMOKE TEST")
    print(f"{_SEP}\n")

    for i, query in enumerate(SMOKE_QUERIES, 1):
        print(f"[Q{i}] {query}\n")

        chunks = retrieve(query, k=5)
        print("  Retrieved context:")
        for j, chunk in enumerate(chunks, 1):
            print(f"    [{j}] {chunk['source_file']} | {chunk['header']} | dist={chunk['distance']:.4f}")
        print()

        answer = generate(query, chunks)
        print("  Generated answer:")
        for line in answer.splitlines():
            print(f"    {line}")
        print()
        print(f"  {'─' * 66}")
        print()


if __name__ == "__main__":
    run_smoke()
```

- [ ] **Step 2: Run the smoke test**

```bash
venv_actual/bin/python -m tests.smoke_generate
```

For each query, check two things:
1. **Grounding check:** The answer should only contain facts visible in the "Retrieved context" lines above it. If the answer discusses something not in those chunks, the system prompt needs a stronger grounding instruction.
2. **Citation check:** The answer should end with a "Sources:" line naming at least one `.md` file.

Example of a **grounded response** (Q2):
```
WMU students can ride all KMetro bus routes for free using their Bronco Card,
including routes 3, 16, and 21. No payment is required at the farebox.

Sources: source6_winter_commuting.md
```

Example of a **non-grounded response** (fail — the model drew on training data):
```
WMU is served by the Kalamazoo Metro Transit (KMetro) system. Like many
university transit partnerships, students typically receive a discount or
free access through a student fee or ID card program.

Sources: source6_winter_commuting.md
```
The second example uses hedging language ("typically", "like many universities") — a signal the model is reasoning from training knowledge rather than the retrieved text.

- [ ] **Step 3: Commit**

```bash
git -c commit.gpgsign=false add tests/smoke_generate.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: add smoke_generate.py — end-to-end grounding verification script

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Verification

Run these checks before marking Milestone 5.1 complete:

1. **All unit tests pass:** `venv_actual/bin/python -m pytest tests/ -v` — 36 tests, all green
2. **Smoke test runs cleanly:** `venv_actual/bin/python -m tests.smoke_generate` — 3 answers printed, no errors
3. **Grounding check passes manually:** For each smoke test answer, every factual claim traces back to one of the printed retrieved chunks
4. **Citation check:** Every answer ends with a `Sources:` line

Milestone 5.1 is complete when `generate()` produces answers that are traceable to retrieved chunks and cite their sources. Milestone 5.2 will build the Gradio/Streamlit interface that wires `retrieve()` → `generate()` into a student-facing chat UI.
