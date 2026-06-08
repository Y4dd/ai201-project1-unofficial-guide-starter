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
