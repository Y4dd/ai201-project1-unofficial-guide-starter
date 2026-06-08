from unittest.mock import patch

from app import handle_query

_SAMPLE_CHUNKS = [
    {
        "text": "## Bus Pass\nWMU students ride all KMetro bus routes for free with their Bronco Card.",
        "source_file": "source6_winter_commuting.md",
        "header": "## Bus Pass",
        "distance": 0.2,
    },
    {
        "text": "## Henry Hall Rates\nHenry Hall double room costs $6,395.50 per semester.",
        "source_file": "source1_wmu_residence_halls.md",
        "header": "## Henry Hall Rates",
        "distance": 0.35,
    },
]


def test_handle_query_returns_answer_and_sources() -> None:
    with patch("app.retrieve", return_value=_SAMPLE_CHUNKS), patch(
        "app.generate",
        return_value="Students ride KMetro for free.\n\nSources: source6_winter_commuting.md",
    ):
        answer, sources = handle_query("Is KMetro free for students?")
    assert "KMetro" in answer
    assert "source6_winter_commuting.md" in sources
    assert "source1_wmu_residence_halls.md" in sources


def test_handle_query_empty_question_returns_prompt() -> None:
    answer, sources = handle_query("   ")
    assert "please" in answer.lower() or "enter" in answer.lower()
    assert sources == ""


def test_handle_query_empty_question_does_not_call_retrieve() -> None:
    with patch("app.retrieve") as mock_retrieve, patch("app.generate") as mock_generate:
        handle_query("")
    mock_retrieve.assert_not_called()
    mock_generate.assert_not_called()
