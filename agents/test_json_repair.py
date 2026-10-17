"""A reply that is almost JSON is mended; one that is not stays unreadable.

``malformed_response`` is what the chat endpoint answers when a model reply
cannot be read as the JSON envelope the prompt asks for, after the retries. The
repairs here fix only how an otherwise good object is written (prose or a fence
around it, a trailing comma, raw line breaks inside a string). Prose with no
object in it, and an object cut off before it closes, are left unreadable: the
first could be a refusal, the second is missing part of its answer, and
``docs/agents_api.md`` forbids publishing either as an answer.

The model is replaced throughout; nothing here calls Vertex.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agents import adk_agents


@pytest.mark.parametrize(
    "text",
    [
        'Here is the answer:\n{"answer": "2", "basis": "x", "unit": ""}\nHope that helps {really}.',
        '```json\n{"answer": "2", "basis": "x", "unit": ""}\n```',
        '{"answer": "2", "basis": "x", "unit": "",}',
        '{"answer": "line one\nline two", "basis": "x", "unit": ""}',
        '{"answer": "2", "claims": [{"id": "c1", "kind": "count", "value": 2,},], "basis": "x"}',
    ],
)
def test_structural_slips_are_repaired(text: str) -> None:
    parsed = adk_agents._parse_json(text)
    assert str(parsed["answer"]).startswith(("2", "line one"))


def test_repair_changes_how_it_is_written_not_what_it_says() -> None:
    parsed = adk_agents._parse_json('{"answer": "a\nb", "n": [1, 2,],}')
    assert parsed == {"answer": "a\nb", "n": [1, 2]}


@pytest.mark.parametrize(
    "text",
    [
        "The ring spans 2 banks.",  # prose with no object in it
        '{"answer": "The ring spans 2 ba',  # cut off before it closes
        "",
    ],
)
def test_prose_and_truncation_are_not_repaired(text: str) -> None:
    with pytest.raises((ValueError, adk_agents.EmptyModelResponse)):
        adk_agents._parse_json(text)


def test_retry_after_an_unreadable_reply_carries_the_format_reminder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    def fake_run_agent(
        name: str,
        message: str,
        cfg: Any = None,
        sink: set[str] | None = None,
        citations: list[dict[str, str]] | None = None,
    ) -> str:
        prompts.append(message)
        if len(prompts) == 1:
            return "The ring spans 2 banks."
        return json.dumps({"answer": "2", "basis": "x", "unit": "banks"})

    monkeypatch.setattr(adk_agents, "_run_agent", fake_run_agent)
    payload = adk_agents._parsed_with_retries("analyst_qa", "How many banks?", None, None)
    assert payload["answer"] == "2"
    assert prompts[0] == "How many banks?"
    assert prompts[1].startswith("How many banks?")
    assert "ONLY the JSON object" in prompts[1]


def test_an_empty_reply_is_retried_without_the_reminder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    def fake_run_agent(
        name: str,
        message: str,
        cfg: Any = None,
        sink: set[str] | None = None,
        citations: list[dict[str, str]] | None = None,
    ) -> str:
        prompts.append(message)
        return "" if len(prompts) == 1 else json.dumps({"answer": "2"})

    monkeypatch.setattr(adk_agents, "_run_agent", fake_run_agent)
    adk_agents._parsed_with_retries("analyst_qa", "How many banks?", None, None)
    assert prompts == ["How many banks?", "How many banks?"]
