"""The context-resolution pass and the system prompt plumbing.

The decision model is a single-pass classifier: a follow-up like "and back off
again" only means something with the previous turns in view, and scores as
unclear. These tests pin the rewrite-then-rerun behaviour and the guarantees
around it - a rewrite that changes nothing must not loop, and a rewrite that
still cannot be routed must fall through to the prose answer.
"""

from __future__ import annotations

import asyncio

import pytest
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
    mock_aiohttp_client,
)

from custom_components.typesafe_conversation.llm_backend import (
    OllamaBackend,
    OpenAICompatBackend,
)

HISTORY = [("turn on the kitchen light", "Done.")]


@pytest.fixture(name="mocker")
def mocker_fixture():
    with mock_aiohttp_client() as mocker:
        yield mocker


@pytest.fixture(name="ollama")
async def ollama_fixture(mocker: AiohttpClientMocker):
    session = mocker.create_session(asyncio.get_running_loop())
    return OllamaBackend(session, "http://127.0.0.1:11434", "llama3.2")


def _chat(content: str) -> dict:
    return {"message": {"role": "assistant", "content": content}}


async def test_rewrite_returns_the_standalone_command(ollama, mocker):
    mocker.post(
        "http://127.0.0.1:11434/api/chat",
        json=_chat("turn off the kitchen light"),
    )
    rewritten = await ollama.rewrite_with_context(
        "and back off again", HISTORY, speaker_area="Kitchen"
    )
    assert rewritten == "turn off the kitchen light"

    _method, _url, body, _headers = mocker.mock_calls[0]
    user_content = body["messages"][-1]["content"]
    assert 'user: "turn on the kitchen light"' in user_content
    assert 'newest utterance: "and back off again"' in user_content


async def test_rewrite_passes_none_through_untouched(ollama, mocker):
    """The agent treats NONE as 'nothing to resolve' and falls through."""
    mocker.post("http://127.0.0.1:11434/api/chat", json=_chat("NONE"))
    rewritten = await ollama.rewrite_with_context("what is your name", HISTORY)
    assert rewritten == "NONE"


async def test_openai_backend_rewrites_too(mocker):
    session = mocker.create_session(asyncio.get_running_loop())
    backend = OpenAICompatBackend(session, "http://127.0.0.1:8000", "qwen3.5-9b", None)
    mocker.post(
        "http://127.0.0.1:8000/v1/chat/completions",
        json={
            "choices": [{"message": {"role": "assistant", "content": " turn it off "}}]
        },
    )
    rewritten = await backend.rewrite_with_context("und wieder aus", HISTORY)
    assert rewritten == "turn it off"


async def test_system_prompt_is_appended_after_the_guardrails(ollama, mocker):
    mocker.post(
        "http://127.0.0.1:11434/api/chat",
        json=_chat("Sure, the light is on."),
    )
    await ollama.answer_freeform(
        "what can you do",
        HISTORY,
        home_state="nothing of note",
        local_time="12:00",
        weekday="Monday",
        speaker_area="Kitchen",
        system_prompt="Answer in German. Be witty.",
    )
    _method, _url, body, _headers = mocker.mock_calls[0]
    system = body["messages"][0]["content"]
    assert "Additional instructions:" in system
    assert system.index("You cannot control any device") < system.index(
        "Additional instructions:"
    ), "user instructions must come after the built-in guardrails"
    assert "Answer in German. Be witty." in system


async def test_system_prompt_absent_leaves_prompt_untouched(ollama, mocker):
    mocker.post(
        "http://127.0.0.1:11434/api/chat",
        json=_chat("Sure, the light is on."),
    )
    await ollama.answer_freeform(
        "what can you do",
        HISTORY,
        home_state="nothing of note",
        local_time="12:00",
        weekday="Monday",
        speaker_area="Kitchen",
    )
    _method, _url, body, _headers = mocker.mock_calls[0]
    assert "Additional instructions:" not in body["messages"][0]["content"]


def test_context_prompt_demands_a_standalone_command():
    from custom_components.typesafe_conversation.llm_backend import (
        CONTEXT_SYSTEM_PROMPT,
    )

    assert "NONE" in CONTEXT_SYSTEM_PROMPT
    assert "standalone" in CONTEXT_SYSTEM_PROMPT
    assert "Keep the user's" in CONTEXT_SYSTEM_PROMPT
