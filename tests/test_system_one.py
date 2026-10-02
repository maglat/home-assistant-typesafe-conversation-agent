"""The decision clients: request shape, error mapping, retries, circuit breaker."""

from __future__ import annotations

import asyncio
import json

import pytest
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
    mock_aiohttp_client,
)

from custom_components.typesafe_conversation.const import (
    CONF_DECISION_API_KEY,
    CONF_DECISION_BACKEND,
    CONF_DECISION_BASE_URL,
    CONF_MODEL,
    DECISION_OPENAI,
    TYPESAFE_API_URL,
)
from custom_components.typesafe_conversation.system_one import (
    ChoiceAnswer,
    DecisionAuthError,
    DecisionRequestError,
    DecisionUnavailableError,
    OpenAIDecisionClient,
    TypeSafeDecisionClient,
    create_decision_client,
)

OK = {
    "model": "jev-1.13.0",
    "answers": {
        "category": {
            "type": "choice",
            "choice": "command",
            "probabilities": {"command": 0.9, "query": 0.08, "information": 0.02},
            "confidence": 0.85,
        },
        "compound": {"type": "noul", "noul": 0.04},
        "frustration": {
            "type": "score",
            "score": 1.05,
            "legend": {"0": "Calm", "1": "Cross"},
            "probabilities": {"0": 0.2, "1": 0.8},
            "confidence": 0.7,
        },
    },
    "usage": {"input_tokens": 6482, "output_tokens": 210},
}

QUESTIONS = {
    "category": {
        "type": "choice",
        "criteria": {"command": "run a device", "query": "ask about state"},
    },
    "compound": {"type": "noul"},
}


@pytest.fixture(name="mocker")
def mocker_fixture():
    with mock_aiohttp_client() as mocker:
        yield mocker


@pytest.fixture(name="client")
async def client_fixture(mocker: AiohttpClientMocker):
    session = mocker.create_session(asyncio.get_running_loop())
    return TypeSafeDecisionClient(session, "sk-test", "jev-latest")


async def test_request_shape_and_typed_answers(client, mocker):
    mocker.post(TYPESAFE_API_URL, json=OK)
    response = await client.async_ask({"request": {"text": "hi"}}, {"category": {}})

    _method, _url, body, headers = mocker.mock_calls[0]
    assert body["model"] == "jev-latest"
    assert body["state"] == {"request": {"text": "hi"}}
    assert headers["Authorization"] == "Bearer sk-test"

    assert response.model == "jev-1.13.0"
    assert response.input_tokens == 6482
    assert response.choice("category").choice == "command"
    assert response.noul("compound") == 0.04
    assert response.score("frustration").score == 1.05
    # Wrong-typed access returns None rather than raising.
    assert response.choice("compound") is None


def test_margin_catches_a_confident_looking_tie():
    """Confidence and margin fail differently, which is why we check both."""
    tied = ChoiceAnswer("a", {"a": 0.45, "b": 0.44, "c": 0.11}, 0.62)
    assert tied.confidence > 0.6
    assert tied.margin < 0.05, "top two are effectively tied"


async def test_auth_failure_is_not_retried(client, mocker):
    mocker.post(TYPESAFE_API_URL, status=401, text="nope")
    with pytest.raises(DecisionAuthError):
        await client.async_ask({}, {})
    assert len(mocker.mock_calls) == 1


async def test_validation_failure_is_not_retried(client, mocker):
    """A 422 is our bug, not a transient one - retrying just wastes time."""
    mocker.post(TYPESAFE_API_URL, status=422, text='{"detail":"questions.x.criteria"}')
    with pytest.raises(DecisionRequestError, match="criteria"):
        await client.async_ask({}, {})
    assert len(mocker.mock_calls) == 1


async def test_rate_limit_is_retried_then_gives_up(client, mocker):
    mocker.post(TYPESAFE_API_URL, status=429, text="slow down")
    with pytest.raises(DecisionUnavailableError):
        await client.async_ask({}, {})
    assert len(mocker.mock_calls) == 3, "three attempts, then the fallback ladder"


async def test_circuit_opens_after_repeated_failure(client, mocker):
    """Three failed requests, then stop trying for a while.

    A dead API must not add six seconds of timeout to every utterance.
    """
    mocker.post(TYPESAFE_API_URL, status=500, text="boom")
    for _ in range(3):
        with pytest.raises(DecisionUnavailableError):
            await client.async_ask({}, {})
    assert client.circuit_open

    before = len(mocker.mock_calls)
    with pytest.raises(DecisionUnavailableError, match="circuit breaker"):
        await client.async_ask({}, {})
    assert len(mocker.mock_calls) == before, "no request while the circuit is open"


async def test_success_resets_the_failure_count(client, mocker):
    mocker.post(TYPESAFE_API_URL, status=500, text="boom")
    with pytest.raises(DecisionUnavailableError):
        await client.async_ask({}, {})
    mocker.clear_requests()
    mocker.post(TYPESAFE_API_URL, json=OK)
    await client.async_ask({}, {})
    assert not client.circuit_open


# --- the OpenAI-compatible decision backend -----------------------------------


def _chat_body(content: str) -> dict:
    return {
        "model": "clef-flash",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30},
    }


GOOD_REPLY = json.dumps(
    {
        "category": {
            "choice": "command",
            "probabilities": {"command": 0.9, "query": 0.1},
        },
        "compound": {"noul": 0.05},
    }
)


@pytest.fixture(name="openai_client")
async def openai_client_fixture(mocker: AiohttpClientMocker):
    session = mocker.create_session(asyncio.get_running_loop())
    return OpenAIDecisionClient(
        session, "http://127.0.0.1:8000", "clef-flash", timeout=5.0
    )


async def test_openai_backend_parses_a_strict_reply(openai_client, mocker):
    mocker.post(
        "http://127.0.0.1:8000/v1/chat/completions", json=_chat_body(GOOD_REPLY)
    )
    response = await openai_client.async_ask({"request": {"text": "hi"}}, QUESTIONS)

    _method, url, body, _headers = mocker.mock_calls[0]
    assert str(url).endswith("/v1/chat/completions")
    assert body["model"] == "clef-flash"
    assert "category" in body["messages"][-1]["content"]

    assert response.choice("category").choice == "command"
    assert response.choice("category").confidence == pytest.approx(0.8, abs=0.01)
    assert response.noul("compound") == 0.05


async def test_openai_backend_tolerates_fenced_prose(openai_client, mocker):
    fenced = "Here you go:\n```json\n" + GOOD_REPLY + "\n```"
    mocker.post("http://127.0.0.1:8000/v1/chat/completions", json=_chat_body(fenced))
    response = await openai_client.async_ask({}, QUESTIONS)
    assert response.choice("category").choice == "command"


async def test_openai_backend_accepts_bare_labels(openai_client, mocker):
    """A weaker model that answers {"category": "command"} still routes."""
    bare = json.dumps({"category": "command", "compound": 0.1})
    mocker.post("http://127.0.0.1:8000/v1/chat/completions", json=_chat_body(bare))
    response = await openai_client.async_ask({}, QUESTIONS)
    assert response.choice("category").choice == "command"
    assert response.noul("compound") == 0.1


async def test_openai_backend_maps_an_unknown_label_to_flat(openai_client, mocker):
    """An option the question never offered must not look confident."""
    bad = json.dumps({"category": "sing_a_song", "compound": 0.1})
    mocker.post("http://127.0.0.1:8000/v1/chat/completions", json=_chat_body(bad))
    response = await openai_client.async_ask({}, QUESTIONS)
    answer = response.choice("category")
    assert answer.choice == "command"  # first option, flat distribution
    assert answer.confidence == 0.0, "flat distribution must not look solid"


async def test_openai_backend_retries_on_5xx(openai_client, mocker):
    mocker.post("http://127.0.0.1:8000/v1/chat/completions", status=500, text="boom")
    with pytest.raises(DecisionUnavailableError):
        await openai_client.async_ask({}, QUESTIONS)
    assert len(mocker.mock_calls) == 3


async def test_openai_backend_garbage_is_a_request_error(openai_client, mocker):
    mocker.post(
        "http://127.0.0.1:8000/v1/chat/completions",
        json=_chat_body("I cannot answer that in JSON, sorry."),
    )
    with pytest.raises(DecisionRequestError):
        await openai_client.async_ask({}, QUESTIONS)


async def test_openai_validate_lists_models(openai_client, mocker):
    mocker.get(
        "http://127.0.0.1:8000/v1/models",
        json={"data": [{"id": "clef-flash"}, {"id": "clef"}]},
    )
    assert await openai_client.async_validate() == ["clef-flash", "clef"]


async def test_create_decision_client_openai_backend():
    client = create_decision_client(
        session=None,
        settings={
            CONF_DECISION_BACKEND: DECISION_OPENAI,
            CONF_DECISION_BASE_URL: "http://127.0.0.1:8000",
            CONF_MODEL: "clef-flash",
            CONF_DECISION_API_KEY: "sk-local",
        },
    )
    assert isinstance(client, OpenAIDecisionClient)
    assert client.model == "clef-flash"


async def test_create_decision_client_defaults_to_typesafe():
    client = create_decision_client(
        session=None, settings={"api_key": "sk-test", CONF_MODEL: "jev-latest"}
    )
    assert isinstance(client, TypeSafeDecisionClient)


# --- the System One backend against a self-hosted endpoint ---------------------


async def test_systemone_backend_accepts_a_custom_base_url(mocker):
    """Kev on the LAN: same wire protocol, different host, no API key."""
    session = mocker.create_session(asyncio.get_running_loop())
    client = TypeSafeDecisionClient(
        session, None, "kev-latest", base_url="http://192.168.178.7:8010"
    )
    mocker.post(
        "http://192.168.178.7:8010/v1/systemone",
        json={
            "model": "kev-latest",
            "answers": {
                "category": {
                    "type": "choice",
                    "choice": "command",
                    "probabilities": {"command": 0.9, "query": 0.1},
                    "confidence": 0.8,
                }
            },
            "usage": {"input_tokens": 100, "output_tokens": 10},
        },
    )
    response = await client.async_ask({}, QUESTIONS)
    assert response.choice("category").choice == "command"

    _method, url, body, headers = mocker.mock_calls[0]
    assert str(url) == "http://192.168.178.7:8010/v1/systemone"
    assert body["model"] == "kev-latest"
    assert "Authorization" not in headers, "no key, no header"


async def test_systemone_backend_strips_a_trailing_systemone_path(mocker):
    """Users paste the full endpoint URL; both shapes must work."""
    session = mocker.create_session(asyncio.get_running_loop())
    client = TypeSafeDecisionClient(
        session,
        None,
        "kev-latest",
        base_url="http://192.168.178.7:8010/v1/systemone",
    )
    mocker.get(
        "http://192.168.178.7:8010/v1/models",
        json={"models": [{"name": "kev-latest"}]},
    )
    assert await client.async_validate() == ["kev-latest"]


async def test_create_decision_client_systemone_with_base_url():
    """The factory wires a self-hosted System One endpoint from settings."""
    client = create_decision_client(
        session=None,
        settings={
            CONF_DECISION_BACKEND: "typesafe",
            CONF_DECISION_BASE_URL: "http://192.168.178.7:8010",
            CONF_MODEL: "kev-latest",
        },
    )
    assert isinstance(client, TypeSafeDecisionClient)
    assert client._base_url == "http://192.168.178.7:8010"
    assert client._api_key is None


async def test_create_decision_client_typesafe_without_key_still_builds():
    """A self-hosted endpoint needs no key; None must not veto the client."""
    client = create_decision_client(
        session=None,
        settings={CONF_DECISION_BACKEND: "typesafe", CONF_MODEL: "kev-latest"},
    )
    assert isinstance(client, TypeSafeDecisionClient)


async def test_timeout_is_not_retried(openai_client, mocker):
    """A slow model must fail fast: one timeout, then the fallback ladder."""

    import pytest

    from custom_components.typesafe_conversation.system_one import (
        DecisionUnavailableError,
        _TimeoutError,
    )

    calls = 0

    async def _slow(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise _TimeoutError("Timed out talking to the decision model")

    openai_client._ask_once = _slow
    with pytest.raises(DecisionUnavailableError):
        await openai_client.async_ask({}, {})
    assert calls == 1, f"expected 1 call, got {calls}"


async def test_openai_validate_rejects_a_systemone_endpoint(mocker):
    """Kev answers /v1/models with a System One payload; validation must
    reject the config instead of letting every utterance 404 later."""
    import pytest

    from custom_components.typesafe_conversation.system_one import (
        DecisionRequestError,
        OpenAIDecisionClient,
    )

    mocker.get(
        "http://127.0.0.1:8000/v1/models",
        json={"models": [{"name": "kev-latest", "device": "cuda"}]},
    )
    client = OpenAIDecisionClient(
        mocker.create_session(asyncio.get_running_loop()),
        "http://127.0.0.1:8000",
        "kev-latest",
        timeout=5.0,
    )
    with pytest.raises(DecisionRequestError):
        await client.async_validate()
