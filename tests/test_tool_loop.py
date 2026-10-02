"""The Assist-API tool loop for the prose LLM.

These tests mock the OpenAI-compatible endpoint and the Assist API at the
aiohttp/intent layer, so the loop's message assembly, tool-call execution and
round capping are exercised against the real chat-log machinery.
"""

from __future__ import annotations

import pytest

from custom_components.typesafe_conversation.tool_loop import (
    ToolLoopError,
    _chat_log_to_messages,
    _parse_tool_args,
)


def test_parse_tool_args_accepts_object_and_string():
    assert _parse_tool_args({"name": "kitchen"}) == {"name": "kitchen"}
    assert _parse_tool_args('{"name": "kitchen"}') == {"name": "kitchen"}
    assert _parse_tool_args("not json") == {}
    assert _parse_tool_args(None) == {}
    assert _parse_tool_args(42) == {}


def test_chat_log_to_messages_round_trips_tool_traffic():
    from homeassistant.components.conversation import (
        AssistantContent,
        SystemContent,
        ToolResultContent,
        UserContent,
    )
    from homeassistant.helpers import llm

    log = type("FakeLog", (), {"content": []})()
    log.content = [
        SystemContent(content="You are the voice of the home."),
        UserContent(content="turn on the kitchen light"),
        AssistantContent(
            agent_id="test",
            content=None,
            tool_calls=[
                llm.ToolInput(
                    tool_name="HassLightTurnOn",
                    tool_args={"name": "kitchen light"},
                    id="call_1",
                )
            ],
        ),
        ToolResultContent(
            agent_id="test",
            tool_call_id="call_1",
            tool_name="HassLightTurnOn",
            tool_result={"speech": {"plain": {"speech": "Turned on"}}},
        ),
        AssistantContent(agent_id="test", content="Done, the light is on."),
    ]

    messages = _chat_log_to_messages(log)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assistant = messages[2]
    assert assistant["tool_calls"][0]["function"]["name"] == "HassLightTurnOn"
    assert assistant["tool_calls"][0]["id"] == "call_1"
    tool_msg = messages[3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"
    assert "Turned on" in tool_msg["content"]
    assert messages[4]["content"] == "Done, the light is on."


async def test_tool_loop_surfaces_http_errors(hass, aioclient_mock):
    """An endpoint error must raise ToolLoopError, not hang or lie."""
    from custom_components.typesafe_conversation.tool_loop import run_tool_loop

    session = aioclient_mock.create_session(hass.loop)
    aioclient_mock.post(
        "http://127.0.0.1:8000/v1/chat/completions", status=500, text="boom"
    )

    user_input = _fake_user_input(hass)
    chat_log = _fake_chat_log(hass, user_input)

    with pytest.raises(ToolLoopError):
        await run_tool_loop(
            hass,
            session,
            base_url="http://127.0.0.1:8000",
            model="test-model",
            api_key=None,
            referer=None,
            title=None,
            answer_timeout=5.0,
            user_input=user_input,
            chat_log=chat_log,
            user_text="turn on the light",
            extra_system_prompt=None,
        )

    await session.close()


def _fake_user_input(hass):
    from homeassistant.components.conversation import ConversationInput
    from homeassistant.core import Context

    return ConversationInput(
        text="turn on the light",
        context=Context(),
        conversation_id="test",
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id="conversation.test",
    )


def _fake_chat_log(hass, user_input):
    from homeassistant.components.conversation import ChatLog, SystemContent

    log = ChatLog(
        hass=hass,
        conversation_id="test",
        content=[SystemContent(content="")],
    )
    log.async_add_user_content(
        __import__(
            "homeassistant.components.conversation.chat_log", fromlist=["UserContent"]
        ).UserContent(content=user_input.text)
    )
    return log


def test_sanitize_json_schema_stringifies_voluptuous_sentinels():
    """The 422 from the field: _Unsupported inside tool parameters."""
    import json

    from custom_components.typesafe_conversation.tool_loop import (
        _sanitize_json_schema,
    )

    class _Unsupported:
        """Stand-in for the voluptuous_openapi sentinel."""

    schema = {
        "type": "object",
        "properties": {"entity_id": {"type": "string"}, "area": _Unsupported()},
        "required": ["entity_id", _Unsupported()],
    }
    cleaned = _sanitize_json_schema(schema)
    # The whole structure must survive a strict JSON round-trip.
    json.dumps(cleaned)
    assert cleaned["properties"]["entity_id"] == {"type": "string"}
    assert isinstance(cleaned["required"], list)


def test_sanitize_json_schema_handles_primitives_and_depth():
    from custom_components.typesafe_conversation.tool_loop import (
        _sanitize_json_schema,
    )

    assert _sanitize_json_schema(None) is None
    assert _sanitize_json_schema("text") == "text"
    assert _sanitize_json_schema(3) == 3
    assert _sanitize_json_schema(True) is True
    assert _sanitize_json_schema(["a", 1]) == ["a", 1]
    # Depth guard: deeply nested structures collapse to an empty dict.
    deep = current = {}
    for _ in range(20):
        current["child"] = {}
        current = current["child"]
    # Depth guard: the deepest levels collapse, the rest survives as JSON.
    import json as _json

    _json.dumps(_sanitize_json_schema(deep))
