"""Let the prose LLM control exposed devices through Home Assistant's Assist API.

The decision model handles commands it is confident about; everything that
falls through to the prose answer has so far been read-only. With the Assist
API wired in, the LLM can act as well: Home Assistant hands over one tool per
intent - HassLightTurnOn, HassClimateSetTemperature, calendar and todo reads -
restricted to the entities exposed to Assist, and the LLM chains tool calls
until it can answer.

The loop follows the same pattern as Home Assistant's own Ollama and OpenAI
agents: append the model's tool calls to the chat log, let the chat log
execute them through the Assist API, feed the results back, repeat. A cap on
rounds bounds a confused model; the fallback ladder bounds a dead one.
"""

from __future__ import annotations

import json
from typing import Any

import aiohttp
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from .const import LOGGER, MAX_TOOL_ROUNDS

try:
    from voluptuous_openapi import convert
except ImportError:  # pragma: no cover - depends on the HA build
    convert = None  # type: ignore[assignment]

TOOL_SYSTEM_SUFFIX = """\
You can control the smart home through the provided tools. The devices you may
act on are listed in the context above; do not invent other devices. Call a
tool when the request asks for an action or a state you can read with one.
After the tools have done their work, answer briefly in spoken text."""


class ToolLoopError(Exception):
    """The tool loop could not be completed."""


def _tool_to_openai(tool: llm.Tool, custom_serializer: Any) -> dict[str, Any]:
    """Format one Assist tool as an OpenAI function tool."""
    if convert is None:
        raise ToolLoopError(
            "voluptuous_openapi is not available in this Home Assistant build"
        )
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
        },
    }


async def run_tool_loop(
    hass: HomeAssistant,
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    referer: str | None,
    title: str | None,
    answer_timeout: float,
    user_input: conversation.ConversationInput,
    chat_log: conversation.ChatLog,
    user_text: str,
    extra_system_prompt: str | None,
) -> str:
    """Give the LLM the Assist tools and let it answer with them.

    Returns the assistant's final spoken text. Raises ToolLoopError when the
    Assist API is not available or the model never produced a usable answer;
    the caller then falls back to the plain prose path.
    """
    # One instance per request: the prompt embeds current states, and the
    # context carries the user so intent calls land on their authority.
    try:
        api_instance = await llm.async_get_api(
            hass,
            llm.LLM_API_ASSIST,
            user_input.as_llm_context("typesafe_conversation"),
        )
    except HomeAssistantError as err:
        raise ToolLoopError(str(err)) from err

    await chat_log.async_provide_llm_data(
        user_input.as_llm_context("typesafe_conversation"),
        llm.LLM_API_ASSIST,
        None,
        extra_system_prompt,
    )

    tools = [
        _tool_to_openai(tool, api_instance.custom_serializer)
        for tool in api_instance.tools
    ]
    if not tools:
        raise ToolLoopError("the Assist API exposed no tools")

    for _round in range(MAX_TOOL_ROUNDS):
        messages = _chat_log_to_messages(chat_log)
        body = await _chat_completion(
            session,
            base_url,
            model,
            api_key,
            referer,
            title,
            messages,
            tools,
            answer_timeout,
        )
        message = body.get("choices", [{}])[0].get("message", {})
        tool_calls = message.get("tool_calls") or []
        content = message.get("content") or ""

        if not tool_calls:
            if not content.strip():
                raise ToolLoopError("the model returned neither text nor tool calls")
            LOGGER.debug(
                "Tool loop finished after text answer (%s tools available)",
                len(tools),
            )
            return content.strip()

        # Record the assistant turn with its calls, then execute each one
        # through the Assist API and append the results.
        assistant_content = conversation.AssistantContent(
            agent_id="typesafe_conversation",
            content=content or None,
            tool_calls=[
                llm.ToolInput(
                    tool_name=call.get("function", {}).get("name", ""),
                    tool_args=_parse_tool_args(
                        call.get("function", {}).get("arguments")
                    ),
                    id=call.get("id") or f"call_{idx}",
                )
                for idx, call in enumerate(tool_calls)
            ],
        )
        tool_results = []
        async for result in chat_log.async_add_assistant_content(assistant_content):
            tool_results.append(result)

        if not tool_results:
            raise ToolLoopError("the tool calls produced no results")
        LOGGER.debug(
            "Tool round %s: %s call(s) -> %s result(s)",
            _round + 1,
            len(tool_calls),
            len(tool_results),
        )

    raise ToolLoopError(
        f"the model kept calling tools for {MAX_TOOL_ROUNDS} rounds without answering"
    )


def _parse_tool_args(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string or an object; accept both."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _chat_log_to_messages(chat_log: conversation.ChatLog) -> list[dict[str, Any]]:
    """Render the chat log into OpenAI chat messages, tools included."""
    messages: list[dict[str, Any]] = []
    for content in chat_log.content:
        if isinstance(content, conversation.SystemContent):
            messages.append({"role": "system", "content": content.content})
        elif isinstance(content, conversation.UserContent):
            messages.append({"role": "user", "content": content.content})
        elif isinstance(content, conversation.AssistantContent):
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": content.content or "",
            }
            if content.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.tool_name,
                            "arguments": json.dumps(call.tool_args),
                        },
                    }
                    for call in content.tool_calls
                ]
            messages.append(entry)
        elif isinstance(content, conversation.ToolResultContent):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": content.tool_call_id,
                    "content": json.dumps(content.tool_result, default=str),
                }
            )
    return messages


def _json_default(value: Any) -> str:
    """Last-resort JSON encoder for objects HA's encoder rejects.

    voluptuous_openapi's converted schemas can carry sentinel objects
    (``_Unsupported``) that HA's json_dumps refuses; they never matter on the
    wire, so render them as their type name and move on.
    """
    return f"<{type(value).__name__}>"


async def _chat_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    api_key: str | None,
    referer: str | None,
    title: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    timeout: float,
) -> dict[str, Any]:
    """One chat completion, with tools, against an OpenAI-compatible endpoint."""
    from .llm_backend import _chat_completions_url

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "stream": False,
        "temperature": 0.3,
    }
    # aiohttp's json= uses HA's json_dumps, whose default hook raises on the
    # voluptuous_openapi sentinel objects that can hide inside converted tool
    # schemas ("Type is not JSON serializable: _Unsupported"). Serialise here
    # with a tolerant encoder instead, and send plain bytes.
    body = json.dumps(payload, default=_json_default).encode("utf-8")
    headers["Content-Type"] = "application/json"
    try:
        async with session.post(
            _chat_completions_url(base_url),
            data=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status >= 400:
                body_text = await response.text()
                raise ToolLoopError(f"HTTP {response.status}: {body_text[:300]}")
            return await response.json()
    except aiohttp.ClientError as err:
        raise ToolLoopError(str(err)) from err


__all__ = ["ToolLoopError", "run_tool_loop"]
