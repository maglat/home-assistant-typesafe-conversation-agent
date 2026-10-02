"""The two things we still need a text-generating model for.

Everything about controlling the home is decided by Jev and executed by code.
The LLM is used only where a string genuinely has to be produced:

1. splitting a compound request into atomic commands, and
2. answering a general-knowledge or prose question.

It is never given tools and never told it can control anything, so a slow or
failing LLM can delay an answer but can never block or corrupt a device action.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from .const import (
    ANSWER_MAX_TOKENS,
    ANSWER_TEMPERATURE,
    ANSWER_TIMEOUT,
    BACKEND_OLLAMA,
    BACKEND_OPENAI_COMPAT,
    LOGGER,
    MAX_SUB_COMMANDS,
    OLLAMA_KEEP_ALIVE,
    PROMPT_LOG_CHARS,
    SPLIT_MAX_TOKENS,
    SPLIT_TIMEOUT,
)

SPLIT_SYSTEM_PROMPT = """\
You split a smart-home voice command into atomic commands.

Rules:
- Output ONLY a JSON array of strings. No prose, no explanation, no markdown, \
no code fences.
- Each string must be a complete, self-contained command that names its own \
target.
- Distribute shared subjects and verbs: "turn off the lights and the fan" -> \
["turn off the lights", "turn off the fan"]
- Keep the user's original wording and their original order.
- Never invent a device, room, value or action that is not in the input.
- Do not split one action applied to several devices: "turn off all the \
lights" is ONE command.
- If the input is a single command, return a one-element array.

Input: turn off the kitchen lights and lock the front door
Output: ["turn off the kitchen lights", "lock the front door"]

Input: dim the living room lights to 30% and start the coffee maker
Output: ["dim the living room lights to 30%", "start the coffee maker"]

Input: turn off all of the lights in the house
Output: ["turn off all of the lights in the house"]

Input: set the thermostat to 21, close the blinds, and play some jazz
Output: ["set the thermostat to 21", "close the blinds", "play some jazz"]"""

ANSWER_SYSTEM_PROMPT = """\
You are the voice assistant for a Home Assistant smart home.

- Reply in plain spoken text. No markdown, no bullet lists, no emoji, no \
headings.
- Be brief: one or two sentences, under 40 words, as if speaking aloud.
- You cannot control any device in this mode. Never claim to have turned \
anything on or off, and never promise to do something.
- Use the home state below when the question is about the home. If the answer \
is not in it, or you are not confident, say you do not know rather than \
guessing.
- Answer general-knowledge questions truthfully and briefly from your own \
knowledge.

Current time: {local_time} on {weekday}. The speaker is in {speaker_area}.

Home state:
{home_state}"""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?\s*", re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"^\s*<think>.*", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove reasoning-model think blocks from a reply.

    GLM, Qwen-thinking and friends wrap their hidden reasoning in
    ``<think>...``. When the token cap cuts generation short, the
    closing tag never arrives and the whole reply is reasoning with no
    answer - which read as "the model did something but nothing came back".
    """
    text = _THINK_RE.sub("", text)
    return _UNCLOSED_THINK_RE.sub("", text)


class LLMBackendError(Exception):
    """Any failure talking to the configured LLM."""


class LLMBackendTimeoutError(LLMBackendError):
    """The endpoint missed its timeout.

    Distinguished from other failures so the fallback ladder can tell
    "server busy" from "server broken": a busy server will miss the next,
    bigger prompt's budget too, so stacking a second timeout only adds
    dead air before the same apology.
    """


CONTEXT_SYSTEM_PROMPT = """\
You resolve follow-up requests in a smart-home voice conversation.

You get the last exchanges and the newest utterance. The newest utterance may
use words like "it", "there", "also", "too", or "again" that only make sense
with the earlier turns in view.

Rewrite the newest utterance as ONE short standalone command that a device
control system can carry out without any of that context. Keep the user's
language. Examples:

history: user "turn on the kitchen light", assistant "Done."
utterance: "and back off again"
output: turn off the kitchen light

history: user "set the thermostat to 21 degrees", assistant "Done."
utterance: "and the living room too"
output: set the living room thermostat to 21 degrees

history: user "play some jazz", assistant "Playing jazz."
utterance: "louder"
output: turn the volume up

If the utterance is not about controlling the home - a question, a new topic,
general chat - output exactly: NONE
Output ONLY the rewritten command or NONE. No quotes, no explanation."""


class LLMBackend(ABC):
    """Two operations. Nothing else is ever asked of the LLM."""

    name: str

    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        model: str,
        api_key: str | None = None,
        answer_timeout: float = ANSWER_TIMEOUT,
    ) -> None:
        self._session = session
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._answer_timeout = answer_timeout

    # Read-only views for the tool loop, which dials the same endpoint with
    # an OpenAI tools payload. Not the provider-specific internals.
    @property
    def session(self) -> aiohttp.ClientSession:
        return self._session

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    @property
    def api_key(self) -> str | None:
        return self._api_key

    @property
    def answer_timeout(self) -> float:
        return self._answer_timeout

    @property
    def _fast_timeout(self) -> float:
        """Budget for the small utility prompts (split, rewrite).

        A slice of the answer timeout rather than a fixed constant, so a
        user who raised the answer timeout for a busy shared server raised
        this one implicitly. The old hardcoded 4s assumed a dedicated
        server; behind a queue - one GPU serving several clients - even a
        300-token request can wait longer than that.
        """
        return max(SPLIT_TIMEOUT, self._answer_timeout / 3.0)

    @abstractmethod
    async def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> tuple[str, dict[str, Any]]:
        """Send a chat completion.

        Returns the assistant's text and a normalized metrics dict. The two
        providers report different things - Ollama gives timings and token
        counts, an OpenAI-compatible endpoint gives token counts only - so
        each backend normalizes into shared key names and the caller never
        has to branch. Every backend supplies ``elapsed_s``, measured here
        rather than taken from the provider: it is what the user actually
        waits, and it is the only figure that also covers connection setup
        and a slow link.
        """

    # An optional hook, no-op by default - deliberately not abstract.
    async def async_warm_up(self) -> None:  # noqa: B027
        """Nudge the model into memory. Overridden where it helps."""

    async def split_compound(self, utterance: str) -> list[str]:
        """Break a compound request into atomic commands.

        Never raises. If the model is slow, unreachable, or returns something
        that is not a JSON array, we fall back to treating the utterance as a
        single command - which is exactly what would have happened without the
        compound question. A failure here must not cost the user their command.
        """
        messages = [
            {"role": "system", "content": SPLIT_SYSTEM_PROMPT},
            {"role": "user", "content": utterance},
        ]
        try:
            raw, metrics = await self._chat(
                messages,
                max_tokens=SPLIT_MAX_TOKENS,
                temperature=0.0,
                timeout=self._fast_timeout,
            )
        except LLMBackendError as err:
            LOGGER.warning("Could not split a compound request (%s)", err)
            return [utterance]
        _log_exchange("split", self.name, self._model, messages, raw, metrics)

        parts = _parse_string_array(raw)
        if not parts:
            LOGGER.warning("LLM split returned no usable array: %r", raw[:200])
            return [utterance]
        if len(parts) > MAX_SUB_COMMANDS:
            LOGGER.warning(
                "LLM split produced %s parts; keeping the first %s",
                len(parts),
                MAX_SUB_COMMANDS,
            )
            parts = parts[:MAX_SUB_COMMANDS]
        LOGGER.debug("Split %r into %s", utterance, parts)
        return parts

    async def answer_freeform(
        self,
        utterance: str,
        history: list[tuple[str, str]],
        *,
        home_state: str,
        local_time: str,
        weekday: str,
        speaker_area: str | None,
        system_prompt: str = "",
    ) -> str:
        """Answer a general or prose question in natural language."""
        system = ANSWER_SYSTEM_PROMPT.format(
            local_time=local_time,
            weekday=weekday,
            speaker_area=speaker_area or "an unknown room",
            home_state=home_state,
        )
        if system_prompt.strip():
            # User instructions come after the built-in guardrails, so a
            # persona or a language rule cannot override the safety lines.
            system = f"{system}\n\nAdditional instructions:\n{system_prompt.strip()}"
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for user_text, assistant_text in history:
            messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": utterance})

        text, metrics = await self._chat(
            messages,
            max_tokens=ANSWER_MAX_TOKENS,
            temperature=ANSWER_TEMPERATURE,
            timeout=self._answer_timeout,
        )
        _log_exchange("answer", self.name, self._model, messages, text, metrics)
        return text.strip()

    async def rewrite_with_context(
        self,
        utterance: str,
        history: list[tuple[str, str]],
        *,
        speaker_area: str | None = None,
    ) -> str:
        """Rewrite a follow-up utterance into a standalone command.

        Returns the rewritten command, or raises LLMBackendError when there is
        nothing to resolve - the caller then falls through to the prose
        answer. Never raises for network problems; those surface as
        LLMBackendError too, and the fallback ladder absorbs them.
        """
        lines: list[str] = []
        for user_text, assistant_text in history:
            lines.append(f'user: "{user_text}"')
            if assistant_text:
                lines.append(f'assistant: "{assistant_text}"')
        if speaker_area:
            lines.append(f"(the speaker is in the {speaker_area})")
        lines.append(f'newest utterance: "{utterance}"')

        messages = [
            {"role": "system", "content": CONTEXT_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ]
        raw, metrics = await self._chat(
            messages,
            max_tokens=SPLIT_MAX_TOKENS,
            temperature=0.0,
            timeout=self._fast_timeout,
        )
        _log_exchange("rewrite", self.name, self._model, messages, raw, metrics)
        return raw.strip()


class OllamaBackend(LLMBackend):
    """Ollama's native chat endpoint."""

    name = BACKEND_OLLAMA

    async def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        started = time.monotonic()
        data = await _post_json(
            self._session, f"{self._base_url}/api/chat", payload, headers, timeout
        )
        elapsed = time.monotonic() - started
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as err:
            raise LLMBackendError(f"Unexpected Ollama response: {data}") from err
        return _strip_think(text), _ollama_metrics(data, elapsed)

    async def async_warm_up(self) -> None:
        """Keep the model resident.

        A cold 7B load is several seconds and would be blamed on this
        integration, so we pay for it in the background instead.
        """
        try:
            _text, metrics = await self._chat(
                [{"role": "user", "content": "hi"}],
                max_tokens=1,
                temperature=0.0,
                timeout=SPLIT_TIMEOUT,
            )
            LOGGER.debug(
                "Ollama warm-up took %.2fs (load %.2fs)",
                metrics.get("elapsed_s", 0.0),
                metrics.get("load_s", 0.0),
            )
        except LLMBackendError as err:
            LOGGER.debug("Ollama warm-up did not succeed: %s", err)


class OpenAICompatBackend(LLMBackend):
    """Any OpenAI-compatible /v1/chat/completions endpoint, such as OpenRouter."""

    name = BACKEND_OPENAI_COMPAT

    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        model: str,
        api_key: str | None = None,
        referer: str | None = None,
        title: str | None = None,
        answer_timeout: float = ANSWER_TIMEOUT,
    ) -> None:
        super().__init__(session, base_url, model, api_key, answer_timeout)
        self._referer = referer
        self._title = title

    async def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        # OpenRouter attributes traffic with these; harmless elsewhere.
        if self._referer:
            headers["HTTP-Referer"] = self._referer
        if self._title:
            headers["X-Title"] = self._title
        started = time.monotonic()
        data = await _post_json(
            self._session,
            _chat_completions_url(self._base_url),
            payload,
            headers,
            timeout,
        )
        elapsed = time.monotonic() - started
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as err:
            raise LLMBackendError(f"Unexpected response: {data}") from err
        return _strip_think(text), _openai_metrics(data, elapsed)


# --- metrics ------------------------------------------------------------
# Each backend normalizes its provider's response into these shared keys, so
# the log helper never branches on which provider answered. A key that the
# provider does not report is simply absent.
#
#   elapsed_s          always - measured client-side
#   prompt_tokens      Ollama prompt_eval_count   | OpenAI usage.prompt_tokens
#   completion_tokens  Ollama eval_count          | OpenAI usage.completion_tokens
#   load_s             Ollama only - a cold model load
#   prompt_s, eval_s   Ollama only - enables tokens/sec
#   total_s            Ollama only - the server's own view of the request


def _ns_to_s(value: Any) -> float | None:
    """Ollama reports durations in nanoseconds."""
    if isinstance(value, (int, float)) and value > 0:
        return value / 1_000_000_000
    return None


def _ollama_metrics(data: dict[str, Any], elapsed: float) -> dict[str, Any]:
    out: dict[str, Any] = {"elapsed_s": elapsed}
    for key, name in (
        ("prompt_eval_count", "prompt_tokens"),
        ("eval_count", "completion_tokens"),
    ):
        if isinstance(data.get(key), int):
            out[name] = data[key]
    for key, name in (
        ("load_duration", "load_s"),
        ("prompt_eval_duration", "prompt_s"),
        ("eval_duration", "eval_s"),
        ("total_duration", "total_s"),
    ):
        if (seconds := _ns_to_s(data.get(key))) is not None:
            out[name] = seconds
    return out


def _openai_metrics(data: dict[str, Any], elapsed: float) -> dict[str, Any]:
    """An OpenAI-compatible endpoint reports tokens but no timing.

    There is no local model to load, so there is no cold-load equivalent
    either; elapsed_s is the only timing available and it is ours.
    """
    out: dict[str, Any] = {"elapsed_s": elapsed}
    usage = data.get("usage")
    if isinstance(usage, dict):
        for key, name in (
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
        ):
            if isinstance(usage.get(key), int):
                out[name] = usage[key]
    return out


def _log_exchange(
    operation: str,
    backend: str,
    model: str,
    messages: list[dict[str, str]],
    reply: str,
    metrics: dict[str, Any],
) -> None:
    """Record one LLM exchange at DEBUG.

    The system prompt embeds the home catalog - entity names, areas and
    current states - and home-assistant.log is what people paste into issue
    reports, so it is truncated here. The reply is short enough
    (ANSWER_MAX_TOKENS is 180) to log whole.
    """
    if not LOGGER.isEnabledFor(logging.DEBUG):
        return

    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    lines = [
        f"llm {backend} {model} {operation}",
        f"  messages  : {len(messages)} (system {len(system)} chars, truncated below)",
        f"  system    : {system[:PROMPT_LOG_CHARS]!r}"
        + ("..." if len(system) > PROMPT_LOG_CHARS else ""),
        f"  reply     : {reply!r} ({len(reply)} chars)",
        f"  elapsed   : {metrics['elapsed_s']:.2f}s",
    ]
    if (load := metrics.get("load_s")) is not None:
        lines.append(f"  load      : {load:.2f}s  <- cold model load")
    for label, tokens_key, seconds_key in (
        ("prompt", "prompt_tokens", "prompt_s"),
        ("eval  ", "completion_tokens", "eval_s"),
    ):
        tokens = metrics.get(tokens_key)
        if tokens is None:
            continue
        seconds = metrics.get(seconds_key)
        if seconds:
            lines.append(
                f"  {label}    : {tokens} tok in {seconds:.2f}s "
                f"({tokens / seconds:.1f} tok/s)"
            )
        else:
            # An OpenAI-compatible endpoint reports no per-phase timing.
            lines.append(f"  {label}    : {tokens} tok")
    if (total := metrics.get("total_s")) is not None:
        lines.append(f"  total     : {total:.2f}s (server-side)")
    LOGGER.debug("\n".join(lines))


def _chat_completions_url(base_url: str) -> str:
    """Accept base URLs with or without a trailing ``/v1``.

    Users paste whatever their server's docs show - ``http://host:8000`` from
    vLLM, ``http://host:8080/v1`` from llama.cpp - and both are correct.
    """
    trimmed = base_url.rstrip("/")
    return (
        f"{trimmed}/chat/completions"
        if trimmed.endswith("/v1")
        else f"{trimmed}/v1/chat/completions"
    )


async def _post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    try:
        async with asyncio.timeout(timeout):
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise LLMBackendError(f"HTTP {response.status}: {body[:300]}")
                return await response.json()
    except LLMBackendError:
        raise
    except TimeoutError as err:
        raise LLMBackendTimeoutError(f"Timed out after {timeout}s") from err
    except aiohttp.ClientError as err:
        raise LLMBackendError(str(err)) from err
    except json.JSONDecodeError as err:
        raise LLMBackendError(f"Response was not JSON: {err}") from err


def _parse_string_array(raw: str) -> list[str]:
    """Get a list of strings out of whatever the model actually returned.

    Models wrap JSON in fences, prefix it with "Output:", or bury it in a
    sentence. Try the strict reading first, then progressively looser ones.
    """
    for candidate in _candidates(raw):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError, TypeError:
            continue
        if isinstance(parsed, list):
            items = [p.strip() for p in parsed if isinstance(p, str) and p.strip()]
            if items:
                return items
    return []


def _candidates(raw: str):
    text = raw.strip()
    yield text
    if (fenced := _FENCE_RE.search(text)) is not None:
        yield fenced.group(1)
    if (array := _ARRAY_RE.search(text)) is not None:
        yield array.group(0)


def create_backend(
    session: aiohttp.ClientSession, settings: dict[str, Any]
) -> LLMBackend | None:
    """Build the configured backend, or None when no LLM is set up.

    Without an LLM the agent still handles every command and query; it just
    cannot split compound requests or answer general questions.
    """
    from .const import (
        CONF_LLM_API_KEY,
        CONF_LLM_BACKEND,
        CONF_LLM_BASE_URL,
        CONF_LLM_MODEL,
        CONF_LLM_REFERER,
        CONF_LLM_TIMEOUT,
        CONF_LLM_TITLE,
        DEFAULT_LLM_REFERER,
        DEFAULT_LLM_TITLE,
    )

    backend = settings.get(CONF_LLM_BACKEND)
    base_url = settings.get(CONF_LLM_BASE_URL)
    model = settings.get(CONF_LLM_MODEL)
    if not backend or not base_url or not model:
        return None

    timeout = float(settings.get(CONF_LLM_TIMEOUT) or ANSWER_TIMEOUT)

    if backend == BACKEND_OLLAMA:
        return OllamaBackend(
            session, base_url, model, settings.get(CONF_LLM_API_KEY), timeout
        )
    if backend == BACKEND_OPENAI_COMPAT:
        return OpenAICompatBackend(
            session,
            base_url,
            model,
            settings.get(CONF_LLM_API_KEY),
            settings.get(CONF_LLM_REFERER) or DEFAULT_LLM_REFERER,
            settings.get(CONF_LLM_TITLE) or DEFAULT_LLM_TITLE,
            timeout,
        )
    LOGGER.error("Unknown LLM backend %r", backend)
    return None


__all__ = [
    "LLMBackend",
    "LLMBackendError",
    "LLMBackendTimeoutError",
    "OllamaBackend",
    "OpenAICompatBackend",
    "create_backend",
]
