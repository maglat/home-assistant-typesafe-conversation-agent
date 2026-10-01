"""Decision backends: the System One layer, against any compatible model.

Two backends share one interface:

- ``TypeSafeDecisionClient`` talks to TypeSafe's hosted ``POST /v1/systemone``
  (Jev, and anything else TypeSafe hosts). The endpoint is a single POST, so
  this deliberately is not the ``typesafe-sdk`` package: that depends on
  ``httpx2``, which Home Assistant does not ship, and using the aiohttp session
  HA already manages keeps the integration dependency-free.
- ``OpenAIDecisionClient`` turns any OpenAI-compatible ``/v1/chat/completions``
  endpoint into the same interface, so Clef, Clef-flash, Von, Laya, Kev or a
  plain LLM can drive the same routing code. It asks the questions in one
  prompt, demands strict JSON back, and derives confidence from the returned
  probabilities.

The router (``router.py``) consumes ``DecisionResponse`` and never learns which
backend answered. The response types, the retries and the circuit breaker are
shared, because a dead model must degrade identically whichever one is
configured.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from .const import (
    API_BACKOFF,
    API_MAX_RETRIES,
    API_TIMEOUT,
    CIRCUIT_FAILURE_THRESHOLD,
    CIRCUIT_RESET_SECONDS,
    LOGGER,
    TYPESAFE_API_URL,
    TYPESAFE_MODELS_URL,
)


class DecisionError(Exception):
    """Any failure talking to the configured decision backend."""


class DecisionAuthError(DecisionError):
    """The API key is missing, invalid, or lacks access."""


class DecisionRequestError(DecisionError):
    """Our request was malformed. This is a bug in our question builder."""


class DecisionUnavailableError(DecisionError):
    """The model is rate limited, overloaded, unreachable, or the breaker is open."""


# Historical names, kept so downstream imports and older issues keep working.
SystemOneError = DecisionError
SystemOneAuthError = DecisionAuthError
SystemOneRequestError = DecisionRequestError
SystemOneUnavailableError = DecisionUnavailableError


@dataclass(slots=True)
class ChoiceAnswer:
    """One Choice answer."""

    choice: str
    probabilities: dict[str, float]
    confidence: float

    @property
    def margin(self) -> float:
        """p(top) - p(second).

        Confidence and margin fail differently: a distribution can be peaked
        enough to look confident while two options remain effectively tied.
        """
        if len(self.probabilities) < 2:
            return 1.0
        top, second = sorted(self.probabilities.values(), reverse=True)[:2]
        return top - second


@dataclass(slots=True)
class ScoreAnswer:
    """One Score answer."""

    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


@dataclass(slots=True)
class NoulAnswer:
    """One Noul answer. Has no confidence - the value carries the uncertainty."""

    noul: float


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


@dataclass(slots=True)
class DecisionResponse:
    """A parsed decision response, whichever backend produced it."""

    model: str
    answers: dict[str, Answer]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def choice(self, key: str) -> ChoiceAnswer | None:
        answer = self.answers.get(key)
        return answer if isinstance(answer, ChoiceAnswer) else None

    def score(self, key: str) -> ScoreAnswer | None:
        answer = self.answers.get(key)
        return answer if isinstance(answer, ScoreAnswer) else None

    def noul(self, key: str) -> float | None:
        answer = self.answers.get(key)
        return answer.noul if isinstance(answer, NoulAnswer) else None


# Historical alias: the response the System One API returns.
SystemOneResponse = DecisionResponse


def _parse_answer(key: str, payload: dict[str, Any]) -> Answer:
    kind = payload.get("type")
    if kind == "choice":
        return ChoiceAnswer(
            choice=payload["choice"],
            probabilities={k: float(v) for k, v in payload["probabilities"].items()},
            confidence=float(payload["confidence"]),
        )
    if kind == "score":
        return ScoreAnswer(
            score=float(payload["score"]),
            legend=dict(payload.get("legend", {})),
            probabilities={k: float(v) for k, v in payload["probabilities"].items()},
            confidence=float(payload["confidence"]),
        )
    if kind == "noul":
        return NoulAnswer(noul=float(payload["noul"]))
    raise DecisionError(f"Unknown answer type {kind!r} for question {key!r}")


def _confidence_from_probabilities(probabilities: dict[str, float]) -> float:
    """Jev's confidence formula: (n * p_top - 1) / (n - 1).

    1.0 for a single option or a sure thing, 0.0 for a uniform distribution
    over n options. Deriving it here keeps the router's thresholds meaningful
    no matter which backend produced the probabilities.
    """
    if not probabilities:
        return 0.0
    n = len(probabilities)
    if n <= 1:
        return 1.0
    top = max(probabilities.values())
    return max(0.0, min(1.0, (n * top - 1) / (n - 1)))


class DecisionClient(ABC):
    """One method: send a state and questions, get typed answers back.

    Retries and the circuit breaker live here, because a dead model must
    degrade identically whichever backend is configured.
    """

    def __init__(self, session: aiohttp.ClientSession, model: str) -> None:
        self._session = session
        self._model = model
        self._consecutive_failures = 0
        self._open_until = 0.0

    @property
    def model(self) -> str:
        return self._model

    @property
    def circuit_open(self) -> bool:
        """True while we are deliberately not calling the API."""
        if self._open_until and time.monotonic() >= self._open_until:
            # Half-open: let the next request through and see what happens.
            self._open_until = 0.0
            self._consecutive_failures = 0
        return bool(self._open_until)

    async def async_validate(self) -> list[str]:
        """Check the credentials or endpoint; return the usable model names.

        The config flow calls this before saving an entry. Backends without a
        validation endpoint report an empty list.
        """
        return []

    @abstractmethod
    async def _ask_once(
        self, state: Any, questions: dict[str, dict[str, Any]]
    ) -> DecisionResponse:
        """One request, no retries. Raise a DecisionError subclass on failure."""

    async def async_ask(
        self, state: Any, questions: dict[str, dict[str, Any]]
    ) -> DecisionResponse:
        """Send one state and every question, and return the typed answers."""
        if self.circuit_open:
            raise DecisionUnavailableError("System One circuit breaker is open")

        started = time.monotonic()
        last_error: Exception | None = None

        for attempt in range(API_MAX_RETRIES):
            try:
                response = await self._ask_once(state, questions)
            except DecisionUnavailableError as err:
                last_error = err
                if attempt == API_MAX_RETRIES - 1:
                    break
                delay = (
                    err.retry_after
                    if isinstance(err, _RetryableError) and err.retry_after
                    else API_BACKOFF[attempt]
                )
                LOGGER.debug(
                    "Request attempt %s/%s failed (%s); retrying in %.2fs",
                    attempt + 1,
                    API_MAX_RETRIES,
                    err,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            except DecisionError:
                # Auth and validation errors are not worth retrying.
                self._record_failure()
                raise

            self._consecutive_failures = 0
            response.latency_ms = (time.monotonic() - started) * 1000
            LOGGER.debug(
                "%s answered %s questions in %.0fms (%s input tokens)",
                response.model,
                len(response.answers),
                response.latency_ms,
                response.input_tokens,
            )
            return response

        self._record_failure()
        raise DecisionUnavailableError(
            str(last_error) if last_error else "System One unavailable"
        )

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= CIRCUIT_FAILURE_THRESHOLD:
            self._open_until = time.monotonic() + CIRCUIT_RESET_SECONDS
            LOGGER.warning(
                "The API failed %s times in a row; pausing calls for %.0fs and "
                "serving the fallback agent instead",
                self._consecutive_failures,
                CIRCUIT_RESET_SECONDS,
            )


class _RetryableError(DecisionUnavailableError):
    """A failure worth another attempt."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


class TypeSafeDecisionClient(DecisionClient):
    """Talks to TypeSafe's hosted ``POST /v1/systemone``."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        model: str,
    ) -> None:
        super().__init__(session, model)
        self._api_key = api_key

    async def async_validate(self) -> list[str]:
        """Check the key and return the model names the account can use."""
        try:
            async with self._session.get(
                TYPESAFE_MODELS_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=aiohttp.ClientTimeout(total=API_TIMEOUT),
            ) as response:
                if response.status in (401, 403):
                    raise DecisionAuthError("Invalid TypeSafe API key")
                response.raise_for_status()
                payload = await response.json()
        except DecisionError:
            raise
        except aiohttp.ClientError as err:
            raise DecisionUnavailableError(str(err)) from err
        except TimeoutError as err:
            raise DecisionUnavailableError("Timed out reaching TypeSafe") from err
        return [m["name"] for m in payload.get("models", [])]

    async def _ask_once(
        self, state: Any, questions: dict[str, dict[str, Any]]
    ) -> DecisionResponse:
        body = {"state": state, "model": self._model, "questions": questions}
        started = time.monotonic()
        try:
            async with self._session.post(
                TYPESAFE_API_URL,
                json=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=API_TIMEOUT),
            ) as response:
                if response.status in (401, 403):
                    raise DecisionAuthError("Invalid TypeSafe API key")
                if response.status == 422:
                    detail = await response.text()
                    # Our question builder produced something invalid. The body
                    # names the offending field, so log it loudly - the
                    # build-time validator should have caught this.
                    raise DecisionRequestError(
                        f"The API rejected the request: {detail}"
                    )
                if response.status in (429, 529):
                    raise _RetryableError(
                        f"The API returned {response.status}",
                        retry_after=_parse_retry_after(
                            response.headers.get("retry-after")
                        ),
                    )
                if response.status >= 500:
                    raise _RetryableError(f"The API returned {response.status}")
                response.raise_for_status()
                payload = await response.json()
        except DecisionError:
            raise
        except aiohttp.ClientError as err:
            raise _RetryableError(str(err)) from err
        except TimeoutError as err:
            raise _RetryableError("Timed out talking to the System One API") from err

        usage = payload.get("usage", {})
        return DecisionResponse(
            model=payload.get("model", self._model),
            answers={
                key: _parse_answer(key, value)
                for key, value in payload.get("answers", {}).items()
            },
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            latency_ms=(time.monotonic() - started) * 1000,
            raw=payload,
        )


# Historical alias: the only client that existed before the pluggable backends.
SystemOneClient = TypeSafeDecisionClient


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _models_url(base_url: str) -> str:
    """Accept base URLs with or without a trailing ``/v1``."""
    trimmed = base_url.rstrip("/")
    return f"{trimmed}/models" if trimmed.endswith("/v1") else f"{trimmed}/v1/models"


def _chat_completions_url(base_url: str) -> str:
    """Same tolerance as _models_url, for the completion endpoint."""
    trimmed = base_url.rstrip("/")
    return (
        f"{trimmed}/chat/completions"
        if trimmed.endswith("/v1")
        else f"{trimmed}/v1/chat/completions"
    )


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    """Pull a JSON object out of whatever the model actually returned.

    Models wrap JSON in fences, prefix it with prose, or append a sentence.
    Try the strict reading first, then progressively looser ones.
    """
    text = raw.strip()
    candidates = [text]
    if (fenced := _FENCE_RE.search(text)) is not None:
        candidates.insert(0, fenced.group(1))
    if (obj := _OBJECT_RE.search(text)) is not None:
        candidates.append(obj.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalise_probabilities(raw: Any) -> dict[str, float]:
    """Coerce whatever the model returned into clean floats summing to 1."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except TypeError, ValueError:
            continue
    total = sum(out.values())
    if total <= 0:
        return {k: 1.0 / len(out) for k in out} if out else {}
    return {k: v / total for k, v in out.items()}


def _coerce_answers(
    payload: dict[str, Any], questions: dict[str, dict[str, Any]]
) -> dict[str, Answer]:
    """Turn a model's JSON object into typed answers, question by question.

    The OpenAI-compatible path has no server-side schema, so every question is
    repaired here against the schema we sent: missing keys become explicit
    unknowns, a bare label becomes a full distribution, and a numeric answer
    to a noul becomes the noul itself.
    """
    answers: dict[str, Answer] = {}
    for key, question in questions.items():
        kind = question.get("type")
        value = payload.get(key)
        if isinstance(value, dict):
            # The model echoed a full answer object; trust its shape.
            try:
                answers[key] = _parse_answer(key, {"type": kind, **value})
                continue
            except KeyError, TypeError, ValueError:
                pass
        if kind == "noul":
            answers[key] = _coerce_noul(value)
        elif kind == "score":
            answers[key] = _coerce_score(value, question)
        else:
            answers[key] = _coerce_choice(value, question)
    return answers


def _coerce_noul(value: Any) -> NoulAnswer:
    """A noul is a probability; accept booleans, numbers or nothing."""
    if isinstance(value, bool):
        return NoulAnswer(noul=1.0 if value else 0.0)
    if isinstance(value, (int, float)):
        return NoulAnswer(noul=float(value))
    return NoulAnswer(noul=0.0)


def _coerce_score(value: Any, question: dict[str, Any]) -> ScoreAnswer:
    """Rebuild a score answer from whatever the model returned."""
    criteria = question.get("criteria") or []
    labels = [str(level) for level in range(len(criteria))]
    probabilities = _normalise_probabilities(
        value.get("probabilities") if isinstance(value, dict) else value
    )
    if not probabilities:
        probabilities = dict.fromkeys(labels, 0.0)
        chosen = ""
    else:
        chosen = max(probabilities, key=lambda k: probabilities[k])
    return ScoreAnswer(
        score=float(chosen) if chosen.lstrip("-").isdigit() else 0.0,
        legend={str(i): str(c) for i, c in enumerate(criteria)},
        probabilities=probabilities,
        confidence=_confidence_from_probabilities(probabilities),
    )


def _coerce_choice(value: Any, question: dict[str, Any]) -> ChoiceAnswer:
    """Rebuild a choice answer from whatever the model returned."""
    criteria = question.get("criteria") or {}
    options = [str(option) for option in criteria]
    probabilities = _normalise_probabilities(
        value.get("probabilities") if isinstance(value, dict) else value
    )
    chosen = ""
    if isinstance(value, str):
        chosen = value
    elif isinstance(value, dict):
        for candidate in (value.get("choice"), value.get("answer")):
            if isinstance(candidate, str):
                chosen = candidate
                break
    if not chosen and probabilities:
        chosen = max(probabilities, key=lambda k: probabilities[k])
    if options and chosen not in options:
        # Not an exact option: match case-insensitively, else drop it.
        lowered = {option.lower(): option for option in options}
        chosen = lowered.get(chosen.strip().lower(), "")
    if not chosen:
        # Nothing usable: a flat distribution over the real options, which
        # every confidence gate in the router treats as "not solid".
        chosen = options[0] if options else ""
        probabilities = (
            {option: 1.0 / len(options) for option in options} if options else {}
        )
    elif not probabilities or chosen not in probabilities:
        probabilities = dict.fromkeys(options, 0.0)
        probabilities[chosen] = 1.0
    return ChoiceAnswer(
        choice=chosen,
        probabilities=probabilities,
        confidence=_confidence_from_probabilities(probabilities),
    )


class OpenAIDecisionClient(DecisionClient):
    """Runs the same question set through an OpenAI-compatible chat endpoint.

    This is the escape hatch that lets any open model drive the router: point
    ``base_url`` at a Clef, Von, Laya or plain-LLM server that speaks
    ``/v1/chat/completions`` (vLLM, llama.cpp, Ollama, TabbyAPI, ...). One
    prompt carries the state and every question; the reply must be one JSON
    object keyed by question id. Confidence is derived here from the returned
    probabilities, so the router's thresholds keep their meaning.
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = API_TIMEOUT,
    ) -> None:
        super().__init__(session, model)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    async def async_validate(self) -> list[str]:
        """Check the endpoint is reachable and the model is served there."""
        try:
            async with self._session.get(
                _models_url(self._base_url),
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as response:
                if response.status in (401, 403):
                    raise DecisionAuthError("The endpoint rejected the API key")
                response.raise_for_status()
                payload = await response.json()
        except DecisionError:
            raise
        except aiohttp.ClientError as err:
            raise DecisionUnavailableError(str(err)) from err
        except TimeoutError as err:
            raise DecisionUnavailableError("Timed out reaching the endpoint") from err
        models = [
            m.get("id", "")
            for m in payload.get("data", [])
            if isinstance(m, dict) and m.get("id")
        ]
        return models

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _ask_once(
        self, state: Any, questions: dict[str, dict[str, Any]]
    ) -> DecisionResponse:
        prompt = _build_decision_prompt(state, questions)
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _DECISION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "temperature": 0.0,
            "max_tokens": _max_tokens_for(questions),
        }
        started = time.monotonic()
        try:
            async with self._session.post(
                _chat_completions_url(self._base_url),
                json=payload,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as response:
                if response.status in (401, 403):
                    raise DecisionAuthError("The endpoint rejected the API key")
                if response.status in (429, 529):
                    raise _RetryableError(
                        f"The endpoint returned {response.status}",
                        retry_after=_parse_retry_after(
                            response.headers.get("retry-after")
                        ),
                    )
                if response.status >= 500:
                    raise _RetryableError(f"The endpoint returned {response.status}")
                response.raise_for_status()
                body = await response.json()
        except DecisionError:
            raise
        except aiohttp.ClientError as err:
            raise _RetryableError(str(err)) from err
        except TimeoutError as err:
            raise _RetryableError("Timed out talking to the decision model") from err

        try:
            content = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as err:
            raise DecisionRequestError(f"Unexpected response shape: {body}") from err
        parsed = _extract_json_object(content)
        if parsed is None:
            raise DecisionRequestError(
                "The model did not return a JSON object: " + content[:200]
            )
        usage = body.get("usage") or {}
        return DecisionResponse(
            model=body.get("model", self._model),
            answers=_coerce_answers(parsed, questions),
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_ms=(time.monotonic() - started) * 1000,
            raw=parsed,
        )


_DECISION_SYSTEM_PROMPT = """\
You are the decision engine of a smart-home voice assistant. You are given a \
state object and a set of questions. Each question has an id, a type and its \
allowed options. Answer EVERY question.

Respond with ONE JSON object and nothing else - no prose, no markdown fences. \
Shape, for one choice question with id "q1":

{"q1": {"choice": "<one option id>", \
"probabilities": {"<option id>": <p>, ...}}}

Rules:
- Every question id from the input must appear exactly once as a key.
- "choice" must be one of that question's option ids, copied exactly.
- "probabilities" must cover that question's options with numbers in [0,1] \
summing to 1.0.
- For true/false questions (type "noul") answer \
{"noul": <probability that the statement is true>} instead.
- For score questions answer like a choice, where the option ids are the \
score levels 0..N-1.
- Base every answer only on the state and the question text. Never invent \
options that were not offered."""


def _build_decision_prompt(state: Any, questions: dict[str, dict[str, Any]]) -> str:
    """Render the state and the full question schema into one prompt."""
    return json.dumps({"state": state, "questions": questions}, ensure_ascii=False)


def _max_tokens_for(questions: dict[str, dict[str, Any]]) -> int:
    """Room for one JSON object: an id, a choice and a probability each."""
    # ~12 tokens per option plus fixed overhead, measured against Qwen-style
    # tokenizers; generous, because truncation is the one failure mode that
    # cannot be repaired downstream.
    option_count = sum(len(q.get("criteria") or ()) for q in questions.values())
    return min(4096, 200 + option_count * 12)


def create_decision_client(
    session: aiohttp.ClientSession | None,
    settings: dict[str, Any],
) -> DecisionClient | None:
    """Build the configured decision backend, or None for the default.

    Without ``decision_backend`` the hosted TypeSafe API is used, exactly as
    before the pluggable backends existed. ``session`` may be None in tests.
    """
    from .const import (
        CONF_API_KEY,
        CONF_DECISION_API_KEY,
        CONF_DECISION_BACKEND,
        CONF_DECISION_BASE_URL,
        CONF_DECISION_TIMEOUT,
        CONF_MODEL,
        DECISION_OPENAI,
        DECISION_TYPESAFE,
        DEFAULT_MODEL,
    )

    backend = settings.get(CONF_DECISION_BACKEND) or DECISION_TYPESAFE
    model = settings.get(CONF_MODEL) or DEFAULT_MODEL

    if backend == DECISION_TYPESAFE:
        api_key = settings.get(CONF_API_KEY)
        if not api_key:
            return None
        return TypeSafeDecisionClient(session, api_key, model)  # type: ignore[arg-type]

    if backend == DECISION_OPENAI:
        base_url = settings.get(CONF_DECISION_BASE_URL)
        if not base_url:
            LOGGER.error("The OpenAI decision backend needs a base URL")
            return None
        return OpenAIDecisionClient(
            session,  # type: ignore[arg-type]
            base_url,
            model,
            settings.get(CONF_DECISION_API_KEY),
            float(settings.get(CONF_DECISION_TIMEOUT) or API_TIMEOUT),
        )

    LOGGER.error("Unknown decision backend %r", backend)
    return None


__all__ = [
    "Answer",
    "ChoiceAnswer",
    "DecisionAuthError",
    "DecisionClient",
    "DecisionError",
    "DecisionRequestError",
    "DecisionResponse",
    "DecisionUnavailableError",
    "NoulAnswer",
    "OpenAIDecisionClient",
    "ScoreAnswer",
    "SystemOneAuthError",
    "SystemOneClient",
    "SystemOneError",
    "SystemOneRequestError",
    "SystemOneResponse",
    "SystemOneUnavailableError",
    "TypeSafeDecisionClient",
]
