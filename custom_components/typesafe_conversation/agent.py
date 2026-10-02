"""The request pipeline: one decision call, then code decides.

Everything the router might need is asked in a single API call, including the
branches that will turn out to be irrelevant. Measured against jev-1.13.0, each
extra question costs about 97 input tokens and almost no extra latency, so
asking a question we discard is close to free - and it saves a round trip on
the requests where it turns out to matter.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from homeassistant.components import conversation
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import intent
from homeassistant.util import dt as dt_util

from . import questions as Q
from .const import (
    CATALOG_SUMMARY_MAX_ENTITIES,
    DEFAULT_ALWAYS_CONFIRM_RISKY,
    LOGGER,
    MAX_HISTORY_TURNS,
)
from .entities import EntityCatalog
from .executor import (
    ExecutionError,
    async_execute,
    async_execute_cancel,
    async_execute_query,
    describe_action,
)
from .extraction import extract
from .llm_backend import LLMBackend, LLMBackendError
from .router import Plan, Route, route, should_try_llm_answer
from .system_one import (
    DecisionClient,
    DecisionError,
    DecisionRequestError,
    DecisionResponse,
)
from .tool_loop import ToolLoopError, run_tool_loop


@dataclass(slots=True)
class AgentSettings:
    """Per-conversation settings, merged from the entry and its subentry."""

    inline_entity_descriptions: bool = False
    always_confirm_risky: bool = DEFAULT_ALWAYS_CONFIRM_RISKY
    bypass_local_intents: bool = False
    llm_control_devices: bool = False
    """Let the prose LLM call Home Assistant tools for the exposed devices.

    Off by default: the decision model is the control plane, and the prose
    path is read-only unless the user opts in. When on, the fallback path can
    act through the Assist API - restricted to the entities exposed to
    Assist, executed under the requesting user's context.
    """
    system_prompt: str = ""
    """Extra instructions for the freeform answers, verbatim from the user.

    Appended after the built-in answer prompt, so the guardrails (never claim
    to have controlled a device, prefer the home state over guessing) survive
    whatever the user adds - a persona, a language rule, a verbosity cap."""


class TypeSafeAgent:
    """Runs one utterance through the decision model and carries out the result."""

    def __init__(
        self,
        hass: HomeAssistant,
        catalog: EntityCatalog,
        decision_client: DecisionClient,
        llm: LLMBackend | None,
        settings: AgentSettings,
        traces: Any = None,
    ) -> None:
        self.hass = hass
        self.catalog = catalog
        self.decision_client = decision_client
        self.llm = llm
        self.settings = settings
        self._traces = traces
        self._questions_cache: tuple[int, bool, dict[str, Any]] | None = None
        self.continue_conversation = False
        """Set per request. Belongs to ConversationResult, not IntentResponse,
        so the entity reads it back after async_process returns."""

    # -- the entry point ------------------------------------------------------

    async def async_process(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> intent.IntentResponse:
        self.continue_conversation = False
        text = user_input.text.strip()
        if not text:
            # No point spending a request on an empty utterance.
            return self._error(
                user_input,
                intent.IntentResponseErrorCode.NO_INTENT_MATCH,
                "Sorry, I didn't catch that.",
            )

        speaker_area_id = self._speaker_area(user_input)
        try:
            response, entities = await self._ask(text, speaker_area_id, chat_log)
        except DecisionRequestError:
            # Our question builder produced something the API rejected. Already
            # logged with the offending field; behave as if Jev were down.
            return await self._fallback(user_input, chat_log, None)
        except DecisionError as err:
            LOGGER.warning(
                "System One unavailable (%s); using the fallback ladder", err
            )
            return await self._fallback(user_input, chat_log, None)

        plan = route(
            response,
            entities_by_id={e.entity_id: e for e in entities},
            extraction=extract(
                text,
                want_media="media_player" in self.catalog.domains,
                want_color="light" in self.catalog.domains,
            ),
            speaker_area_id=speaker_area_id,
            available_domains=frozenset(self.catalog.domains),
            always_confirm_risky=self.settings.always_confirm_risky,
            catalog_floors={
                a.area_id: a.floor_name for a in self.catalog.areas if a.floor_name
            },
            unavailable_ids=self._unavailable_ids(),
        )
        record = {
            "utterance": text,
            # Where the request came from. Standard ConversationInput fields,
            # so these are populated whatever the front end: a satellite fills
            # both, typed input leaves both None. Recording them is what turns
            # "something keeps arming the alarm" into a one-step answer.
            "device_id": user_input.device_id,
            "satellite_id": user_input.satellite_id,
            "from_satellite": user_input.satellite_id is not None,
            "route": plan.route.value,
            "reason": plan.reason,
            "domain": plan.domain,
            "action": plan.action,
            "target": plan.target.entity.entity_id
            if plan.target.entity is not None
            else (
                f"area:{plan.target.area_id}"
                if plan.target.area_id
                else ("whole_house" if plan.target.whole_house else None)
            ),
            "value": plan.value,
            "relative_step": plan.relative_step,
            "text_slot": plan.text_slot,
            **plan.trace,
        }
        LOGGER.debug(
            "Routed %r from %s -> %s (%s) in %sms, %s input tokens",
            text,
            user_input.satellite_id or user_input.device_id or "text input",
            plan.route.value,
            plan.reason,
            plan.trace.get("latency_ms"),
            plan.trace.get("input_tokens"),
        )
        for key, value in plan.trace.items():
            if isinstance(value, dict) and "choice" in value:
                LOGGER.debug(
                    "  %-16s %-28s conf %.2f margin %.2f  %s",
                    key,
                    value["choice"],
                    value["confidence"],
                    value["margin"],
                    value["top"],
                )
        if self._traces is not None:
            self._traces.append(record)
        conversation.async_conversation_trace_append(
            conversation.ConversationTraceEventType.AGENT_DETAIL, record
        )
        return await self._carry_out(plan, response, user_input, chat_log)

    # -- the single Jev call --------------------------------------------------

    async def _ask(
        self,
        text: str,
        speaker_area_id: str | None,
        chat_log: conversation.ChatLog | None,
    ) -> tuple[DecisionResponse, tuple]:
        entities, _narrowed = self.catalog.prefilter(text, speaker_area_id)
        extraction = extract(
            text,
            want_media="media_player" in self.catalog.domains,
            want_color="light" in self.catalog.domains,
        )
        questions = dict(self._structural_questions(entities))
        # Conditional questions depend on the utterance, not the catalog, so
        # they are built fresh and never cached.
        if extraction.values:
            questions[Q.Q_VALUE_PICK] = Q._value_pick_question(extraction)
        if extraction.colors_mentioned:
            questions[Q.Q_COLOR_PICK] = Q._color_pick_question()
        if extraction.media_chunks:
            questions[Q.Q_MEDIA_SPAN] = Q._media_span_question(extraction)
        Q.validate_questions(questions)

        state = {
            "request": {
                "text": text,
                "language": self.hass.config.language,
                "spoken_from_area": speaker_area_id,
                "local_time": dt_util.now().strftime("%Y-%m-%dT%H:%M"),
                "weekday": dt_util.now().strftime("%A"),
            },
            "home": self.catalog.snapshot(entities),
        }
        if chat_log is not None and (history := self._history(chat_log)):
            state["conversation"] = history

        return await self.decision_client.async_ask(state, questions), entities

    def _structural_questions(self, entities: tuple) -> dict[str, Any]:
        """Cache the catalog-derived questions against the catalog generation.

        They are a pure function of the catalog, so rebuilding twenty question
        dicts on every utterance would be wasted work.
        """
        generation = self.catalog.generation
        inline = self.settings.inline_entity_descriptions
        if (
            self._questions_cache is not None
            and self._questions_cache[0] == generation
            and self._questions_cache[1] == inline
            and len(entities) == len(self.catalog.entities)
        ):
            return self._questions_cache[2]

        built = Q.build_questions(
            entities=entities,
            areas=self.catalog.areas,
            domains=self.catalog.domains,
            extraction=extract("", want_media=False, want_color=False),
            inline_descriptions=inline,
        )
        if len(entities) == len(self.catalog.entities):
            self._questions_cache = (generation, inline, built)
        return built

    # -- acting on the plan ---------------------------------------------------

    async def _carry_out(
        self,
        plan: Plan,
        response: DecisionResponse,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> intent.IntentResponse:
        match plan.route:
            case Route.CANCEL:
                return await async_execute_cancel(self.hass, user_input)

            case Route.COMPOUND:
                return await self._handle_compound(user_input, chat_log)

            case Route.INFORMATION:
                return await self._answer_freeform(user_input, chat_log)

            case Route.QUERY:
                # hassil's GetState speech is well phrased and localized, and
                # because we advertise CONTROL the pipeline withheld exactly
                # this intent from it. Give it the first try - it costs ~5ms.
                if (
                    local := await conversation.async_handle_intents(
                        self.hass, user_input, chat_log
                    )
                ) is not None:
                    return local
                if plan.query_kind == "needs_prose":
                    return await self._answer_freeform(user_input, chat_log)
                try:
                    return await async_execute_query(self.hass, plan, user_input)
                except (ExecutionError, intent.IntentError) as err:
                    LOGGER.debug("Query execution failed (%s)", err)
                    return await self._answer_freeform(user_input, chat_log)

            case Route.CLARIFY:
                names = " or ".join(name for _, name in plan.options)
                return self._speech(
                    user_input, f"Did you mean the {names}?", continue_conversation=True
                )

            case Route.UNAVAILABLE:
                # Terminal. The fallback ladder would resolve the same dead
                # entity via hassil, then have the LLM apologise vaguely.
                return self._speech(user_input, plan.speech or "That is unavailable.")

            case Route.CONFIRM:
                return self._speech(
                    user_input, _confirm_question(plan), continue_conversation=True
                )

            case Route.COMMAND:
                return await self._run_command(plan, user_input, chat_log)

        return await self._fallback(user_input, chat_log, response)

    async def _run_command(
        self,
        plan: Plan,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> intent.IntentResponse:
        try:
            response = await async_execute(self.hass, plan, user_input, self.catalog)
        except intent.MatchFailedError as err:
            LOGGER.debug("No match for %s (%s); falling back", plan.reason, err)
            return await self._fallback(user_input, chat_log, None)
        except (ExecutionError, intent.IntentError) as err:
            LOGGER.warning("Could not carry out %s: %s", plan.reason, err)
            return self._error(
                user_input,
                intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                f"Sorry, I couldn't do that. {err}",
            )

        if failed := _wholly_failed(response):
            # Home Assistant records the *area* it matched in success_results,
            # so a response in which every entity refused the call still comes
            # back as action_done with no error. Reading that as a win would
            # have us cheerfully announce something that did not happen.
            LOGGER.warning(
                "%s reached no entity: %s rejected the call",
                plan.reason,
                ", ".join(failed),
            )
            return self._error(
                user_input,
                intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                f"{_join(failed)} could not do that.",
            )

        if not response.speech:
            # A successful command must always say something. Nothing else
            # will: the intent handlers set targets and states but no speech,
            # so without this the pipeline skips TTS and the user cannot tell
            # a command that worked from one that hung. name_target_in_speech
            # only decides how specific to be - in the middle confidence band
            # we name the target so a wrong guess can be corrected at once.
            response.async_set_speech(describe_action(plan, response))
        return response

    async def _handle_compound(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> intent.IntentResponse:
        if self.llm is None:
            return await self._fallback(user_input, chat_log, None)

        parts = await self.llm.split_compound(user_input.text)
        if len(parts) <= 1:
            # Not actually compound, or the split failed. Either way, one more
            # pass without the compound branch resolves it.
            return await self._rerun_single(
                parts[0] if parts else user_input.text, user_input, chat_log
            )

        speaker_area_id = self._speaker_area(user_input)
        results = await asyncio.gather(
            *(self._ask(part, speaker_area_id, None) for part in parts),
            return_exceptions=True,
        )

        done: list[str] = []
        failed: list[str] = []
        # Sequential, in the order the user said them: "turn on the AC and set
        # it to 20" only works one way round.
        for part, result in zip(parts, results, strict=True):
            if isinstance(result, BaseException):
                failed.append(part)
                continue
            sub_response, entities = result
            plan = route(
                sub_response,
                entities_by_id={e.entity_id: e for e in entities},
                extraction=extract(
                    part,
                    want_media="media_player" in self.catalog.domains,
                    want_color="light" in self.catalog.domains,
                ),
                speaker_area_id=speaker_area_id,
                available_domains=frozenset(self.catalog.domains),
                always_confirm_risky=self.settings.always_confirm_risky,
                unavailable_ids=self._unavailable_ids(),
            )
            # Never stop mid-way to ask a question: the user said four things
            # and is not expecting an interrogation about the second.
            if plan.route not in (Route.COMMAND, Route.QUERY):
                failed.append(part)
                continue
            sub_input = _with_text(user_input, part)
            try:
                if plan.route is Route.QUERY:
                    await async_execute_query(self.hass, plan, sub_input)
                else:
                    await async_execute(self.hass, plan, sub_input, self.catalog)
                done.append(plan.target.described)
            except (ExecutionError, intent.IntentError) as err:
                LOGGER.debug("Sub-command %r failed: %s", part, err)
                failed.append(part)

        return self._compound_response(user_input, done, failed)

    def _compound_response(
        self,
        user_input: conversation.ConversationInput,
        done: list[str],
        failed: list[str],
    ) -> intent.IntentResponse:
        if not done:
            return self._error(
                user_input,
                intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                "Sorry, I couldn't do any of that.",
            )
        speech = f"Done: {_join(done)}."
        if failed:
            # The user needs to know precisely what did not happen.
            speech = f"Done: {_join(done)}. But I couldn't {_join(failed)}."
        return self._speech(user_input, speech)

    async def _rerun_single(
        self,
        text: str,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> intent.IntentResponse:
        try:
            response, entities = await self._ask(
                text, self._speaker_area(user_input), chat_log
            )
        except DecisionError:
            return await self._fallback(user_input, chat_log, None)
        plan = route(
            response,
            entities_by_id={e.entity_id: e for e in entities},
            extraction=extract(
                text,
                want_media="media_player" in self.catalog.domains,
                want_color="light" in self.catalog.domains,
            ),
            speaker_area_id=self._speaker_area(user_input),
            available_domains=frozenset(self.catalog.domains),
            always_confirm_risky=self.settings.always_confirm_risky,
            catalog_floors={
                a.area_id: a.floor_name for a in self.catalog.areas if a.floor_name
            },
            unavailable_ids=self._unavailable_ids(),
        )
        if plan.route is Route.COMPOUND:
            plan.route = Route.FALLBACK
        return await self._carry_out(plan, response, user_input, chat_log)

    # -- fallbacks ------------------------------------------------------------

    async def _fallback(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        response: DecisionResponse | None,
    ) -> intent.IntentResponse:
        """hassil, then context resolution, then the LLM, then admit defeat.

        No intent_filter: because we advertise CONTROL, the pipeline's
        prefer-local pass withheld HassGetState and HassMediaSearchAndPlay from
        the matcher. This is a genuinely new attempt, not a repeat of one.
        """
        if (
            local := await conversation.async_handle_intents(
                self.hass, user_input, chat_log
            )
        ) is not None:
            LOGGER.debug("Fallback: handled locally by the sentence matcher")
            return local

        if self.llm is not None and (
            response is None or should_try_llm_answer(response)
        ):
            # The utterance may only be unclear *in isolation*. A follow-up
            # like "and back off again" resolves once the previous turns are
            # in view - so let the LLM rewrite it into a standalone command
            # and run the decision model once more on the result.
            try:
                return await self._resolve_context(user_input, chat_log)
            except LLMBackendError as err:
                LOGGER.debug("Context resolution unavailable: %s", err)

            if self.settings.llm_control_devices:
                # Full Assist API: the LLM may call tools for the exposed
                # devices. Tried before the read-only prose answer; a failure
                # here only costs latency, never the answer.
                try:
                    return await self._answer_with_tools(user_input, chat_log)
                except LLMBackendError as err:
                    LOGGER.warning("LLM tool path failed: %s", err)

            try:
                return await self._answer_freeform(user_input, chat_log)
            except LLMBackendError as err:
                LOGGER.warning("LLM fallback failed: %s", err)

        return self._error(
            user_input,
            intent.IntentResponseErrorCode.NO_INTENT_MATCH,
            "Sorry, I'm not sure what you'd like me to do.",
        )

    async def _resolve_context(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> intent.IntentResponse:
        """Rewrite a context-dependent utterance into a standalone one.

        The decision model sees every turn, but it is a single-pass
        classifier: follow-ups that only make sense with the previous turns
        in view ("and back off again", "and the kitchen too?") score as
        unclear. The LLM is good at exactly this, so it rewrites the
        utterance with the history in view and the rewrite runs through the
        normal pipeline. If nothing is left after resolving - the user was
        chatting, not commanding - the answer path takes over.
        """
        history = self._history_pairs(chat_log)
        if not history:
            raise LLMBackendError("no history to resolve against")

        rewritten = await self.llm.rewrite_with_context(
            user_input.text, history, speaker_area=self._speaker_area_name(user_input)
        )
        rewritten = rewritten.strip()
        if not rewritten or rewritten.casefold() == user_input.text.casefold():
            # Nothing to resolve: the utterance was already standalone, so a
            # second decision pass would only repeat the same fallback.
            raise LLMBackendError("rewrite did not change the utterance")

        LOGGER.debug("Context resolution: %r -> %r", user_input.text, rewritten)
        try:
            response, entities = await self._ask(
                rewritten, self._speaker_area(user_input), None
            )
        except DecisionError as err:
            raise LLMBackendError(f"decision pass failed: {err}") from err
        plan = route(
            response,
            entities_by_id={e.entity_id: e for e in entities},
            extraction=extract(
                rewritten,
                want_media="media_player" in self.catalog.domains,
                want_color="light" in self.catalog.domains,
            ),
            speaker_area_id=self._speaker_area(user_input),
            available_domains=frozenset(self.catalog.domains),
            always_confirm_risky=self.settings.always_confirm_risky,
            catalog_floors={
                a.area_id: a.floor_name for a in self.catalog.areas if a.floor_name
            },
            unavailable_ids=self._unavailable_ids(),
        )
        if plan.route is Route.COMPOUND:
            plan.route = Route.FALLBACK
        if plan.route in (Route.FALLBACK, Route.COMPOUND):
            # Still unclear with the context in view: let the freeform answer
            # try, rather than looping through resolution again.
            raise LLMBackendError(f"still unresolved after rewrite: {plan.reason}")
        return await self._carry_out(plan, response, user_input, chat_log)

    async def _answer_freeform(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> intent.IntentResponse:
        if self.llm is None:
            return self._error(
                user_input,
                intent.IntentResponseErrorCode.NO_INTENT_MATCH,
                "I can only control the home right now.",
            )
        now = dt_util.now()
        entities = self.catalog.entities[:CATALOG_SUMMARY_MAX_ENTITIES]
        try:
            answer = await self.llm.answer_freeform(
                user_input.text,
                self._history_pairs(chat_log),
                home_state=self.catalog.summarize(entities),
                local_time=now.strftime("%H:%M"),
                weekday=now.strftime("%A"),
                speaker_area=self._speaker_area_name(user_input),
                system_prompt=self.settings.system_prompt,
            )
        except LLMBackendError as err:
            LOGGER.warning("LLM could not answer: %s", err)
            return self._error(
                user_input,
                intent.IntentResponseErrorCode.UNKNOWN,
                "Sorry, I can't answer that right now.",
            )
        return self._speech(user_input, answer or "I'm not sure.")

    async def _answer_with_tools(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> intent.IntentResponse:
        """Answer through the Assist API: the LLM may call HA tools.

        Used as the first prose attempt when device control for the LLM is
        enabled. Any failure falls back to the read-only prose answer, so a
        broken tool path can only cost latency, never the answer itself.
        """
        if self.llm is None:
            raise LLMBackendError("no LLM configured")
        try:
            answer = await run_tool_loop(
                self.hass,
                self.llm.session,
                base_url=self.llm.base_url,
                model=self.llm.model,
                api_key=self.llm.api_key,
                referer=getattr(self.llm, "referer", None),
                title=getattr(self.llm, "title", None),
                answer_timeout=self.llm.answer_timeout,
                user_input=user_input,
                chat_log=chat_log,
                user_text=user_input.text,
                extra_system_prompt=self.settings.system_prompt or None,
            )
        except ToolLoopError as err:
            LOGGER.warning("Tool loop failed, falling back to prose: %s", err)
            raise LLMBackendError(str(err)) from err
        return self._speech(user_input, answer or "Done.")

    # -- small helpers --------------------------------------------------------

    def _unavailable_ids(self) -> frozenset[str]:
        """Entities that exist but cannot act, read fresh.

        The catalog deliberately caches structure and never state, and route()
        is pure, so availability is computed here and passed in. `unknown` is
        not included: that entity is alive, its value simply is not known yet.
        """
        return frozenset(
            entity.entity_id
            for entity in self.catalog.entities
            if (state := self.hass.states.get(entity.entity_id)) is None
            or state.state == STATE_UNAVAILABLE
        )

    def _speaker_area(self, user_input: conversation.ConversationInput) -> str | None:
        if user_input.device_id is None:
            return None
        from homeassistant.helpers import device_registry as dr

        device = dr.async_get(self.hass).async_get(user_input.device_id)
        return device.area_id if device else None

    def _speaker_area_name(
        self, user_input: conversation.ConversationInput
    ) -> str | None:
        area_id = self._speaker_area(user_input)
        if area_id is None:
            return None
        area = ar.async_get(self.hass).async_get_area(area_id)
        return area.name if area else None

    def _history(self, chat_log: conversation.ChatLog) -> list[dict[str, str]]:
        return [
            {"user": user_text, "assistant": assistant_text}
            for user_text, assistant_text in self._history_pairs(chat_log)
        ]

    def _history_pairs(self, chat_log: conversation.ChatLog) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        pending: str | None = None
        for content in chat_log.content:
            if isinstance(content, conversation.UserContent):
                pending = content.content
            elif isinstance(content, conversation.AssistantContent) and pending:
                pairs.append((pending, content.content or ""))
                pending = None
        # Drop the turn currently in flight, then keep the last few.
        return pairs[-MAX_HISTORY_TURNS:]

    def _speech(
        self,
        user_input: conversation.ConversationInput,
        text: str,
        continue_conversation: bool = False,
    ) -> intent.IntentResponse:
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(text)
        if continue_conversation:
            self.continue_conversation = True
        return response

    def _error(
        self,
        user_input: conversation.ConversationInput,
        code: intent.IntentResponseErrorCode,
        message: str,
    ) -> intent.IntentResponse:
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_error(code, message)
        return response


def _confirm_question(plan: Plan) -> str:
    """Phrase the confirmation.

    A script carries its meaning in its name, not its action, so "run the
    Disarm the alarm?" reads badly - name it directly instead.
    """
    target = plan.target.described
    if plan.domain in ("script", "scene"):
        return f"Do you want me to run {target}?"
    verb = (plan.action or "do that").replace("_", " ")
    return f"Do you want me to {verb} the {target}?"


def _wholly_failed(response: intent.IntentResponse) -> list[str]:
    """Names of the entities that refused, when *none* accepted.

    async_handle_states puts the matched area in success_results whatever
    becomes of the entities inside it, so response_type stays action_done and
    error_code stays unset even when every service call was rejected. The only
    honest signal is that failed_results holds entities and success_results
    holds none. A partial success is left alone - something did happen.
    """
    if not response.failed_results:
        return []
    if any(
        target.type == intent.IntentResponseTargetType.ENTITY
        for target in response.success_results
    ):
        return []
    return [
        target.name
        for target in response.failed_results
        if target.type == intent.IntentResponseTargetType.ENTITY
    ]


def _with_text(
    user_input: conversation.ConversationInput, text: str
) -> conversation.ConversationInput:
    return conversation.ConversationInput(
        text=text,
        context=user_input.context,
        conversation_id=user_input.conversation_id,
        device_id=user_input.device_id,
        satellite_id=user_input.satellite_id,
        language=user_input.language,
        agent_id=user_input.agent_id,
        extra_system_prompt=user_input.extra_system_prompt,
    )


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


__all__ = ["AgentSettings", "TypeSafeAgent"]
