"""Config and subentry flows for TypeSafe Conversation."""

from __future__ import annotations

from typing import Any, override

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    ANSWER_TIMEOUT,
    BACKEND_OLLAMA,
    BACKEND_OPENAI_COMPAT,
    CONF_ALWAYS_CONFIRM_RISKY,
    CONF_API_KEY,
    CONF_BYPASS_LOCAL_INTENTS,
    CONF_DECISION_API_KEY,
    CONF_DECISION_BACKEND,
    CONF_DECISION_BASE_URL,
    CONF_DECISION_TIMEOUT,
    CONF_INLINE_ENTITY_DESCRIPTIONS,
    CONF_LLM_API_KEY,
    CONF_LLM_BACKEND,
    CONF_LLM_BASE_URL,
    CONF_LLM_CONTROL_DEVICES,
    CONF_LLM_MODEL,
    CONF_LLM_TIMEOUT,
    CONF_MODEL,
    CONF_SYSTEM_PROMPT,
    DECISION_OPENAI,
    DECISION_TYPESAFE,
    DEFAULT_ALWAYS_CONFIRM_RISKY,
    DEFAULT_DECISION_TIMEOUT,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DOMAIN,
    LOGGER,
    TYPESAFE_CONSOLE_URL,
)
from .system_one import (
    DecisionAuthError,
    DecisionClient,
    DecisionError,
)

_BACKEND_OPTIONS = [
    SelectOptionDict(value=DECISION_TYPESAFE, label="TypeSafe hosted (Jev)"),
    SelectOptionDict(
        value=DECISION_OPENAI,
        label="OpenAI-compatible (Clef, Von, Laya, vLLM, Ollama, ...)",
    ),
]


def _decision_schema(backend: str | None) -> vol.Schema:
    """The fields that follow the backend choice.

    The hosted backend needs only an API key; the OpenAI-compatible one takes
    a base URL, a model name and an optional key. The model field is shown for
    both, because TypeSafe will host more than Jev.
    """
    fields: dict[Any, Any] = {
        vol.Required(
            CONF_DECISION_BACKEND, default=backend or DECISION_TYPESAFE
        ): SelectSelector(SelectSelectorConfig(options=_BACKEND_OPTIONS)),
    }
    if backend == DECISION_OPENAI:
        fields[vol.Required(CONF_DECISION_BASE_URL)] = TextSelector()
        fields[vol.Required(CONF_MODEL)] = TextSelector()
        fields[vol.Optional(CONF_DECISION_API_KEY)] = TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        )
        fields[
            vol.Optional(CONF_DECISION_TIMEOUT, default=DEFAULT_DECISION_TIMEOUT)
        ] = NumberSelector(
            NumberSelectorConfig(min=2, max=120, step=1, unit_of_measurement="s")
        )
    else:
        fields[vol.Optional(CONF_MODEL, default=DEFAULT_MODEL)] = TextSelector()
        fields[vol.Required(CONF_API_KEY)] = TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        )
    return vol.Schema(fields)


class TypeSafeConfigFlow(ConfigFlow, domain=DOMAIN):
    """Pick the decision backend, then the optional prose LLM."""

    VERSION = 1
    """Deliberately 1: every field added for the pluggable backends is
    optional, so entries created before them stay valid without a migration."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    @override
    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            client = _build_validator(async_get_clientsession(self.hass), user_input)
            if client is None:
                errors["base"] = "unknown"
            else:
                try:
                    await client.async_validate()
                except DecisionAuthError:
                    errors["base"] = "invalid_auth"
                except DecisionError:
                    errors["base"] = "cannot_connect"
                except Exception:
                    LOGGER.exception("Unexpected error validating the decision backend")
                    errors["base"] = "unknown"
                else:
                    self._data = dict(user_input)
                    return await self.async_step_llm()

        return self.async_show_form(
            step_id="user",
            data_schema=_decision_schema((user_input or {}).get(CONF_DECISION_BACKEND)),
            errors=errors,
            # hassfest rejects a literal URL inside a translated string, so the
            # console link is supplied here instead.
            description_placeholders={"console_url": TYPESAFE_CONSOLE_URL},
        )

    async def async_step_llm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure the LLM used for compound splits and prose answers.

        Optional: without it the agent still handles every command and query,
        it just cannot split compound requests or answer general questions.
        """
        if user_input is not None:
            self._data.update(
                {k: v for k, v in user_input.items() if v not in (None, "")}
            )
            return self.async_create_entry(
                title="TypeSafe Conversation",
                data=self._data,
                subentries=[
                    {
                        "subentry_type": "conversation",
                        "title": "TypeSafe Conversation",
                        "data": {},
                        "unique_id": None,
                    }
                ],
            )
        return self.async_show_form(step_id="llm", data_schema=STEP_LLM_SCHEMA)

    @override
    async def async_step_reauth(self, entry_data: dict[str, Any]) -> ConfigFlowResult:
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        entry = self._get_reauth_entry()
        if user_input is not None:
            data = {**entry.data, **user_input}
            client = _build_validator(async_get_clientsession(self.hass), data)
            try:
                if client is not None:
                    await client.async_validate()
            except DecisionAuthError:
                errors["base"] = "invalid_auth"
            except DecisionError:
                errors["base"] = "cannot_connect"
            else:
                return self.async_update_reload_and_abort(
                    entry, data_updates=user_input
                )
        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_API_KEY): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    )
                }
            ),
            errors=errors,
        )

    @classmethod
    @callback
    @override
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        return {"conversation": TypeSafeSubentryFlowHandler}


def _build_validator(session: Any, data: dict[str, Any]) -> DecisionClient | None:
    """Build whichever client can validate the chosen backend's credentials."""
    from .system_one import create_decision_client

    return create_decision_client(session, data)


STEP_LLM_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_LLM_BACKEND, default=BACKEND_OLLAMA): SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(value=BACKEND_OLLAMA, label="Ollama"),
                    SelectOptionDict(
                        value=BACKEND_OPENAI_COMPAT,
                        label="OpenAI-compatible (OpenRouter, vLLM, ...)",
                    ),
                ]
            )
        ),
        vol.Optional(CONF_LLM_BASE_URL, default=DEFAULT_OLLAMA_URL): TextSelector(),
        vol.Optional(CONF_LLM_MODEL): TextSelector(),
        vol.Optional(CONF_LLM_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
        vol.Optional(CONF_LLM_TIMEOUT, default=ANSWER_TIMEOUT): NumberSelector(
            NumberSelectorConfig(min=5, max=180, step=5, unit_of_measurement="s")
        ),
    }
)


class TypeSafeSubentryFlowHandler(ConfigSubentryFlow):
    """One conversation agent, with its own tuning."""

    @property
    def _is_new(self) -> bool:
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        return await self.async_step_set_options(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        return await self.async_step_set_options(user_input)

    async def async_step_set_options(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if user_input is not None:
            title = user_input.pop("name", "TypeSafe Conversation")
            if self._is_new:
                return self.async_create_entry(title=title, data=user_input)
            return self.async_update_and_abort(
                self._get_entry(), self._get_reconfigure_subentry(), data=user_input
            )

        current = {} if self._is_new else dict(self._get_reconfigure_subentry().data)
        schema = vol.Schema(
            {
                vol.Required(
                    "name",
                    default=(
                        "TypeSafe Conversation"
                        if self._is_new
                        else self._get_reconfigure_subentry().title
                    ),
                ): TextSelector(),
                vol.Optional(
                    CONF_ALWAYS_CONFIRM_RISKY,
                    default=current.get(
                        CONF_ALWAYS_CONFIRM_RISKY, DEFAULT_ALWAYS_CONFIRM_RISKY
                    ),
                ): BooleanSelector(),
                vol.Optional(
                    CONF_INLINE_ENTITY_DESCRIPTIONS,
                    default=current.get(CONF_INLINE_ENTITY_DESCRIPTIONS, False),
                ): BooleanSelector(),
                vol.Optional(
                    CONF_SYSTEM_PROMPT,
                    default=current.get(CONF_SYSTEM_PROMPT, ""),
                ): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)
                ),
                vol.Optional(
                    CONF_LLM_CONTROL_DEVICES,
                    default=current.get(CONF_LLM_CONTROL_DEVICES, False),
                ): BooleanSelector(),
                vol.Optional(
                    CONF_BYPASS_LOCAL_INTENTS,
                    default=current.get(CONF_BYPASS_LOCAL_INTENTS, False),
                ): BooleanSelector(),
            }
        )
        return self.async_show_form(step_id="set_options", data_schema=schema)
