"""The TypeSafe Conversation integration.

Puts a TypeSafe System One model in front of the LLM as the decision layer:
device commands and state queries resolve from typed answers in code, and only genuinely
generative work - splitting compound requests, answering general questions -
reaches an LLM.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_track_time_interval

from .const import (
    CONF_DECISION_BACKEND,
    CONF_MODEL,
    CONVERSATION_DOMAIN,
    DECISION_TYPESAFE,
    DEFAULT_MODEL,
    DOMAIN,
    TRACE_HISTORY,
    WARMUP_INTERVAL_SECONDS,
)
from .entities import EntityCatalog
from .llm_backend import LLMBackend, create_backend
from .system_one import (
    DecisionAuthError,
    DecisionClient,
    DecisionError,
    OpenAIDecisionClient,
    create_decision_client,
)

PLATFORMS = [Platform.CONVERSATION]
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


@dataclass
class TypeSafeRuntimeData:
    """Everything one config entry needs at runtime."""

    client: DecisionClient
    catalog: EntityCatalog
    llm: LLMBackend | None
    model: str
    questions_cache: tuple[int, bool, dict[str, Any]] | None = None
    """Shared across turns: the question set is a pure function of the catalog."""

    traces: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=TRACE_HISTORY)
    )
    """Recent request traces, newest last.

    Home Assistant defines conversation traces but nothing reads them back -
    there is no websocket command and no UI - so they are write-only. Keeping
    our own ring buffer is what makes the diagnostics download useful."""


type TypeSafeConfigEntry = ConfigEntry[TypeSafeRuntimeData]


async def async_setup_entry(hass: HomeAssistant, entry: TypeSafeConfigEntry) -> bool:
    """Set up TypeSafe Conversation from a config entry."""
    session = async_get_clientsession(hass)
    client = create_decision_client(session, {**entry.data})
    if client is None:
        raise ConfigEntryNotReady("No usable decision backend configured")

    # Only the hosted backend has a key to check; a self-hosted endpoint is
    # validated by reachability, and a failure there is worth a retry.
    if entry.data.get(CONF_DECISION_BACKEND, DECISION_TYPESAFE) == DECISION_TYPESAFE:
        try:
            await client.async_validate()
        except DecisionAuthError as err:
            raise ConfigEntryAuthFailed(str(err)) from err
        except DecisionError as err:
            raise ConfigEntryNotReady(f"Could not reach TypeSafe: {err}") from err
    elif isinstance(client, OpenAIDecisionClient):
        try:
            await client.async_validate()
        except DecisionError as err:
            raise ConfigEntryNotReady(
                f"Could not reach the decision model: {err}"
            ) from err

    store = hass.data.setdefault(DOMAIN, {})
    catalog: EntityCatalog | None = store.get("catalog")
    if catalog is None:
        # One catalog for the whole instance: it describes Home Assistant, not
        # any particular config entry.
        catalog = EntityCatalog(hass, CONVERSATION_DOMAIN)
        catalog.async_start()
        store["catalog"] = catalog

    llm = create_backend(session, {**entry.data})
    entry.runtime_data = TypeSafeRuntimeData(
        client=client,
        catalog=catalog,
        llm=llm,
        model=entry.data.get(CONF_MODEL, DEFAULT_MODEL),
    )

    if llm is not None:
        # A cold model load is seconds long and would be blamed on us, so pay
        # for it in the background and keep paying every 20 minutes.
        entry.async_create_background_task(
            hass, llm.async_warm_up(), "typesafe_llm_warmup", eager_start=False
        )

        async def _async_warm_up(_now: datetime) -> None:
            """Keep the model resident.

            This must be a coroutine function. async_track_time_interval
            classifies its action as a HassJob, and a plain sync callable is
            run in an executor thread - from which hass.async_create_task is
            not safe to call.
            """
            await llm.async_warm_up()

        entry.async_on_unload(
            async_track_time_interval(
                hass,
                _async_warm_up,
                timedelta(seconds=WARMUP_INTERVAL_SECONDS),
                name="typesafe_llm_warmup",
            )
        )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: TypeSafeConfigEntry) -> bool:
    """Unload a config entry."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded and not [
        other
        for other in hass.config_entries.async_entries(DOMAIN)
        if other.entry_id != entry.entry_id
    ]:
        # Last entry gone: stop listening for registry changes.
        store = hass.data.get(DOMAIN, {})
        if (catalog := store.pop("catalog", None)) is not None:
            catalog.async_stop()
    return unloaded


async def _async_update_listener(
    hass: HomeAssistant, entry: TypeSafeConfigEntry
) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


__all__ = ["TypeSafeConfigEntry", "TypeSafeRuntimeData"]
