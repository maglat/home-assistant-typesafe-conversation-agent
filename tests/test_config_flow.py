"""The config flow's step sequence.

The reported bug: a single form whose fields changed with the backend select
only re-rendered after a submit, so switching to OpenAI-compatible showed no
URL field until a failed submit forced a re-render. These tests pin the
two-step shape that fixes it: pick the backend, then see its fields.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from homeassistant.config_entries import SOURCE_USER
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.typesafe_conversation.const import (
    CONF_DECISION_API_KEY,
    CONF_DECISION_BACKEND,
    CONF_DECISION_BASE_URL,
    CONF_MODEL,
    DECISION_OPENAI,
    DOMAIN,
)
from custom_components.typesafe_conversation.system_one import (
    DecisionUnavailableError,
)


@pytest.fixture(autouse=True)
def _custom_integrations(enable_custom_integrations):
    return None


async def test_user_step_offers_the_backend_choice(hass: HomeAssistant):
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"
    # Only the selector is on the first form - no URL, no key.
    assert list(result["data_schema"].schema) == [CONF_DECISION_BACKEND]


async def test_choosing_openai_shows_the_url_fields_immediately(hass: HomeAssistant):
    """The fix: the next form carries the endpoint fields, no submit needed."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_DECISION_BACKEND: DECISION_OPENAI}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "openai"

    field_keys = [str(k) for k in result["data_schema"].schema]
    assert CONF_DECISION_BASE_URL in field_keys
    assert CONF_MODEL in field_keys
    assert CONF_DECISION_API_KEY in field_keys


async def test_choosing_typesafe_asks_for_the_key(hass: HomeAssistant):
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_DECISION_BACKEND: "typesafe"}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "typesafe"
    field_keys = [str(k) for k in result["data_schema"].schema]
    assert "api_key" in field_keys


async def test_openai_step_validates_and_reaches_the_llm_step(
    hass: HomeAssistant,
):
    """A reachable endpoint carries the flow on to the optional LLM step."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_DECISION_BACKEND: DECISION_OPENAI}
    )

    with patch(
        "custom_components.typesafe_conversation.config_flow._build_validator"
    ) as validator:
        # async_validate is a coroutine on the real client.
        validator.return_value.async_validate = _fake_validate(["clef-flash"])
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_DECISION_BASE_URL: "http://127.0.0.1:8010",
                CONF_MODEL: "kev-latest",
            },
        )

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "llm"


async def test_unreachable_endpoint_shows_an_error(hass: HomeAssistant):
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_DECISION_BACKEND: DECISION_OPENAI}
    )

    with patch(
        "custom_components.typesafe_conversation.config_flow._build_validator"
    ) as validator:
        validator.return_value.async_validate = _fake_validate_error(
            DecisionUnavailableError("down")
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_DECISION_BASE_URL: "http://127.0.0.1:9",
                CONF_MODEL: "kev-latest",
            },
        )

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "openai"
    assert result["errors"] == {"base": "cannot_connect"}


def _fake_validate(models: list[str]):
    async def _validate():
        return models

    return _validate


def _fake_validate_error(err: Exception):
    async def _validate():
        raise err

    return _validate


@pytest.mark.parametrize(
    ("backend", "step"),
    [("typesafe", "typesafe"), (DECISION_OPENAI, "openai")],
)
async def test_backend_steps_route_to_the_llm_step(hass, backend, step):
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_DECISION_BACKEND: backend}
    )
    assert result["step_id"] == step
