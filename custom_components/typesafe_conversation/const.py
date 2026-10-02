"""Constants for the TypeSafe Conversation integration.

The decision layer talks to a System One API: Jev at TypeSafe by default, or
any Jev-compatible endpoint (Cloudflare Clef, a self-hosted Clef, an OpenAI-
compatible model behind the adapter) — nothing outside DEFAULT_MODEL assumes
Jev, so pointing CONF_MODEL at a later model is a config change rather than a
code change.
"""

from __future__ import annotations

import logging
from typing import Final

DOMAIN: Final = "typesafe_conversation"

CONVERSATION_DOMAIN: Final = "conversation"
"""Home Assistant's conversation domain, which is the assistant entities are
exposed to.

Imported as a constant rather than read off the ``conversation`` module: this
package has its own ``conversation.py`` platform, and forwarding the platform
setup rebinds that name on the package, so ``conversation.DOMAIN`` inside this
integration can silently become our own domain instead."""
LOGGER: Final = logging.getLogger(__package__)

# --- System One decision API ---------------------------------------------------
TYPESAFE_API_URL: Final = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_MODELS_URL: Final = "https://api.typesafe.ai/v1/models"
TYPESAFE_CONSOLE_URL: Final = "https://console.typesafe.ai/"
DEFAULT_MODEL: Final = "jev-latest"
"""Jev is the only hosted System One model today. An alias, so it follows releases."""

API_TIMEOUT: Final = 6.0
API_MAX_RETRIES: Final = 3
API_BACKOFF: Final = (0.25, 0.75, 2.0)

# Circuit breaker: after this many consecutive failures, stop calling the API for
# CIRCUIT_RESET_SECONDS and serve the fallback ladder instead.
CIRCUIT_FAILURE_THRESHOLD: Final = 3
CIRCUIT_RESET_SECONDS: Final = 60.0

# --- Config keys -------------------------------------------------------------
CONF_API_KEY: Final = "api_key"
CONF_MODEL: Final = "model"
CONF_DECISION_BACKEND: Final = "decision_backend"
CONF_DECISION_BASE_URL: Final = "decision_base_url"
CONF_DECISION_API_KEY: Final = "decision_api_key"
CONF_DECISION_TIMEOUT: Final = "decision_timeout"
CONF_LLM_BACKEND: Final = "llm_backend"
CONF_LLM_BASE_URL: Final = "llm_base_url"
CONF_LLM_MODEL: Final = "llm_model"
CONF_LLM_API_KEY: Final = "llm_api_key"
CONF_LLM_REFERER: Final = "llm_referer"
CONF_LLM_TITLE: Final = "llm_title"
CONF_LLM_CONTROL_DEVICES: Final = "llm_control_devices"
CONF_BYPASS_LOCAL_INTENTS: Final = "bypass_local_intents"
CONF_SYSTEM_PROMPT: Final = "system_prompt"
CONF_INLINE_ENTITY_DESCRIPTIONS: Final = "inline_entity_descriptions"
CONF_ALWAYS_CONFIRM_RISKY: Final = "always_confirm_risky"
CONF_LLM_TIMEOUT: Final = "llm_timeout"

DEFAULT_ALWAYS_CONFIRM_RISKY: Final = True
"""Ask before unlocking or opening the house, however sure the model is.

The model answers "unlock the front door" at confidence 1.0, so the confidence gate
below would let it through silently. One extra turn is cheap; an unlock the
user did not intend is not. Turn this off to get the pure confidence gate."""

DECISION_TYPESAFE: Final = "typesafe"
DECISION_OPENAI: Final = "openai"

DEFAULT_DECISION_BASE_URL: Final = TYPESAFE_API_URL
"""Only used by the OpenAI-compatible decision backend."""
DEFAULT_DECISION_TIMEOUT: Final = 12.0
"""Seconds for one decision request.

A hosted System One model answers in a few hundred milliseconds; a large
self-hosted model on shared hardware needs more. The retries below multiply
this, so the worst case is roughly timeout x API_MAX_RETRIES before the
fallback ladder takes over.
"""

DEFAULT_OLLAMA_URL: Final = "http://localhost:11434"
DEFAULT_OPENAI_COMPAT_URL: Final = "https://openrouter.ai/api"
BACKEND_OLLAMA: Final = "ollama"
BACKEND_OPENAI_COMPAT: Final = "openai_compatible"
DEFAULT_LLM_REFERER: Final = (
    "https://github.com/maglat/home-assistant-typesafe-conversation-agent"
)
DEFAULT_LLM_TITLE: Final = "HA TypeSafe Conversation"

# --- LLM behaviour -----------------------------------------------------------
SPLIT_TIMEOUT: Final = 4.0
"""Floor for the small utility prompts (split, rewrite).

The actual budget is max(SPLIT_TIMEOUT, answer_timeout / 3): a user who
raised the answer timeout for a busy shared server raised this one
implicitly. The old hardcoded 4s assumed a dedicated server; behind a
queue - one GPU serving several clients - even a 300-token request can
wait longer than that.
"""
ANSWER_TIMEOUT: Final = 30.0
"""Seconds to wait for a freeform answer.

A large local model can take well over the old 20s, especially on the first
call after a restart. Raise it with CONF_LLM_TIMEOUT, or point the LLM at a
smaller model - this path is only used for prose, so it does not need to be
the same model you would pick for reasoning."""
SPLIT_MAX_TOKENS: Final = 512
ANSWER_MAX_TOKENS: Final = 1024
"""Token caps for the prose paths.

Reasoning models (GLM, Qwen-thinking, DeepSeek-R1-style) spend their budget
on hidden reasoning before writing the visible answer - a 180-token cap was
consumed entirely by reasoning, returning an empty reply. The caps only bound
runaway generation; normal answers stay far below them."""
ANSWER_TEMPERATURE: Final = 0.3
MAX_SUB_COMMANDS: Final = 6
OLLAMA_KEEP_ALIVE: Final = "30m"
WARMUP_INTERVAL_SECONDS: Final = 20 * 60
MAX_TOOL_ROUNDS: Final = 8
"""How many tool-call rounds the freeform LLM may chain.

One round is one model response; a round that calls tools feeds the results
back and asks again. Eight is generous - Home Assistant's own agents cap at
ten - and the cap is what stops a confused model from looping forever.
"""

PROMPT_LOG_CHARS: Final = 200
"""How much of a system prompt to write to the debug log.

The freeform prompt embeds the home catalog - entity names, areas and current
states - and home-assistant.log is what gets pasted into issue reports."""

# --- Catalog -----------------------------------------------------------------
# The binding constraint is the API's 255-option cap on a Choice, not tokens.
MAX_CHOICE_OPTIONS: Final = 250
BEAM_WIDTH: Final = 2
"""Domain/area paths kept when a home is too large for one entity question.

Ported from upstream's hierarchical-classification work: when a home has more
exposed entities than the server's option cap, the first request classifies
domain and area, and a second, narrower request picks the entity among the
surviving paths. The beam keeps the two most probable paths.
"""
MAX_HISTORY_TURNS: Final = 6
CATALOG_SUMMARY_MAX_ENTITIES: Final = 120
TRACE_HISTORY: Final = 20
"""How many recent request traces to keep for the diagnostics download."""

# --- Routing thresholds ------------------------------------------------------
# Every value below is a first guess calibrated from the TypeSafe demo's
# described behaviour. Re-fit them with scripts/calibrate.py against a live key
# on a real home before relying on them.
MIN_MARGIN: Final = 0.15
"""Minimum p(top) - p(second) for a Choice to count as solid."""

CONF_ACT_TERSE: Final = 0.75
"""At or above this, act and acknowledge briefly."""

CONF_ACT_EXPLICIT: Final = 0.50
"""At or above this, act but name the target explicitly in the speech."""

CONF_CLARIFY_FLOOR: Final = 0.30
"""Below this a target is not worth clarifying; fall back instead."""

T_CATEGORY_CANCEL: Final = 0.60
T_CATEGORY_INFORMATION: Final = 0.55
T_CATEGORY_UNCLEAR: Final = 0.55
T_CATEGORY_QUERY: Final = 0.45
T_CATEGORY_COMPOUND: Final = 0.50
T_QUERY_KIND: Final = 0.50
T_DOMAIN: Final = 0.50
T_ENTITY: Final = 0.60
T_AREA: Final = 0.55
T_SCOPE: Final = 0.55
T_SCOPE_WHOLE_HOUSE: Final = 0.60
T_ACTION: Final = 0.40
"""Kept for reference; the action gate uses T_ACTION_PROBABILITY instead."""

T_ACTION_PROBABILITY: Final = 0.55
"""Minimum probability for the chosen action.

Gating the action on *probability* rather than confidence, because confidence is
derived as (n * p_top - 1) / (n - 1) and so depends on how many options the
question offered. `action_script` has two options (run / not_targeted), so
p=0.66 scores only 0.31 confidence, while the same 0.66 on the seven-option
`action_light` scores 0.60. Thresholding confidence therefore systematically
under-trusts the small-option domains - scripts, scenes and buttons - which on a
script-heavy home is most of the traffic. Probability is comparable across
option counts."""

# Noul dead bands. Between the low and high value we treat the answer as "no"
# and log for calibration, because a Noul near 0.5 means the model splits
# yes/no - not that the condition half-holds.
NOUL_COMPOUND_HIGH: Final = 0.70
NOUL_COMPOUND_LOW: Final = 0.45
NOUL_HERE_RELATIVE: Final = 0.60
NOUL_RISKY: Final = 0.50
"""The risky question alone decides whether to confirm.

It used to be ANDed with an allowlist of risky actions, which exempted every
script: a script's action is always "run", so no script could reach the gate
whatever it did. Security actions are commonly implemented as scripts, and
those would have run unconfirmed. Measured against jev-1.13.0, the question
separates them on its own: "disarm the alarm" scores 0.95 and "open the
driveway gate" 0.96, while "arm the alarm in home mode" scores 0.03."""

NOUL_RISKY_EXPLICIT: Final = 0.30
"""Lower bar when the action itself is literally an unlock, open or disarm.

The allowlist is a floor now, never a filter - it can only make the gate fire
more readily, never suppress it."""

# A risky action (unlocking, opening an exterior door) needs more than the
# normal confidence before we do it without asking.
T_RISKY_ACTION: Final = 0.85
T_RISKY_TARGET: Final = 0.75
RISKY_ACTIONS: Final = frozenset({"unlock", "open", "disarm"})
RISKY_DOMAINS: Final = frozenset({"lock", "cover"})
RISKY_DEVICE_CLASSES: Final = frozenset({"garage", "gate", "door", "window"})

# Fall back to the LLM for a freeform answer if the category leaned at all
# towards something the LLM could answer.
T_LLM_LEAN: Final = 0.25

# --- Relative-change magnitudes (percentage points) --------------------------
MAGNITUDE_STEPS: Final = {
    "slight": 10,
    "moderate": 25,
    "large": 50,
    "maximum": 100,
}
