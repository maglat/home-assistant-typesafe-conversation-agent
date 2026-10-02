# TypeSafe Conversation for Home Assistant

[![tests](https://github.com/maglat/home-assistant-typesafe-conversation-agent/actions/workflows/test.yml/badge.svg)](https://github.com/maglat/home-assistant-typesafe-conversation-agent/actions/workflows/test.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![HACS: custom](https://img.shields.io/badge/HACS-custom-orange.svg)](https://hacs.xyz/docs/faq/custom_repositories/)

A Home Assistant conversation agent that decides with a
[TypeSafe System One](https://docs.typesafe.ai) model instead of an LLM — or
with any OpenAI-compatible model you point it at.

A System One model returns typed, calibrated judgements rather than text. Every
device command and state query is resolved from those judgements in ordinary
Python, in a single API call. An LLM is used for exactly two things — splitting
a request that contains several commands, and answering a general question — so
the slow path is only taken when something actually has to be written in prose.

Measured against the same utterances, this replaces a 2–8 s LLM turn with a
~250 ms one (hosted model) or ~1 s (a local GPU model) for anything that
controls or inspects the home.

> Found a bug? Please [report it](../../issues/new?template=bug_report.yml).

## Features

- **Fast, deterministic device control** — commands and state queries are
  resolved from typed judgements in code, never by free-text parsing.
- **Pluggable decision backend** — hosted TypeSafe Jev, or any
  OpenAI-compatible endpoint: a self-hosted
  [Kev](https://github.com/jaredpalmer/kev),
  [Clef](https://huggingface.co/Cloudflare/clef),
  [Von](https://github.com/wfzyx/von),
  [Laya](https://huggingface.co/convaiinnovations/laya), or a plain LLM served
  by vLLM, llama.cpp, Ollama or TabbyAPI. A System One endpoint configured
  under the OpenAI-compatible backend is detected automatically.
- **Optional language model** — Ollama or any OpenAI-compatible endpoint, used
  only for compound splitting and prose answers. Reasoning models (GLM,
  Qwen-thinking, DeepSeek-R1-style) are handled: think blocks are stripped and
  token budgets account for hidden reasoning.
- **Follow-up resolution** — "and back off again" is rewritten into a
  standalone command with the conversation in view, then routed normally.
- **Compound commands** — "turn off the lights and lock the door" is split,
  executed in order, and reported per part.
- **Calibrated routing** — every branch is gated on confidence *and* margin;
  clarification is asked only for an ambiguous target; risky actions (unlock,
  open, disarm) always confirm first.
- **Optional device control for the LLM** — the prose path can gain Home
  Assistant's Assist tools, restricted to exposed entities. Off by default.
- **Localised acknowledgements** — command confirmations follow the language
  of the request (English and German built in).
- **Safe degradation** — a circuit breaker, fail-fast timeouts and a layered
  fallback mean a dead or slow model never blocks a command the sentence
  matcher can handle.
- **Privacy-conscious diagnostics** — API keys, base URLs and entity IDs are
  redacted in the diagnostics download.

## How it works

```
utterance
   │   Home Assistant's own sentence matcher has already taken the phrasings
   │   it recognises; what reaches us is the fuzzy and the compound.
   ▼
entity catalog  ── cached, rebuilt on registry and exposure changes
   ▼
ONE decision call:  state     = the request + every exposed entity and area
                    questions = 8 fixed + 3 built from the catalog
                              + 1 per device kind present + a few conditional ones
   ▼
route()  ── a pure function over the answers
   ├─ command   → intent.async_handle → your services (risky ones ask first)
   ├─ query     → sentence matcher, then HassGetState / HassClimateGetTemperature
   ├─ compound  → LLM splits it → N decision calls → run in order
   ├─ general   → LLM answers in prose
   └─ unsure    → fallback ladder (below)
```

### The fallback ladder

When the decision model is not confident enough, the request walks a ladder —
each rung is tried only if the previous one did not resolve:

1. **Sentence matcher** — Home Assistant's own matcher gets a second chance.
2. **Context rewrite** — with conversation history present, the LLM rewrites
   the utterance into a standalone command ("and back off again" + the earlier
   "turn on the kitchen light" → "turn off the kitchen light") and the decision
   model runs once more on the result. This runs whenever there is history,
   regardless of how the category leaned.
3. **LLM with tools** — only if *Let the language model control devices* is
   enabled; the LLM may call Assist tools for the exposed entities.
4. **Prose answer** — the LLM answers the utterance as a general question.
5. **Apology** — an honest "I'm not sure", never a guessed action.

A timeout short-circuits the ladder: if the rewrite timed out, the prose answer
against the same busy server is skipped — the outcome is already decided.

### Speculative fan-out

Every question the router might need is asked up front, including the branches
that will turn out to be irrelevant. An extra question costs about **97 input
tokens** and almost no extra latency, while a second round trip would cost
hundreds of milliseconds. So we ask "what should happen to the locks?" on every
single request and simply do not read the answer unless the request turned out
to be about locks.

A confident but wrong answer on a branch nobody reads is free. That is the
contract, and there is a test that holds it: `is everything locked up` returns
`action_lock = lock` at 0.99 confidence, and the router correctly treats the
utterance as a question and locks nothing.

### Two axes, not one

A branch is acted on only when **confidence** clears its bar *and* the
**margin** between the top two options is at least 0.15. They fail differently:
a distribution can look confident overall while the top two options remain
effectively tied.

| confidence | behaviour |
| --- | --- |
| ≥ 0.75 | act, brief acknowledgement |
| 0.50–0.75 | act, but name the target out loud, so a wrong guess is correctable |
| 0.30–0.50 (target only) | ask which device was meant |
| below, or a near-tie | hand back to the sentence matcher, then the LLM |

Clarification is only ever used for an ambiguous *target*. "Which lamp?" is a
short, natural question. "Did you want to turn something on?" is not, so a weak
category or action goes to the fallback ladder instead.

### Values

The decision model never invents a number. Code finds the candidate spans in
the utterance and the model picks which one the user meant:

- `"set the living room lights to 30%"` → candidates `["30%"]` → picked → `brightness_pct: 30`
- `"close the blinds halfway"` → `halfway` → `position: 50`
- `"turn the volume down a bit"` → no value in the text, so a direction and a
  magnitude question → `volume_step: -10`
- `"make the lights a bit warmer"` → a closed set of colour names → `2700 K`

Temperature units come from the entity, or failing that from your Home
Assistant configuration. They are never asked of the model.

## Decision backends

The decision layer is pluggable. Pick the backend when adding the integration:

| backend | what it talks to | needs |
| --- | --- | --- |
| **TypeSafe hosted** | `https://api.typesafe.ai/v1/systemone` — Jev, and whatever TypeSafe hosts later | an API key |
| **OpenAI-compatible** | any `/v1/chat/completions` endpoint: a self-hosted Kev, Clef, Von or Laya, or a plain LLM served by vLLM, llama.cpp, Ollama, TabbyAPI, … | a base URL and a model name; an API key only if the endpoint wants one |

The OpenAI-compatible backend sends the same state and question schema in one
prompt and expects one JSON object back, keyed by question id, with a
probability distribution per question. Confidence is derived from those
probabilities with Jev's own formula, so every routing threshold behaves the
same whichever backend answers. Models that answer loosely (bare labels
instead of distributions, fenced JSON, prose around the object) are parsed
tolerantly; an answer that cannot be repaired is treated as "not confident"
and falls back, never as a wrong command.

**Automatic protocol detection:** if an endpoint configured under the
OpenAI-compatible backend answers `/v1/models` with a System One payload (the
shape Kev and Clef serve), the integration transparently serves every decision
through the System One protocol instead. Either backend works against either
server type — pick whichever matches how you think about your server.

Base URLs are accepted with or without a trailing `/v1` — `http://host:8000`
and `http://host:8000/v1` both work.

## Language model

The optional LLM covers exactly two jobs: splitting compound requests and
answering general questions. It is never given device control unless you opt
in, and it is never on the fast path.

| backend | what it talks to |
| --- | --- |
| **OpenAI-compatible** (default) | any `/v1/chat/completions` endpoint: TabbyAPI, vLLM, llama.cpp, LM Studio, OpenRouter, … |
| **Ollama** | Ollama's native `/api/chat` |

Reasoning models are supported: `<think>` blocks (including unclosed ones cut
off by the token cap) are stripped from replies, `reasoning_content` is ignored,
and the token budgets are sized so hidden reasoning cannot eat the visible
answer.

Timeouts: the answer timeout (default 30 s, configurable) bounds the prose
path; the small utility prompts (split, rewrite) get `max(4 s, answer
timeout / 3)`. Decision timeouts are fail-fast — a timed-out request is not
retried, because a slow model stays slow and the fallback ladder serves the
user faster than a second timeout would. After three consecutive failures the
circuit breaker pauses calls for 60 s and serves the fallback ladder instead.

## What gets sent

On every request this integration sends:

- the text of the utterance
- **every entity you have exposed to Assist** — its name, area, floor, domain,
  device class and current state
- the areas and floors in your home, by name

That is the whole point of the design: the model is given the home as state and
answers questions about it, rather than being asked to write code or call tools.
But it means your device and room names, and what is currently on or off, leave
your network when the hosted backend is used. If that is not acceptable to you,
point both backends at self-hosted models and nothing leaves the house.

What does **not** happen: nothing is stored by this project, there is no
telemetry of its own, and the optional LLM backend is configured separately —
point it at a server on your own machine and the prose path never leaves the
house either. Diagnostics downloads redact the API keys, the base URLs and
your entity IDs.

TypeSafe's own handling of what you send is governed by their
[terms](https://typesafe.ai/legal/mca) and
[data processing agreement](https://typesafe.ai/data-processing), not by this
project.

## Cost

Applies to the hosted backend only; self-hosted endpoints are free.

Input tokens only. State is billed once per request rather than once per
question — it is what makes the speculative fan-out cheap: asking a question
you end up discarding costs almost nothing. A 29-entity home came to ~6.5k
tokens per request against `jev-1.13.0`, about **$0.00027** per request at the
then-current rate. Check [your console](https://console.typesafe.ai/) for what
you will actually be billed.

## Install

**Requires Home Assistant 2026.5.0 or newer.** Earlier releases do not report a
failed service call back to the conversation agent, so a command that no entity
could carry out would be announced as if it had worked.

### Through HACS

Not in the default HACS store, so add it as a custom repository: in HACS,
**⋮ → Custom repositories**, paste

```
https://github.com/maglat/home-assistant-typesafe-conversation-agent
```

choose category **Integration**, then **Add**. It will then appear in HACS for
install and for update notifications.

### By hand

Copy `custom_components/typesafe_conversation` into your Home Assistant `config`
directory.

### Setup

1. Restart Home Assistant.
2. **Settings → Devices & Services → Add Integration → TypeSafe Conversation.**
3. **Decision backend** — pick one:
   - *TypeSafe hosted (Jev)*: enter your API key from
     [console.typesafe.ai](https://console.typesafe.ai/).
   - *OpenAI-compatible*: enter the base URL (`http://homeassistant.local:8000`)
     and the model name your endpoint serves (`kev-latest`, `clef-flash`, …).
     Local servers need no API key. A System One endpoint is detected
     automatically.
4. **Language model** (optional) — pick *OpenAI-compatible* or *Ollama*, enter
   base URL and model, and raise the answer timeout if your server is shared.
   Skip this step to run without an LLM: every command and query still works,
   you only lose compound splitting and prose answers.
5. Set it as the conversation agent under **Settings → Voice assistants**.

## Options

Entry options are set in the config flow and changeable afterwards under
**Settings → Devices & Services → TypeSafe Conversation → Configure**. Options
marked ◆ live on each conversation agent (subentry) and are edited per agent.

| option | default | what it does |
| --- | --- | --- |
| `decision_backend` | `typesafe` | `typesafe` (hosted Jev) or `openai` (any OpenAI-compatible endpoint; System One endpoints are detected automatically). |
| `api_key` | — | Your TypeSafe API key. Required for the hosted backend. |
| `model` | `jev-latest` | Which model to use — a TypeSafe model name, or the model name your own endpoint serves (`kev-latest`, `clef-flash`, …). |
| `decision_base_url` | — | Root of your OpenAI-compatible endpoint, with or without `/v1`. |
| `decision_api_key` | — | Only if that endpoint needs one. Redacted in diagnostics. |
| `decision_timeout` | `12` s | Seconds for one decision request. Self-hosted models need more than the hosted API. |
| ◆ `always_confirm_risky` | **on** | Ask before unlocking a door, opening a garage or disarming an alarm, however sure the model is. **Turning this off lets confident requests through silently.** |
| ◆ `bypass_local_intents` | off | Send every command here, including ones Home Assistant's own sentence matcher recognises. Off is recommended; see [below](#prefer-handling-commands-locally). |
| ◆ `inline_entity_descriptions` | off | Describe every entity inside each question rather than once in the shared state. Roughly doubles the tokens. Only worth it if the agent picks the wrong device. |
| ◆ `system_prompt` | — | Extra instructions for prose answers — a persona, a language rule, how brief to be. Appended after the built-in guardrails. |
| ◆ `llm_control_devices` | off | Give the language model Assist tools so it can act on exposed devices when the decision model is unsure. |
| `llm_backend` | `openai_compatible` | `openai_compatible` or `ollama`. Used only for compound requests and general questions. |
| `llm_base_url` | — | Where that backend lives, with or without `/v1`. |
| `llm_model` | — | Model name on that backend. |
| `llm_api_key` | — | If the backend needs one. Redacted in diagnostics. |
| `llm_timeout` | `30` s | How long to wait for a prose answer before giving up. The split/rewrite prompts get a third of this, at least 4 s. |
| `llm_referer`, `llm_title` | project defaults | Sent as `HTTP-Referer` and `X-Title`; OpenRouter uses them for attribution. |

### Prefer handling commands locally

The agent advertises `ConversationEntityFeature.CONTROL`. With prefer-local on,
Home Assistant's sentence matcher keeps every phrasing it recognises, and only
`HassGetState`, `HassMediaSearchAndPlay` and everything it *failed* to parse
reach this agent. That is deliberate: there is no point spending an API call
beating hassil at "turn off the kitchen lights", and it concentrates the traffic
on the requests where the decision model earns its keep. Set
`bypass_local_intents` if you want to compare the two regimes.

### Follow-ups ("and back off again")

The decision model is a single-pass classifier, so a follow-up that only makes
sense with the previous turns in view scores as unclear. Whenever conversation
history exists, the LLM rewrites the utterance into a standalone command with
that history in view and the decision model runs once more on the result. The
fast path is untouched: follow-ups cost one extra LLM round trip, direct
commands none.

If the rewrite comes back as "not a device command" (a question, a new topic),
the prose answer takes over instead.

### Let the language model control devices

Off by default, the prose path is read-only: it can answer in text but never
act. With **Let the language model control devices** enabled, the fallback
path gains Home Assistant's Assist tools — one tool per intent, restricted to
the entities exposed to Assist, executed under the requesting user's context.
That is the same permission model as Home Assistant's own OpenAI/Ollama
agents: the LLM can chain tool calls ("turn off the kitchen and then set the
thermostat to 20") for requests the decision model could not place.

The decision model stays the fast path; the tool loop only runs when the
decision model was not confident enough, and a failed tool call falls back to
the read-only prose answer.

## Debugging

### Is the agent even being asked?

With *Prefer handling commands locally* on, Home Assistant's own sentence
matcher keeps every phrasing it recognises, so a well-formed command like
"turn off the living room light" never reaches this integration. In the
**Settings → Voice assistants → Debug** trace that shows up as:

```
processed_locally: true
Natural language processing   0.01s
```

That is the design working, not a failure. A request this agent handled looks
like `processed_locally: false`. To exercise it, use phrasings the matcher
cannot parse — "it's too dark in here", "get the coffee going", "add milk to
the shopping list", "turn off the lamp and start the vacuum" — or turn on
`bypass_local_intents` to send everything here.

### Download diagnostics

**Settings → Devices & Services → TypeSafe Conversation → ⋮ → Download
diagnostics** gives the last 20 requests with the route taken, the reason, the
resolved target, and every answer's probability distribution and confidence:

```json
{
  "utterance": "get the coffee going",
  "route": "command", "reason": "switch.turn_on",
  "target": "switch.coffee_maker",
  "category":      {"choice": "command", "confidence": 1.0,  "margin": 1.0},
  "target_entity": {"choice": "switch.coffee_maker", "confidence": 1.0},
  "action":        {"choice": "turn_on", "confidence": 0.99,
                    "top": {"turn_on": 0.99, "not_targeted": 0.01}},
  "latency_ms": 244, "input_tokens": 6482
}
```

API keys and base URLs are redacted, and the catalog is reported as counts
rather than entity ids, so the file is safe to attach to an issue.

### Debug log

```yaml
# configuration.yaml
logger:
  default: warning
  logs:
    custom_components.typesafe_conversation: debug
```

or, without a restart, **Developer Tools → Actions → `logger.set_level`** with
`custom_components.typesafe_conversation: debug`. Each request then logs the
route, latency and token count, followed by one line per answer:

```
Routed 'get the coffee going' -> command (switch.turn_on) in 244ms, 6482 input tokens
  category         command                      conf 1.00 margin 1.00  {'command': 1.0}
  target_entity    switch.coffee_maker          conf 1.00 margin 1.00  {...}
  action           turn_on                      conf 0.99 margin 0.98  {...}
```

### If the LLM path times out

`LLM could not answer: Timed out after 30.0s` means the model backing the prose
path is too slow — often because a shared server was busy with another client.
That path only writes sentences, so it does not need to be the model you would
choose for reasoning. Raise `llm_timeout` in the integration options if you
would rather wait; the split and rewrite budgets scale with it.

## Development

```sh
python3.14 -m venv .venv        # Home Assistant 2026.5+ requires Python 3.14
.venv/bin/pip install "pytest-homeassistant-custom-component==0.13.348" syrupy ruff
.venv/bin/pip install "hassil==3.8.0" "home-assistant-intents==2026.6.24"
.venv/bin/python -m pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

CI runs the suite against both the oldest supported Home Assistant and the
current one, plus Home Assistant's `hassfest` and the HACS validator. The
routing tests replay real recorded model responses from `tests/fixtures/`, so
they describe how the models actually behave rather than how we imagine they
do. See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

### A note on confidence

Jev derives confidence as `(n * p_top - 1) / (n - 1)`, so the *same* probability
scores differently depending on how many options the question offered. A
two-option Choice at p=0.66 scores 0.31; a seven-option Choice at the same
probability scores 0.60. The action gate therefore thresholds the chosen
option's **probability**, not its confidence — thresholding confidence made
every script command fall back on a script-heavy home.