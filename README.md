# TypeSafe Conversation for Home Assistant

[![tests](https://github.com/the-sof/home-assistant-typesafe-conversation-agent/actions/workflows/test.yml/badge.svg)](https://github.com/the-sof/home-assistant-typesafe-conversation-agent/actions/workflows/test.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![HACS: custom](https://img.shields.io/badge/HACS-custom-orange.svg)](https://hacs.xyz/docs/faq/custom_repositories/)

A Home Assistant conversation agent that decides with a
[TypeSafe System One](https://docs.typesafe.ai) model instead of an LLM — or
with any OpenAI-compatible open-source model you point it at.

> **Status: early.** Expect rough edges, and please
> [report them](../../issues/new?template=bug_report.yml). The default backend
> uses a [TypeSafe](https://console.typesafe.ai/) API key, which is metered —
> see [Cost](#cost). A self-hosted
> [Clef](https://huggingface.co/Cloudflare/clef),
> [Von](https://github.com/wfzyx/von) or
> [Laya](https://huggingface.co/convaiinnovations/laya) endpoint works just as
> well and is free.

A System One model returns typed, calibrated judgements rather than text. Jev is
the one available today and the default; the integration is not written around
it, so a later model is a config change. Every device command and state query
is resolved from those judgements in ordinary Python, in a single API call. An
LLM is used for exactly two things — splitting a request that contains several
commands, and answering a general question — so the slow path is only taken when
something actually has to be written in prose.

Measured against the same utterances, this replaces a 2–8 s LLM turn with a
~250 ms one for anything that controls or inspects the home.

## How it works

```
utterance
   │   Home Assistant's own sentence matcher has already taken the phrasings
   │   it recognises; what reaches us is the fuzzy and the compound.
   ▼
entity catalog  ── cached, rebuilt on registry and exposure changes
   ▼
ONE System One call:  state    = the request + every exposed entity and area
               questions = 8 fixed + 3 built from the catalog
                         + 1 per device kind present + a few conditional ones
   ▼
route()  ── a pure function over the answers
   ├─ command   → intent.async_handle → your services
   ├─ query     → sentence matcher, then HassGetState / HassClimateGetTemperature
   ├─ compound  → LLM splits it → N parallel Jev calls → run in order
   ├─ general   → LLM answers in prose
   └─ unsure    → sentence matcher → LLM → "I'm not sure"
```

### Speculative fan-out

Every question the router might need is asked up front, including the branches
that will turn out to be irrelevant. Measured against `jev-1.13.0`, an extra
question costs about **97 input tokens** and almost no extra latency, while a
second round trip would cost ~250 ms. So we ask "what should happen to the
locks?" on every single request and simply do not read the answer unless the
request turned out to be about locks.

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

Jev cannot emit text, so it never invents a number. Code finds the candidate
spans in the utterance and Jev picks which one the user meant:

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
| **TypeSafe hosted** (default) | `https://api.typesafe.ai/v1/systemone` — Jev, and whatever TypeSafe hosts later | an API key |
| **OpenAI-compatible** | any `/v1/chat/completions` endpoint: a self-hosted [Clef](https://huggingface.co/Cloudflare/clef) or [Clef-flash](https://huggingface.co/Cloudflare/clef-flash), [Von](https://github.com/wfzyx/von), [Laya](https://huggingface.co/convaiinnovations/laya), or a plain LLM served by vLLM, llama.cpp, Ollama, TabbyAPI, … | a base URL and a model name; an API key only if the endpoint wants one |

The OpenAI-compatible backend sends the same state and question schema in one
prompt and expects one JSON object back, keyed by question id, with a
probability distribution per question. Confidence is derived from those
probabilities with Jev's own formula, so every routing threshold behaves the
same whichever backend answers. Models that answer loosely (bare labels
instead of distributions, fenced JSON, prose around the object) are parsed
tolerantly; an answer that cannot be repaired is treated as "not confident"
and falls back, never as a wrong command.

Base URLs are accepted with or without a trailing `/v1` — `http://host:8000`
and `http://host:8000/v1` both work.

## What gets sent

On every request this integration sends, to `https://api.typesafe.ai/v1/systemone`:

- the text of the utterance
- **every entity you have exposed to Assist** — its name, area, floor, domain,
  device class and current state
- the areas and floors in your home, by name

That is the whole point of the design: the model is given the home as state and
answers questions about it, rather than being asked to write code or call tools.
But it means your device and room names, and what is currently on or off, leave
your network. If that is not acceptable to you, this integration is not the
right choice, and a fully local LLM agent is.

What does **not** happen: nothing is stored by this project, there is no
telemetry of its own, and the optional LLM backend is configured separately —
point it at Ollama on your own machine and the prose path never leaves the
house either. Diagnostics downloads redact the API key, the LLM base URL and
your entity IDs.

TypeSafe's own handling of what you send is governed by their
[terms](https://typesafe.ai/legal/mca) and
[data processing agreement](https://typesafe.ai/data-processing), not by this
project.

## Cost

Input tokens only. State is billed once per request rather than once per
question — verified against the API, and it is what makes the speculative
fan-out cheap: asking a question you end up discarding costs almost nothing.

Measured against `jev-1.13.0` at $42/Btok, a 29-entity home came to ~6.5k tokens
per request, about **$0.00027**. Both the rate and a model's token accounting
are TypeSafe's to change, and a later model will price differently — treat these
as an order of magnitude, and check
[your console](https://console.typesafe.ai/) and
[typesafe.ai](https://typesafe.ai/) for what you will actually be billed.

## Install

**Requires Home Assistant 2026.5.0 or newer.** Earlier releases do not report a
failed service call back to the conversation agent, so a command that no entity
could carry out would be announced as if it had worked. The integration is
tested against 2026.5.0 and the current release on every change.

### Through HACS

Not in the default HACS store yet, so add it as a custom repository: in HACS,
**⋮ → Custom repositories**, paste this repository's URL, choose category
**Integration**, then **Add**. It will then appear in HACS for install and for
update notifications.

### By hand

Copy `custom_components/typesafe_conversation` into your Home Assistant `config`
directory.

### Either way

Restart Home Assistant, then **Settings → Devices & Services → Add Integration →
TypeSafe Conversation**. You will need an API key from
[console.typesafe.ai](https://console.typesafe.ai/). Finally, set it as the
conversation agent under **Settings → Voice assistants**.

The LLM step is optional. Without it the agent still handles every command and
query; it just cannot split compound requests or answer general questions.
Ollama and any OpenAI-compatible endpoint (OpenRouter, vLLM, …) are supported.

### Options

Set when you add the integration, and changeable afterwards under
**Settings → Devices & Services → TypeSafe Conversation → Configure**.

| option | default | what it does |
| --- | --- | --- |
| `decision_backend` | `typesafe` | `typesafe` (hosted Jev) or `openai` (any OpenAI-compatible endpoint). |
| `api_key` | — | Your TypeSafe API key. Required for the hosted backend. |
| `model` | `jev-latest` | Which model to use — a TypeSafe model name, or the model name your own endpoint serves (`clef-flash`, `von`, …). |
| `decision_base_url` | — | Root of your OpenAI-compatible endpoint, with or without `/v1`. |
| `decision_api_key` | — | Only if that endpoint needs one. Redacted in diagnostics. |
| `decision_timeout` | `12` s | Seconds for one decision request. Self-hosted models on modest hardware need more than the hosted API. |
| `always_confirm_risky` | **on** | Ask before unlocking a door, opening a garage or disarming an alarm, however sure the model is. **Turning this off lets confident requests through silently** — the model's judgement becomes the only gate. |
| `bypass_local_intents` | off | Send every command here, including ones Home Assistant's own sentence matcher recognises. Off is recommended; see [below](#leave-prefer-handling-commands-locally-on). |
| `inline_entity_descriptions` | off | Describe every entity inside each question rather than once in the shared state. Roughly doubles the tokens. Only worth it if the agent picks the wrong device. |
| `llm_backend` | none | `ollama`, an OpenAI-compatible endpoint, or unset. Used only for compound requests and general questions. |
| `llm_base_url` | `http://localhost:11434` (Ollama) | Where that backend lives. Point it at your own machine to keep the prose path local. |
| `llm_model` | — | Model name on that backend. |
| `llm_api_key` | — | If the backend needs one. Redacted in diagnostics. |
| `llm_timeout` | `30` s | How long to wait for the LLM before giving up. Only the prose path is affected. |
| `llm_referer`, `llm_title` | project defaults | Sent as `HTTP-Referer` and `X-Title`; OpenRouter uses them for attribution. |

### Leave "prefer handling commands locally" on

The agent advertises `ConversationEntityFeature.CONTROL`. With prefer-local on,
Home Assistant's sentence matcher keeps every phrasing it recognises, and only
`HassGetState`, `HassMediaSearchAndPlay` and everything it *failed* to parse
reach this agent. That is deliberate: there is no point spending an API call
beating hassil at "turn off the kitchen lights", and it concentrates the traffic
on the requests where Jev earns its keep. Set `bypass_local_intents` if you want
to compare the two regimes.

### Follow-ups ("and back off again")

The decision model is a single-pass classifier, so a follow-up that only makes
sense with the previous turns in view scores as unclear. When that happens and
an LLM is configured, the integration asks the LLM to rewrite the utterance
into a standalone command with the conversation in view — "and back off again"
plus the earlier "turn on the kitchen light" becomes "turn off the kitchen
light" — and runs the decision model once more on the rewrite. The fast path
is untouched: follow-ups cost one extra LLM round trip, direct commands none.

If the rewrite comes back as "not a device command" (a question, a new topic),
the prose answer takes over instead.

### System prompt

Each conversation agent takes an optional **system prompt** (per-subentry
option): extra instructions for the general-knowledge answers — a persona, a
language rule, how brief to be. It is appended *after* the built-in
guardrails, so it cannot make the assistant claim to have controlled a device.

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
like `processed_locally: false` and roughly 250ms. To exercise it, use
phrasings the matcher cannot parse — "it's too dark in here", "get the coffee
going", "add milk to the shopping list", "turn off the lamp and start the
vacuum" — or turn on `bypass_local_intents` to send everything here.

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

API keys and the LLM base URL are redacted, and the catalog is reported as
counts rather than entity ids, so the file is safe to attach to an issue.

Home Assistant has a conversation-trace mechanism too, and this integration
writes to it — but nothing in Home Assistant reads it back (there is no
websocket command and no UI), so diagnostics is the usable route.

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
path is too slow. That path only writes sentences, so it does not need to be
the model you would choose for reasoning — a small local model, or a hosted
one, is usually the right trade. Raise `llm_timeout` in the integration options
if you would rather wait.

## Development

```sh
python3.14 -m venv .venv        # Home Assistant 2026.5+ requires Python 3.14
.venv/bin/pip install "pytest-homeassistant-custom-component==0.13.348" syrupy ruff
.venv/bin/pip install "hassil==3.8.0" "home-assistant-intents==2026.6.24"
.venv/bin/python -m pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

CI runs those three against both the oldest supported Home Assistant and the
current one, plus Home Assistant's `hassfest` and the HACS validator. See
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

The harness version is a **test-environment** choice, not the supported range —
see *Install* for that. Each release pins exactly one core version
(`0.13.348` → `homeassistant==2026.7.4`, `0.13.329` → `2026.5.0`), and a mismatch
against an already-installed `homeassistant` produces confusing import errors.

The third line is the `conversation` component's own requirements, which the
harness does not pull in — without them the import fails at `hassil`. Those pins
move with the core version, so if you change the harness, read them off the
`homeassistant` you actually installed:

```sh
.venv/bin/python -c "import json,pathlib,homeassistant as h; \
  print(json.loads((pathlib.Path(h.__file__).parent/'components'/'conversation'/'manifest.json').read_text())['requirements'])"
```

CI runs the suite against both ends of the supported range on every push and
pull request, deriving those requirements the same way.

The routing tests replay **real recorded Jev responses** from
`tests/fixtures/answers/`, so they describe how the model actually behaves
rather than how we imagine it does. To re-record them, or to re-fit the
thresholds in `const.py` against your own home:

```sh
set -a; . ./.env; set +a
.venv/bin/python scripts/calibrate.py --csv out.csv --record tests/fixtures/answers
```

Every threshold is a named constant in `const.py` precisely so it can be
re-fitted rather than argued about.

To pull your own home's catalog and calibrate against it:

```sh
.venv/bin/python scripts/pull_home.py --out my_home.json
.venv/bin/python scripts/calibrate.py --home my_home.json \
    --cases tests/fixtures/scripted_cases.json --area <your-area-id>
```

`pull_home.py` reads exposure settings over the WebSocket API (they are not in
the REST API) and redacts latitude/longitude by default, because weather
integrations name entities after the station's coordinates.

A catalog pulled from a live instance describes the home it came from — room
layout, device brands, which security devices exist — so keep it out of version
control. `.gitignore` already excludes `tests/fixtures/real_*` and
`docs/calibration-real-*.csv` for that reason. The fixtures that ship with this
repo are synthetic.

### A note on confidence

Jev derives confidence as `(n * p_top - 1) / (n - 1)`, so the *same* probability
scores differently depending on how many options the question offered. A
two-option Choice at p=0.66 scores 0.31; a seven-option Choice at the same
probability scores 0.60. The action gate therefore thresholds the chosen
option's **probability**, not its confidence — thresholding confidence made
every script command fall back on a script-heavy home.
