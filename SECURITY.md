# Security policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Use GitHub's private vulnerability reporting instead: go to the
[Security tab](https://github.com/maglat/home-assistant-typesafe-conversation-agent/security)
and click **Report a vulnerability**. That opens a private thread visible only
to the maintainer.

This integration is worth reporting against carefully. It holds a TypeSafe API
key and, optionally, credentials for an LLM backend. It can also call any
Home Assistant intent that the user has exposed to Assist — which may include
unlocking doors, opening garage doors and disarming alarms.

Please include what you would put in a bug report: how to reproduce it, what
you expected, and what happened. Redact your own entity names and any tokens.

I maintain this on my own time, so I cannot promise a response window. I will
acknowledge a report as soon as I see it and tell you honestly whether and when
I can fix it.

## Supported versions

Only the latest release. There are no backports.

The minimum supported Home Assistant version is stated in
[`hacs.json`](hacs.json) and in the README; older cores are not supported
because they do not report failed service calls back to the conversation agent.

## Things that are not vulnerabilities

- **The catalogue is sent to a third-party API.** By design, and documented in
  the README under *What gets sent*. If you cannot accept that, this
  integration is not for you.
- **A risky action ran without confirmation while `always_confirm_risky` was
  off.** That option exists to allow exactly this; the default is on.
- **An API key is visible in Home Assistant's own config entry storage.** That
  is how every Home Assistant integration stores credentials. Diagnostics
  downloads *are* redacted — a leak there would be a real bug.
