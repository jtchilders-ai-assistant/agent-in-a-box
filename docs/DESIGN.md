# ALCF Agent in a Box — Design

## Goal

Ship a Docker image that an **ALCF user** can `docker run` on a laptop and
immediately get an AI agent that (a) knows ALCF, (b) thinks using an
ALCF-hosted model, and (c) can act on ALCF systems (job submission, filesystem)
— all through a local web chat, with no external LLM provider and no API key to
manage.

## Why Hermes

[Hermes Agent](https://github.com/NousResearch/hermes-agent) already is the
thing we'd otherwise build: a provider-agnostic agent core with persistent
memory, a skills system, a terminal/tool loop, and — critically — a built-in
**web dashboard** (`hermes dashboard`) that serves a local chat UI. So the image
is Hermes + ALCF-specific content + a thin runtime wrapper, not a new framework.

## Architecture

```
        ┌─────────────────────────── container ───────────────────────────┐
        │                                                                  │
 laptop │  browser ─http://localhost:8787──▶ Caddy (HTTP) ─▶ hermes        │
 ───────┼──────────────────────────────────────┬── dashboard (web chat)── │
        │  (loopback secure context = clipboard)│    agent core (tools)    │
        │   /opt/data (named volume) ───────────┤                          │
        │     ├─ .globus  → Globus tokens       │                          │
        │     └─ memory + config + sessions     │                          │
        └───────────────────────────────────────┼─────────────────────────┘
                                                 │
                    ┌────────────────────────────┼───────────────────────────┐
                    ▼                             ▼                            ▼
        ALCF Inference Service          IRI Facility API             Globus Compute
        inference-api.alcf.anl.gov      api.alcf.anl.gov              (compute-node bash)
        (LLM brain: muse-spark-bf16     (job submit / fs ops)
         on Tara)
```

A single named volume (`alcf-agent-home` → `/opt/data`) holds Globus tokens
*and* Hermes memory/config/sessions; Caddy serves HTTP on the container port,
which Docker publishes only on host `127.0.0.1`. Browsers treat localhost as a
trustworthy secure context (so clipboard works) without local certificate setup.

## The LLM source: ALCF Inference Service (not Argo)

Argo is ANL-staff-only; this agent is for **users**. The ALCF Inference Service
(Sophia vLLM / Metis SambaNova) is open to ALCF users and OpenAI-compatible.

- API host: `https://inference-api.alcf.anl.gov` (NOT `inference.alcf.anl.gov`,
  which serves the Open WebUI SPA).
- Sophia base_url: `/resource_server/sophia/vllm/v1`; Metis:
  `/resource_server/metis/api/v1`.
- It's public-facing, so the agent works from an off-network laptop.

## Auth model (the load-bearing constraint)

The three runtime services use distinct Globus resource-server scopes. The
entrypoint uses the official **`alcf-tokens`** package (pinned at `==0.3.0`) to
request them all in **one combined Globus login** (`alcf-tokens login`):

| Service | Python API | Token used for |
|---|---|---|
| Inference | `alcf_tokens.auth.get_access_token('inference')` | The agent's LLM calls (minted per request through each provider's `key_cmd`) |
| IRI API | `alcf_tokens.auth.get_access_token('iri')` | Job submission, filesystem ops |
| Globus Compute | `alcf_tokens.auth.get_service_authorizer('globus-compute')` | Compute-node command execution |

The default `alcf-tokens login` always requests all four ALCF services
(Inference, IRI, Globus Compute, and Transfer) per ALCF policy — feature flags
(`ALCF_ENABLE_IRI`, `ALCF_ENABLE_GLOBUS_COMPUTE`) gate runtime feature exposure
but do not narrow the consent. Tokens last 48h and auto-refresh; a full
re-auth via `alcf-tokens login` is required every 30 days. Note: `alcf-tokens
login` has no `--force` flag; simply re-run to trigger a new login.

Consequences baked into the design:
- **No credentials in the image.** First run does one interactive `alcf-tokens login`;
  tokens live in the mounted `~/.globus` volume.
- **The rendered config holds no inference token.** Every ALCF provider (the
  generated block and the template fallback) carries
  `key_cmd: "/opt/hermes/.venv/bin/alcf-tokens get-token inference"` and no
  `api_key`; `model.api_key` renders empty in ALCF mode. Hermes v0.23.2 turns
  `key_cmd` into a callable credential (`agent/command_token_source.py`) that
  the wire clients call per request, caching the token and re-minting at most
  every 15 minutes (`_NO_TTL_REFRESH_SECONDS = 900`), so a token `alcf-tokens`
  refreshed on disk reaches the running agent with no re-render.
- **Live login check at startup** (ALCF mode). `get-token inference` succeeds
  with a cached token even when ALCF rejects it (e.g. after a high-assurance
  timeout), so it cannot gate startup. `scripts/inference_token_check.py` runs
  `alcf-tokens test-token inference` (GET `whoami`) under a timeout
  (`ALCF_TOKEN_CHECK_TIMEOUT`, default 20s) and prints one sanitized
  `STATUS<TAB>reason` line: `ok`, `auth` (no login / 401 / 403), `unavailable`
  (5xx / 429 / 408 / network / timeout) or `unknown`. It never echoes a
  response body, stdout or stderr. `auth` with a TTY runs `alcf-tokens login`
  and re-checks (exit 1 unless ok); `auth` with no TTY **exits 77** with the
  reason and the `docker run -it` / `docker exec -it ... alcf-tokens login`
  fix, and never starts a login. `unavailable` / `unknown` warn, write the
  status file and continue. A failing `get-token` counts as `auth`.
  `ALCF_TOKEN_CHECK=0` restores the old local-`get-token` gate. Generic mode
  keeps the local gate.
- The entrypoint runs a **login-check loop** (every `ALCF_TOKEN_CHECK_INTERVAL`
  seconds, default 6h): `get-token`, then the live check. `ok` re-renders the
  config and clears `$HERMES_HOME/.inference_token_status`; `auth` logs a
  banner and writes `EXPIRED`; `unavailable` / `unknown` write `UNREACHABLE` /
  `UNKNOWN`. Neither the loop nor `key_cmd` can renew an expired login (30-day
  limit or high-assurance timeout): that needs a new interactive login.

### Secret hygiene

The rendered `config.yaml` is the only intended durable home of an inference
credential; the entrypoint forces it to mode 0600 on every render. In ALCF
mode it holds none (providers mint through `key_cmd`); the credential is the
`alcf-tokens` store on the volume. In generic
mode, `INFERENCE_TOKEN` arrives through `docker run -e`; once it is in the
config, the entrypoint `unset`s it before launching anything. The generic-mode
invariant is therefore: **the token is not inherited by long-lived descendants;
it is persisted only in the mode-0600 config.** The dashboard, Caddy, the MCP
`bash` server, and a gateway the dashboard starts never see it in their
environment. (Short-lived setup helpers the entrypoint runs before the unset,
such as `generic_inference.py`, do receive it.)

**Known limit (`docker exec`).** Docker gives every process started with
`docker exec` the container's configured environment (`Config.Env`, i.e. the
`-e` values from `docker run`), not the entrypoint's. The entrypoint cannot
change that. A gateway that `docker exec <c> hermes -z ...` auto-starts
therefore inherits `INFERENCE_TOKEN` (seen in the v0.23.2 image smoke test).
If that matters, use the dashboard, or blank the variable for the exec:
`docker exec -e INFERENCE_TOKEN= <c> hermes ...`. ALCF mode is unaffected,
because it passes no inference token through the environment.

## Startup pipeline (what the launch scripts do)

`scripts/entrypoint.sh` runs on every container start and renders the live config
before launching the dashboard. The design principle throughout is **fail-open**:
every ALCF-service call is best-effort and falls back to a safe default rather
than aborting the launch. The intentional hard stops are configuration errors,
not service failures: the context-floor guard only fires on a positive,
confirmed sub-floor reading, launch/delegation-provider resolution refuses a
model that does not map to one unambiguous generated provider, and a set but
non-positive-integer `ALCF_MAX_TOKENS` / `ALCF_REASONING_MAX_TOKENS` is refused
up front rather than replaced by the default cap.

Diagnostic and state files for a start (`ctxlen.log`, `populate_models.log`,
`launch_provider.log`, `generic_inference.log`, `launch_source`) live in one
private run directory, `mktemp -d "${TMPDIR:-/tmp}/aib-run.XXXXXX"` (mode 0700,
files 0600), never at fixed `/tmp` paths. `startup.log` names it.

Order of operations in `render_config()` (re-run by the 6h login-check loop):

1. **Fresh inference token** for the catalog and context-window lookups below.
   It is not written to the config (providers use `key_cmd`).
2. **Launch-model context window** (`resolve_context_length.py`). Reads the real
   `max_model_len` from the gateway because ALCF caps some models below their
   published spec and Hermes' family table would otherwise over-estimate it.
   Defaults to 128000 on any failure (so the floor guard fails open).
3. **Model-list generation** (`populate_models.py`). Queries the live catalog on
   every enabled cluster (sophia, metis, minerva, tara), filters to chat models with a real ≥64k window, and emits a
   keyed `providers:` block (the Hermes v0.23.2 schema). Each cluster is split
   into a baseline provider and a `-reasoning` provider (see quirk #1), each
   carrying its own output cap as `extra_body.max_tokens`.
   Reasoning classification uses the catalog `reasoning_parser` field plus an id
   heuristic. On any discovery failure it prints the committed static fallback
   block from the template, so the container always comes up with a usable list.
   Two launch-model safeguards live here. First, the launch cluster
   (`ALCF_CLUSTER`) is always generated, even if its `ALCF_ENABLE_<CLUSTER>`
   flag is `0`; the override is logged. Second, if the launch cluster fell back
   to its static list and that list lacks `ALCF_MODEL`, the model is added to
   the matching plain or `-reasoning` provider, using the step-2 context window
   (only if ≥ 64k), with a loud stderr notice. This injection is a best-effort
   emergency fallback: the model has passed only the local context-floor check,
   and whether it is actually served stays unverified until the catalog
   recovers. The launch cluster's state (`live` / `fallback` /
   `fallback-injected` / `fallback-missing`) is handed to the entrypoint through
   a state file in the private run directory (below), for step 5; any other
   content in it is refused with exit 78.
4. **Context-floor guard.** Hermes hard-refuses a model whose window is below
   `MINIMUM_CONTEXT_LENGTH` (64000). If the *launch* model is sub-floor, exit 78
   with an actionable list of valid models rather than a raw Hermes stacktrace.
5. **Launch-provider resolution.** Parse the *generated block* (single source of
   truth, via `populate_models.py --launch-provider --base-url`) for the one
   provider on the launch endpoint that lists the launch model, and set the
   top-level `model.provider` to it (`custom:alcf-sophia[-reasoning]`). This is
   how turn one inherits the correct per-model output cap given that Hermes has
   no per-model `max_tokens`. This is the second intentional hard stop: if no
   provider, or more than one, lists the model, exit 78 with the valid model
   ids. When the launch cluster is on its static fallback and the model could
   not be added in step 3, or generation failed outright, the refusal instead
   says "catalog unavailable; '<model>' is not in the static fallback list".
   A distinct `ALCF_DELEGATION_MODEL` is endpoint-pinned too
   (`--delegation-provider --base-url`): the provider on the launch endpoint
   that lists it (exactly one there must), else the one provider elsewhere that
   lists it. It exits 78 when unresolvable, or listed by several providers that
   are not on the launch endpoint. (Before the
   v0.23.2 migration this step fell back to an id heuristic and then to
   `custom:alcf-sophia`; that silent fallback was removed.)
6. **Splice + substitute.** The generated block replaces the sentinel-to-EOF
   region of `config.template.yaml`, then `${VAR}` placeholders are substituted.

After the render, a **model availability banner** (`--status-report`, gated by
`ALCF_SHOW_MODEL_STATUS`) classifies every offered model as LIVE / QUEUED /
OFFLINE against the live `<cluster>/jobs` state and annotates each with its
context window. QUEUED carries the scheduler's estimated start time, which may be
hours away — distinct from the ~10–15 min GPU warm-up of a model that is merely
cold. Then skills / `MEMORY.md`
/ `SOUL.md` are seeded-or-refreshed (checksum-stamped so user edits are never
clobbered), the 6h refresh loop starts, and the dashboard launches behind Caddy.

## Two provider quirks (both handled)

1. **Reasoning models return empty content with a small output cap.** Reasoning
   models (`gpt-oss-*`, `gemma-4-*`, `nemotron-3-super`, `*-Thinking`) spend part
   of the `max_tokens` output budget on a hidden reasoning channel, so a small cap
   can leave them returning empty content (`finish_reason=length`). Hermes has no
   *per-model* `max_tokens` (unlike `context_length`), and v0.23.2 ignores a bare
   `max_tokens` key on a `providers:` entry ("unknown config keys ignored"). It
   does send a provider's `extra_body` with every request to that provider's
   models, at launch and after a `/model` switch, and the OpenAI-compatible wire
   carries `extra_body.max_tokens` as the request's `max_tokens`. Fix
   (`populate_models.py`): split each cluster into a baseline provider
   (`alcf-sophia`, `extra_body.max_tokens` 2048) and a `-reasoning` provider
   (`alcf-sophia-reasoning`, `extra_body.max_tokens` 12288, env-overridable via
   `ALCF_MAX_TOKENS` / `ALCF_REASONING_MAX_TOKENS`); leave top-level
   `model.max_tokens` unset so the per-provider cap is the only one in play;
   and point the launch model's top-level `model.provider` at the provider that
   lists it so turn one gets the right cap too. (Before v0.23.2 the cap was a
   provider-level `max_tokens` in the legacy `custom_providers:` list.) Reasoning is
   detected from the gateway's `reasoning_parser` field, with an id heuristic for
   models served without it (e.g. gpt-oss).
2. **The vLLM gateway rejects `name` on `role: tool` messages** with HTTP 422
   `extra_forbidden`, breaking all agentic tool use. `name` is a *valid* Chat
   Completions field that compliant providers accept, so we can't strip it
   unconditionally. **Current fix: none needed.** Upstream Hermes v0.23.2 drops
   `name` from `role: tool` messages itself (role-qualified, in
   `agent/transports/chat_completions.py`), so the image uses Hermes unmodified.
   - *Historical (retired with the v0.23.2 base):* earlier images applied a local
     patch, `patches/0001-strip-tool-message-name.patch`, from the fork branch
     `feat/strip-tool-message-name`. It added a config-gated flag,
     `model.strip_tool_message_name` (default off), which the image set to
     `true`. The patch file and the flag are gone.

## Verification log (spike, 2026-07-29)

All proven with real execution, not description:
- ✅ Headless Globus auth → 48h inference token.
- ✅ `curl` chat completion to Sophia `gpt-oss-120b` → HTTP 200, correct content
  (with adequate `max_tokens`).
- ✅ A/B curl proof of the `name`/422 blocker and the drop-`name` fix.
- ✅ Hermes (isolated `HERMES_HOME`) plain chat → correct answer.
- ✅ Hermes agentic tool call that previously 422'd → now returns the right
  answer with the flag on (the since-retired patch flag; see quirk #2).
- ✅ `hermes dashboard` web chat drove `gpt-oss-120b` through autonomous terminal
  tool calls end-to-end.

## Repo layout

```
Dockerfile                         official Hermes image (unmodified) + ALCF content, layered
config/config.template.yaml        single ALCF inference target + static providers fallback
config/SOUL.md                     agent identity + "what can I do" greeting (seeded to $HERMES_HOME/SOUL.md)
scripts/entrypoint.sh              first-run alcf-tokens login, config render + dynamic model list, floor/launch-provider guards, refresh loop, launch
scripts/populate_models.py         generates the switchable model list from the live ALCF catalog (reasoning split, 64k floor, --hot-report, --launch-provider)
scripts/resolve_context_length.py  resolves a model's real serving window (max_model_len) for the context-floor guard
scripts/fetch_docs.py              pulls ALCF user-guides markdown into docs/
scripts/iri_hello_world.py         one-shot IRI job submitter (write; consumes allocation)
scripts/alcf_facility.py           read-only helper: system status, my jobs, job output, allocations
skills/                            alcf-inference-service, alcf-iri-facility-api, alcf-pbs..., alcf-facility-status-and-jobs
memory/MEMORY.md                   curated, sanitized ALCF knowledge (seeded to built-in memory)
docs/                              ALCF docs snapshot (nightly CI)
.github/workflows/build.yml        build + push image to GHCR
.github/workflows/refresh-docs.yml nightly docs refresh
```

## Sanitization checklist (before publishing)

The image is shared, so it must carry only generic ALCF knowledge:
- [x] No personal ANL username / API keys in skills or config.
- [x] Memory seed is a curated public-facts file, NOT an export of a personal
      Mnemosyne bank.
- [x] Review `Vendor_Support` project-name references in the IRI skill —
      genericized to `<your project>` / "look it up via GET /account/projects"
      (removed the specific project name + allocation node-hour figures).
- [x] No Hermes fork or local patch: the image pins the official
      `nousresearch/hermes-agent` release tag. (Historical: the retired
      `strip_tool_message_name` fork patch carried no unrelated personal changes.)

## Open items

Everything in the original build plan is done and verified (image built and
run end-to-end, default model chosen — see below — GitHub repo + CI live). The
remaining items are genuinely optional follow-ups:

1. Serve the Hermes CLI/TUI in addition to the dashboard (currently
   dashboard-only; the CLI still works if you `docker exec` in).

### Resolved (kept for history)

- ✅ Tool-message `name` fix: upstream Hermes v0.23.2 strips `name` from tool
  messages, so the local `strip_tool_message_name` patch layer was retired and
  deleted with the move to that base.
- ✅ One combined consent validated against inference, IRI, and Globus Compute.
- ✅ Default model decided: `google/gemma-4-31B-it` — a non-reasoning model kept
  consistently *hot* on Sophia, so it avoids both the cold-start HTTP 503 and
  gpt-oss's reasoning-token thrash under agentic load. All models stay
  switchable in the dashboard.
- ✅ Default model changed (2026-11-23): `muse-spark-bf16` on Tara (1M context,
  reasoning, with the `-reasoning` provider's 12288-token cap). Trade-off: no
  Tara model is kept always hot, so the first request may wait for a cold
  start. The default model depends on `ALCF_CLUSTER` (the image sets no
  `ALCF_MODEL`; the entrypoint holds the one table), so `ALCF_CLUSTER=sophia`
  alone restores the previous always-hot default (`google/gemma-4-31B-it`).
- ✅ Repo created under `jtchilders-ai-assistant` with GHCR publish + nightly
  docs-refresh CI. (Move to `argonne-lcf` if this becomes official.)
