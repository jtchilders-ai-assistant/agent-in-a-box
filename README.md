# ALCF Agent in a Box

A ready-to-run AI agent for **ALCF users**, packaged as an OCI container image.
Run it locally with Docker, Podman, or Apptainer, log in with your ALCF/Globus
credentials, and get a local web chat that can:

- **Answer questions about ALCF** using the latest ALCF documentation and a
  curated knowledge base (baked into the image).
- **Run inference on the [ALCF Inference Service]** (Sophia / Metis) — no
  external LLM provider, no API key to manage. The agent's own brain *is* an
  ALCF-hosted open model (default `google/gemma-4-31B-it`).
- **Submit and manage jobs** on ALCF systems (Polaris, Crux, Aurora) through the
  [IRI Facility API], plus filesystem operations on Home/Eagle.
- **Run commands on ALCF compute nodes** through Globus Compute, using the
  user's own account and allocation—without giving the container SSH access.

It is built on [Hermes Agent] (Nous Research) — an open-source, provider-agnostic
agent framework with persistent memory, skills, and a built-in web dashboard.

> ⚠️ **Independent tool — not an official ALCF/Argonne/DOE product**, and it runs
> an **autonomous AI agent** that can be wrong and that takes real actions with
> your credentials (jobs, node-hours, files). **You are responsible for what it
> does.** Please read [DISCLAIMER.md](DISCLAIMER.md) before using it.

[ALCF Inference Service]: https://docs.alcf.anl.gov/services/inference-endpoints/
[IRI Facility API]: https://docs.alcf.anl.gov/services/iri-api/
[Hermes Agent]: https://github.com/NousResearch/hermes-agent

---

The agent runs on your local machine. It sends inference requests to ALCF and,
when you ask it to act remotely, uses IRI and Globus Compute under your ALCF
identity. These instructions do **not** run the agent on an ALCF login or compute
node and do not give the container SSH access.

## Quick start with Docker

```bash
docker run --pull=always -it --rm \
  -p 127.0.0.1:8787:8787 \
  -e ALCF_DASHBOARD_PASSWORD='choose-a-password' \
  -v alcf-agent-home:/opt/data \
  ghcr.io/jtchilders-ai-assistant/alcf-agent:latest
```

On first run the container will:

1. Start **ONE combined Globus login** for the ALCF Inference Service, IRI
   Facility API, and Globus Compute. It prints one URL; you log in and paste
   back one authorization code. The official package's combined login always
   requests all four supported services (including Globus Transfer). The
   `ALCF_ENABLE_IRI` and `ALCF_ENABLE_GLOBUS_COMPUTE` flags disable the box's
   corresponding runtime features, not scopes in that combined consent.
2. Launch the web chat at **<http://localhost:8787>**. Chrome treats localhost
   as a trustworthy secure context, so chat copy/paste works without a local
   certificate or browser warning. Log in with username `alcf` (override
   with `-e ALCF_DASHBOARD_USER=...`) and the password you set. If you don't set
   `ALCF_DASHBOARD_PASSWORD`, the container generates one and prints it at
   startup.

The single `alcf-agent-home` volume persists your Globus tokens **and** the
agent's memory across restarts, so you only log in occasionally (tokens last 48h
and auto-refresh; a full re-auth is required every 30 days).

## Run locally with Podman

Podman accepts the same OCI image and run options. This command keeps the web
dashboard bound to your machine's loopback interface and stores state in a
Podman-managed named volume:

```bash
podman run -it --rm \
  --name alcf-agent \
  -p 127.0.0.1:8787:8787 \
  -e ALCF_DASHBOARD_PASSWORD='choose-a-password' \
  -v alcf-agent-home:/opt/data \
  ghcr.io/jtchilders-ai-assistant/alcf-agent:latest
```

Open <http://localhost:8787>. First-run authentication and subsequent behavior
are the same as with Docker. To renew expired credentials while the named
container is running:

```bash
podman exec -it alcf-agent \
  /opt/hermes/.venv/bin/alcf-tokens login
```

## Run locally with Apptainer

This path is for a **local Linux machine** with Apptainer installed. Apptainer
uses the host network by default, so the dashboard's existing loopback listener
is available at <http://localhost:8787>; do not add a network namespace or port
mapping. Unlike Docker and Podman, Apptainer normally exposes your home and
current directory inside the container. The commands below replace the home
mount with a private state directory, disable the current-directory mount, and
bind that same private directory to `/opt/data`.

Pull the public OCI image into a local SIF, then create persistent state:

```bash
apptainer pull alcf-agent.sif \
  docker://ghcr.io/jtchilders-ai-assistant/alcf-agent:latest
mkdir -p "$HOME/.local/share/alcf-agent"
chmod 700 "$HOME/.local/share/alcf-agent"
APPTAINER_HOME="$HOME/.local/share/alcf-agent"
```

Run the remaining commands from the same shell so `APPTAINER_HOME` remains set.

Set the dashboard password through Apptainer's explicit container-environment
prefix and start the agent:

```bash
export APPTAINERENV_ALCF_DASHBOARD_PASSWORD='choose-a-password'
apptainer run --containall --no-mount cwd \
  --home "$APPTAINER_HOME:$HOME" \
  --cwd /opt/data \
  --bind "$HOME/.local/share/alcf-agent:/opt/data" \
  alcf-agent.sif
```

`apptainer run` executes the OCI image's built-in entrypoint; do not append an
entrypoint command. The private home mapping is required because Globus stores
credentials below `$HOME/.globus`, while the `/opt/data` bind stores Hermes
state. Both resolve to the same host directory. Apptainer runs as your host UID,
so the directory you create is writable without `--fakeroot`.

The SIF itself is read-only. To expose a dedicated local work directory, create
it first and add one narrow bind—never bind your entire home:

```bash
mkdir -p "$HOME/alcf-work"
apptainer run --containall --no-mount cwd \
  --home "$APPTAINER_HOME:$HOME" \
  --cwd /opt/data \
  --bind "$HOME/.local/share/alcf-agent:/opt/data" \
  --bind "$HOME/alcf-work:/work" \
  alcf-agent.sif
```

To renew expired credentials using the same persistent state:

```bash
apptainer exec --containall --no-mount cwd \
  --home "$APPTAINER_HOME:$HOME" \
  --cwd /opt/data \
  --bind "$HOME/.local/share/alcf-agent:/opt/data" \
  alcf-agent.sif /opt/hermes/.venv/bin/alcf-tokens login
```

Apptainer receives most image environment defaults from the OCI image. For any
override listed under **Configuration knobs**, export the corresponding
`APPTAINERENV_` variable—for example,
`APPTAINERENV_ALCF_BASH_ACCOUNT=my-project`.

## Data & filesystem access (sandboxed by design)

**The agent cannot see or touch your laptop's files.** It runs fully inside the
container. Its file and terminal tools only reach the container's own
filesystem:

- `/opt/data` — persistent private state. Docker and Podman use the named volume
  `alcf-agent-home`; the Apptainer command above binds the dedicated
  `$HOME/.local/share/alcf-agent` directory. It holds the agent's config, memory,
  Globus tokens, and session history so they survive restarts.
- `/opt/alcf`, `/opt/hermes` — baked-in ALCF content and the agent code.

Nothing else under your host home (`~/Documents`, `~/anl`, etc.) is mounted by
the documented commands, so the agent can't read or modify your other local
files. Docker and Podman discard the container on exit with `--rm`; Apptainer's
SIF is read-only. In every case only the explicit state storage persists. This
is intentional: a support tool shouldn't have ambient access to a user's
machine.

If you *want* the agent to work with local files, add an explicit bind mount of
a **dedicated** directory (never your whole home):

```bash
# creates/uses ~/alcf-work on your host, visible to the agent at /work
docker run --pull=always -it --rm -p 127.0.0.1:8787:8787 \
  -e ALCF_DASHBOARD_PASSWORD='choose-a-password' \
  -v alcf-agent-home:/opt/data \
  -v "$HOME/alcf-work:/work" \
  ghcr.io/jtchilders-ai-assistant/alcf-agent:latest
```

Then ask the agent to read/write under `/work`. Only that directory is exposed.

> **Security:** publish the dashboard exactly as shown, on host `127.0.0.1`
> only with Docker or Podman. Apptainer uses the host network and the image's
> Caddy configuration listens only on `localhost` / `127.0.0.1`. Browsers grant
> `http://localhost` secure-context privileges for clipboard access, while the
> dashboard username/password gate remains enabled. Do not replace the Docker or
> Podman mapping with `-p 8787:8787`, which exposes it on every interface.
>
> **Reauthentication:** all enabled ALCF services share the combined login. If
> the agent reports expired or missing credentials, renew them with `alcf-tokens login`:
> ```bash
> docker exec -it <container> \
>   /opt/hermes/.venv/bin/alcf-tokens login
> ```
> The official `alcf-tokens login` default always requests all four ALCF services
> (Inference, IRI, Globus Compute, and Transfer) in one browser visit. One fresh
> login is sufficient after an upgrade (the client ID changed).

### Compute-node shell (MCP `bash` tool) & subagents

When the base Hermes supports MCP, the agent's compute-node shell is also
exposed to the model as a native **`bash` tool**: one warm node held across the
conversation, with `cd` / `export` / `module load` persisting between commands
like a real login shell. Recommended container environment overrides:

- `-e ALCF_BASH_ACCOUNT=<project>` — default ALCF project to charge (otherwise
  the agent looks one up or asks you, then it sticks for the session).
- `-e ALCF_BASH_ENDPOINT=polaris|crux|sophia|edith`, `-e ALCF_BASH_QUEUE=...`,
  `-e ALCF_BASH_WALLTIME=H:MM:SS` — where the node comes from (defaults:
  polaris / debug / 1:00:00). `-e ALCF_ENABLE_MCP_BASH=0` removes the tool.
- `-e ALCF_DELEGATION_MODEL=google/gemma-4-26B-A4B-it` — run `delegate_task`
  subagents (e.g. build/test workers) on a **262k-context** model while your
  chat stays on the default. Defaults to the chat model if unset.
> The token is stored under `/opt/data`, so it persists in the configured volume
> or state directory.
>
> **Network:** the ALCF Inference Service endpoint (`inference-api.alcf.anl.gov`)
> is public-facing, so this works from a laptop off the ALCF network. Some IRI
> API operations may require membership in the relevant ALCF project.

---

## What's inside

| Piece | Source | Purpose |
|---|---|---|
| Hermes Agent (official image) | `nousresearch/hermes-agent:v2026.7.30` | Agent core + web dashboard (pinned base image) |
| Tool-message patch | `patches/0001-strip-tool-message-name.patch` | Enables agentic tool use on the ALCF gateway |
| ALCF skills | `skills/` | How to call ALCF inference / IRI / PBS |
| Knowledge seed | `memory/MEMORY.md` | Curated, **sanitized** ALCF facts (always injected) |
| Docs snapshot | `docs/` | Latest ALCF user docs (refreshed nightly) |
| Config template | `config/config.template.yaml` | Points Hermes at ALCF inference; carries the static model-list fallback |
| Model-list generator | `scripts/populate_models.py` | Builds the switchable model list from the live ALCF catalog at start (reasoning split, 64k floor, availability report) |
| Entrypoint | `scripts/entrypoint.sh` | First-run auth, config render, dynamic model list, launch-provider + context-floor guard, token-refresh loop, launch |

### Architecture: built on the official Hermes image

This image is `FROM nousresearch/hermes-agent:v2026.7.30` plus three thin layers
(patch, Globus auth helpers, ALCF content). We deliberately do **not** rebuild
Hermes — the official image already handles the fixed SQLite build, s6-overlay
supervision, the editable install, and the dashboard. See
[docs/DESIGN.md](docs/DESIGN.md).

### Why a patched Hermes?

The ALCF Inference Service's vLLM gateway validates the Chat Completions
tool-message schema strictly and rejects a `name` field on `role: tool`
messages with HTTP 422, which breaks agentic (tool-using) sessions. Hermes was
extended with a config flag `model.strip_tool_message_name` (default off) that
strips that field for such gateways. The image applies the patch and sets the
flag to `true`. The fix is upstreamable; once merged the patch layer becomes a
no-op. See [docs/DESIGN.md](docs/DESIGN.md) for the full root-cause writeup.

---

## Configuration knobs

Everything is driven by `config/config.template.yaml`, rendered into the
running config at container start. Pass these with `-e` to Docker/Podman or as
`APPTAINERENV_...` variables to Apptainer:

| Env var | Default | Meaning |
|---|---|---|
| `ALCF_MODEL` | `google/gemma-4-31B-it` | Model id on the inference service |
| `ALCF_CLUSTER` | `sophia` | `sophia` (vLLM) or `metis` (SambaNova) |
| `ALCF_MAX_TOKENS` | `2048` | Baseline per-response output cap for **plain chat** models |
| `ALCF_REASONING_MAX_TOKENS` | `12288` | Per-response output cap for **reasoning** models (gpt-oss, gemma-4, nemotron-3-super, *-Thinking). They spend part of the output budget on a hidden reasoning channel, so they need more headroom than chat models. |
| `ALCF_DASHBOARD_PORT` | `8787` | Web chat port inside the container |
| `ALCF_DASHBOARD_USER` | `alcf` | Dashboard login username |
| `ALCF_DASHBOARD_PASSWORD` | *(auto-generated + printed)* | Dashboard login password (hashed at start; plaintext never stored) |
| `ALCF_ENABLE_IRI` | `1` | Enable IRI job/filesystem tools (the official combined login still requests the IRI credential when disabled) |
| `ALCF_ENABLE_GLOBUS_COMPUTE` | `1` | Enable compute-node execution (the official combined login still requests the Compute credential when disabled) |
| `ALCF_ENABLE_METIS` | `1` | Include the Metis cluster's models in the switchable list |
| `ALCF_ENABLE_MINERVA` | `1` | Include the Minerva cluster's models in the switchable list |
|  `ALCF_SHOW_MODEL_STATUS` | `1` | Print the model availability banner (LIVE/QUEUED/OFFLINE + context windows) at startup |
|  `ALCF_BASH_ACCOUNT` | *(unset)* | **Recommended:** your default ALCF project for the compute-node `bash` tool. When set, the agent holds ONE warm compute node across the whole conversation instead of paying repeated cold starts; when unset, it must ask/look up a project first. |
| `AGENT_BOX_FACILITIES` | `alcf` | Facilities to enable (`alcf`, `nersc`, `esnet-east`, `esnet-west`). Default ALCF only. Multi-facility needs the shipped registry and separate authentication (`nersc-tokens` for NERSC). ESnet has registry metadata only; its validated PAT route is not released. |

## Facility configuration (multi-facility reality)

On first start the container runs a facility-configuration step before ALCF login. With a TTY it prompts for facilities; pressing `Enter` (or running non-interactive / no TTY) retains ALCF only. The registry (`config/facilities.yaml`) is the source of truth. Multi-facility example stays loopback, named volume, `--pull always`:

```bash
AGENT_BOX_FACILITIES=alcf,nersc \
docker run --pull=always -it --rm \
  -p 127.0.0.1:8787:8787 \
  -e ALCF_DASHBOARD_PASSWORD='choose-a-password' \
  -v alcf-agent-home:/opt/data \
  ghcr.io/jtchilders-ai-assistant/alcf-agent:latest
```

The product config (`/opt/data/agent-box/config.yaml`) reflects the resolved settings. Inspect it with:

```bash
docker run --rm -v alcf-agent-home:/opt/data --entrypoint /opt/hermes/.venv/bin/python ghcr.io/jtchilders-ai-assistant/alcf-agent:latest /opt/alcf/agent_box.py config show
docker run --rm -v alcf-agent-home:/opt/data --entrypoint /opt/hermes/.venv/bin/python ghcr.io/jtchilders-ai-assistant/alcf-agent:latest /opt/alcf/agent_box.py config validate
```

Capabilities and limits (truthful):
- ALCF remains the default identity and working runtime.
- NERSC (`nersc-tokens==0.1.1`) is packaged. If NERSC is selected, startup performs a separate NERSC Globus login (`nersc-tokens login iri --facilities nersc`) after the ALCF login; it uses a separate token cache. Auth/config guidance exists, but no generic NERSC IRI operation tool is wired — the agent explains NERSC endpoint/auth boundaries but does not claim it can submit NERSC IRI calls.
- ESnet (`esnet-east`/`esnet-west`) endpoint selection is registry-only; the validated PAT route is not released, validation/operations are unavailable, and the agent must fail closed rather than claim ESnet submission works.

## Generic OpenAI-compatible inference mode (optional)

**The default is unchanged:** with no `INFERENCE_*` variables set, the box
behaves exactly as before — ALCF Inference Service on `ALCF_CLUSTER`, the
combined Globus login, the live catalog, the availability banner, and the
6-hour token refresh. Everything below only applies when you opt in.

Setting `INFERENCE_URL` points the agent's own brain at **any**
OpenAI-compatible endpoint (`GET <base>/models` + Bearer auth) instead of the
ALCF gateway. IRI job/filesystem tools, Globus Compute, Transfer, and the
dashboard are untouched — the combined `alcf-tokens` login still happens
because those features keep working. Generic mode skips only the ALCF-specific
catalog/status/refresh machinery.

Required settings (any partial combination fails closed with exit 78):

| Env var | Required | Meaning |
|---|---|---|
| `INFERENCE_URL` | *(unset = ALCF mode)* | OpenAI-compatible API base URL (http/https; credentials, query strings, and fragments are rejected). `/models` is appended exactly once. |
| `INFERENCE_TOKEN` | **yes** in generic mode | Bearer token for the endpoint. Never logged, never written to the manifest. |
| `INFERENCE_MODEL` | **yes** in generic mode | Model id to launch; must exist in the endpoint's catalog with context ≥ 64000. |
| `INFERENCE_MAX_TOKENS` | optional (default `4096`) | Per-response output cap; must be a positive integer. |
| `INFERENCE_CONTEXT_LENGTHS` | optional | JSON object overriding per-model context windows for models the endpoint doesn't report correctly, e.g. `-e INFERENCE_CONTEXT_LENGTHS='{"my-model": 200000}'`. An override may not invent a model absent from the catalog, and overrides below 64000 still disable the model. |

Docker:

```bash
docker run --pull=always -it --rm \
  -p 127.0.0.1:8787:8787 \
  -e ALCF_DASHBOARD_PASSWORD='choose-a-password' \
  -e INFERENCE_URL='https://my-vllm.example.com/v1' \
  -e INFERENCE_TOKEN='...' \
  -e INFERENCE_MODEL='my-model' \
  -v alcf-agent-home:/opt/data \
  ghcr.io/jtchilders-ai-assistant/alcf-agent:latest
```

Podman accepts the identical options (`podman run -it --rm \ -p
127.0.0.1:8787:8787 …`). For Apptainer, keep the existing confinement
(`--containall --no-mount cwd`, private home, host network — no port mapping)
and pass the settings with the `APPTAINERENV_` prefix:

```bash
export APPTAINERENV_INFERENCE_URL='https://my-vllm.example.com/v1'
export APPTAINERENV_INFERENCE_TOKEN='...'
export APPTAINERENV_INFERENCE_MODEL='my-model'
apptainer run --containall --no-mount cwd \
  --home "$APPTAINER_HOME:$HOME" \
  --cwd /opt/data \
  --bind "$HOME/.local/share/alcf-agent:/opt/data" \
  alcf-agent.sif
```

### Example: AmSC Model Access Gateway (MAG)

MAG is one example of the generic OpenAI-compatible mode; it does not require
MAG-specific code. Obtain a MAG bearer token through your organization's normal
credential process, keep it out of shell history and the repository, and expose
it to the command through an environment variable:

```bash
read -rsp 'MAG bearer token: ' MAG_TOKEN && printf '\n'
export MAG_TOKEN

docker run --pull=always -it --rm \
  -p 127.0.0.1:8787:8787 \
  -e ALCF_DASHBOARD_PASSWORD='choose-a-password' \
  -e INFERENCE_URL='https://mag.genesis.americansciencecloud.org/v1' \
  -e INFERENCE_TOKEN="$MAG_TOKEN" \
  -e INFERENCE_MODEL='gpt-5.6-sol@openai' \
  -e INFERENCE_MAX_TOKENS='8192' \
  -v alcf-agent-home:/opt/data \
  ghcr.io/jtchilders-ai-assistant/alcf-agent:latest

unset MAG_TOKEN
```

The model must be present in MAG's live `/models` catalog and report at least a
64,000-token context window. If you choose a catalog model without recognized
context metadata, provide an explicit `INFERENCE_CONTEXT_LENGTHS` override as
described above. The generic-mode credential-persistence warning below applies
to MAG too.

At startup the container prints an ENABLED/DISABLED report for every catalog
model (a model is disabled if its context window is missing or below 64000;
model *names* are never used to infer a window — only `max_model_len`,
`context_length`, `context_window`, `max_context_length`, `max_input_tokens`,
`capabilities.context_window_tokens`, `metadata.context_length`, or an
explicit override count).

> ⚠️ **Credential persistence:** `INFERENCE_TOKEN` is rendered into
> `$HERMES_HOME/config.yaml` (forced to mode 0600), which lives on your
> `alcf-agent-home` volume / Apptainer state directory and persists after the
> container exits. Anyone with access to that volume can read the token. It is
> static — there is no auto-refresh in generic mode; restart the container with
> an updated `-e INFERENCE_TOKEN=...` to rotate it. Keep publishing the
> dashboard on `127.0.0.1` only, exactly as above.

## What happens at container start

`scripts/entrypoint.sh` is the launch script. On every start (not just the first
run) it performs the following steps before handing off to the Hermes dashboard.
All of the ALCF-service calls below are **best-effort and non-fatal** — if the
network or catalog is unavailable, each step falls back to a safe default rather
than aborting the launch.

1. **Globus authentication.** On first run, the pinned official
   `alcf-tokens==0.3.0` package performs ONE combined Globus login for Inference,
   IRI, Globus Compute, and Globus Transfer. It applies the ALCF all-services
   policy and stores refreshable per-service credentials in `/opt/data`.
2. **Dashboard auth gate.** Hashes the dashboard password (from
   `ALCF_DASHBOARD_PASSWORD`, or an auto-generated one printed once). Hermes
   refuses a non-loopback bind without this gate.
3. **Config render** (`render_config`), which builds the running config from
   `config/config.template.yaml`:
   - **Fetch a fresh inference token** — it's the `api_key`, and it rotates.
   - **Resolve the launch model's real context window**
     (`resolve_context_length.py`) from the gateway's `max_model_len`, because
     ALCF caps some models below their published spec and Hermes must be told the
     true window. Falls back to 128000 on any lookup failure.
   - **Generate the switchable model list** (`populate_models.py`) from the live
     catalog on both clusters: filter to chat models, drop anything below the 64k
     context floor, split each cluster into baseline vs. `-reasoning` providers,
     and stamp each provider's per-response output cap (`ALCF_MAX_TOKENS` /
     `ALCF_REASONING_MAX_TOKENS`). On discovery failure it emits the committed
     static fallback block instead. (See **Switching models** above.)
   - **Context-floor guard.** If the *launch* model's real window is under 64k,
     refuse to start (exit 78) with an actionable list of valid ≥64k models,
     instead of letting Hermes throw a raw stacktrace at load.
   - **Launch-provider resolution.** Find which generated provider lists the
     launch model and point the top-level `model:` block at it
     (`custom:alcf-sophia`, `custom:alcf-sophia-reasoning`, …), so the very first
     turn inherits the correct per-model output cap.
   - **Splice + substitute** the providers block into the template and write the
     final config.
4. **Model availability banner** (`populate_models.py --status-report`, unless
   `ALCF_SHOW_MODEL_STATUS=0`): classifies every offered model as **LIVE**
   (loaded on GPU now), **QUEUED** (a job exists but hasn't started — the
   scheduler's estimated start is printed, and can be hours out) or **OFFLINE**
   (not loaded, not scheduled — requests 503), and prints each model's context
   window. Models dropped for being under the 64k floor are listed too, so the
   box's list can be reconciled against the ALCF web UI.
5. **Seed / refresh skills, memory, and SOUL.** ALCF skills, `MEMORY.md`, and the
   agent's `SOUL.md` identity are image-managed: refreshed from the image on each
   start **only if you haven't edited your copy** (tracked by checksum stamps), so
   knowledge-base fixes reach existing volumes without clobbering user edits. A
   stock/legacy Hermes `SOUL.md` with no ALCF stamp is replaced so the ALCF
   identity always lands.
6. **Token-refresh loop + launch.** A background loop re-renders the config with a
   fresh inference token every 6h (tokens last 48h; a full re-auth is required
   every 30 days). If a refresh fails — usually the 30-day limit — it logs a loud
   banner and drops a status file the agent surfaces *in chat*. Then the Hermes
   dashboard starts behind the loopback-only Caddy HTTP proxy.

## Memory & documentation

The agent's ALCF knowledge is delivered in two complementary tiers:

1. **`MEMORY.md` — always-injected knowledge base.** A curated distillation of
   ALCF facts (endpoints, auth model, systems, "you are an ALCF support agent")
   that Hermes injects into every turn. Seeded from `memory/MEMORY.md` into the
   data volume on first run; the user can edit it and their edits are preserved.
   Keep it compact — it costs tokens on every message.

2. **`docs/` — full ALCF docs snapshot, read on demand.** `scripts/fetch_docs.py`
   pulls full ALCF user-guide pages (inference, IRI API, running jobs, example
   scripts, Polaris/Aurora getting-started, filesystems, allocations) into
   `docs/`, baked into the image at `/opt/alcf/docs/`. These are the
   authoritative detail source: the agent searches/reads them with its file tool
   when a question needs specifics. `MEMORY.md` carries an index of these pages
   so the agent knows what's available and when to read each. A nightly GitHub
   Action refreshes the snapshot, so `:latest` is never more than a day stale.

Why not a vector/RAG memory store? The official Hermes image ships only the
built-in `MEMORY.md` memory plus opt-in external providers; there is no bundled
local RAG backend. The `MEMORY.md` (orientation) + on-disk docs (deep lookup by
the file tool) design is backend-agnostic, needs no extra infrastructure, and
works out of the box. To add retrieval later, configure an external memory
provider (`hermes memory setup`) and ingest `docs/` into it.

## Switching models

The ALCF Inference Service serves models across four clusters (**Sophia**, vLLM;
**Metis**, **Minerva**, and **Tara**, API), but its gateway doesn't expose the standard OpenAI
`/v1/models` discovery path in the place Hermes expects. So instead of shipping a
hand-maintained list, the container **generates the switchable model list at
startup** from the live ALCF catalog (`scripts/populate_models.py`) and writes it
into the running config. This means the dropdown tracks what ALCF actually serves,
without a rebuild.

In the web chat, click the **MODEL** selector and pick a provider + model. The
providers are named by cluster and reasoning class:

- **`alcf-sophia`** — Sophia plain-chat models (e.g. `argonne/AuroraGPT-IT-v4-0125`)
- **`alcf-sophia-reasoning`** — Sophia reasoning models (`google/gemma-4-31B-it`
  (default), `openai/gpt-oss-120b`, `openai/gpt-oss-20b`,
  `nvidia/nemotron-3-super-120b`, `arcee-ai/Trinity-Large-Thinking-…`)
- **`alcf-metis`** / **`alcf-metis-reasoning`** — the Metis equivalents (drop the
  Metis providers with `-e ALCF_ENABLE_METIS=0`)
- **`alcf-minerva-reasoning`** — Minerva models (`gpt-oss-120b`, `inkling-bf16`,
  `nemotron-3-ultra`; drop with `-e ALCF_ENABLE_MINERVA=0`)
- **`alcf-tara-reasoning`** — Tara models (`nemotron-3-ultra`,
  `openai/gpt-oss-20b`)

Tara is now serving both listed models. Its catalog does not yet publish complete
context-window/reasoning metadata, so the box includes only these independently
verified models with conservative safe context windows rather than guessing for
future catalog additions.

Three behaviors are worth knowing:

1. **64k context floor.** Hermes refuses to load any model whose *real* serving
   window is below 64,000 tokens. ALCF caps many models well under that (all
   Llama 3.x/4, Mixtral, Devstral, Mistral-Large-2407 serve at 16k–32k; Metis
   `Mistral-Large-3-675B` serves at just **8192**), so those are **intentionally
   excluded** from the list — they would be broken dropdown entries. The
   generator reads each model's true window from the gateway rather than trusting
   the published spec, and the startup banner names the excluded models with
   their windows so you can reconcile the box's list against the ALCF web UI.

2. **Reasoning vs. plain chat is a separate provider, with a bigger output cap.**
   Reasoning models (gpt-oss, gemma-4, nemotron-3-super, `*-Thinking`) spend part
   of the `max_tokens` output budget on a hidden reasoning channel, so a small cap
   can leave them returning empty responses. The generator detects reasoning
   models (Minerva's `capabilities.reasoning.supported` where available, else the
   gateway's `reasoning_parser` field, plus an id heuristic for models served
   without either) and puts them in the `-reasoning` provider with a larger
   per-response cap — `ALCF_REASONING_MAX_TOKENS` (default **12288**) vs.
   `ALCF_MAX_TOKENS` (default **2048**) for plain chat. The launch model is
   automatically pointed at whichever provider matches its class, so the first
   turn already gets the right cap.

3. **LIVE vs. QUEUED vs. OFFLINE (the HTTP 503 you might see).** All models share
   the same endpoint + Globus token, so *switching* is instant — but ALCF only
   keeps a subset loaded on GPU at any moment. At startup the container prints an
   **availability banner** (`populate_models.py --status-report`) that classifies
   every offered model and shows its context window:

   - **LIVE** — loaded on GPU, answers immediately.
   - **QUEUED** — a job exists but hasn't started. The banner prints the
     scheduler's estimated start, which can be **hours** away when the cluster has
     no free nodes.
   - **OFFLINE** — not loaded and not scheduled; requests return
     `HTTP 503 "... is offline."`

   A model that is merely warming up returns `HTTP 503 "online but not ready"` for
   ~10–15 min and then works — that looks like a failure but isn't. A whole
   cluster can also be down, in which case *every* model on it shows OFFLINE.
   Suppress the banner with `-e ALCF_SHOW_MODEL_STATUS=0`.

If the live catalog is unreachable at startup, the generator falls back to a
committed static list in `config/config.template.yaml`, so the container always
comes up with a usable (if possibly slightly stale) set of models.

For the **full live catalog** (embeddings, GenSLM science models, everything —
including the sub-64k models that can't be the agent's brain), just ask the
agent: *"what models are available on ALCF inference?"* — it queries the service
directly via its `alcf-inference-service` skill.

---

## Development

See [docs/DESIGN.md](docs/DESIGN.md) for architecture, the auth model, and the
verification log (spike results proving inference + agentic tool use work).
Experimental cluster-resident agents and development campaigns are maintained
separately in the private `alcf-agent-experiments` repository; this repository
contains only the laptop product.

Build locally:

```bash
docker build -t alcf-agent:dev .
```

## Status

**Working, verified end-to-end in a real container.** A `docker run` does the
Globus login, persists the token to the volume (subsequent starts skip re-auth),
serves an auth-gated web chat, and the in-container agent drives ALCF inference
(default `google/gemma-4-31B-it`) *with working tool calls*. Remaining polish is
tracked in
[docs/DESIGN.md](docs/DESIGN.md) → "Open items".

---

## License & disclaimer

Licensed under the **Apache License, Version 2.0** — see [LICENSE](LICENSE) and
[NOTICE](NOTICE). This includes an explicit disclaimer of warranty and
limitation of liability.

This is an **independent community project**, not an official ALCF/Argonne/DOE
product, and it runs an **autonomous AI agent that can be wrong and that acts
with your credentials**. Before using it, read **[DISCLAIMER.md](DISCLAIMER.md)** —
it covers the non-affiliation, the agent's autonomy and fallibility, and your
responsibility for the actions it takes on your behalf.
