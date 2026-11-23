# ALCF Agent in a Box  (Option A: extend the official Hermes image)
#
# We build ON TOP of the official, hardened Hermes image
# (nousresearch/hermes-agent) rather than reinventing its build. That image
# already handles the fixed SQLite build, s6-overlay supervision, editable
# install, config/skills seeding (docker/stage2-hook.sh), the /opt/data volume,
# and the web dashboard. On top of it we add, in order:
#
#   0. Caddy, a loopback HTTP proxy in front of the dashboard, + its Caddyfile.
#   1. The ALCF Python deps (alcf-tokens, globus-compute-sdk, requests),
#      installed into the sealed base venv with the base's own pinned uv, behind
#      a version-drift guard and a smoke import.
#   2. A record of the base's Hermes version (/opt/alcf/.hermes_version) for
#      the startup banner.
#   3. ALCF content baked as image defaults: skills, curated MEMORY.md, a docs
#      snapshot, the config template, SOUL.md, the runtime helper scripts
#      (Globus/IRI facility helpers, remote bash + its MCP server, context-window
#      and model-catalog resolvers, generic-inference validator) and our
#      entrypoint, which does first-run Globus auth, renders the config with a
#      fresh token, and launches the dashboard.
#   4. The ALCF-agent-box git revision + build date (/opt/alcf/.alcf_version).
#   5. ALCF_* ENV defaults, a healthcheck through Caddy, and the entrypoint,
#      run as the non-root hermes user.
#
# The Hermes install itself is used unmodified: upstream Hermes strips `name`
# from role:tool messages (which the ALCF vLLM gateway rejects with HTTP 422),
# so no local source patch is needed.
#
# Pin the base by an explicit release tag for reproducibility (not :latest,
# which drifts and could move the venv/plugin paths or config schema the
# entrypoint relies on). Bump deliberately after re-verifying tool calls and the
# generated provider config still work.
ARG HERMES_BASE=nousresearch/hermes-agent:v0.23.2
FROM ${HERMES_BASE}

# The base image sets USER/ENV/ENTRYPOINT for the stock Hermes runtime. We need
# root to install deps into its venv and drop in content.
USER root

# ---------------------------------------------------------------------------
# 0. Caddy (loopback HTTP proxy). Browsers treat http://localhost as a secure
#    context, so clipboard access works without local certificates. The host
#    must publish the port on 127.0.0.1 only, as shown in the README.
# ---------------------------------------------------------------------------
COPY --from=caddy:2 /usr/bin/caddy /usr/bin/caddy
COPY config/Caddyfile /opt/alcf/Caddyfile

# ---------------------------------------------------------------------------
# 1. Official alcf-tokens package + Globus Compute SDK. alcf-tokens provides
#    the `alcf-tokens` CLI (login / get-token) and the `alcf_tokens.auth`
#    Python API (get_access_token / get_service_authorizer) used by the runtime
#    helpers. Pin the exact released wheel for reproducibility; bump
#    deliberately after verifying the new version's scope/policy contract.
#
#    How: the base's /opt/hermes/.venv is built and sealed by upstream's
#    package manager (pyvenv.cfg: hermes-pm-built = true); it has no pip, and
#    uv is internal PM tooling, NOT on PATH. We run the base's own pinned uv
#    (/opt/hermes/tools/uv-<version>-linux-<arch>/uv, located by glob so it
#    works on amd64 and arm64; exactly one must exist), so the installer is the
#    same one that built the venv. Version-drift guard: every package already
#    in the venv must keep its exact version (additions are fine), otherwise
#    the build fails and prints what changed -- our deps may never silently
#    upgrade or remove the upstream sealed environment.
# ---------------------------------------------------------------------------
RUN set -eu; \
    py=/opt/hermes/.venv/bin/python; \
    uv=""; n=0; \
    for c in /opt/hermes/tools/uv-*/uv; do \
        if [ -f "$c" ] && [ -x "$c" ]; then uv="$c"; n=$((n + 1)); fi; \
    done; \
    if [ "$n" -ne 1 ]; then \
        echo "ERROR: expected exactly one executable /opt/hermes/tools/uv-*/uv in the base image, found $n" >&2; \
        ls -d /opt/hermes/tools/uv-* >&2 || true; \
        exit 1; \
    fi; \
    echo "Using the base image's pinned uv: $uv ($("$uv" --version))"; \
    "$uv" pip freeze --python "$py" > /tmp/venv-before.txt; \
    if [ ! -s /tmp/venv-before.txt ]; then \
        echo "ERROR: empty package snapshot of $py; refusing to install without a drift guard" >&2; \
        exit 1; \
    fi; \
    "$uv" pip install --python "$py" --no-cache \
        alcf-tokens==0.3.0 globus-compute-sdk requests; \
    "$uv" pip freeze --python "$py" > /tmp/venv-after.txt; \
    changed="$(grep -vxFf /tmp/venv-after.txt /tmp/venv-before.txt || true)"; \
    if [ -n "$changed" ]; then \
        echo "ERROR: installing the ALCF deps changed or removed packages in the sealed base venv:" >&2; \
        echo "$changed" >&2; \
        diff /tmp/venv-before.txt /tmp/venv-after.txt >&2 || true; \
        exit 1; \
    fi; \
    rm -f /tmp/venv-before.txt /tmp/venv-after.txt; \
    "$py" -c "import hermes_cli, alcf_tokens.auth, globus_compute_sdk, requests, ruamel.yaml"; \
    /opt/hermes/.venv/bin/alcf-tokens --help >/dev/null

# Record the base's Hermes version line (e.g. "Hermes Agent v0.23.2
# (2026.9.24) · upstream ae2a81cc") for the startup banner; v0.23.2 no longer
# ships /opt/hermes/.hermes_build_sha. Captured once here so the entrypoint
# never runs `hermes --version` at boot. Uses the venv binary (not the
# /opt/hermes/bin exec shim) and a throwaway HOME/HERMES_HOME so no cache files
# land in the image. An empty version line fails the build.
RUN set -eu; \
    tmp="$(mktemp -d)"; \
    out="$(HOME="$tmp" HERMES_HOME="$tmp" /opt/hermes/.venv/bin/hermes --version)"; \
    rm -rf "$tmp"; \
    line="$(printf '%s\n' "$out" | sed -n 1p)"; \
    if [ -z "$line" ]; then \
        echo "ERROR: 'hermes --version' printed no version line" >&2; \
        exit 1; \
    fi; \
    printf '%s\n' "$line" > /opt/alcf/.hermes_version; \
    echo "Recorded Hermes version: $line"

# ---------------------------------------------------------------------------
# 2. ALCF content (changes most often -> last for cache friendliness).
#    /opt/alcf is the staging area; the entrypoint copies skills + MEMORY.md
#    into $HERMES_HOME (/opt/data) on first run so they land on the durable
#    volume without clobbering user edits.
# ---------------------------------------------------------------------------
COPY skills/  /opt/alcf/skills/
COPY memory/  /opt/alcf/memory/
COPY docs/    /opt/alcf/docs/
COPY config/config.template.yaml /opt/alcf/config.template.yaml
COPY config/SOUL.md /opt/alcf/SOUL.md
COPY scripts/entrypoint.sh /opt/alcf/entrypoint.sh
COPY scripts/iri_hello_world.py /opt/alcf/iri_hello_world.py
COPY scripts/alcf_facility.py /opt/alcf/alcf_facility.py
COPY scripts/alcf_remote_bash.py /opt/alcf/alcf_remote_bash.py
COPY scripts/alcf_bash_mcp.py /opt/alcf/alcf_bash_mcp.py
COPY scripts/resolve_context_length.py /opt/alcf/resolve_context_length.py
COPY scripts/populate_models.py /opt/alcf/populate_models.py
COPY scripts/generic_inference.py /opt/alcf/generic_inference.py
RUN chmod +x /opt/alcf/entrypoint.sh /opt/alcf/iri_hello_world.py \
             /opt/alcf/alcf_facility.py /opt/alcf/alcf_remote_bash.py \
             /opt/alcf/alcf_bash_mcp.py \
             /opt/alcf/resolve_context_length.py /opt/alcf/populate_models.py \
             /opt/alcf/generic_inference.py \
    && chown -R hermes:hermes /opt/alcf

# Bake the ALCF-agent-box git revision + build date so the container can print
# an unambiguous version banner at startup ("am I running the right image?").
# CI passes ALCF_GIT_SHA=${{ github.sha }}; a bare local build leaves it "dev".
ARG ALCF_GIT_SHA=dev
ARG ALCF_BUILD_DATE=unknown
RUN printf '%s\n%s\n' "${ALCF_GIT_SHA}" "${ALCF_BUILD_DATE}" > /opt/alcf/.alcf_version \
    && chown hermes:hermes /opt/alcf/.alcf_version

# ALCF defaults (override at `docker run` with -e).
# Default model: muse-spark-bf16 on Tara (1M-token context, reasoning; it gets
# the alcf-tara-reasoning provider's 12288-token output cap). TRADE-OFF: Tara
# models are not kept "always hot", so the first request after a quiet period
# may wait for a cold start (HTTP 503 until loaded). To restore the previous
# always-hot default, run with -e ALCF_CLUSTER=sophia -e
# ALCF_MODEL=google/gemma-4-31B-it. All models remain switchable in the
# dashboard.
#
# Notable OPTIONAL run-time envs (unset by default; see entrypoint.sh):
#   ALCF_BASH_ACCOUNT       default ALCF project for the MCP `bash` tool (else
#                           the model asks for / looks up one and it sticks)
#   ALCF_BASH_ENDPOINT/QUEUE/WALLTIME/TIMEOUT/MAX_OUTPUT  MCP bash tuning
#   ALCF_ENABLE_MCP_BASH=0  drop the MCP bash tool from the config
#   ALCF_DELEGATION_MODEL   model for delegate_task subagents (default: the
#                           launch model). Recommended for heavy build/test
#                           sessions: google/gemma-4-26B-A4B-it (262k context).
#   TERMINAL_MAX_FOREGROUND_TIMEOUT  Hermes terminal-tool ceiling (seconds) on a
#                           per-call `timeout`. Stock Hermes caps it at 600,
#                           which is shorter than a remote-bash cold start +
#                           queue wait + build (helper --timeout default 1200).
#                           1800 here lets both the config default below and a
#                           model-requested long wait through. Must stay >= the
#                           `terminal.timeout` in config.template.yaml.
ENV ALCF_MODEL=muse-spark-bf16 \
    ALCF_CLUSTER=tara \
    ALCF_MAX_TOKENS=2048 \
    ALCF_REASONING_MAX_TOKENS=12288 \
    ALCF_DASHBOARD_PORT=8787 \
    ALCF_DASHBOARD_INTERNAL_PORT=9119 \
    ALCF_ENABLE_IRI=1 \
    ALCF_ENABLE_GLOBUS_COMPUTE=1 \
    TERMINAL_MAX_FOREGROUND_TIMEOUT=1800 \
    HERMES_HOME=/opt/data \
    XDG_DATA_HOME=/opt/data/.local/share \
    XDG_CONFIG_HOME=/opt/data/.config

EXPOSE 8787

# Health: the public port is HTTP through Caddy. We accept ANY HTTP
# response (including the 401 auth challenge) as "server up" — we only care
# that Caddy + the dashboard behind it are answering. start-period covers
# first-run Globus auth + dashboard warmup.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -s -o /dev/null -w '%{http_code}' \
        "http://127.0.0.1:${ALCF_DASHBOARD_PORT}/" 2>/dev/null \
      | grep -qE '^[1-5][0-9]{2}$' || exit 1

# Our entrypoint does the ALCF first-run flow, then hands off to the stock
# `hermes dashboard`. We run it as the non-root hermes user (the base image's
# supervised services also drop to this user).
USER hermes
WORKDIR /opt/data
ENTRYPOINT ["/opt/alcf/entrypoint.sh"]
