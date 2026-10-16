#!/usr/bin/env bash
# ALCF Agent container entrypoint (Option A: on top of nousresearch/hermes-agent).
#
#   1. Resolve the ALCF inference base_url from ALCF_CLUSTER.
#   2. First-run: ONE combined Globus login via the official alcf-tokens CLI
#      authenticates ALL ALCF services (Inference + IRI + Globus Compute +
#      Transfer) in a single browser visit and a single pasted code. The
#      default login always requests all four services; feature flags
#      (ALCF_ENABLE_IRI, ALCF_ENABLE_GLOBUS_COMPUTE) gate RUNTIME exposure
#      but do not narrow the consent.
#   3. Render the Hermes config with a fresh inference access token.
#   4. Seed skills + curated MEMORY.md into $HERMES_HOME (idempotent).
#   5. Background token-refresh loop (tokens last 48h; refresh every 6h).
#   6. Launch the Hermes web dashboard (the local web chat).
#
# $HERMES_HOME (/opt/data) and ~/.globus persist across restarts, so step 2
# is skipped once tokens exist.
#
# GENERIC MODE: setting INFERENCE_URL switches the LLM brain from the ALCF
# Inference Service to any OpenAI-compatible endpoint (see step 1b). The
# combined Globus login still happens because IRI, Globus Compute, and
# Transfer keep working unchanged; only the ALCF-specific inference catalog,
# status banner, and token refresh are skipped.
set -euo pipefail

ALCF_DIR=/opt/alcf
VENV_BIN=/opt/hermes/.venv/bin
PY="$VENV_BIN/python"
ALCF_TOKENS="$VENV_BIN/alcf-tokens"
NERSC_TOKENS="$VENV_BIN/nersc-tokens"
HERMES_HOME="${HERMES_HOME:-/opt/data}"
CONFIG_OUT="$HERMES_HOME/config.yaml"

log() { printf '\033[36m[alcf-agent]\033[0m %s\n' "$*"; }
err() { printf '\033[31m[alcf-agent]\033[0m %s\n' "$*" >&2; }

# --- 0. Version banner ------------------------------------------------------
# Print the ALCF-agent-box git revision + build date baked at image build, plus
# the underlying Hermes SHA, so you can confirm exactly which image is running.
_alcf_sha="dev"; _alcf_date="unknown"
if [[ -f "$ALCF_DIR/.alcf_version" ]]; then
  _alcf_sha="$(sed -n 1p "$ALCF_DIR/.alcf_version")"
  _alcf_date="$(sed -n 2p "$ALCF_DIR/.alcf_version")"
fi
_hermes_sha="$(cat /opt/hermes/.hermes_build_sha 2>/dev/null | cut -c1-12 || true)"
printf '\033[1;36m╔══════════════════════════════════════════════════════════╗\033[0m\n'
printf '\033[1;36m║\033[0m  ALCF Agent in a Box\n'
printf '\033[1;36m║\033[0m  version : %s  (built %s)\n' "$_alcf_sha" "$_alcf_date"
printf '\033[1;36m║\033[0m  hermes  : %s\n' "${_hermes_sha:-unknown}"
printf '\033[1;36m╚══════════════════════════════════════════════════════════╝\033[0m\n'
# Drop a chat-readable copy on the volume (refreshed every start so it always
# reflects the running image; the agent reads it when asked "what version?").
mkdir -p "${HERMES_HOME:-/opt/data}" 2>/dev/null || true
printf 'ALCF Agent in a Box\nalcf-agent-box commit: %s\nbuilt: %s\nhermes commit: %s\nrepo: https://github.com/jtchilders-ai-assistant/alcf-agent-box\n' \
  "$_alcf_sha" "$_alcf_date" "${_hermes_sha:-unknown}" > "${HERMES_HOME:-/opt/data}/ALCF_VERSION.txt" 2>/dev/null || true

# --- 1. Resolve inference base_url ------------------------------------------
case "${ALCF_CLUSTER:-sophia}" in
  sophia) ALCF_BASE_URL="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1" ;;
  metis)  ALCF_BASE_URL="https://inference-api.alcf.anl.gov/resource_server/metis/api/v1" ;;
  *) err "Unknown ALCF_CLUSTER='$ALCF_CLUSTER' (expected sophia|metis)"; exit 1 ;;
esac
export ALCF_BASE_URL

mkdir -p "$HERMES_HOME"

# First-run multi-facility setup: only if no product config exists.
if [[ ! -f "$HERMES_HOME/agent-box/config.yaml" ]]; then
  PYTHONPATH="/opt/alcf:${PYTHONPATH:-}" "$PY" /opt/alcf/agent_box_setup.py --home "$HERMES_HOME"
fi

# --- 1b. Generic inference mode (opt-in via INFERENCE_URL) --------------------
# Setting INFERENCE_URL points the agent's LLM brain at ANY OpenAI-compatible
# endpoint instead of the ALCF Inference Service. This changes ONLY the model
# provider: the combined Globus login below still happens (step 2), and IRI,
# Globus Compute, Transfer, and the dashboard behave exactly as in ALCF mode.
# What generic mode skips is the ALCF-specific part of inference: the gateway
# catalog/status banner, max_model_len resolution, and the 6h Globus token
# refresh (the supplied INFERENCE_TOKEN is static and user-managed).
# FAIL-CLOSED: any partial generic configuration is refused with exit 78.
GENERIC_MODE=0
if [[ -n "${INFERENCE_URL:-}" ]]; then
  GENERIC_MODE=1
  missing=""
  [[ -n "${INFERENCE_TOKEN:-}" ]] || missing+=" INFERENCE_TOKEN"
  [[ -n "${INFERENCE_MODEL:-}" ]] || missing+=" INFERENCE_MODEL"
  if [[ -n "$missing" ]]; then
    err "INFERENCE_URL is set (generic inference mode) but required setting(s) are missing:$missing"
    err "In generic mode INFERENCE_TOKEN and INFERENCE_MODEL are both mandatory."
    err "Unset INFERENCE_URL to use the default ALCF Inference Service."
    exit 78   # EX_CONFIG: launch configuration is invalid
  fi
  if ! printf '%s' "${INFERENCE_MAX_TOKENS:-4096}" | grep -Eq '^[1-9][0-9]*$'; then
    err "INFERENCE_MAX_TOKENS must be a positive integer (got '${INFERENCE_MAX_TOKENS}')."
    exit 78   # EX_CONFIG
  fi
  log "Generic inference mode: endpoint from INFERENCE_URL, model '$INFERENCE_MODEL'"
  log "(IRI / Globus Compute / Transfer still use the combined ALCF Globus login; the token itself is never logged)"
elif [[ -n "${INFERENCE_TOKEN:-}" || -n "${INFERENCE_MODEL:-}" ]]; then
  # PARTIAL GENERIC CONFIGURATION: generic settings are set but INFERENCE_URL
  # is absent. Silently entering default ALCF mode here would ignore settings
  # the user clearly meant for a generic endpoint (and would then use the ALCF
  # gateway with them pending), so refuse the state outright. Values are never
  # printed — only the names of the offending variables.
  err "INFERENCE_TOKEN and/or INFERENCE_MODEL are set but INFERENCE_URL is not — this is a partial generic inference configuration."
  err "Either set INFERENCE_URL to complete generic mode, or unset INFERENCE_TOKEN/INFERENCE_MODEL to use the default ALCF Inference Service."
  exit 78   # EX_CONFIG: partial generic configuration is refused
fi

# --- 2. First-run Globus auth (interactive, idempotent) ---------------------
# ONE alcf-tokens login covers ALL ALCF services (Inference, IRI, Globus
# Compute, and Transfer) in a single browser visit + single pasted code.
# This is the official ALCF policy as of alcf-tokens==0.3.0: the default
# login always requests all four services. Feature flags (ALCF_ENABLE_IRI,
# ALCF_ENABLE_GLOBUS_COMPUTE) gate RUNTIME feature exposure but do NOT alter
# the consent scope — one fresh login is all that is needed on upgrade.
# Note: alcf-tokens login has no --force flag; re-run to trigger a new login.
authed_inference() { "$ALCF_TOKENS" get-token inference >/dev/null 2>&1; }

if ! authed_inference; then
  log "First-time setup: ONE Globus login for all ALCF services"
  log "(Inference + IRI + Globus Compute + Transfer — ALCF policy default)"
  log "A single URL will be printed — open it, log in with your ALCF/Globus"
  log "account, and paste the ONE authorization code back here."
  echo
  "$ALCF_TOKENS" login
  echo
  if ! authed_inference; then
    err "ALCF authentication failed. Re-run the container to try again."
    exit 1
  fi
  log "ALCF authentication OK (single login covered all services)."
fi

# --- 2a. NERSC IRI Globus auth (conditional, separate token store) ---------
# NERSC uses a distinct Globus client and token cache. Selecting it in the
# persisted product config must therefore produce its own browser flow; an ALCF
# login never authorizes NERSC. Check the resolved config with the packaged
# helper rather than parsing YAML in shell. `get-token` checks/refreshes only
# the NERSC token cache and has no user-facing output here; it avoids prompting
# again on later container starts while preserving a real NERSC login on first
# use or after expiry.
nersc_enabled="$(PYTHONPATH="/opt/alcf:${PYTHONPATH:-}" "$PY" - "$HERMES_HOME/agent-box/config.yaml" <<'PY'
import sys
from agent_box_config import load_product_config, load_registry

config_path = sys.argv[1]
registry, _ = load_registry('/opt/alcf/config/facilities.yaml')
config = load_product_config(config_path, registry=registry)
print('1' if config['facilities'].get('nersc', {}).get('enabled') is True else '0')
PY
)"
nersc_authed() { "$NERSC_TOKENS" get-token iri --facilities nersc >/dev/null 2>&1; }
if [[ "$nersc_enabled" == "1" ]] && ! nersc_authed; then
  log "NERSC selected: separate Globus login is required for NERSC IRI."
  log "A NERSC authorization URL will be printed; it forces identity-provider login so use your NERSC account."
  "$NERSC_TOKENS" login iri --facilities nersc --prompt-login
  if ! nersc_authed; then
    err "NERSC authentication failed. Re-run the container to try again."
    exit 1
  fi
  log "NERSC authentication OK."
fi

# --- 3. Dashboard auth (required for a container 0.0.0.0 bind) ---------------
# Hermes refuses a non-loopback dashboard bind without an auth gate. We hash a
# password at start (plaintext never persisted). Password source, in order:
#   ALCF_DASHBOARD_PASSWORD env  ->  else auto-generate one and print it once.
ALCF_DASHBOARD_USER="${ALCF_DASHBOARD_USER:-alcf}"
if [[ -z "${ALCF_DASHBOARD_PASSWORD:-}" ]]; then
  ALCF_DASHBOARD_PASSWORD="$("$PY" -c 'import secrets; print(secrets.token_urlsafe(12))')"
  log "No ALCF_DASHBOARD_PASSWORD set — generated one for this session:"
  printf '\033[33m    dashboard login:  user=%s  password=%s\033[0m\n' "$ALCF_DASHBOARD_USER" "$ALCF_DASHBOARD_PASSWORD"
  log "(set -e ALCF_DASHBOARD_PASSWORD=... to choose your own and keep it stable)"
fi
export ALCF_DASHBOARD_USER
export ALCF_DASHBOARD_PASSWORD_HASH="$("$PY" -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('$ALCF_DASHBOARD_PASSWORD'))")"

# --- 4. Render config (Python; no envsubst in the base image) ---------------
render_config() {
  local token ctxlen providers_block
  local launch_model launch_provider render_base_url block_for_scan
  if [[ "$GENERIC_MODE" == "1" ]]; then
    # GENERIC MODE: static user-supplied endpoint/token/model. We skip the
    # ALCF-specific catalog discovery (populate_models.py), the gateway
    # max_model_len resolution, and the floor-guard fallback machinery —
    # generic_inference.py already validated the URL, authenticated against
    # <base>/models, enforced the 64k context floor, and refused (exit 78)
    # anything unusable. The token is never echoed to any log.
    token="$INFERENCE_TOKEN"
    launch_model="$INFERENCE_MODEL"
    launch_provider="custom:generic-inference"
    local gmanifest="$HERMES_HOME/.generic_inference_manifest.json"
    if ! INFERENCE_TOKEN="$token" "$PY" "$ALCF_DIR/generic_inference.py" \
           --base-url "$INFERENCE_URL" --model "$INFERENCE_MODEL" \
           --max-tokens "${INFERENCE_MAX_TOKENS:-4096}" \
           --output "$gmanifest" >/tmp/generic_inference.log 2>&1; then
      err "Generic inference setup failed — refusing to start. Details (token-free):"
      sed 's/^/  /' /tmp/generic_inference.log >&2 || true
      exit 78   # EX_CONFIG: endpoint/token/model/catalog is unusable
    fi
    # Surface the per-model ENABLED/DISABLED report in the startup log.
    while IFS= read -r _gline; do log "$_gline"; done </tmp/generic_inference.log
    ctxlen="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_context_length"])' "$gmanifest")"
    render_base_url="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_url"])' "$gmanifest")"
    # One named provider carrying EVERY catalog-enabled model with its real
    # context window (static mapping — discover_models off, like ALCF mode).
    # The api_key stays a ${...} placeholder; the token is substituted once at
    # template render, exactly like the ALCF path.
    providers_block="$("$PY" - "$gmanifest" "${INFERENCE_MAX_TOKENS:-4096}" <<'GPYEOF'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
lines = [
    "custom_providers:",
    "  - name: generic-inference",
    "    base_url: " + json.dumps(manifest["base_url"]),
    '    api_key: "${ALCF_ACCESS_TOKEN}"',
    "    discover_models: false",
    "    max_tokens: %d" % int(sys.argv[2]),
    "    models:",
]
for model_id, ctx in sorted(manifest["enabled_models"].items()):
    lines.append("      " + json.dumps(model_id) + ":")
    lines.append("        context_length: %d" % ctx)
print("\n".join(lines))
GPYEOF
)"
    case "$providers_block" in
      custom_providers:*) : ;;
      *) err "generic_inference produced no providers block; refusing to start"; exit 78 ;;
    esac
    block_for_scan="$providers_block"
  else
  token="$("$ALCF_TOKENS" get-token inference)"
  # Resolve the REAL serving context window for the LAUNCH model from the
  # server's max_model_len (ALCF caps some models below their published spec;
  # Hermes' family table would otherwise over-estimate it). Falls back to
  # $ALCF_CONTEXT_LENGTH / 128000 if the lookup fails. Never fatal.
  ctxlen="$(ALCF_INFER_TOKEN="$token" ALCF_PY="$PY" \
            "$PY" "$ALCF_DIR/resolve_context_length.py" \
            "$ALCF_BASE_URL" "${ALCF_MODEL:-google/gemma-4-31B-it}" 2>>/tmp/ctxlen.log)"
  # Guard: if the resolver printed nothing usable, hard-default here too.
  case "$ctxlen" in ''|*[!0-9]*) ctxlen=128000 ;; esac
  log "Model context window: $ctxlen tokens (from max_model_len; see /tmp/ctxlen.log)"

  # Generate the switchable-model `custom_providers:` block from the LIVE ALCF
  # catalog on every serving cluster (sophia + metis + minerva). populate_models.py
  # filters to chat models and assigns each its real serving context window; it
  # always prints a valid block (committed static fallback on any discovery
  # failure), so this is never fatal. ALCF_ENABLE_METIS=0 / ALCF_ENABLE_MINERVA=0
  # drop those providers.
  providers_block="$(ALCF_INFER_TOKEN="$token" ALCF_PY="$PY" \
            ALCF_ENABLE_METIS="${ALCF_ENABLE_METIS:-1}" \
            ALCF_ENABLE_MINERVA="${ALCF_ENABLE_MINERVA:-1}" \
            ALCF_MAX_TOKENS="${ALCF_MAX_TOKENS:-2048}" \
            ALCF_REASONING_MAX_TOKENS="${ALCF_REASONING_MAX_TOKENS:-12288}" \
            "$PY" "$ALCF_DIR/populate_models.py" 2>>/tmp/populate_models.log)"
  # Guard: a generated block MUST start with 'custom_providers:'. If somehow it
  # didn't (script crashed hard), leave $providers_block empty so the splice step
  # keeps the template's committed static fallback instead.
  case "$providers_block" in
    custom_providers:*) log "Model list generated from live catalog (see /tmp/populate_models.log)" ;;
    *) providers_block=""; err "populate_models produced no block; keeping static fallback list" ;;
  esac

  # --- Launch-model context floor guard ------------------------------------
  # Hermes hard-refuses to start any model whose context window is below its
  # MINIMUM_CONTEXT_LENGTH (64000). The ALCF gateway caps many models well below
  # that (all Llama 3.x/4, Mixtral, Devstral, Mistral-Large-2407 serve at
  # 16k-32k), so a user who overrides ALCF_MODEL to one of those would otherwise
  # hit a raw Hermes stacktrace at launch. Refuse EARLY with an actionable list
  # of valid models instead.
  #
  # ROBUST-AGAINST-ITS-OWN-FAILURE by construction:
  #   * $ctxlen was already defaulted to 128000 on ANY resolver failure (see the
  #     `case` above), and 128000 >= floor, so a broken/unreachable resolver
  #     NEVER trips this guard -- it fails OPEN (launch proceeds), never closed.
  #   * We only block on a genuine, positive sub-floor integer reading.
  #   * The valid-model list is derived from the block we just generated; if that
  #     extraction yields nothing (block empty/unparseable) we fall back to a
  #     hardcoded hint so the message is always useful. No step here can abort
  #     the launch except the one intentional `exit 78`.
  local floor=64000 launch_model valid_list
  launch_model="${ALCF_MODEL:-google/gemma-4-31B-it}"
  # Compare defensively: only treat $ctxlen as sub-floor when it is a clean
  # positive integer AND strictly less than the floor. `case` already guarantees
  # $ctxlen is all-digits, but re-check so this block is safe if moved.
  if printf '%s' "$ctxlen" | grep -Eq '^[0-9]+$' && [ "$ctxlen" -gt 0 ] && [ "$ctxlen" -lt "$floor" ]; then
    # Extract the sophia provider's model ids from the generated block for the
    # hint. `|| true` so a grep miss can't abort under `set -e`.
    valid_list="$(printf '%s\n' "$providers_block" \
      | sed -n '/name: alcf-sophia/,/name: alcf-metis/p' \
      | grep -E '^      [A-Za-z0-9].*:$' \
      | sed -E 's/^ +//; s/:$//' \
      | sort | awk 'NR>1{printf ", "}{printf "%s",$0}END{print ""}' 2>/dev/null || true)"
    if [ -z "$valid_list" ]; then
      valid_list="openai/gpt-oss-120b, openai/gpt-oss-20b, google/gemma-4-31B-it, google/gemma-4-26B-A4B-it, nvidia/nemotron-3-super-120b, argonne/AuroraGPT-IT-v4-0125"
    fi
    err "======================================================================"
    err "REFUSING TO START: model '$launch_model' has a context window of"
    err "$ctxlen tokens on the ALCF gateway, below Hermes' required minimum of"
    err "$floor tokens (Hermes rejects smaller windows at load)."
    err ""
    err "The ALCF inference service caps many models (all Llama 3.x/4, Mixtral,"
    err "Devstral, Mistral-Large-2407) below 64k, which makes them unusable in"
    err "Hermes regardless of their published spec."
    err ""
    err "Set ALCF_MODEL to one of these >=64k models and restart the container:"
    err "  $valid_list"
    err "======================================================================"
    exit 78   # EX_CONFIG: launch configuration is invalid
  fi
  # --- end launch-model context floor guard --------------------------------

  # --- Resolve the launch model's named provider (per-model output cap) -----
  # The top-level model: block resolves through a NAMED custom provider so the
  # LAUNCH turn inherits that provider's max_tokens (reasoning models get 12288,
  # plain chat 2048). Hermes has no top-level per-model max_tokens, so this is how
  # turn-one gets the right cap; /model switches on the dashboard then resolve the
  # cap per-provider on their own.
  #
  # Derive the provider from the SAME block that will be written (single source of
  # truth): find which `- name: alcf-*` section lists the launch model id, so a
  # model the live catalog classified as reasoning (e.g. nemotron via its
  # reasoning_parser) lands in the -reasoning provider even though its id has no
  # reasoning keyword. Falls back to the id-heuristic Python resolver if the scan
  # finds nothing (e.g. block unreadable), and finally to custom:alcf-sophia.
  local launch_provider block_for_scan
  # Scan the generated block if we have one; else the template's static fallback.
  if [ -n "$providers_block" ]; then
    block_for_scan="$providers_block"
  else
    block_for_scan="$(cat "$ALCF_DIR/config.template.yaml" 2>/dev/null || true)"
  fi
  launch_provider="$(printf '%s\n' "$block_for_scan" | awk -v m="$launch_model" '
    /^  - name: / { prov = $3; next }
    $0 ~ "^      " m ":$" { print prov; exit }
  ')"
  if [ -z "$launch_provider" ]; then
    # Fallback: ask the Python resolver (id heuristic + best-effort catalog).
    launch_provider="$(ALCF_INFER_TOKEN="$token" ALCF_PY="$PY" \
        ALCF_BASE_URL="$ALCF_BASE_URL" \
        "$PY" "$ALCF_DIR/populate_models.py" --launch-provider "$launch_model" \
        2>>/tmp/populate_models.log || true)"
    # awk printed a bare provider name (alcf-sophia-reasoning); the Python path
    # already prints the custom: prefix. Normalise: prefix bare names.
    case "$launch_provider" in custom:*) : ;; ?*) launch_provider="custom:$launch_provider" ;; esac
  else
    launch_provider="custom:$launch_provider"
  fi
  # Last-resort default so the config always has a valid provider.
  [ -n "$launch_provider" ] || launch_provider="custom:alcf-sophia"
  log "Launch model '$launch_model' -> provider $launch_provider (per-model output cap)"
  # --- end launch-model provider resolution --------------------------------
  render_base_url="$ALCF_BASE_URL"
  fi
  # --- end generic/ALCF branch ----------------------------------------------

  # --- Optional feature blocks (delegation subagents + MCP bash tool) -------
  # Both are appended AFTER the providers block and each is gated on a runtime
  # grep of the INSTALLED Hermes source: if this base image predates the
  # feature, we skip the block (with a log line) instead of risking an unknown
  # config key at dashboard startup. Same fail-open philosophy as the rest of
  # this script, and the same grep-gate pattern the Dockerfile patch layer uses.
  local extra_block=""

  # (a) delegation: route delegate_task subagents. Subagents get their own
  # FRESH context, so this is the context firewall for build/test loops; set
  # ALCF_DELEGATION_MODEL to a long-context model (e.g. google/gemma-4-26B-A4B-it,
  # 262k on Sophia) to give workers 2x the default window. Defaults to the
  # launch model (always valid, always hot enough to have been chosen).
  if grep -rqs "delegate_task" /opt/hermes/agent 2>/dev/null; then
    local delegation_model delegation_provider
    delegation_model="${ALCF_DELEGATION_MODEL:-$launch_model}"
    if [ "$delegation_model" = "$launch_model" ]; then
      delegation_provider="$launch_provider"
    elif [[ "$GENERIC_MODE" == "1" ]]; then
      # GENERIC MODE delegation: the ONLY provider that exists is
      # generic-inference, and the ONLY models on it are the catalog-enabled
      # ones generic_inference.py verified (the manifest's enabled_models
      # mapping — the provider block quotes model ids, so membership is
      # checked against the manifest, not by grepping the YAML block).
      # A delegation model not in that mapping has NO valid provider in
      # generic mode, and the ALCF path below (populate_models.py with
      # $token, plus the custom:alcf-sophia fallback) would leak the generic
      # INFERENCE_TOKEN to the ALCF gateway and route subagents to a provider
      # that does not exist in this config. Fail closed instead.
      if ! "$PY" -c 'import json,sys; sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1], encoding="utf-8"))["enabled_models"] else 1)' \
             "$gmanifest" "$delegation_model"; then
        err "======================================================================"
        err "REFUSING TO START: generic delegation model '$delegation_model'"
        err "(ALCF_DELEGATION_MODEL) is not one of the models served by the"
        err "generic inference endpoint. In generic mode there is no ALCF provider"
        err "to resolve it against, and silently falling back would leak the"
        err "generic token to the ALCF gateway."
        err ""
        err "Set ALCF_DELEGATION_MODEL to one of the ENABLED models reported above,"
        err "or unset it to delegate on the launch model '$launch_model'."
        err "======================================================================"
        exit 78   # EX_CONFIG: generic delegation model is unresolvable
      fi
      delegation_provider="custom:generic-inference"
    else
      # Same provider scan as the launch model (which named provider lists it),
      # with the same Python-resolver + static fallbacks.
      delegation_provider="$(printf '%s\n' "$block_for_scan" | awk -v m="$delegation_model" '
        /^  - name: / { prov = $3; next }
        $0 ~ "^      " m ":$" { print prov; exit }
      ')"
      if [ -z "$delegation_provider" ]; then
        delegation_provider="$(ALCF_INFER_TOKEN="$token" ALCF_PY="$PY" \
            ALCF_BASE_URL="$ALCF_BASE_URL" \
            "$PY" "$ALCF_DIR/populate_models.py" --launch-provider "$delegation_model" \
            2>>/tmp/populate_models.log || true)"
        case "$delegation_provider" in custom:*) : ;; ?*) delegation_provider="custom:$delegation_provider" ;; esac
      else
        delegation_provider="custom:$delegation_provider"
      fi
      [ -n "$delegation_provider" ] || delegation_provider="custom:alcf-sophia"
    fi
    extra_block+="$(cat <<EOF

delegation:
  provider: "$delegation_provider"
  model: "$delegation_model"
EOF
)"
    log "Delegation (subagents) -> $delegation_model via $delegation_provider"
  else
    log "Base Hermes has no delegate_task; skipping delegation config"
  fi

  # (b) mcp_servers: expose remote-bash to the model as a native `bash` tool
  # (one required arg, persistent shell semantics, warm node held by the
  # server) — the tool SHAPE these models were trained on, instead of the
  # error-prone alcf_remote_bash.py CLI syntax. See scripts/alcf_bash_mcp.py.
  # Values are baked at render time; ALCF_BASH_ACCOUNT may be empty, in which
  # case the model is told to pass account=<project> once (sticky).
  if [[ "${ALCF_ENABLE_GLOBUS_COMPUTE:-1}" != "0" && "${ALCF_ENABLE_MCP_BASH:-1}" != "0" ]] \
     && grep -rqs "mcp_servers" /opt/hermes/agent 2>/dev/null; then
    extra_block+="$(cat <<EOF

mcp_servers:
  alcf-bash:
    command: "/opt/hermes/.venv/bin/python"
    args: ["/opt/alcf/alcf_bash_mcp.py"]
    # Hermes-side (CLIENT) timeouts. These are SEPARATE from ALCF_BASH_TIMEOUT
    # below, which is the server's own per-command budget. Hermes defaults to a
    # 300s tool-call timeout, so without this a command that the server is
    # happily waiting 1200s for is killed by the client at 300s and the model
    # gets a bare TimeoutError with no result. Keep timeout > ALCF_BASH_TIMEOUT
    # so the server's own error (which explains what happened) always wins.
    timeout: $(( ${ALCF_BASH_TIMEOUT:-1200} + 120 ))
    connect_timeout: 60
    # Hermes pings idle MCP servers to detect stale sessions. This server is
    # single-threaded stdio: while a bash command is in flight it CANNOT answer
    # a ping, so a keepalive landing mid-command reads as a dead session,
    # triggers a reconnect and kills the in-flight call. At the stock 180s
    # cadence that fires during essentially every real build. Push the cadence
    # past the call budget so pings only ever land while genuinely idle.
    keepalive_interval: $(( ${ALCF_BASH_TIMEOUT:-1200} + 180 ))
    env:
      ALCF_BASH_ACCOUNT: "${ALCF_BASH_ACCOUNT:-}"
      ALCF_BASH_ENDPOINT: "${ALCF_BASH_ENDPOINT:-polaris}"
      ALCF_BASH_QUEUE: "${ALCF_BASH_QUEUE:-debug}"
      ALCF_BASH_WALLTIME: "${ALCF_BASH_WALLTIME:-1:00:00}"
      ALCF_BASH_TIMEOUT: "${ALCF_BASH_TIMEOUT:-1200}"
      ALCF_BASH_MAX_OUTPUT: "${ALCF_BASH_MAX_OUTPUT:-20000}"
      ALCF_ENABLE_GLOBUS_COMPUTE: "${ALCF_ENABLE_GLOBUS_COMPUTE:-1}"
EOF
)"
    log "MCP bash tool enabled (endpoint=${ALCF_BASH_ENDPOINT:-polaris}, account=${ALCF_BASH_ACCOUNT:-<unset — model asks/looks up>})"
    if [[ -z "${ALCF_BASH_ACCOUNT:-}" ]]; then
      log "TIP: set -e ALCF_BASH_ACCOUNT=<project> at docker run — the bash tool then charges that project by default and holds one warm compute node without asking."
    fi
  else
    if [[ "${ALCF_ENABLE_GLOBUS_COMPUTE:-1}" == "0" || "${ALCF_ENABLE_MCP_BASH:-1}" == "0" ]]; then
      log "MCP bash tool disabled by env flag"
    else
      log "Base Hermes has no mcp_servers support; skipping MCP bash tool"
    fi
  fi
  # --- end optional feature blocks ------------------------------------------

  # Render: substitute ${...} placeholders, and splice the generated providers
  # block over the sentinel-to-EOF region of the template. The generated block
  # still contains ${ALCF_ACCESS_TOKEN}, so we splice FIRST then substitute.
  ALCF_ACCESS_TOKEN="$token" \
  ALCF_BASE_URL="$render_base_url" \
  ALCF_MODEL="$launch_model" \
  ALCF_LAUNCH_PROVIDER="$launch_provider" \
  ALCF_MAX_TOKENS="${ALCF_MAX_TOKENS:-2048}" \
  ALCF_CONTEXT_LENGTH="$ctxlen" \
  ALCF_DASHBOARD_USER="$ALCF_DASHBOARD_USER" \
  ALCF_DASHBOARD_PASSWORD_HASH="$ALCF_DASHBOARD_PASSWORD_HASH" \
  ALCF_PROVIDERS_BLOCK="$providers_block" \
  ALCF_EXTRA_CONFIG_BLOCK="$extra_block" \
  "$PY" - "$ALCF_DIR/config.template.yaml" "$CONFIG_OUT" <<'PYEOF'
import os, sys, string
src, dst = sys.argv[1], sys.argv[2]
tmpl = open(src, encoding="utf-8").read()

# Splice: replace from the sentinel line (inclusive) to EOF with the generated
# custom_providers block, when one was produced. The sentinel is a YAML comment,
# so the template stays valid even if this splice is skipped.
SENTINEL = "#__ALCF_CUSTOM_PROVIDERS__"
block = os.environ.get("ALCF_PROVIDERS_BLOCK", "").strip()
if block:
    idx = tmpl.find(SENTINEL)
    if idx != -1:
        # cut back to the start of the sentinel's line
        line_start = tmpl.rfind("\n", 0, idx) + 1
        tmpl = tmpl[:line_start] + block + "\n"
    else:
        sys.stderr.write("[entrypoint] sentinel not found; using template as-is\n")

# Append the optional feature blocks (delegation / mcp_servers) AFTER the
# providers splice — they are top-level keys and gated upstream on the base
# image actually supporting them.
extra = os.environ.get("ALCF_EXTRA_CONFIG_BLOCK", "").strip()
if extra:
    tmpl = tmpl.rstrip("\n") + "\n\n" + extra + "\n"

# Substitute ${VAR} placeholders from the environment; leave unknown ones as-is.
out = string.Template(tmpl).safe_substitute(os.environ)
open(dst, "w", encoding="utf-8").write(out)
PYEOF
  # The rendered config embeds the live inference api_key. FORCE mode 0600 on
  # every render — including re-renders over an existing file, where the file
  # mode comes from the pre-existing inode and open(mode=) alone would not
  # tighten it.
  chmod 600 "$CONFIG_OUT"
}
render_config
if [[ "$GENERIC_MODE" == "1" ]]; then
  log "Config rendered -> $CONFIG_OUT (mode 0600; generic endpoint from INFERENCE_URL, model '$INFERENCE_MODEL')"
else
  log "Config rendered -> $CONFIG_OUT (mode 0600; cluster=$ALCF_CLUSTER model=${ALCF_MODEL:-google/gemma-4-31B-it})"
fi

# SECRET HYGIENE (generic mode): INFERENCE_TOKEN arrived in the container
# environment (docker -e) and render_config has already substituted it into
# the mode-0600 config — its ONLY intended durable home. Left in place it
# would be inherited by the dashboard, Caddy, the MCP server, terminal tools,
# and every long-lived descendant, so drop it from the entrypoint environment
# BEFORE any of those start. `unset` removes it from the shell's parameters
# AND the exported environment. No top-level shell variable retains the value
# ($token inside render_config is `local` and died with the function). The
# token remains intentionally persisted in $CONFIG_OUT (chmod 600 above).
# ALCF mode never has INFERENCE_TOKEN set (partial config exits 78 above), so
# this is scoped to generic mode and cannot affect the ALCF refresh loop's
# re-render of render_config.
if [[ "$GENERIC_MODE" == "1" ]]; then
  unset INFERENCE_TOKEN
fi

# --- 4b. Model availability status banner ------------------------------------
# The model dropdown lists the full ALCF catalog, but ALCF only keeps a subset
# loaded on GPU at any time. populate_models.py --status-report classifies every
# OFFERED model as LIVE / QUEUED / OFFLINE against the live <cluster>/jobs state
# and annotates each with its context window, so the user can pick an instant
# model, see a queued model's real estimated start (which can be HOURS, not the
# ~10-15 min GPU warm-up), or know a cluster is simply down. Purely informational
# and best-effort: never fatal (|| true), and the script itself reports "unknown"
# rather than guessing if /jobs is unreachable.
# GENERIC MODE skips this ALCF-specific gateway banner (generic_inference.py's
# ENABLED/DISABLED report above is the equivalent in generic mode).
if [ "$GENERIC_MODE" == "0" ] && [ "${ALCF_SHOW_MODEL_STATUS:-1}" != "0" ]; then
  ALCF_INFER_TOKEN="$("$ALCF_TOKENS" get-token inference 2>/dev/null || true)" ALCF_PY="$PY" \
    ALCF_ENABLE_METIS="${ALCF_ENABLE_METIS:-1}" \
    ALCF_ENABLE_MINERVA="${ALCF_ENABLE_MINERVA:-1}" \
    "$PY" "$ALCF_DIR/populate_models.py" --status-report 2>>/tmp/populate_models.log \
    | while IFS= read -r line; do log "$line"; done || true
fi

# --- 5. Seed / refresh skills + memory ---------------------------------------
# ALCF skills and the knowledge base are IMAGE-MANAGED: we refresh them from the
# image on every start so knowledge-base fixes land for users who keep their
# existing volume — WITHOUT clobbering user edits. We track the checksum of what
# we last wrote; if the on-disk copy still matches that (user hasn't touched it)
# we overwrite with the image's newer version, otherwise we leave the user's
# edited copy alone and log a note.
STAMP_DIR="$HERMES_HOME/.alcf_seed_stamps"
mkdir -p "$STAMP_DIR" "$HERMES_HOME/skills/research" "$HERMES_HOME/memories"

managed_seed() {  # $1=source file, $2=dest file, $3=label
  local src="$1" dst="$2" label="$3" stamp
  stamp="$STAMP_DIR/$(echo "$dst" | md5sum | cut -d' ' -f1).sha"
  [[ -f "$src" ]] || return 0
  if [[ ! -f "$dst" ]]; then
    cp "$src" "$dst"; sha256sum "$src" | cut -d' ' -f1 > "$stamp"
    log "Seeded $label"
  elif [[ -f "$stamp" ]] && sha256sum -c <(echo "$(cat "$stamp")  $dst") >/dev/null 2>&1; then
    # on-disk copy is byte-identical to what we last seeded => safe to refresh
    if ! cmp -s "$src" "$dst"; then
      cp "$src" "$dst"; sha256sum "$src" | cut -d' ' -f1 > "$stamp"
      log "Updated $label from image"
    fi
  else
    log "Kept your edited $label (image has a newer version at $src)"
  fi
}

# Some Hermes code paths (config load during startup) write a STOCK default
# SOUL.md into $HERMES_HOME before this seed step runs, and older volumes carry
# one from a pre-ALCF-SOUL image. A stock/legacy SOUL.md has no ALCF seed stamp,
# so managed_seed would treat it as "user-edited" and KEEP it -- the ALCF
# identity would never land, and a fresh "what can you do?" would answer as
# generic Hermes with no mention of ALCF. Detect the stock/legacy Hermes SOUL
# (which carries zero ALCF/user intent) and delete it so managed_seed can seed
# the real ALCF identity. This is order-independent: it fixes both the
# fresh-volume race and stale volumes. A genuinely user-customized SOUL.md
# (anything not matching these known-stock signatures) is left untouched.
is_stock_hermes_soul() {  # $1=path -> exit 0 if it's the stock/legacy Hermes default
  local f="$1"
  [[ -f "$f" ]] || return 1
  # Signature 1: the current DEFAULT_SOUL_MD one-liner (identity by its opening).
  if head -c 200 "$f" | grep -q "You are Hermes Agent, an intelligent AI assistant created by Nous Research"; then
    return 0
  fi
  # Signature 2: the legacy comment-only scaffold (no persona text at all) --
  # every non-blank line is an HTML comment / markdown heading, no real content.
  if grep -q "This file defines the agent's personality and tone" "$f" 2>/dev/null; then
    return 0
  fi
  return 1
}
SOUL_DST="$HERMES_HOME/SOUL.md"
SOUL_STAMP="$STAMP_DIR/$(echo "$SOUL_DST" | md5sum | cut -d' ' -f1).sha"
if [[ -f "$SOUL_DST" && ! -f "$SOUL_STAMP" ]] && is_stock_hermes_soul "$SOUL_DST"; then
  rm -f "$SOUL_DST"
  log "Replacing stock Hermes SOUL.md with the ALCF Agent identity"
fi

managed_seed "$ALCF_DIR/memory/MEMORY.md" "$HERMES_HOME/memories/MEMORY.md" "ALCF knowledge base (memories/MEMORY.md)"
managed_seed "$ALCF_DIR/SOUL.md" "$HERMES_HOME/SOUL.md" "ALCF Agent identity (SOUL.md)"
# Skills: refresh each SKILL.md tree wholesale when unmodified (skills are
# reference material, not typically user-edited). Simple approach: mirror the
# baked skills dir, overwriting — skills live under skills/research/ which is
# image-owned here.
cp -r "$ALCF_DIR/skills/." "$HERMES_HOME/skills/research/" 2>/dev/null || true

# --- 6. Token refresh loop (tokens last 48h; refresh every 6h) --------------
# The inference token IS the api_key, and it rotates. We re-render the config
# with a fresh token every 6h. GENERIC MODE SKIPS THIS LOOP: the supplied
# INFERENCE_TOKEN is static and user-managed (restarting the container with an
# updated -e INFERENCE_TOKEN=... is how you rotate it). If a refresh FAILS, the token has almost
# certainly hit the 30-day hard re-auth limit — from then on every LLM call
# will fail upstream (typically HTTP 401). A chat-only user never sees the
# container log, so we (a) print a loud, actionable banner to the log AND (b)
# drop a status file on the volume that the agent reads and surfaces IN CHAT
# (see memory/MEMORY.md → "Inference token expiry"). On success we clear it.
TOKEN_STATUS="$HERMES_HOME/.inference_token_status"
: > "$TOKEN_STATUS" 2>/dev/null || true   # start clean (empty = healthy)
reauth_cmd="docker exec -it <container> \
  /opt/hermes/.venv/bin/alcf-tokens login"
if [[ "$GENERIC_MODE" == "0" ]]; then
(
  while true; do
    sleep 21600
    if render_config 2>/dev/null; then
      log "Refreshed inference access token."
      : > "$TOKEN_STATUS" 2>/dev/null || true   # healthy again
    else
      err "╔═══════════════════════════════════════════════════════════════╗"
      err "║  INFERENCE TOKEN REFRESH FAILED                               ║"
      err "║  Your ALCF inference login has expired (30-day re-auth limit). ║"
      err "║  The agent's LLM calls will fail until you re-authenticate.    ║"
      err "║  Run this on your host, then the agent works again:           ║"
      err "╚═══════════════════════════════════════════════════════════════╝"
      err "    $reauth_cmd"
      # Machine-readable note the agent reads to explain the failure in chat.
      printf 'EXPIRED\n%s\n' \
        "Inference Globus login expired (30-day re-auth). Re-authenticate on the host: $reauth_cmd" \
        > "$TOKEN_STATUS" 2>/dev/null || true
    fi
  done
) &
fi   # end ALCF-mode token-refresh loop guard (generic mode uses a static INFERENCE_TOKEN)

# --- 7. Launch: dashboard behind Caddy on a loopback-published HTTP port ------
# Browsers treat http://localhost as a trustworthy secure context, so clipboard
# access works without a local CA or certificate warning. Docker must publish
# this public port on host 127.0.0.1 only, as shown in the README.
PUB_PORT="${ALCF_DASHBOARD_PORT:-8787}"
INT_PORT="${ALCF_DASHBOARD_INTERNAL_PORT:-9119}"
mkdir -p "$XDG_DATA_HOME" "$XDG_CONFIG_HOME" 2>/dev/null || true

log "Starting dashboard (internal) on 0.0.0.0:${INT_PORT} (auth gate ON)"
# Bind 0.0.0.0 so Hermes engages the auth gate (loopback bind would skip it).
# Only the public proxy port is published from the container, so the internal
# port is not directly reachable from the host — Caddy proxies to it.
hermes dashboard --host 0.0.0.0 --port "$INT_PORT" --no-open &
DASH_PID=$!

# Wait for the dashboard to come up (up to ~60s) before starting the proxy.
for i in $(seq 1 60); do
  if curl -sf -o /dev/null "http://127.0.0.1:${INT_PORT}/api/health" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$DASH_PID" 2>/dev/null; then
    err "Dashboard exited during startup."; exit 1
  fi
  sleep 1
done

# Render the Caddyfile ports and launch Caddy on the public port (foreground).
export ALCF_DASHBOARD_PORT="$PUB_PORT" ALCF_DASHBOARD_INTERNAL_PORT="$INT_PORT"
"$PY" - "$ALCF_DIR/Caddyfile" /opt/data/Caddyfile <<'PYEOF'
import os, sys, string
src, dst = sys.argv[1], sys.argv[2]
open(dst, "w").write(string.Template(open(src).read()).safe_substitute(os.environ))
PYEOF

# Make the user-facing URL visually dominant; Hermes and Caddy emit additional
# startup lines immediately afterward, so a single ordinary log line is easy to
# miss in terminal output.
log "================================================================"
log "  OPEN THE WEB UI:  http://localhost:${PUB_PORT}"
log "  LOGIN: user=${ALCF_DASHBOARD_USER}  (use your dashboard password)"
log "================================================================"
exec caddy run --config /opt/data/Caddyfile --adapter caddyfile
