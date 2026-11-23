#!/usr/bin/env python3
"""Generate the Hermes ``providers:`` block from the LIVE ALCF catalog.

At container start we query every serving ALCF Inference Service cluster and emit
keyed ``providers`` entries per cluster, each with a curated, chat-only model
mapping that carries every model's REAL serving context window:

  - ``alcf-sophia``   (vLLM,  full OpenAI-compat)  api .../sophia/vllm/v1
  - ``alcf-metis``    (SambaNova, chat-only)       api .../metis/api/v1
  - ``alcf-minerva``  (api framework)              api .../minerva/api/v1
  - ``alcf-tara``     (api framework)              api .../tara/api/v1

Tara + ``muse-spark-bf16`` is the default launch (Dockerfile ALCF_CLUSTER /
ALCF_MODEL). A cluster that appears in list-endpoints but serves nothing yet
belongs in PROVISIONING_CLUSTERS (no provider; the banner reports it as
provisioning); none does today.

Why this exists
---------------
The ALCF gateway 404s the standard OpenAI ``/v1/models`` path, so Hermes' own
live discovery finds nothing and the model picker shows only the launch model.
We instead query the ALCF-specific ``.../<cluster>/models`` endpoint ourselves
and write the result into a static ``models:`` MAPPING (``discover_models:false``)
so the dropdown reflects the live catalog and every model gets its correct
``context_length`` (Hermes honours per-model
``providers.<key>.models.<id>.context_length`` on an in-session ``/model``
switch). Without the right per-model window a switch overflows the real context
and the gateway silently drops the SSE stream (EmptyStreamError).

Four clusters, three catalog shapes
-----------------------------------
* Sophia entries report ``framework`` + (for served LLMs) ``max_model_len``. We
  keep ``framework == "vllm"`` chat models and derive ``context_length`` from
  ``max_model_len``. We EXCLUDE non-chat frameworks (triton/dinoserver/sam3service)
  and, by id heuristic, embedding + science models (``*-embed``, ``embedding*``,
  ``genslm*``) that lack a chat interface and pollute the picker.
* Metis entries are ``framework == "api"`` (SambaNova) and report NO
  ``max_model_len``. They are all chat models, so we keep them all and assign
  ``context_length`` from a small verified table (METIS_CONTEXT), falling back to
  METIS_DEFAULT_CONTEXT for an unknown new Metis model.

  IMPORTANT: Metis windows differ from the same-named Sophia model — verified
  live 2026-08-10, re-verified 2026-08-24 by reading the gateway's
  ``context_length_exceeded`` error: Mistral-Large-3-675B=8192,
  gpt-oss-120b=131072, gemma-4-31B-it=131072. That is exactly why the clusters
  are separate providers, not one merged list.
* Minerva entries are ``framework == "api"`` but publish a STRUCTURED
  ``capabilities`` object (schema_version 1) — the richest of the three:
  ``capabilities.context_window_tokens`` and ``capabilities.reasoning.supported``
  are authoritative, so Minerva needs neither a hardcoded window table nor the
  reasoning id heuristic. Read them; the heuristic is only a last resort there.
* Tara entries use the same capabilities schema as Minerva and share its
  selector (``_select_capabilities``).

Allowlist add-backs
-------------------
A few Sophia vLLM CHAT models report no ``max_model_len`` (e.g. gpt-oss-20b,
AuroraGPT variants). A strict "must have max_model_len" filter would drop them,
but ALCF users look for them (AuroraGPT is Argonne's own model). SOPHIA_ALLOWLIST
maps such ids to a fallback context so they still appear.

Never fatal
-----------
Any failure (network, auth, bad JSON, empty result for a cluster) degrades to the
committed static fallback for that cluster, and the script always prints a valid
``providers:`` block so container start cannot break on a lookup failure.

Usage
-----
    populate_models.py [--out PATH] [--no-metis] [--no-minerva] [--no-tara]
    populate_models.py --status-report        # LIVE/QUEUED/OFFLINE + windows
    populate_models.py --launch-provider ID [--base-url URL] < providers-block
    populate_models.py --list-models [--base-url URL]       < providers-block

Prints the YAML block to stdout (and to --out if given). --launch-provider and
--list-models read an already-generated block (or a whole config) on stdin,
parse it with ruamel.yaml (the Hermes runtime venv's parser) or PyYAML, exit 1
if it cannot be parsed, and never touch the network. Reads:
    ALCF_INFER_TOKEN    optional pre-fetched inference token
    ALCF_PY             python used to run the auth helper (default: this python)
    ALCF_ENABLE_METIS   include the Metis provider?   "1" (default) / "0"
    ALCF_ENABLE_MINERVA include the Minerva provider? "1" (default) / "0"
    ALCF_ENABLE_TARA    include the Tara provider?    "1" (default) / "0"
    ALCF_LAUNCH_CLUSTER the launch cluster: always included, whatever its
                        ALCF_ENABLE_<CLUSTER> flag says
    ALCF_LAUNCH_MODEL   the launch model; with ALCF_LAUNCH_CONTEXT (its resolved
    ALCF_LAUNCH_CONTEXT window), added to the launch cluster's static fallback if
                        discovery failed and the fallback lacks it (>= 64k only)
    ALCF_LAUNCH_SOURCE_FILE  where to write the launch cluster's state: live |
                        fallback | fallback-injected | fallback-missing
"""
import json
import os
import re
import sys
import textwrap
import urllib.error
import urllib.request

INFER_HOST = "https://inference-api.alcf.anl.gov/resource_server"

# Per-cluster provider identity: name + base_url template.
SOPHIA_BASE = f"{INFER_HOST}/sophia/vllm/v1"
METIS_BASE = f"{INFER_HOST}/metis/api/v1"
MINERVA_BASE = f"{INFER_HOST}/minerva/api/v1"
TARA_BASE = f"{INFER_HOST}/tara/api/v1"

# Clusters that appear in list-endpoints but are NOT yet serving models. We do
# not build a provider for these; the status banner reports them as
# "provisioning" so the user knows to check back rather than assuming a bug.
# Empty today: tara (provisioning on 2026-08-24) serves models per the ALCF
# models doc of 2026-11-23 and is a real cluster in _clusters().
PROVISIONING_CLUSTERS = ()

PY = os.environ.get("ALCF_PY", sys.executable)
# Fallback context window for a chat model that reports no server value.
DEFAULT_CONTEXT = 32768

# Hermes refuses to load any model whose resolved context window is below this
# (agent/model_metadata.py: MINIMUM_CONTEXT_LENGTH). A model under the floor is a
# BROKEN dropdown entry — selecting it raises at load/switch — so we exclude any
# model whose real serving window is < MIN_CONTEXT. Keep this in sync with
# Hermes' constant; a mismatch just means we're slightly conservative/liberal at
# the boundary, never that we ship an unusable entry (we err on excluding).
MIN_CONTEXT = 64000

# --- Per-model output cap (max_tokens) ---------------------------------------
# ALCF serves several REASONING models whose hidden chain-of-thought is emitted
# on a separate channel that DRAWS FROM THE SAME max_tokens budget as the visible
# answer. With a small cap, the model can burn the whole budget thinking and
# return empty content (finish_reason=length) — verified on gpt-oss at ALCF. So
# reasoning models need a bigger output cap than plain chat models.
#
# Hermes has NO per-model max_tokens (unlike context_length), and Hermes
# v0.23.2 IGNORES a provider-level max_tokens on a ``providers:`` entry (it logs
# "unknown config keys ignored: max_tokens"). What it does honour is the entry's
# ``extra_body``: it becomes the provider's request_overrides and is sent with
# every request to a model selected under that provider (at launch and after an
# in-session /model switch). So we express "per-model" caps by SPLITTING each
# cluster into two provider entries — a baseline one and a "-reasoning" one —
# each carrying its own ``extra_body: {max_tokens: N}``. The entrypoint points
# the top-level model.provider at the one entry that lists the launch model, so
# the launch turn gets the same cap as a later switch to that model.
#
# Both caps are env-overridable (entrypoint passes them through):
#   ALCF_MAX_TOKENS            baseline output cap  (default 2048)
#   ALCF_REASONING_MAX_TOKENS  reasoning output cap (default 12288)
BASELINE_MAX_TOKENS = int(os.environ.get("ALCF_MAX_TOKENS") or 2048)
REASONING_MAX_TOKENS = int(os.environ.get("ALCF_REASONING_MAX_TOKENS") or 12288)

# Reasoning detection is two-layered:
#  (1) AUTHORITATIVE — the Sophia /models entry advertises a `reasoning_parser`
#      (e.g. nemotron-3-super="super_v3"; gemma-4*="gemma4"). vLLM only sets this
#      when the model is served with a reasoning parser, i.e. it emits a separate
#      reasoning channel. This is captured per-model during selection.
#  (2) ID HEURISTIC — some reasoning models are served WITHOUT a reasoning_parser
#      field (gpt-oss uses the built-in Harmony format + the `openai` tool-call
#      parser; Trinity-Large-Thinking exposes none). Catch these by id. Also the
#      only signal we have on Metis (framework: api, no parser field at all).
REASONING_ID_PATTERNS = (
    re.compile(r"gpt-oss", re.IGNORECASE),
    re.compile(r"gemma-4", re.IGNORECASE),  # gemma-4* served with reasoning_parser=gemma4 on Sophia; Metis omits the field
    re.compile(r"thinking", re.IGNORECASE),
    re.compile(r"reasoning", re.IGNORECASE),
    re.compile(r"deepseek-?r1", re.IGNORECASE),
    re.compile(r"\bqwq\b", re.IGNORECASE),
    re.compile(r"(^|[/-])o[13]([-.]|$)", re.IGNORECASE),  # o1 / o3 families
)


def _is_reasoning_id(mid: str) -> bool:
    return any(p.search(mid) for p in REASONING_ID_PATTERNS)

# Sophia chat models that are real vLLM chat endpoints but report no
# max_model_len from the server. Keep them (allowlist) with a sane window.
SOPHIA_ALLOWLIST = {
    "openai/gpt-oss-20b": 128000,
    "argonne/AuroraGPT-IT-v4-0125": 128000,
    "argonne/AuroraGPT-Tulu3-SFT-0125": 128000,
    "argonne/AuroraGPT-DPO-UFB-0225": 128000,
    "argonne/AuroraGPT-KTO-UFB-0325": 128000,
}

# Metis models report no max_model_len; these were verified live from the
# gateway's context_length_exceeded error (2026-08-10). Note they differ from
# the same-named Sophia models.
METIS_CONTEXT = {
    "Mistral-Large-3-675B-Instruct-2512": 8192,
    "gpt-oss-120b": 131072,
    "gemma-4-31B-it": 131072,
}
METIS_DEFAULT_CONTEXT = 32768

# Minerva models report their window + reasoning class in a structured
# `capabilities` object (schema_version 1) — richer and more authoritative than
# Sophia's `max_model_len`/`reasoning_parser` pair:
#   capabilities.context_window_tokens   int
#   capabilities.reasoning.supported     bool
# Verified live 2026-08-24: gpt-oss-120b 131072, inkling-bf16 262144,
# nemotron-3-ultra 262144 — all reasoning:true, all tool_calling:true.
# NOTE the id heuristic MISSES inkling-bf16 and nemotron-3-ultra, so without the
# capabilities read they would land in the 2048-token baseline provider and
# return empty content (their reasoning channel eats the whole budget —
# reproduced: inkling-bf16 @ max_tokens=30 -> content:null finish_reason:length).
MINERVA_DEFAULT_CONTEXT = 131072
# Tara (framework: api) publishes the same capabilities schema as Minerva, so it
# shares Minerva's selector; this window is only for an entry without
# capabilities.context_window_tokens.
TARA_DEFAULT_CONTEXT = 131072

# Frameworks that are NOT chat completion endpoints — exclude outright.
NON_CHAT_FRAMEWORKS = {"triton", "dinoserver", "sam3service"}

# Id substrings that mark a non-chat model (embeddings / science) — exclude even
# when the framework looks like vllm.
NON_CHAT_ID_PATTERNS = (
    re.compile(r"embed", re.IGNORECASE),
    re.compile(r"^genslm", re.IGNORECASE),
    re.compile(r"/genslm", re.IGNORECASE),
)

# --- Static fallbacks (used only when live discovery for a cluster fails) -----
# Only models at/above MIN_CONTEXT (64k) belong here — a sub-floor model is a
# broken entry Hermes will reject on load. Values are (context_length, is_reasoning),
# reflecting the real serving windows + reasoning class verified live 2026-08-10.
SOPHIA_FALLBACK = {
    "openai/gpt-oss-120b": (65536, True),
    "openai/gpt-oss-20b": (128000, True),
    "google/gemma-4-31B-it": (128000, True),
    "google/gemma-4-26B-A4B-it": (262144, True),
    "arcee-ai/Trinity-Large-Thinking-W4A16": (131072, True),
    "nvidia/nemotron-3-super-120b": (262144, True),
    "argonne/AuroraGPT-IT-v4-0125": (128000, False),
}
# Metis fallback: only the Metis models at/above the floor. Mistral-Large-3
# serves at 8192 (verified) so it is intentionally NOT here — it's unusable.
METIS_FALLBACK = {
    mid: (ctx, _is_reasoning_id(mid))
    for mid, ctx in METIS_CONTEXT.items() if ctx >= MIN_CONTEXT
}
# Minerva fallback: verified live 2026-08-24 from capabilities.*. All three are
# reasoning models (capabilities.reasoning.supported = true), all well above the
# 64k floor.
MINERVA_FALLBACK = {
    "gpt-oss-120b": (131072, True),
    "inkling-bf16": (262144, True),
    "nemotron-3-ultra": (262144, True),
}
# Tara fallback: the Tara table of the ALCF "Available Models" doc as of
# 2026-11-23 (plans/evidence/tara/alcf-models-doc-2026-11-23.md). Every model is
# marked R (reasoning) and T, none H: Tara models are not kept always hot, so a
# first request may wait for a cold start. Note the "openai/" prefix on the
# gpt-oss ids, unlike Metis/Minerva.
TARA_FALLBACK = {
    "google/gemma-4-31B-it-262k": (262144, True),
    "muse-spark-bf16": (1048576, True),
    "mistral-large-4": (1048576, True),
    "nemotron-3-ultra": (1048576, True),
    "nemotron-35-super-vl": (1048576, True),
    "openai/gpt-oss-120b": (131072, True),
    "openai/gpt-oss-20b": (65536, True),
    "beam-1.0-fp8": (1048576, True),
    "inkling-bf16": (1048576, True),
}


from alcf_tokens.auth import get_access_token


def _exc_summary(e: BaseException) -> str:
    """The exception class and HTTP status (when there is one) for a log line.
    Never str(e): that may echo a URL, a response body or a credential."""
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status is None and isinstance(e, urllib.error.HTTPError):
        status = e.code
    return type(e).__name__ + (f", HTTP {status}" if status is not None else "")


def _get_token() -> str:
    # Support passing a pre-fetched token via env (set by entrypoint to avoid
    # extra alcf-tokens calls during the same render_config invocation).
    pre = os.environ.get("ALCF_INFER_TOKEN", "").strip()
    if pre:
        return pre
    return get_access_token("inference")


def _fetch_models(cluster: str, token: str) -> list:
    """GET .../<cluster>/models -> list of model dicts (raises on failure)."""
    url = f"{INFER_HOST}/{cluster}/models"
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=30) as r:
        doc = json.load(r)
    items = doc.get("data", doc) if isinstance(doc, dict) else doc
    return [m for m in items if isinstance(m, dict)] if isinstance(items, list) else []


def _is_non_chat_id(mid: str) -> bool:
    return any(p.search(mid) for p in NON_CHAT_ID_PATTERNS)


# Models dropped because their real serving window is below Hermes' 64k floor,
# recorded per cluster as {cluster: {model_id: context_length}} while selecting.
# The status banner prints these so a user comparing the box against the ALCF web
# UI can see WHY a model they can see there is missing here (e.g. Metis
# Mistral-Large-3-675B serves at only 8192). Populated as a side effect of the
# selectors; harmless when unread.
EXCLUDED_BY_FLOOR: dict = {}


def _note_excluded(cluster: str, mid: str, ctx: int) -> None:
    EXCLUDED_BY_FLOOR.setdefault(cluster, {})[mid] = ctx
    print(f"[populate_models] {cluster}: skip {mid} (window {ctx} < {MIN_CONTEXT})",
          file=sys.stderr)


def _select_sophia(models: list) -> dict:
    """{model_id: (context_length, is_reasoning)} for Sophia chat models.

    Excludes non-chat frameworks, embedding/science ids, AND any model whose real
    serving window is below Hermes' MIN_CONTEXT floor (an unusable dropdown entry).
    is_reasoning is True when the server advertises a `reasoning_parser` OR the id
    matches a known reasoning family (see _is_reasoning_id).
    """
    out = {}
    for m in models:
        mid = m.get("id")
        if not mid:
            continue
        fw = (m.get("framework") or "").lower()
        if fw in NON_CHAT_FRAMEWORKS:
            continue
        if _is_non_chat_id(mid):
            continue
        mml = m.get("max_model_len")
        if isinstance(mml, int) and mml > 0:
            ctx = mml
        elif mid in SOPHIA_ALLOWLIST:
            ctx = SOPHIA_ALLOWLIST[mid]
        else:
            # a vllm model with no window and not allowlisted -> skip
            continue
        if ctx < MIN_CONTEXT:
            _note_excluded("sophia", mid, ctx)
            continue
        is_reasoning = bool(m.get("reasoning_parser")) or _is_reasoning_id(mid)
        out[mid] = (ctx, is_reasoning)
    return out


def _select_metis(models: list) -> dict:
    """{model_id: (context_length, is_reasoning)} for Metis models (all chat).

    Metis reports no max_model_len, so context comes from METIS_CONTEXT (verified)
    or METIS_DEFAULT_CONTEXT. Any model below Hermes' MIN_CONTEXT floor is excluded
    (e.g. Mistral-Large-3 serves at only 8192 — unusable in Hermes). Metis exposes
    no reasoning_parser field, so reasoning is detected by id heuristic only.
    """
    out = {}
    for m in models:
        mid = m.get("id")
        if not mid:
            continue
        fw = (m.get("framework") or "").lower()
        if fw in NON_CHAT_FRAMEWORKS:
            continue
        if _is_non_chat_id(mid):
            continue
        ctx = METIS_CONTEXT.get(mid, METIS_DEFAULT_CONTEXT)
        if ctx < MIN_CONTEXT:
            _note_excluded("metis", mid, ctx)
            continue
        out[mid] = (ctx, _is_reasoning_id(mid))
    return out


def _select_capabilities(cluster: str, default_ctx: int):
    """Selector for a cluster that publishes the structured ``capabilities``
    object (schema_version 1): Minerva and Tara.

    Returns ``select(models) -> {model_id: (context_length, is_reasoning)}``.
    Unlike Sophia/Metis we do NOT have to guess: read
    ``capabilities.context_window_tokens`` for the window (``default_ctx`` when
    absent) and ``capabilities.reasoning.supported`` for the reasoning class,
    falling back to the id heuristic only when the field is absent. This
    matters — inkling-bf16 and nemotron-3-ultra match no reasoning id pattern
    but ARE reasoning models, and the id heuristic alone would starve them of
    output budget. A model whose ``api_protocols`` lacks ``chat_completions`` is
    skipped (Hermes talks Chat Completions to these endpoints); sub-floor models
    are recorded under ``cluster`` for the status banner.
    """
    def select(models: list) -> dict:
        out = {}
        for m in models:
            mid = m.get("id")
            if not mid:
                continue
            fw = (m.get("framework") or "").lower()
            if fw in NON_CHAT_FRAMEWORKS:
                continue
            if _is_non_chat_id(mid):
                continue
            caps = m.get("capabilities") or {}
            protos = caps.get("api_protocols")
            if isinstance(protos, list) and protos and "chat_completions" not in protos:
                continue
            ctx = caps.get("context_window_tokens")
            if not (isinstance(ctx, int) and ctx > 0):
                ctx = default_ctx
            if ctx < MIN_CONTEXT:
                _note_excluded(cluster, mid, ctx)
                continue
            reasoning = (caps.get("reasoning") or {}).get("supported")
            is_reasoning = bool(reasoning) if reasoning is not None else _is_reasoning_id(mid)
            out[mid] = (ctx, is_reasoning)
        return out
    return select


_select_minerva = _select_capabilities("minerva", MINERVA_DEFAULT_CONTEXT)
_select_tara = _select_capabilities("tara", TARA_DEFAULT_CONTEXT)


def _resolve_cluster(cluster: str, token: str, selector, fallback: dict) -> tuple:
    """Return (mapping, source) where source is 'live' or 'fallback'."""
    try:
        models = _fetch_models(cluster, token)
        mapping = selector(models)
        if mapping:
            return mapping, "live"
        print(f"[populate_models] {cluster}: empty selection; using fallback",
              file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[populate_models] {cluster} discovery failed ({_exc_summary(e)}); using fallback",
              file=sys.stderr)
    return dict(fallback), "fallback"


_PLAIN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")


def _yaml_key(mid: str) -> str:
    """A model id as a YAML mapping key: bare when safe, else double-quoted."""
    return mid if _PLAIN_ID.fullmatch(mid) else json.dumps(mid)


def _emit_provider(name: str, base_url: str, mapping: dict,
                   max_tokens: int) -> list:
    """Render one keyed ``providers`` entry, indented to sit under ``providers:``.

    ``mapping`` is {model_id: (context_length, is_reasoning)}; only context_length
    is emitted per model (Hermes has no per-model max_tokens). ``max_tokens`` is
    the output cap for every model selected under this provider. It goes in
    ``extra_body``, which Hermes v0.23.2 sends as request overrides on each
    request; a provider-level ``max_tokens`` key would be ignored.
    """
    lines = [
        f"  {name}:",
        f"    name: {name}",
        f'    api: "{base_url}"',
        '    api_key: "${ALCF_ACCESS_TOKEN}"',
        "    discover_models: false",
        "    extra_body:",
        f"      max_tokens: {max_tokens}",
        "    models:",
    ]
    for mid in sorted(mapping):
        ctx = mapping[mid][0]
        lines.append(f"      {_yaml_key(mid)}:")
        lines.append(f"        context_length: {ctx}")
    return lines


def _split_reasoning(mapping: dict) -> tuple:
    """Split {id:(ctx,is_reasoning)} into (baseline_map, reasoning_map)."""
    baseline = {mid: v for mid, v in mapping.items() if not v[1]}
    reasoning = {mid: v for mid, v in mapping.items() if v[1]}
    return baseline, reasoning


def _emit_cluster(name: str, base_url: str, mapping: dict) -> list:
    """Return the lines of one cluster's keyed ``providers`` entries.

    Every line is indented two spaces so build_block() can nest the result
    directly under its single ``providers:`` root (no list items, no root key).
    Non-reasoning models go in ``<name>`` with the baseline output cap; reasoning
    models go in ``<name>-reasoning`` with the larger reasoning cap. An entry is
    omitted entirely if its bucket is empty, so a cluster with only one class
    yields a single provider (no empty "-reasoning" entry cluttering the picker).
    """
    baseline, reasoning = _split_reasoning(mapping)
    lines = []
    if baseline:
        lines += _emit_provider(name, base_url, baseline,
                                max_tokens=BASELINE_MAX_TOKENS)
    if reasoning:
        lines += _emit_provider(f"{name}-reasoning", base_url, reasoning,
                                max_tokens=REASONING_MAX_TOKENS)
    return lines


def _clusters(include_metis: bool = True, include_minerva: bool = True,
              include_tara: bool = True) -> list:
    """Ordered cluster registry: (cluster, provider_name, base_url, selector, fallback).

    Single source of truth shared by build_block() and the status banner, so
    adding a cluster is a one-line change here.
    Provisioning clusters (see PROVISIONING_CLUSTERS) are deliberately absent —
    they have no models to offer; the banner reports them separately.
    """
    out = [("sophia", "alcf-sophia", SOPHIA_BASE, _select_sophia, SOPHIA_FALLBACK)]
    if include_metis:
        out.append(("metis", "alcf-metis", METIS_BASE, _select_metis, METIS_FALLBACK))
    if include_minerva:
        out.append(("minerva", "alcf-minerva", MINERVA_BASE,
                    _select_minerva, MINERVA_FALLBACK))
    if include_tara:
        out.append(("tara", "alcf-tara", TARA_BASE, _select_tara, TARA_FALLBACK))
    return out


def _include_flags(include_metis: bool, include_minerva: bool,
                   include_tara: bool = True) -> tuple:
    """Apply the launch-cluster override to the ALCF_ENABLE_<CLUSTER> flags.

    The launch model must resolve on the launch endpoint, so the launch cluster
    ($ALCF_LAUNCH_CLUSTER) is always generated; the flags only trim the OTHER
    clusters. Logs one line when a flag is overridden.
    """
    launch = os.environ.get("ALCF_LAUNCH_CLUSTER", "")
    flags = (("metis", include_metis), ("minerva", include_minerva), ("tara", include_tara))
    for cluster, enabled in flags:
        if launch == cluster and not enabled:
            print(f"[populate_models] ALCF_ENABLE_{cluster.upper()}=0 overridden: "
                  f"{cluster} is the launch cluster", file=sys.stderr)
    return tuple(enabled or launch == cluster for cluster, enabled in flags)


def _launch_context() -> int:
    """$ALCF_LAUNCH_CONTEXT as a positive int, else 0 (unknown)."""
    raw = os.environ.get("ALCF_LAUNCH_CONTEXT", "").strip()
    return int(raw) if raw.isdigit() else 0


def _ensure_launch_model(mapping: dict, src: str, cluster: str) -> str:
    """Keep the launch model resolvable on its cluster during a catalog outage.

    Returns the launch cluster's state: 'live' (discovered; an absent model is
    really not served, so nothing is added), 'fallback' (static list already
    lists it), 'fallback-injected' (the static list lacked it, so it was added
    to ``mapping`` with the resolved launch context and its id-heuristic
    reasoning class) or 'fallback-missing' (lacking, and the context is unknown
    or below MIN_CONTEXT, so the entrypoint refuses).
    """
    model = os.environ.get("ALCF_LAUNCH_MODEL", "")
    if src == "live":
        return "live"
    if model in mapping:
        return "fallback"
    ctx = _launch_context()
    if ctx < MIN_CONTEXT:
        print(f"[populate_models] {cluster}: catalog unavailable and launch model "
              f"{model!r} is not in the static fallback list; its context ({ctx or 'unknown'}) "
              f"is below the {MIN_CONTEXT} floor, so it is NOT added", file=sys.stderr)
        return "fallback-missing"
    mapping[model] = (ctx, _is_reasoning_id(model))
    print(f"[populate_models] {cluster}: catalog unavailable; added launch model "
          f"{model!r} to the static fallback list (context {ctx})", file=sys.stderr)
    return "fallback-injected"


def build_block(include_metis: bool = True, include_minerva: bool = True,
                include_tara: bool = True) -> str:
    try:
        token = _get_token()
    except Exception as e:  # noqa: BLE001
        print(f"[populate_models] token fetch failed ({_exc_summary(e)}); full static fallback",
              file=sys.stderr)
        token = None

    include_metis, include_minerva, include_tara = _include_flags(
        include_metis, include_minerva, include_tara)
    launch_cluster = os.environ.get("ALCF_LAUNCH_CLUSTER", "")
    launch_state = None
    lines = ["providers:"]
    for cluster, provider, base, selector, fallback in _clusters(
            include_metis=include_metis, include_minerva=include_minerva,
            include_tara=include_tara):
        if token:
            mapping, src = _resolve_cluster(cluster, token, selector, fallback)
        else:
            mapping, src = dict(fallback), "fallback"
        if cluster == launch_cluster and os.environ.get("ALCF_LAUNCH_MODEL"):
            launch_state = _ensure_launch_model(mapping, src, cluster)
        lines += _emit_cluster(provider, base, mapping)
        print(f"[populate_models] {cluster}: {len(mapping)} models ({src})",
              file=sys.stderr)

    state_file = os.environ.get("ALCF_LAUNCH_SOURCE_FILE", "")
    if state_file and launch_state:
        with open(state_file, "w", encoding="utf-8") as f:
            f.write(launch_state + "\n")
    return "\n".join(lines) + "\n"


def _split_models_field(job: dict) -> list:
    """A /jobs entry's "Models" field may list several comma-joined ids."""
    return [m.strip() for m in str(job.get("Models", "")).split(",") if m.strip()]


def _fetch_job_states(cluster: str, token: str) -> tuple:
    """Return (live_ids, queued_info) from .../<cluster>/jobs.

    ``live_ids`` is the set of model ids currently RUNNING (loaded on GPU,
    instant first token). ``queued_info`` maps model id -> a short human string
    describing the pending job (estimated start time and/or the scheduler's
    comment), for models that are scheduled but NOT yet running.

    Raises on any transport/parse failure so the caller can report "unknown"
    rather than mislabel every model as offline.
    """
    url = f"{INFER_HOST}/{cluster}/jobs"
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=30) as r:
        doc = json.load(r)
    if not isinstance(doc, dict):
        return set(), {}

    live: set = set()
    for job in doc.get("running", []) or []:
        if not isinstance(job, dict):
            continue
        if str(job.get("Model Status", "")).lower() not in ("", "running"):
            continue
        live.update(_split_models_field(job))

    queued: dict = {}
    for job in doc.get("queued", []) or []:
        if not isinstance(job, dict):
            continue
        eta = str(job.get("Estimated Start Time") or "").strip()
        comment = str(job.get("Job Comments") or "").strip()
        # Strip the scheduler's redundant "Not Running: " prefix.
        if comment.lower().startswith("not running:"):
            comment = comment.split(":", 1)[1].strip()
        detail = f"starts ~{eta}" if eta else ""
        if comment:
            detail = f"{detail}; {comment}" if detail else comment
        for mid in _split_models_field(job):
            if mid not in live:
                queued[mid] = detail or "queued"
    return live, queued


def _fetch_hot(cluster: str, token: str) -> set:
    """Back-compat alias: the set of model ids currently RUNNING (hot)."""
    return _fetch_job_states(cluster, token)[0]


def _fmt_model(mid: str, mapping: dict, extra: str = "") -> str:
    """'model-id (128k ctx)' — the offered window, for at-a-glance reference.

    ``mapping`` is the cluster's {id: (context_length, is_reasoning)} selection;
    the window shown is exactly the ``context_length`` we write into the config
    for that model, so the banner and the dropdown can never disagree.
    """
    ctx = (mapping.get(mid) or (0,))[0]
    if ctx >= 1000:
        # 128000 -> 128k, 131072 -> 131k, 262144 -> 262k
        win = f"{round(ctx / 1000)}k ctx"
    elif ctx:
        win = f"{ctx} ctx"
    else:
        win = "ctx unknown"
    return f"{mid} ({win}{'; ' + extra if extra else ''})"


def status_report() -> int:
    """Print a per-cluster model availability banner.

    Classifies every OFFERED model against the live ``.../<cluster>/jobs`` state
    into three buckets, and annotates each with its context window:

      LIVE     — in ``running``: answers immediately.
      QUEUED   — in ``queued``: a PBS job exists but has not started; we print the
                 scheduler's estimated start time, which can be many HOURS away.
      OFFLINE  — offered by the catalog but in no job bucket: not loaded and not
                 scheduled. A request returns HTTP 503 "... is offline."

    This replaces the old HOT/cold split, which lumped QUEUED and OFFLINE together
    and hardcoded the label "cold (~10-15m)". That was actively wrong whenever a
    cluster was down: on 2026-08-24 Sophia had zero running models and a queue
    whose estimated start was ~31 hours out, yet every Sophia model was advertised
    as ready in 10-15 minutes.

    Best-effort: any failure downgrades a cluster line to "status unknown" rather
    than emitting a wrong availability claim.
    """
    try:
        token = _get_token()
    except Exception as e:  # noqa: BLE001
        print(f"[status-report] token fetch failed ({_exc_summary(e)}); skipping status banner",
              file=sys.stderr)
        return 0

    include_metis, include_minerva, include_tara = _include_flags(
        os.environ.get("ALCF_ENABLE_METIS", "1") != "0",
        os.environ.get("ALCF_ENABLE_MINERVA", "1") != "0",
        os.environ.get("ALCF_ENABLE_TARA", "1") != "0")

    lines = ["Model availability (context window shown per model):"]
    any_live = False
    for cluster, _provider, _base, selector, fallback in _clusters(
            include_metis=include_metis, include_minerva=include_minerva,
            include_tara=include_tara):
        offered, _src = _resolve_cluster(cluster, token, selector, fallback)
        try:
            live, queued = _fetch_job_states(cluster, token)
            known = True
        except Exception as e:  # noqa: BLE001
            print(f"[status-report] {cluster} /jobs failed ({_exc_summary(e)})", file=sys.stderr)
            live, queued, known = set(), {}, False

        lines.append(f"  {cluster}:")
        if not known:
            lines.append(f"    status unknown (could not read /jobs) — "
                         f"{len(offered)} models offered")
            continue

        live_ids = sorted(m for m in offered if m in live)
        queued_ids = sorted(m for m in offered if m not in live and m in queued)
        offline_ids = sorted(m for m in offered
                             if m not in live and m not in queued)
        any_live = any_live or bool(live_ids)

        if live_ids:
            lines.append("    LIVE (instant):")
            for mid in live_ids:
                lines.append(f"      - {_fmt_model(mid, offered)}")
        else:
            lines.append("    LIVE (instant): (none)")
        if queued_ids:
            lines.append("    QUEUED (job pending — see estimated start):")
            for mid in queued_ids:
                lines.append(f"      - {_fmt_model(mid, offered, queued[mid])}")
        if offline_ids:
            lines.append("    OFFLINE (not loaded, not scheduled — requests 503):")
            for mid in offline_ids:
                lines.append(f"      - {_fmt_model(mid, offered)}")

        dropped = EXCLUDED_BY_FLOOR.get(cluster) or {}
        if dropped:
            lines.append(f"    not offered — below Hermes' {MIN_CONTEXT // 1000}k "
                         f"context floor ({len(dropped)}):")
            detail = ", ".join(f"{mid} ({ctx})" for mid, ctx in sorted(dropped.items()))
            for chunk in textwrap.wrap(detail, width=76):
                lines.append(f"      {chunk}")

    for cluster in PROVISIONING_CLUSTERS:
        lines.append(f"  {cluster}:")
        lines.append("    provisioning at ALCF — no models yet; check back later")

    lines.append("  Note: a LIVE model answers immediately. Selecting an OFFLINE or")
    lines.append("  QUEUED model returns HTTP 503 until ALCF loads it — a warm-up is")
    lines.append("  ~10-15 min, but a queued job may wait HOURS for free nodes.")
    if include_tara:
        lines.append("  Tara models are not kept always hot: the first request to one")
        lines.append("  may wait for a cold start.")
    if not any_live:
        lines.append("  WARNING: no model is LIVE on any cluster right now.")
    # Emit to stdout so the entrypoint can `log` it line-by-line.
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


# Back-compat alias for the previous public name.
hot_report = status_report


class LaunchProviderError(Exception):
    """The launch/delegation model does not map to exactly one provider."""


class ProviderParseError(LaunchProviderError):
    """The providers text cannot be read: no YAML parser, bad YAML or bad shape."""


def _yaml_load(text: str):
    """Parse ``text`` with ruamel.yaml (what the Hermes runtime venv ships), else
    PyYAML. With neither importable, or on any parse error, raise
    ProviderParseError: a half-read block must never pick a provider."""
    try:
        from ruamel.yaml import YAML
    except ImportError:
        YAML = None
    if YAML is not None:
        try:
            return YAML(typ="safe", pure=True).load(text)
        except Exception as e:  # ruamel raises several unrelated error types
            raise ProviderParseError(f"providers text is not valid YAML: {e}") from None
    try:
        import yaml
    except ImportError:
        raise ProviderParseError(
            "no YAML parser available (need ruamel.yaml or PyYAML)") from None
    class _UniqueKeyLoader(yaml.SafeLoader):
        """SafeLoader that, like ruamel, rejects duplicate mapping keys."""

        def construct_mapping(self, node, deep=False):
            seen = set()
            for key_node, _value in node.value:
                key = self.construct_object(key_node, deep=deep)
                if key in seen:
                    raise yaml.constructor.ConstructorError(
                        None, None, f"duplicate key {key!r}", key_node.start_mark)
                seen.add(key)
            return super().construct_mapping(node, deep=deep)

    try:
        return yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 (SafeLoader subclass)
    except Exception as e:
        raise ProviderParseError(f"providers text is not valid YAML: {e}") from None


def parse_providers(text: str) -> dict:
    """Read the ``providers:`` mapping of a config or block with a real YAML parser.

    Returns ``{provider_key: {"api": url, "models": {model_id: context_length}}}``.
    Text with no top-level ``providers:`` key (e.g. the legacy
    ``custom_providers:`` list, or empty text) yields ``{}``. Anything else
    that is not that shape -- invalid YAML, a non-mapping ``providers``, an
    entry without a string ``api`` and a ``models`` mapping, a model without an
    positive integer ``context_length`` -- raises ProviderParseError (fail closed).
    """
    doc = _yaml_load(text or "")
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise ProviderParseError("providers text is not a YAML mapping")
    if "providers" not in doc:
        return {}
    providers = doc["providers"]
    if not isinstance(providers, dict):
        raise ProviderParseError("'providers' is not a mapping")
    out: dict = {}
    for key, entry in providers.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            raise ProviderParseError(f"provider {key!r} is not a named mapping")
        api, models = entry.get("api"), entry.get("models")
        if not isinstance(api, str) or not isinstance(models, dict):
            raise ProviderParseError(
                f"provider {key!r} needs a string 'api' and a 'models' mapping")
        out[key] = {"api": api, "models": {}}
        for mid, meta in models.items():
            ctx = meta.get("context_length") if isinstance(meta, dict) else None
            if (not isinstance(mid, str) or not isinstance(ctx, int)
                    or isinstance(ctx, bool) or ctx <= 0):
                raise ProviderParseError(
                    f"provider {key!r} model {mid!r} needs a positive integer context_length")
            out[key]["models"][mid] = ctx
    return out


def _on_endpoint(providers: dict, base_url) -> dict:
    """The providers whose ``api`` is base_url (trailing slash ignored); all if None."""
    if not base_url:
        return providers
    want = base_url.rstrip("/")
    return {k: e for k, e in providers.items() if e["api"].rstrip("/") == want}


def list_models(providers_text: str, base_url=None) -> list:
    """Sorted model ids listed by the providers on base_url (all if None)."""
    scoped = _on_endpoint(parse_providers(providers_text), base_url)
    return sorted({mid for e in scoped.values() for mid in e["models"]})


def launch_provider(model_id: str, providers_text: str, base_url=None) -> str:
    """Return the named provider the launch (or delegation) model resolves through.

    The top-level model: block resolves through a NAMED provider so the launch
    turn gets that provider's output cap (its extra_body), exactly as a later
    /model switch to the same model would. The answer is read from the generated
    block itself, never guessed: among the providers whose ``api`` is base_url
    (the launch endpoint; all providers when None) exactly one
    must list model_id. Zero or several matches raise LaunchProviderError, since
    a guess could silently apply the wrong cap or endpoint. Returns e.g.
    'custom:alcf-sophia-reasoning'. Never touches the network.
    """
    scoped = _on_endpoint(parse_providers(providers_text), base_url)
    matches = sorted(k for k, e in scoped.items() if model_id in e["models"])
    where = f" on {base_url}" if base_url else ""
    if not matches:
        raise LaunchProviderError(
            f"model {model_id!r} is not listed by any generated provider{where}")
    if len(matches) > 1:
        raise LaunchProviderError(
            f"model {model_id!r} is listed by several providers{where} "
            f"({', '.join(matches)}); cannot pick one")
    return f"custom:{matches[0]}"


def delegation_provider(model_id: str, providers_text: str, launch_base_url) -> str:
    """Return the named provider the delegation (subagent) model resolves through.

    Endpoint-pinned like the launch model: if any provider on the launch
    endpoint lists model_id, the launch_provider() rule applies there (exactly
    one, else raise) -- so a model served on several clusters (e.g. gpt-oss-120b)
    delegates on the cluster the user launched on. Otherwise exactly one provider
    on any OTHER endpoint must list it. None, or several elsewhere, raise
    LaunchProviderError naming the candidates; there is no guess. Never touches
    the network.
    """
    providers = parse_providers(providers_text)
    on_launch = _on_endpoint(providers, launch_base_url) if launch_base_url else {}
    if any(model_id in e["models"] for e in on_launch.values()):
        return launch_provider(model_id, providers_text, launch_base_url)
    matches = sorted(k for k, e in providers.items()
                     if k not in on_launch and model_id in e["models"])
    if not matches:
        raise LaunchProviderError(
            f"model {model_id!r} is not listed by any generated provider")
    if len(matches) > 1:
        raise LaunchProviderError(
            f"model {model_id!r} is not on the launch endpoint and is listed by several "
            f"other providers ({', '.join(matches)}); cannot pick one")
    return f"custom:{matches[0]}"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="also write the block to this path")
    ap.add_argument("--no-metis", action="store_true",
                    help="omit the Metis provider")
    ap.add_argument("--no-minerva", action="store_true",
                    help="omit the Minerva provider")
    ap.add_argument("--no-tara", action="store_true",
                    help="omit the Tara provider")
    ap.add_argument("--status-report", "--hot-report", dest="status_report",
                    action="store_true",
                    help="print a per-cluster LIVE/QUEUED/OFFLINE availability "
                         "banner (with each model's context window) instead of "
                         "the providers block (queries /jobs). "
                         "--hot-report is the deprecated alias.")
    ap.add_argument("--launch-provider", default="", metavar="MODEL_ID",
                    help="read a providers block on stdin and print the one named "
                         "provider that lists MODEL_ID (e.g. "
                         "custom:alcf-sophia-reasoning); exit 1 if none or several")
    ap.add_argument("--delegation-provider", default="", metavar="MODEL_ID",
                    help="read a providers block on stdin and print the named "
                         "provider for the delegation model: the one on --base-url "
                         "(the launch endpoint) if it lists MODEL_ID, else the one "
                         "other provider that does; exit 1 if none or ambiguous")
    ap.add_argument("--list-models", action="store_true",
                    help="read a providers block on stdin and print its model ids "
                         "(comma-separated, sorted); exit 1 if none")
    ap.add_argument("--base-url", default="",
                    help="with --launch-provider/--list-models: only consider "
                         "providers whose api is this endpoint; with "
                         "--delegation-provider: the launch endpoint")
    args = ap.parse_args()

    if args.launch_provider or args.delegation_provider:
        try:
            if args.launch_provider:
                name = launch_provider(args.launch_provider, sys.stdin.read(),
                                       args.base_url or None)
            else:
                name = delegation_provider(args.delegation_provider, sys.stdin.read(),
                                           args.base_url or None)
        except LaunchProviderError as e:
            print(f"[populate_models] {e}", file=sys.stderr)
            return 1
        sys.stdout.write(name + "\n")
        return 0

    if args.list_models:
        try:
            ids = list_models(sys.stdin.read(), args.base_url or None)
        except ProviderParseError as e:
            print(f"[populate_models] {e}", file=sys.stderr)
            return 1
        if not ids:
            return 1
        sys.stdout.write(", ".join(ids) + "\n")
        return 0

    if args.status_report:
        return status_report()

    include_metis = not args.no_metis and os.environ.get("ALCF_ENABLE_METIS", "1") != "0"
    include_minerva = (not args.no_minerva
                       and os.environ.get("ALCF_ENABLE_MINERVA", "1") != "0")
    include_tara = not args.no_tara and os.environ.get("ALCF_ENABLE_TARA", "1") != "0"
    block = build_block(include_metis=include_metis, include_minerva=include_minerva,
                        include_tara=include_tara)
    sys.stdout.write(block)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
