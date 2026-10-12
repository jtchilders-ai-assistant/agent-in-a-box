#!/usr/bin/env python3
"""Generic OpenAI-compatible inference catalog helper.

Used by scripts/entrypoint.sh when INFERENCE_URL is set (generic mode). It
discovers the model catalog from a standard OpenAI-compatible endpoint
(GET <base>/models with ``Authorization: Bearer $INFERENCE_TOKEN``), decides
which models Hermes may use (serving context window >= 64000 tokens), and
writes a NON-SECRET manifest (mode 0600) that the entrypoint turns into a
single named custom provider block.

Security posture:
  * The bearer token is only ever placed in the request header. It is never
    printed, logged, or serialized into the manifest.
  * The endpoint URL is validated BEFORE any network call: http/https only,
    no embedded credentials, no query, no fragment, a real host.
  * Every failure mode is fail-closed: exit 78 (EX_CONFIG) with an actionable
    message, and a stale manifest at --output is removed up front so a failed
    run can never leave a previous manifest behind.

Exit codes: 0 success, 78 configuration/catalog error.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from typing import NoReturn
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlparse

EXIT_CONFIG = 78
CONTEXT_FLOOR = 64000
# Direct context-window fields honoured, in priority order, then the nested
# shapes below. Only positive non-bool integers are accepted; the model NAME
# is never used to infer a context window.
DIRECT_CONTEXT_FIELDS = (
    "max_model_len",
    "context_length",
    "context_window",
    "max_context_length",
)
NESTED_CONTEXT_FIELDS = (
    ("capabilities", "context_window_tokens"),
    ("metadata", "context_length"),
)


def fail(message: str) -> "NoReturn":
    """Print an actionable error (NEVER containing the token) and exit 78."""
    print(f"generic_inference: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CONFIG)


def is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def normalize_base_url(raw: str) -> tuple[str, str]:
    """Validate the endpoint URL and return (base_url, models_url).

    ``/models`` is appended exactly once: a base that already ends in
    ``/models`` (with or without trailing slash) is trimmed back to the API
    base first. Validation happens before any network access.
    """
    try:
        parsed = urlparse(raw)
    except ValueError:
        fail("INFERENCE_URL is not a valid URL")
    if parsed.scheme not in ("http", "https"):
        fail("INFERENCE_URL must use the http or https scheme")
    if parsed.username is not None or parsed.password is not None:
        fail("INFERENCE_URL must not embed credentials (user:pass@host)")
    if parsed.query:
        fail("INFERENCE_URL must not carry a query string")
    if parsed.fragment:
        fail("INFERENCE_URL must not carry a fragment")
    if not parsed.netloc or not parsed.hostname:
        fail("INFERENCE_URL must include a host")
    path = parsed.path.rstrip("/")
    if path == "/models":
        path = ""
    elif path.endswith("/models"):
        path = path[: -len("/models")]
    base = f"{parsed.scheme}://{parsed.netloc}{path}"
    return base, base + "/models"


def parse_max_tokens(raw: str) -> int:
    if not re.fullmatch(r"[0-9]+", raw.strip()) or int(raw.strip()) <= 0:
        fail(f"INFERENCE_MAX_TOKENS must be a positive integer (got '{raw}')")
    return int(raw.strip())


def parse_context_overrides(raw: str | None) -> dict[str, int]:
    """Validate INFERENCE_CONTEXT_LENGTHS: JSON object of model-id -> ctx."""
    if raw is None or raw.strip() == "":
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        fail("INFERENCE_CONTEXT_LENGTHS must be a JSON object")
    if not isinstance(parsed, dict):
        fail("INFERENCE_CONTEXT_LENGTHS must be a JSON object")
    overrides: dict[str, int] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or key == "":
            fail("INFERENCE_CONTEXT_LENGTHS keys must be non-empty model ids")
        if not is_positive_int(value):
            fail(
                "INFERENCE_CONTEXT_LENGTHS values must be positive integers "
                f"(model '{key}')"
            )
        overrides[key] = value
    return overrides


def fetch_catalog(models_url: str, token: str) -> object:
    request = urlrequest.Request(
        models_url, headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urlrequest.urlopen(request, timeout=30) as response:
            body = response.read()
    except urlerror.HTTPError as exc:
        # The reason is the HTTP status + endpoint only — never the token.
        fail(
            f"catalog request for {models_url} failed with HTTP {exc.code} "
            "— check that INFERENCE_TOKEN is valid for this endpoint"
        )
    except urlerror.URLError as exc:
        fail(f"catalog request for {models_url} failed: {exc.reason}")
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError):
        fail("catalog response is not valid JSON (malformed catalog)")


def parse_catalog(payload: object) -> list[dict]:
    """Validate the OpenAI-style {data:[...]} catalog; return model entries."""
    if not isinstance(payload, dict):
        fail("malformed catalog: expected a JSON object with a 'data' list")
    data = payload.get("data")
    if not isinstance(data, list):
        fail("malformed catalog: 'data' must be a list of model objects")
    entries: list[dict] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            fail("malformed catalog: each 'data' entry must be an object")
        model_id = item.get("id")
        if not isinstance(model_id, str) or model_id == "":
            fail("malformed catalog: every model entry needs a non-empty 'id'")
        if model_id in seen:
            fail(f"malformed catalog: duplicate model id '{model_id}'")
        seen.add(model_id)
        entries.append(item)
    return entries


def read_context(entry: dict) -> int | None:
    """Serving context window from known fields; None if not reported.

    Only positive non-bool integers count. An unusable value in one shape
    falls through to the next shape rather than being trusted.
    """
    for field in DIRECT_CONTEXT_FIELDS:
        value = entry.get(field)
        if is_positive_int(value):
            return value
    for outer, inner in NESTED_CONTEXT_FIELDS:
        container = entry.get(outer)
        if isinstance(container, dict):
            value = container.get(inner)
            if is_positive_int(value):
                return value
    return None


def write_manifest(path: str, manifest: dict) -> None:
    """Atomically write a non-secret manifest with mode 0600."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".generic_inference.", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
    except OSError as exc:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        fail(f"cannot write manifest to {path}: {exc.strerror or exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Discover a generic OpenAI-compatible model catalog."
    )
    parser.add_argument("--base-url", required=True,
                        help="OpenAI-compatible API base URL (env INFERENCE_URL)")
    parser.add_argument("--model", required=True,
                        help="model id to select (env INFERENCE_MODEL)")
    parser.add_argument("--max-tokens",
                        default=os.environ.get("INFERENCE_MAX_TOKENS", "4096"),
                        help="per-response output cap (env INFERENCE_MAX_TOKENS)")
    parser.add_argument("--output", required=True,
                        help="path for the non-secret manifest (mode 0600)")
    args = parser.parse_args(argv)

    # Documented invariant: a failed run must NEVER leave a stale manifest
    # behind — so the requested output is removed immediately after argument
    # parsing, BEFORE any validation (max-tokens, context overrides, URL),
    # not just before the network call. Removal errors are fail-closed: if we
    # cannot guarantee absence, we refuse to proceed.
    try:
        os.unlink(args.output)
    except FileNotFoundError:
        pass
    except OSError as exc:
        fail(f"cannot remove existing manifest {args.output}: {exc.strerror or exc}")

    max_tokens = parse_max_tokens(args.max_tokens)

    token = os.environ.get("INFERENCE_TOKEN", "")
    if not token:
        fail("INFERENCE_TOKEN is required in generic inference mode")

    overrides = parse_context_overrides(os.environ.get("INFERENCE_CONTEXT_LENGTHS"))

    base_url, models_url = normalize_base_url(args.base_url)

    catalog = parse_catalog(fetch_catalog(models_url, token))

    for override_id in sorted(overrides):
        if not any(entry["id"] == override_id for entry in catalog):
            fail(
                "INFERENCE_CONTEXT_LENGTHS names model "
                f"'{override_id}' which is not present in the catalog — "
                "an explicit override may not invent a model"
            )

    enabled: dict[str, int] = {}
    disabled: dict[str, str] = {}
    for entry in catalog:
        model_id = entry["id"]
        context = overrides.get(model_id, read_context(entry))
        if context is None:
            disabled[model_id] = "context length not reported"
        elif context < CONTEXT_FLOOR:
            disabled[model_id] = f"context length below {CONTEXT_FLOOR}"
        else:
            enabled[model_id] = context

    if not any(entry["id"] == args.model for entry in catalog):
        if enabled:
            hint = "set INFERENCE_MODEL to one of: " + ", ".join(sorted(enabled))
        else:
            hint = (
                "no catalog model passed the "
                f"{CONTEXT_FLOOR}-token context floor"
            )
        fail(
            f"selected model '{args.model}' (INFERENCE_MODEL) is not present "
            f"in the catalog at {models_url} — {hint}"
        )
    if args.model in disabled:
        fail(
            f"selected model '{args.model}' cannot be used: "
            f"{disabled[args.model]} — Hermes requires a context window of at "
            f"least {CONTEXT_FLOOR} tokens"
        )

    manifest = {
        "base_url": base_url,
        "selected_model": args.model,
        "selected_context_length": enabled[args.model],
        "max_tokens": max_tokens,
        "context_floor": CONTEXT_FLOOR,
        "enabled_models": dict(sorted(enabled.items())),
        "disabled_models": dict(sorted(disabled.items())),
    }
    write_manifest(args.output, manifest)

    # Human-readable availability report (the token is never printed).
    for model_id, context in sorted(enabled.items()):
        print(f"ENABLED — {model_id} — context length {context}")
    for model_id, reason in sorted(disabled.items()):
        print(f"DISABLED — {model_id} — {reason}")
    print(
        f"SELECTED — {args.model} — context length {enabled[args.model]} "
        f"(manifest: {args.output})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
