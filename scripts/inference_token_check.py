#!/usr/bin/env python3
"""Live validity check of the stored ALCF inference login.

    inference_token_check.py <path-to-alcf-tokens>

`alcf-tokens get-token inference` exits 0 with a locally cached/refreshed token
even when ALCF REJECTS it (e.g. a high-assurance timeout), so it cannot gate
startup. This runs `alcf-tokens test-token inference` (a GET of the gateway's
/resource_server/whoami with the stored token) and classifies the outcome.

Prints EXACTLY one line `STATUS<TAB>reason` and exits 0 for every
classification -- the classification is the API, not the exit code:

    ok           the gateway accepted the token
    auth         no usable stored login, or the gateway said 401/403
    unavailable  gateway 5xx/429/408, our timeout, or a network error
    unknown      anything else

SECURITY: `reason` is built from fixed phrases (plus an HTTP status code or an
httpx exception class name). The response body, stdout and stderr of
alcf-tokens are NEVER echoed: they can carry a token. The reason is still
sanitized (token-like strings redacted, control characters dropped) and
truncated to 120 characters as defence in depth.

Stdlib only. Timeout: ALCF_TOKEN_CHECK_TIMEOUT seconds (default 20).
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys

DEFAULT_TIMEOUT = 20.0
MAX_REASON = 120

# httpx exception class names / resolver errors that mean "could not reach the
# gateway" (alcf-tokens does not catch these: traceback on stderr, no JSON).
NETWORK_MARKERS = (
    "ConnectError", "ConnectTimeout", "ReadTimeout", "TimeoutException",
    "NetworkError", "RemoteProtocolError",
    "Name or service not known", "Temporary failure in name resolution",
)
UNAVAILABLE_HTTP = {408, 429}

_TOKENISH = (
    re.compile(r"(?i)bearer\s+\S+"),
    re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]*)?"),  # JWT
    re.compile(r"[A-Za-z0-9._~+/=-]{24,}"),  # long opaque token
)


def sanitize(reason: str) -> str:
    """One short, token-free line."""
    text = "".join(ch if ch.isprintable() else " " for ch in str(reason))
    for pattern in _TOKENISH:
        text = pattern.sub("[redacted]", text)
    text = " ".join(text.split()) or "no detail"
    return text[:MAX_REASON]


def _timeout() -> float:
    raw = os.environ.get("ALCF_TOKEN_CHECK_TIMEOUT", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_TIMEOUT
    except ValueError:
        return DEFAULT_TIMEOUT
    return value if value > 0 else DEFAULT_TIMEOUT


def _ready_payload(stdout: str):
    """The last stdout line that is a JSON object carrying `ready`, else None."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict) and "ready" in payload:
            return payload
    return None


def _auth_phrase(error: str) -> str:
    low = error.lower()
    if "high-assurance" in low or "high assurance" in low:
        return "high-assurance timeout"
    if "no valid tokens" in low or "authenticate" in low or "login" in low:
        return "no valid stored login"
    return "stored login not usable"


def classify(returncode: int, stdout: str, stderr: str) -> tuple:
    """(status, reason) for one finished `alcf-tokens test-token inference`."""
    payload = _ready_payload(stdout or "")
    if payload is not None:
        if payload.get("ready") is True:
            if returncode == 0:
                return "ok", "whoami OK"
            return "unknown", f"ready but exit {returncode}"
        error = str(payload.get("error") or "")
        match = re.match(r"\s*HTTP (\d{3})\b", error)
        if match:
            code = int(match.group(1))
            if code in (401, 403):
                detail = _auth_phrase(error)
                if detail == "high-assurance timeout":
                    return "auth", f"HTTP {code}: high-assurance timeout"
                return "auth", f"HTTP {code}"
            if code >= 500 or code in UNAVAILABLE_HTTP:
                return "unavailable", f"HTTP {code}"
            return "unknown", f"HTTP {code}"
        return "auth", _auth_phrase(error)
    for marker in NETWORK_MARKERS:
        if marker in (stderr or ""):
            return "unavailable", f"network error: {marker}"
    return "unknown", f"unrecognised test-token output (exit {returncode})"


def run_check(alcf_tokens: str, timeout: float) -> tuple:
    try:
        proc = subprocess.Popen(
            [alcf_tokens, "test-token", "inference"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="replace", start_new_session=True)
    except OSError:
        return "unknown", "alcf-tokens could not be executed"
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill the whole process group: a grandchild holding the pipes open
        # would otherwise make the second communicate() hang.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.communicate()
        return "unavailable", f"token check timed out after {timeout:g}s"
    return classify(proc.returncode, stdout, stderr)


def main(argv: list) -> int:
    if len(argv) != 2:
        status, reason = "unknown", "usage: inference_token_check.py <alcf-tokens>"
    else:
        try:
            status, reason = run_check(argv[1], _timeout())
        except Exception:  # noqa: BLE001 - classification is the API; never crash
            status, reason = "unknown", "token check failed internally"
    print(f"{status}\t{sanitize(reason)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
