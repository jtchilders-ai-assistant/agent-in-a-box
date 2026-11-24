#!/usr/bin/env python3
"""Behavioral tests for scripts/inference_token_check.py (live ALCF login check).

Each case runs the real script against a fake `alcf-tokens` that replays a
recorded `test-token inference` outcome (stdout / stderr / exit code), shaped
after alcf_tokens/cli.py test_token: JSON {"ready", "error"} on stdout, an
uncaught httpx traceback on stderr for network failures.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "inference_token_check.py"
FAKE_TOKEN = "fake-inference-token-0123456789abcdef"
STATUSES = {"ok", "auth", "unavailable", "unknown"}


def _err(text: str) -> str:
    return json.dumps({"ready": False, "error": text}) + "\n"


CASES = {
    "ready": dict(stdout=json.dumps({"ready": True, "error": None}) + "\n", rc=0,
                  want=("ok", "whoami OK")),
    "auth_error": dict(
        stdout=_err('No valid tokens found for inference. Please authenticate by running '
                    f'"alcf-tokens login". (cached {FAKE_TOKEN})'),
        rc=1, want=("auth", "no valid stored login")),
    "http_401_high_assurance": dict(
        stdout=_err('HTTP 401: {"detail": "Token rejected: high-assurance timeout", '
                    f'"token": "{FAKE_TOKEN}"}}'),
        rc=1, want=("auth", "HTTP 401: high-assurance timeout")),
    "http_403": dict(stdout=_err(f"HTTP 403: forbidden for Bearer {FAKE_TOKEN}"), rc=1,
                     want=("auth", "HTTP 403")),
    "http_503": dict(stdout=_err(f"HTTP 503: <html>upstream {FAKE_TOKEN} down</html>"), rc=1,
                     want=("unavailable", "HTTP 503")),
    "network_traceback": dict(
        stdout="", rc=1,
        stderr=("Traceback (most recent call last):\n"
                '  File "cli.py", line 144, in test_token\n'
                f"    headers={{'Authorization': 'Bearer {FAKE_TOKEN}'}}\n"
                "httpx.ConnectError: [Errno 8] nodename nor servname provided, or not known\n"),
        want=("unavailable", "network error: ConnectError")),
    "garbage": dict(stdout=f"lol {FAKE_TOKEN} \x1b[31mwhat\n", rc=0,
                    want=("unknown", "unrecognised test-token output (exit 0)")),
}


def _traceback(last: str) -> str:
    return ("Traceback (most recent call last):\n"
            '  File "alcf_tokens/cli.py", line 95, in get_token\n'
            "    token = get_access_token(service)\n"
            f"    refresh_token='{FAKE_TOKEN}'\n"
            f"{last}\n")


# `alcf-tokens get-token inference` failures (alcf_tokens/cli.py:94-98 catches
# ONLY AuthError: message on stderr, exit 1; anything else from the refresh
# (globus_sdk / requests) is an uncaught traceback, exit 1).
GET_TOKEN_CASES = {
    "auth_error": dict(
        stderr=(f"No valid tokens found for 'inference' (no token for {FAKE_TOKEN}). "
                'Please authenticate by running "alcf-tokens login".\n'),
        want=("auth", "get-token inference failed: no valid stored login")),
    "requests_connection_error": dict(
        stderr=_traceback(
            "requests.exceptions.ConnectionError: HTTPSConnectionPool(host='auth.globus.org', "
            "port=443): Max retries exceeded with url: /v2/oauth2/token (Caused by "
            "NameResolutionError(\"<urllib3.connection.HTTPSConnection>: Failed to resolve "
            "'auth.globus.org'\"))"),
        want=("unavailable", "get-token network error: NameResolutionError")),
    "globus_connection_error": dict(
        stderr=_traceback("globus_sdk.transport.requests.GlobusConnectionError: "
                          "network failure"),
        want=("unavailable", "get-token network error: GlobusConnectionError")),
    "max_retries": dict(
        stderr=_traceback("urllib3.exceptions.MaxRetryError: Max retries exceeded with url: /x"),
        want=("unavailable", "get-token network error: Max retries exceeded")),
    "globus_503": dict(
        stderr=_traceback("globus_sdk.services.auth.errors.AuthAPIError: ('POST', "
                          "'https://auth.globus.org/v2/oauth2/token', None, 503, 'Error', "
                          "'Service Unavailable')"),
        want=("unavailable", "get-token: Globus HTTP 503")),
    "globus_429": dict(
        stderr=_traceback("globus_sdk.services.auth.errors.AuthAPIError: ('POST', "
                          "'https://auth.globus.org/v2/oauth2/token', None, 429, 'Error', "
                          "'Too Many Requests')"),
        want=("unavailable", "get-token: Globus HTTP 429")),
    "refresh_token_rejected": dict(
        stderr=_traceback("globus_sdk.services.auth.errors.AuthAPIError: ('POST', "
                          "'https://auth.globus.org/v2/oauth2/token', None, 400, "
                          "'invalid_grant', 'invalid_grant')"),
        want=("auth", "get-token inference failed: refresh token rejected (invalid_grant)")),
    "unknown_traceback": dict(
        stderr=_traceback("KeyError: 'access_token'"),
        want=("unknown", "get-token failed unexpectedly (exit 1)")),
    "no_token_printed": dict(
        stdout="\n", stderr=f"warning: cache {FAKE_TOKEN}\n", rc=0,
        want=("unknown", "get-token printed no token")),
}


class InferenceTokenCheckTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.args_log = self.dir / "args"

    def fake_cli(self, *, stdout: str = "", stderr: str = "", rc: int = 0,
                 hang: bool = False, gt_stdout: str = FAKE_TOKEN + "\n", gt_stderr: str = "",
                 gt_rc: int = 0, gt_hang: bool = False) -> Path:
        """`get-token inference` replays gt_* (by default: prints the token);
        `test-token inference` replays stdout/stderr/rc. Each call's args are
        appended to self.args_log."""
        for name, text in (("out", stdout), ("err", stderr),
                           ("gt_out", gt_stdout), ("gt_err", gt_stderr)):
            (self.dir / name).write_text(text, encoding="utf-8")
        cli = self.dir / "alcf-tokens"
        cli.write_text(
            "#!/bin/sh\n"
            f"printf '%s\\n' \"$*\" >> '{self.args_log}'\n"
            'if [ "$1" = get-token ]; then\n'
            + ("  sleep 30\n" if gt_hang else "")
            + f"  cat '{self.dir / 'gt_out'}'\n"
            f"  cat '{self.dir / 'gt_err'}' >&2\n"
            f"  exit {gt_rc}\n"
            "fi\n"
            + ("sleep 30\n" if hang else "")
            + f"cat '{self.dir / 'out'}'\n"
            f"cat '{self.dir / 'err'}' >&2\n"
            f"exit {rc}\n",
            encoding="utf-8")
        cli.chmod(0o755)
        return cli

    def calls(self) -> list:
        return self.args_log.read_text(encoding="utf-8").splitlines()

    def run_check(self, cli: Path | str, timeout: str | None = None):
        env = {k: v for k, v in os.environ.items() if k != "ALCF_TOKEN_CHECK_TIMEOUT"}
        if timeout is not None:
            env["ALCF_TOKEN_CHECK_TIMEOUT"] = timeout
        return subprocess.run([sys.executable, str(SCRIPT), str(cli)], text=True,
                              capture_output=True, env=env, timeout=30)

    def assert_one_line(self, res) -> tuple:
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stderr, "")
        self.assertTrue(res.stdout.endswith("\n"))
        lines = res.stdout.splitlines()
        self.assertEqual(len(lines), 1, res.stdout)
        status, sep, reason = lines[0].partition("\t")
        self.assertEqual(sep, "\t", res.stdout)
        self.assertIn(status, STATUSES)
        self.assertLessEqual(len(reason), 120)
        self.assertTrue(reason.isprintable())
        self.assertNotIn(FAKE_TOKEN, res.stdout)
        return status, reason

    def test_recorded_outcomes_classify(self):
        for name, case in CASES.items():
            with self.subTest(case=name):
                self.args_log.unlink(missing_ok=True)
                cli = self.fake_cli(stdout=case["stdout"], stderr=case.get("stderr", ""),
                                    rc=case["rc"])
                got = self.assert_one_line(self.run_check(cli))
                self.assertEqual(got, case["want"])
                self.assertEqual(self.calls(), ["get-token inference", "test-token inference"])

    def test_get_token_failures_classify_without_running_test_token(self):
        for name, case in GET_TOKEN_CASES.items():
            with self.subTest(case=name):
                self.args_log.unlink(missing_ok=True)
                cli = self.fake_cli(stdout=CASES["ready"]["stdout"], gt_stdout=case.get("stdout", ""),
                                    gt_stderr=case["stderr"], gt_rc=case.get("rc", 1))
                got = self.assert_one_line(self.run_check(cli))
                self.assertEqual(got, case["want"])
                self.assertEqual(self.calls(), ["get-token inference"])

    def test_positive_control_get_token_fakes_carry_the_token(self):
        for name, case in GET_TOKEN_CASES.items():
            with self.subTest(case=name):
                cli = self.fake_cli(gt_stdout=case.get("stdout", ""), gt_stderr=case["stderr"],
                                    gt_rc=case.get("rc", 1))
                raw = subprocess.run([str(cli), "get-token", "inference"], text=True,
                                     capture_output=True, timeout=10)
                self.assertIn(FAKE_TOKEN, raw.stdout + raw.stderr)

    def test_get_token_hang_is_unavailable(self):
        cli = self.fake_cli(stdout=CASES["ready"]["stdout"], gt_hang=True)
        start = time.monotonic()
        got = self.assert_one_line(self.run_check(cli, timeout="0.5"))
        self.assertLess(time.monotonic() - start, 10, "the hung get-token was not killed")
        self.assertEqual(got, ("unavailable", "get-token timed out after 0.5s"))
        self.assertEqual(self.calls(), ["get-token inference"])

    def test_positive_control_fakes_really_carry_the_token(self):
        """The no-token assertions are not vacuous: the fake CLI output that
        the check consumed does contain the token in every non-ok case."""
        for name, case in CASES.items():
            if name == "ready":
                continue
            with self.subTest(case=name):
                cli = self.fake_cli(stdout=case["stdout"], stderr=case.get("stderr", ""),
                                    rc=case["rc"])
                raw = subprocess.run([str(cli), "test-token", "inference"], text=True,
                                     capture_output=True, timeout=10)
                self.assertIn(FAKE_TOKEN, raw.stdout + raw.stderr)

    def test_hang_past_timeout_is_unavailable(self):
        cli = self.fake_cli(stdout=CASES["ready"]["stdout"], hang=True)
        start = time.monotonic()
        got = self.assert_one_line(self.run_check(cli, timeout="0.5"))
        self.assertLess(time.monotonic() - start, 10, "the hung CLI was not killed")
        self.assertEqual(got, ("unavailable", "token check timed out after 0.5s"))

    def test_unexecutable_cli_is_unknown(self):
        got = self.assert_one_line(self.run_check(self.dir / "missing-alcf-tokens"))
        self.assertEqual(got, ("unknown", "alcf-tokens could not be executed"))

    def test_ready_with_nonzero_exit_is_not_ok(self):
        cli = self.fake_cli(stdout=CASES["ready"]["stdout"], rc=1)
        self.assertEqual(self.assert_one_line(self.run_check(cli)), ("unknown", "ready but exit 1"))

    def test_other_http_status_is_unknown_and_408_429_unavailable(self):
        for code, want in ((404, "unknown"), (408, "unavailable"), (429, "unavailable"),
                           (500, "unavailable")):
            with self.subTest(code=code):
                cli = self.fake_cli(stdout=_err(f"HTTP {code}: {FAKE_TOKEN}"), rc=1)
                self.assertEqual(self.assert_one_line(self.run_check(cli)),
                                 (want, f"HTTP {code}"))

    def test_bad_timeout_value_falls_back_to_default(self):
        cli = self.fake_cli(stdout=CASES["ready"]["stdout"])
        for value in ("abc", "0", "-3"):
            with self.subTest(value=value):
                self.assertEqual(self.assert_one_line(self.run_check(cli, timeout=value)),
                                 ("ok", "whoami OK"))

    def test_wrong_usage_still_prints_one_line(self):
        res = subprocess.run([sys.executable, str(SCRIPT)], text=True, capture_output=True,
                             timeout=30)
        self.assertEqual(self.assert_one_line(res)[0], "unknown")


class SanitizeTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(SCRIPT.parent))
        self.addCleanup(sys.path.remove, str(SCRIPT.parent))
        import inference_token_check
        self.mod = inference_token_check

    def test_redacts_token_like_strings_and_control_chars(self):
        out = self.mod.sanitize(f"Bearer {FAKE_TOKEN}\n\x1b[31m eyJabc.def.ghi {FAKE_TOKEN}")
        self.assertNotIn(FAKE_TOKEN, out)
        self.assertNotIn("eyJabc", out)
        self.assertTrue(out.isprintable())

    def test_truncates_to_120(self):
        self.assertEqual(len(self.mod.sanitize("word " * 100)), 120)

    def test_positive_control_plain_reason_is_kept(self):
        self.assertEqual(self.mod.sanitize("HTTP 401: high-assurance timeout"),
                         "HTTP 401: high-assurance timeout")


if __name__ == "__main__":
    unittest.main()
