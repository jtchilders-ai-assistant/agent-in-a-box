#!/usr/bin/env python3
"""Behavioral tests for the generic OpenAI-compatible inference catalog helper."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generic_inference.py"
SECRET = "token-$-must-never-appear"


class _CatalogHandler(BaseHTTPRequestHandler):
    payload: object = {"object": "list", "data": []}
    expected_path = "/v1/models"
    expected_auth = f"Bearer {SECRET}"
    status = 200
    seen: list[tuple[str, str | None]] = []

    def do_GET(self):  # noqa: N802
        type(self).seen.append((self.path, self.headers.get("Authorization")))
        if self.path != type(self).expected_path:
            self.send_error(404)
            return
        if self.headers.get("Authorization") != type(self).expected_auth:
            self.send_error(401)
            return
        body = json.dumps(type(self).payload).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *_args):  # noqa: A002
        pass


class GenericInferenceTests(unittest.TestCase):
    def setUp(self):
        _CatalogHandler.payload = {"object": "list", "data": []}
        _CatalogHandler.expected_path = "/v1/models"
        _CatalogHandler.expected_auth = f"Bearer {SECRET}"
        _CatalogHandler.status = 200
        _CatalogHandler.seen = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _CatalogHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host = self.server.server_address[0]
        port = self.server.server_address[1]
        self.base_url = f"http://{host}:{port}/v1"
        self.tmp = tempfile.TemporaryDirectory(prefix="generic_inference_")
        self.manifest = Path(self.tmp.name) / "manifest.json"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    def run_helper(
        self,
        *,
        model="model-enabled",
        base_url=None,
        token=SECRET,
        context_lengths=None,
        max_tokens="4096",
    ):
        env = os.environ.copy()
        env["INFERENCE_TOKEN"] = token
        if context_lengths is not None:
            env["INFERENCE_CONTEXT_LENGTHS"] = context_lengths
        else:
            env.pop("INFERENCE_CONTEXT_LENGTHS", None)
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--base-url",
                base_url or self.base_url,
                "--model",
                model,
                "--max-tokens",
                max_tokens,
                "--output",
                str(self.manifest),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def manifest_data(self):
        return json.loads(self.manifest.read_text(encoding="utf-8"))

    def test_discovers_catalog_with_bearer_auth_and_records_enabled_models(self):
        _CatalogHandler.payload = {
            "object": "list",
            "data": [
                {"id": "model-enabled", "max_model_len": 131072},
                {"id": "model-low", "context_length": 32768},
                {"id": "model-missing"},
            ],
        }
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(_CatalogHandler.seen, [("/v1/models", f"Bearer {SECRET}")])
        manifest = self.manifest_data()
        self.assertEqual(manifest["base_url"], self.base_url)
        self.assertEqual(manifest["selected_model"], "model-enabled")
        self.assertEqual(manifest["selected_context_length"], 131072)
        self.assertEqual(manifest["max_tokens"], 4096)
        self.assertEqual(manifest["enabled_models"], {"model-enabled": 131072})
        self.assertEqual(
            manifest["disabled_models"],
            {
                "model-low": "context length below 64000",
                "model-missing": "context length not reported",
            },
        )
        self.assertIn("ENABLED — model-enabled", result.stdout)
        self.assertIn("DISABLED — model-low — context length below 64000", result.stdout)
        self.assertIn("DISABLED — model-missing — context length not reported", result.stdout)

    def test_accepts_supported_context_metadata_shapes(self):
        _CatalogHandler.payload = {
            "data": [
                {"id": "model-enabled", "context_window": 64000},
                {"id": "b", "max_context_length": 65000},
                {"id": "c", "capabilities": {"context_window_tokens": 66000}},
                {"id": "d", "metadata": {"context_length": 67000}},
            ]
        }
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.manifest_data()["enabled_models"],
            {"b": 65000, "c": 66000, "d": 67000, "model-enabled": 64000},
        )

    def test_explicit_context_override_wins_and_can_enable_missing_metadata(self):
        _CatalogHandler.payload = {
            "data": [
                {"id": "model-enabled", "max_model_len": 32000},
                {"id": "other"},
            ]
        }
        result = self.run_helper(
            context_lengths=json.dumps({"model-enabled": 200000, "other": 70000})
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.manifest_data()["enabled_models"],
            {"model-enabled": 200000, "other": 70000},
        )

    def test_selected_model_must_exist_in_catalog_even_if_override_names_it(self):
        _CatalogHandler.payload = {"data": [{"id": "other", "context_length": 70000}]}
        result = self.run_helper(context_lengths='{"model-enabled": 100000}')
        self.assertEqual(result.returncode, 78)
        self.assertIn("not present in the catalog", result.stderr)
        self.assertFalse(self.manifest.exists())

    def test_selected_model_with_missing_context_is_rejected(self):
        _CatalogHandler.payload = {"data": [{"id": "model-enabled"}]}
        result = self.run_helper()
        self.assertEqual(result.returncode, 78)
        self.assertIn("context length not reported", result.stderr)

    def test_selected_model_below_floor_is_rejected(self):
        _CatalogHandler.payload = {"data": [{"id": "model-enabled", "context_length": 63999}]}
        result = self.run_helper()
        self.assertEqual(result.returncode, 78)
        self.assertIn("context length below 64000", result.stderr)

    def test_url_is_normalized_and_models_suffix_is_not_duplicated(self):
        _CatalogHandler.payload = {"data": [{"id": "model-enabled", "context_length": 64000}]}
        result = self.run_helper(base_url=self.base_url + "/models/")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(_CatalogHandler.seen[0][0], "/v1/models")
        self.assertEqual(self.manifest_data()["base_url"], self.base_url)

    def test_rejects_unsafe_or_ambiguous_urls_before_network(self):
        for url in (
            "ftp://example.invalid/v1",
            "https://user:pass@example.invalid/v1",
            "https://example.invalid/v1?x=1",
            "https://example.invalid/v1#fragment",
            "https:///v1",
        ):
            with self.subTest(url=url):
                result = self.run_helper(base_url=url)
                self.assertEqual(result.returncode, 78)
                self.assertIn("INFERENCE_URL", result.stderr)
        self.assertEqual(_CatalogHandler.seen, [])

    def test_rejects_malformed_context_override_and_nonpositive_max_tokens(self):
        for context_lengths in ("not-json", "[]", '{"m": true}', '{"m": 0}', '{"": 70000}'):
            with self.subTest(context_lengths=context_lengths):
                result = self.run_helper(context_lengths=context_lengths)
                self.assertEqual(result.returncode, 78)
                self.assertIn("INFERENCE_CONTEXT_LENGTHS", result.stderr)
        for max_tokens in ("0", "-1", "1.5", "abc"):
            with self.subTest(max_tokens=max_tokens):
                result = self.run_helper(max_tokens=max_tokens)
                self.assertEqual(result.returncode, 78)
                self.assertIn("INFERENCE_MAX_TOKENS", result.stderr)

    def test_every_failure_removes_a_stale_manifest(self):
        failure_cases = (
            {"context_lengths": "not-json"},
            {"base_url": "ftp://example.invalid/v1"},
            {"max_tokens": "0"},
        )
        for kwargs in failure_cases:
            with self.subTest(kwargs=kwargs):
                self.manifest.write_text('{"stale": true}\n', encoding="utf-8")
                result = self.run_helper(**kwargs)
                self.assertEqual(result.returncode, 78)
                self.assertFalse(self.manifest.exists())

    def test_rejects_malformed_catalog_and_duplicate_model_ids(self):
        for payload in (
            [],
            {},
            {"data": "not-a-list"},
            {"data": [{"id": "model-enabled", "context_length": 70000}, {"id": "model-enabled", "context_length": 80000}]},
            {"data": [{"id": "", "context_length": 70000}]},
        ):
            with self.subTest(payload=payload):
                _CatalogHandler.payload = payload
                result = self.run_helper()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("catalog", result.stderr.lower())
                self.assertFalse(self.manifest.exists())

    def test_authentication_failure_is_actionable_without_leaking_token(self):
        _CatalogHandler.expected_auth = "Bearer some-other-token"
        result = self.run_helper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("401", result.stderr)
        self.assertNotIn(SECRET, result.stdout + result.stderr)

    def test_token_never_appears_in_output_or_noncredential_manifest(self):
        _CatalogHandler.payload = {"data": [{"id": "model-enabled", "context_length": 70000}]}
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        all_visible = result.stdout + result.stderr + self.manifest.read_text(encoding="utf-8")
        self.assertNotIn(SECRET, all_visible)
        self.assertEqual(stat.S_IMODE(self.manifest.stat().st_mode), 0o600)

    def test_control_characters_in_catalog_model_ids_are_rejected(self):
        hostile = 'model"\n  provider: "custom:attacker"'
        _CatalogHandler.payload = {
            "data": [{"id": hostile, "context_length": 70000}]
        }
        result = self.run_helper(model=hostile)
        self.assertEqual(result.returncode, 78)
        self.assertIn("control characters", result.stderr)
        self.assertFalse(self.manifest.exists())


class EntrypointAndWorkflowContractTests(unittest.TestCase):
    def test_entrypoint_keeps_default_alcf_mode_and_combined_globus_login(self):
        body = (ROOT / "scripts" / "entrypoint.sh").read_text(encoding="utf-8")
        self.assertIn('if [[ -n "${INFERENCE_URL:-}" ]]', body)
        self.assertIn('"$ALCF_TOKENS" login', body)
        self.assertIn("authed_inference", body)
        self.assertIn("GENERIC_MODE", body)
        self.assertIn("render_config", body)
        self.assertIn("populate_models.py", body)
        self.assertIn("ALCF_SHOW_MODEL_STATUS", body)

    def test_partial_generic_configuration_fails_closed(self):
        body = (ROOT / "scripts" / "entrypoint.sh").read_text(encoding="utf-8")
        self.assertIn("INFERENCE_TOKEN", body)
        self.assertIn("INFERENCE_MODEL", body)
        self.assertIn("exit 78", body)
        self.assertIn('elif [[ -n "${INFERENCE_TOKEN:-}" || -n "${INFERENCE_MODEL:-}"', body)

    def test_generic_delegation_never_falls_back_to_alcf_inference_resolution(self):
        body = (ROOT / "scripts" / "entrypoint.sh").read_text(encoding="utf-8")
        self.assertIn('if [[ "$GENERIC_MODE" == "1" ]]', body)
        self.assertIn("generic delegation model", body.lower())
        self.assertIn("refusing to start", body.lower())

    def test_rendered_credential_bearing_config_is_forced_to_mode_0600(self):
        body = (ROOT / "scripts" / "entrypoint.sh").read_text(encoding="utf-8")
        self.assertRegex(body, r"(?:chmod 600|install -m 600).*(?:CONFIG_OUT|config\.yaml)")

    def test_generic_token_is_removed_before_long_lived_processes_start(self):
        body = (ROOT / "scripts" / "entrypoint.sh").read_text(encoding="utf-8")
        unset_match = re.search(r"(?m)^\s*unset INFERENCE_TOKEN\s*$", body)
        if unset_match is None:
            self.fail("entrypoint must unset INFERENCE_TOKEN")
        unset_at = unset_match.start()
        dashboard_at = body.index("hermes dashboard --host")
        caddy_at = body.index("exec caddy run")
        self.assertLess(unset_at, dashboard_at)
        self.assertLess(unset_at, caddy_at)

    def test_generic_helper_is_copied_into_image(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("scripts/generic_inference.py", dockerfile)

    def test_feature_branch_builds_multiarch_test_tag_without_latest(self):
        workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
        self.assertIn("feat/generic-inference", workflow)
        self.assertIn("linux/amd64,linux/arm64", workflow)
        self.assertIn("type=ref,event=branch", workflow)
        self.assertIn("type=raw,value=latest,enable={{is_default_branch}}", workflow)


if __name__ == "__main__":
    unittest.main()
