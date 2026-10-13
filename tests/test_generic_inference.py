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
                {"id": "e", "max_input_tokens": 68000},
            ]
        }
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.manifest_data()["enabled_models"],
            {"b": 65000, "c": 66000, "d": 67000, "e": 68000, "model-enabled": 64000},
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
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runtime = Path(self._tmp.name)
        self.alcf_dir = self.runtime / "alcf"
        self.venv_bin = self.runtime / "venv" / "bin"
        self.hermes_agent = self.runtime / "hermes-agent"
        self.home = self.runtime / "home"
        self.alcf_dir.mkdir(parents=True)
        self.venv_bin.mkdir(parents=True)
        self.hermes_agent.mkdir(parents=True)
        self.home.mkdir(parents=True)
        _CatalogHandler.payload = {"object": "list", "data": []}
        _CatalogHandler.expected_path = "/v1/models"
        _CatalogHandler.expected_auth = f"Bearer {SECRET}"
        _CatalogHandler.status = 200
        _CatalogHandler.seen = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _CatalogHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address[:2]
        self.base_url = f"http://{host}:{port}/v1"
        self.addCleanup(lambda: self.thread.join(timeout=5))
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

        (self.alcf_dir / "generic_inference.py").write_bytes(SCRIPT.read_bytes())
        (self.alcf_dir / "config.template.yaml").write_bytes(
            (ROOT / "config" / "config.template.yaml").read_bytes()
        )
        (self.hermes_agent / "capabilities.py").write_text(
            "delegate_task = True\n", encoding="utf-8"
        )

        python_wrapper = self.venv_bin / "python"
        python_wrapper.write_text(
            "#!/usr/bin/env bash\n"
            "if [[ ${1:-} == -c && ${2:-} == *plugins.dashboard_auth.basic* ]]; then\n"
            "  printf '%s\\n' test-password-hash\n"
            "  exit 0\n"
            "fi\n"
            f"exec {sys.executable!s} \"$@\"\n",
            encoding="utf-8",
        )
        python_wrapper.chmod(0o755)

        token_cli = self.venv_bin / "alcf-tokens"
        token_cli.write_text(
            "#!/usr/bin/env bash\n"
            "case ${1:-} in\n"
            "  get-token) printf '%s\\n' test-alcf-token ;;\n"
            "  login) exit 0 ;;\n"
            "  *) exit 2 ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        token_cli.chmod(0o755)

        body = (ROOT / "scripts" / "entrypoint.sh").read_text(encoding="utf-8")
        body = body.replace("ALCF_DIR=/opt/alcf", f"ALCF_DIR={self.alcf_dir}")
        body = body.replace("VENV_BIN=/opt/hermes/.venv/bin", f"VENV_BIN={self.venv_bin}")
        body = body.replace("/opt/hermes/agent", str(self.hermes_agent))
        stop_marker = "# --- 4b. Model availability status banner"
        self.assertIn(stop_marker, body)
        body = body.replace(stop_marker, "exit 0\n" + stop_marker, 1)
        self.entrypoint = self.runtime / "entrypoint-under-test.sh"
        self.entrypoint.write_text(body, encoding="utf-8")
        self.entrypoint.chmod(0o755)

    def run_entrypoint(self, **overrides):
        env = os.environ.copy()
        env.update(
            {
                "HERMES_HOME": str(self.home),
                "ALCF_CLUSTER": "sophia",
                "ALCF_DASHBOARD_PASSWORD": "test-dashboard-password",
                "ALCF_ENABLE_GLOBUS_COMPUTE": "0",
                "INFERENCE_URL": "",
                "INFERENCE_TOKEN": "",
                "INFERENCE_MODEL": "",
                "ALCF_DELEGATION_MODEL": "",
            }
        )
        env.update(overrides)
        return subprocess.run(
            ["bash", str(self.entrypoint)],
            text=True,
            capture_output=True,
            env=env,
            timeout=15,
        )

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
        for partial in (
            {"INFERENCE_TOKEN": "only-token"},
            {"INFERENCE_MODEL": "only-model"},
        ):
            with self.subTest(partial=tuple(partial)):
                result = self.run_entrypoint(**partial)
                self.assertEqual(result.returncode, 78, result.stderr)
                self.assertIn("partial generic inference configuration", result.stderr)
                self.assertFalse((self.home / "config.yaml").exists())

    def test_generic_delegation_never_falls_back_to_alcf_inference_resolution(self):
        _CatalogHandler.payload = {
            "data": [{"id": "launch-model", "context_length": 131072}]
        }
        result = self.run_entrypoint(
            INFERENCE_URL=self.base_url,
            INFERENCE_TOKEN=SECRET,
            INFERENCE_MODEL="launch-model",
            ALCF_DELEGATION_MODEL="missing-delegation-model",
        )
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertIn("generic delegation model", result.stderr.lower())
        self.assertIn("refusing to start", result.stderr.lower())
        self.assertFalse((self.home / "config.yaml").exists())

    def test_generic_render_config_maps_selected_manifest_model(self):
        _CatalogHandler.payload = {
            "data": [
                {"id": "selected/model", "context_length": 131072},
                {"id": "other-model", "context_length": 70000},
            ]
        }
        result = self.run_entrypoint(
            INFERENCE_URL=self.base_url,
            INFERENCE_TOKEN=SECRET,
            INFERENCE_MODEL="selected/model",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        config = (self.home / "config.yaml").read_text(encoding="utf-8")
        self.assertIn('provider: "custom:generic-inference"', config)
        self.assertIn(f'base_url: "{self.base_url}"', config)
        self.assertIn('model: "selected/model"', config)
        self.assertIn('default: "selected/model"', config)
        self.assertIn(f'api_key: "{SECRET}"', config)
        self.assertEqual(stat.S_IMODE((self.home / "config.yaml").stat().st_mode), 0o600)

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

    def test_main_release_builds_multiarch_latest_without_temporary_feature_trigger(self):
        workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
        self.assertNotIn("feat/generic-inference", workflow)
        self.assertIn("branches: [main]", workflow)
        self.assertIn("linux/amd64,linux/arm64", workflow)
        self.assertNotIn("type=ref,event=branch", workflow)
        self.assertIn("type=raw,value=latest,enable={{is_default_branch}}", workflow)

    def test_generic_docker_example_uses_the_default_release_tag(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("alcf-agent:feat-generic-inference", readme)
        self.assertIn("alcf-agent:latest", readme)

    def test_readme_includes_a_safe_mag_example(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("### Example: AmSC Model Access Gateway (MAG)", readme)
        self.assertIn("INFERENCE_URL='https://mag.genesis.americansciencecloud.org/v1'", readme)
        self.assertIn("INFERENCE_TOKEN=\"$MAG_TOKEN\"", readme)
        self.assertIn("INFERENCE_MODEL='gpt-5.6-sol@openai'", readme)
        self.assertIn("INFERENCE_MAX_TOKENS='8192'", readme)
        self.assertIn("-p 127.0.0.1:8787:8787", readme)
        self.assertNotIn("INFERENCE_TOKEN='YOUR_MAG_TOKEN'", readme)


if __name__ == "__main__":
    unittest.main()
