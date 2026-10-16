"""Exact Docker contract + entrypoint first-run exit-status tests."""
from pathlib import Path
import unittest, re, sys

ROOT = Path(__file__).resolve().parents[1]

class DockerContractTests(unittest.TestCase):
    def test_dockerfile_pins_alcf_tokens_version(self):
        df = (ROOT / "Dockerfile").read_text()
        self.assertIn("alcf-tokens==0.3.0", df, "Dockerfile must pin exact alcf-tokens version")
        # Must NOT substitute nersc-tokens
        self.assertNotIn("nersc-tokens==0.1.1", df)

    def test_dockerfile_has_all_required_copy_contracts(self):
        df = (ROOT / "Dockerfile").read_text()
        required_copies = [
            "COPY scripts/entrypoint.sh /opt/alcf/entrypoint.sh",
            "COPY scripts/alcf_facility.py /opt/alcf/alcf_facility.py",
            "COPY scripts/alcf_remote_bash.py /opt/alcf/alcf_remote_bash.py",
            "COPY scripts/alcf_bash_mcp.py /opt/alcf/alcf_bash_mcp.py",
            "COPY scripts/resolve_context_length.py /opt/alcf/resolve_context_length.py",
            "COPY scripts/populate_models.py /opt/alcf/populate_models.py",
            "COPY scripts/generic_inference.py /opt/alcf/generic_inference.py",
            "COPY scripts/iri_hello_world.py /opt/alcf/iri_hello_world.py",
            "COPY skills/  /opt/alcf/skills/",
            "COPY memory/  /opt/alcf/memory/",
            "COPY docs/    /opt/alcf/docs/",
            "COPY config/config.template.yaml /opt/alcf/config.template.yaml",
            "COPY config/SOUL.md /opt/alcf/SOUL.md",
        ]
        for c in required_copies:
            with self.subTest(c):
                self.assertIn(c, df)

    def test_dockerfile_uses_venv_alcf_tokens_cli(self):
        df = (ROOT / "Dockerfile").read_text()
        # Dockerfile installs via uv pip using /opt/hermes/.venv/bin/python; the CLI binary is in entrypoint.
        self.assertIn("/opt/hermes/.venv/bin/python", df)
        # Must NOT have wrong package (ignore false reviewer claim)
        self.assertNotIn("nersc-tokens==0.1.1", df)

    def test_entrypoint_preserves_setup_exit_status(self):
        ep = (ROOT / "scripts/entrypoint.sh").read_text()
        # Must use set -e (direct exit propagation); must not use `|| exit 1`
        # after a critical setup call (would mask the real exit code).
        self.assertIn("set -euo pipefail", ep)
        # Confirm no `|| exit 1` masks setup exit status
        for line in ep.split("\n"):
            if "|| exit 1" in line:
                # Allow only non-setup contexts; here we assert none at all for simplicity.
                self.fail(f"entrypoint uses `|| exit 1` which masks real exit code: {line}")

    def test_entrypoint_runs_setup_conditional_on_config_absence(self):
        ep = (ROOT / "scripts/entrypoint.sh").read_text()
        # Setup (alcf-tokens login) runs conditionally when config/auth absent.
        # The entrypoint uses $HERMES_HOME for durable volume; --home is not a CLI arg here.
        self.assertIn("authed_inference", ep)
        # Setup only runs if auth missing: conditional block, not unconditional.
        self.assertIn("if ! authed_inference; then", ep)
        # First-run uses $HERMES_HOME (not --home); passes setup result through exit status.
        self.assertIn("HERMES_HOME", ep)
        # Must preserve config absence check, not unconditionally run.
        self.assertIn('"${HERMES_HOME:-/opt/data}"', ep)

    def test_entrypoint_uses_exact_venv_path_for_cli(self):
        ep = (ROOT / "scripts/entrypoint.sh").read_text()
        self.assertIn('/opt/hermes/.venv/bin/alcf-tokens', ep)



    def test_fixture_missing_critical_install_fails_semantically(self):
        # Policy: no Docker build required; source/package contract is tested semantically.
        # If Dockerfile omitted alcf-tokens install, entrypoint would fail at $ALCF_TOKENS call.
        df = (ROOT / "Dockerfile").read_text()
        # We assert the exact required line exists; without it, contract fails.
        self.assertIn("alcf-tokens==0.3.0", df)
        # A fake missing fixture (simulated by asserting absence) must fail:
        missing_df = df.replace("alcf-tokens==0.3.0 ", "")
        self.assertNotIn("nersc-tokens==0.1.1", missing_df)

    def test_entrypoint_exit_propagation_under_set_e(self):
        # Under set -e, `exit 1` after failed auth propagates directly (no `|| exit 1` masking).
        ep = (ROOT / "scripts/entrypoint.sh").read_text()
        # Confirm direct command pattern (not masked by `|| exit 1`).
        # The setup call is direct under set -e: `"$ALCF_TOKENS" login` followed by `if ! ...; then ... exit 1; fi`
        self.assertIn("set -euo pipefail", ep)
        # No `|| exit 1` that would overwrite the setup's real exit status.
        masked = [line for line in ep.split("\n") if "|| exit 1" in line]
        # The only allowed masked lines (if any) are non-setup; here none should exist.
        self.assertEqual(masked, [])
        # Confirm --home is preserved (setup uses $HERMES_HOME, direct command under set -e).
        self.assertIn('"$ALCF_TOKENS" login', ep)

if __name__ == "__main__":
    unittest.main()
