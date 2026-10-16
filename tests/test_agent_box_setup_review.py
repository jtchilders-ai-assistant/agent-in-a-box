import io
import os
import stat
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from agent_box_config import ConfigError, atomic_yaml_write, load_registry
from agent_box_setup import (
    AMSC_TOKEN_FILE_ENV,
    LEGACY_CONFIG_RELATIVE_PATH,
    _is_safe_token_path,
    main,
    run_first_setup,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(ROOT, "config", "facilities.yaml")


class SetupReviewRegressionTests(unittest.TestCase):
    def setUp(self):
        self.registry, _ = load_registry(REGISTRY_PATH)

    def _write_product_config(self, home, facilities):
        path = os.path.join(home, LEGACY_CONFIG_RELATIVE_PATH)
        os.makedirs(os.path.dirname(path), mode=0o700)
        atomic_yaml_write(path, {"schema_version": 1, "facilities": facilities})
        return path

    def test_migration_preserves_real_persisted_facility_state(self):
        with tempfile.TemporaryDirectory() as home:
            self._write_product_config(
                home,
                {
                    "alcf": {"enabled": True, "iri": {"version": "v1"}},
                    "nersc": {"enabled": True, "iri": {"version": "v1"}},
                },
            )
            result = run_first_setup(
                home=home,
                registry=self.registry,
                input_stream=io.StringIO(""),
                output_stream=io.StringIO(),
                is_tty=True,
                environ={},
            )
            self.assertEqual(result.status, "migrated")
            self.assertTrue(result.resolved["nersc"]["enabled"])
            self.assertEqual(result.resolved["nersc"]["version"], "v1")

    def test_migration_writes_durable_alcf_config_when_only_legacy_tokens_exist(self):
        with tempfile.TemporaryDirectory() as home:
            token_path = os.path.join(home, ".globus", "app", "tokens.json")
            os.makedirs(os.path.dirname(token_path))
            open(token_path, "w", encoding="utf-8").close()
            result = run_first_setup(
                home=home,
                registry=self.registry,
                input_stream=io.StringIO(""),
                output_stream=io.StringIO(),
                is_tty=True,
                environ={},
            )
            self.assertEqual(result.status, "migrated")
            self.assertTrue(os.path.isfile(os.path.join(home, LEGACY_CONFIG_RELATIVE_PATH)))

    def test_environment_override_is_honored_during_migration(self):
        with tempfile.TemporaryDirectory() as home:
            self._write_product_config(
                home,
                {"alcf": {"enabled": True, "iri": {"version": "v1"}}},
            )
            result = run_first_setup(
                home=home,
                registry=self.registry,
                input_stream=io.StringIO(""),
                output_stream=io.StringIO(),
                is_tty=True,
                environ={"AGENT_BOX_FACILITIES": "nersc"},
            )
            self.assertFalse(result.resolved["alcf"]["enabled"])
            self.assertTrue(result.resolved["nersc"]["enabled"])

    def test_safe_token_path_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "token")
            link = os.path.join(directory, "link")
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("not-a-real-secret")
            os.chmod(target, 0o600)
            os.symlink(target, link)
            self.assertFalse(_is_safe_token_path(link))

    def test_plain_amsc_token_environment_is_never_honored(self):
        with tempfile.TemporaryDirectory() as home:
            with self.assertRaises(ConfigError) as raised:
                run_first_setup(
                    home=home,
                    registry=self.registry,
                    input_stream=io.StringIO(""),
                    output_stream=io.StringIO(),
                    is_tty=False,
                    environ={
                        "AGENT_BOX_FACILITIES": "esnet-east",
                        "AMSC_TOKEN": "must-not-be-consumed",
                    },
                )
            self.assertEqual(raised.exception.exit_code, 77)
            self.assertNotIn("must-not-be-consumed", str(raised.exception))

    def test_main_fails_closed_when_home_is_unset(self):
        old_home = os.environ.pop("HOME", None)
        old_registry = os.environ.get("AGENT_BOX_REGISTRY")
        os.environ["AGENT_BOX_REGISTRY"] = REGISTRY_PATH
        try:
            self.assertNotEqual(main([]), 0)
        finally:
            if old_home is not None:
                os.environ["HOME"] = old_home
            if old_registry is None:
                os.environ.pop("AGENT_BOX_REGISTRY", None)
            else:
                os.environ["AGENT_BOX_REGISTRY"] = old_registry

    def test_management_cli_config_show_and_validate_are_secret_free(self):
        from agent_box import main as agent_box_main

        with tempfile.TemporaryDirectory() as home:
            config_path = self._write_product_config(
                home,
                {"alcf": {"enabled": True, "iri": {"version": "v1"}}},
            )
            output = io.StringIO()
            self.assertEqual(
                agent_box_main(
                    ["config", "validate"],
                    home=home,
                    registry_path=REGISTRY_PATH,
                    output_stream=output,
                    error_stream=io.StringIO(),
                ),
                0,
            )
            self.assertIn("valid", output.getvalue().lower())
            output = io.StringIO()
            self.assertEqual(
                agent_box_main(
                    ["config", "show"],
                    home=home,
                    registry_path=REGISTRY_PATH,
                    output_stream=output,
                    error_stream=io.StringIO(),
                ),
                0,
            )
            text = output.getvalue().lower()
            self.assertIn("alcf", text)
            self.assertNotIn("token", text)
            self.assertNotIn("password", text)
            self.assertEqual(stat.S_IMODE(os.stat(config_path).st_mode), 0o600)

    def test_management_cli_configure_runs_stream_injected_wizard(self):
        from agent_box import main as agent_box_main

        with tempfile.TemporaryDirectory() as home:
            output = io.StringIO()
            exit_code = agent_box_main(
                ["configure"],
                home=home,
                registry_path=REGISTRY_PATH,
                input_stream=io.StringIO("1,2\n\ny\n"),
                output_stream=output,
                error_stream=io.StringIO(),
                is_tty=True,
                environ={},
            )
            self.assertEqual(exit_code, 0)
            config_path = os.path.join(home, LEGACY_CONFIG_RELATIVE_PATH)
            self.assertTrue(os.path.isfile(config_path))
            with open(config_path, encoding="utf-8") as handle:
                serialized = handle.read().lower()
            self.assertIn("nersc", serialized)
            self.assertNotIn("token", serialized)
            self.assertNotIn("password", serialized)


if __name__ == "__main__":
    unittest.main()
