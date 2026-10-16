import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from agent_box_config import ConfigError, load_registry
from agent_box_setup import (
    LEGACY_CONFIG_RELATIVE_PATH,
    run_first_setup,
    select_facilities,
    select_versions,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(ROOT, "config", "facilities.yaml")


class FirstRunWizardTests(unittest.TestCase):
    def setUp(self):
        self.registry, _ = load_registry(REGISTRY_PATH)

    def test_enter_selects_alcf_only(self):
        output = io.StringIO()
        selected = select_facilities(self.registry, io.StringIO("\n"), output)
        self.assertEqual(selected, ["alcf"])
        self.assertIn("[x] ALCF", output.getvalue())

    def test_comma_selection_and_invalid_retry(self):
        output = io.StringIO()
        selected = select_facilities(
            self.registry,
            io.StringIO("9\n1,2,3\n"),
            output,
        )
        self.assertEqual(selected, ["alcf", "nersc", "esnet-east"])
        self.assertIn("Invalid selection", output.getvalue())

    def test_recommended_versions_are_used_by_default(self):
        output = io.StringIO()
        versions = select_versions(
            self.registry,
            ["alcf", "nersc", "esnet-east"],
            io.StringIO("\n"),
            output,
        )
        self.assertEqual(versions, {"alcf": "v1", "nersc": "v2", "esnet-east": "v2"})

    def test_advanced_versions_validate_and_retry(self):
        output = io.StringIO()
        versions = select_versions(
            self.registry,
            ["nersc"],
            io.StringIO("n\nv9\nv1\n"),
            output,
        )
        self.assertEqual(versions, {"nersc": "v1"})
        self.assertIn("Unsupported version", output.getvalue())

    def test_cancel_does_not_write_configuration(self):
        with tempfile.TemporaryDirectory() as home:
            output = io.StringIO()
            result = run_first_setup(
                home=home,
                registry=self.registry,
                input_stream=io.StringIO("\n\n\nn\n"),
                output_stream=output,
                is_tty=True,
                environ={},
            )
            self.assertEqual(result.status, "cancelled")
            self.assertFalse(os.path.exists(os.path.join(home, LEGACY_CONFIG_RELATIVE_PATH)))

    def test_confirmed_setup_writes_nonsecret_schema(self):
        with tempfile.TemporaryDirectory() as home:
            result = run_first_setup(
                home=home,
                registry=self.registry,
                input_stream=io.StringIO("1,2\n\ny\n"),
                output_stream=io.StringIO(),
                is_tty=True,
                environ={},
            )
            self.assertEqual(result.status, "configured")
            self.assertEqual(result.configuration["schema_version"], 1)
            self.assertTrue(result.configuration["facilities"]["alcf"]["enabled"])
            self.assertTrue(result.configuration["facilities"]["nersc"]["enabled"])
            with open(
                os.path.join(home, LEGACY_CONFIG_RELATIVE_PATH), encoding="utf-8"
            ) as handle:
                serialized = handle.read()
            self.assertNotIn("token", serialized.lower())
            self.assertNotIn("password", serialized.lower())

    def test_no_tty_defaults_to_alcf_without_hanging(self):
        with tempfile.TemporaryDirectory() as home:
            result = run_first_setup(
                home=home,
                registry=self.registry,
                input_stream=io.StringIO(""),
                output_stream=io.StringIO(),
                is_tty=False,
                environ={},
            )
            self.assertEqual(result.status, "configured")
            self.assertTrue(result.resolved["alcf"]["enabled"])
            self.assertFalse(result.resolved["nersc"]["enabled"])

    def test_explicit_noninteractive_selection_is_honored(self):
        with tempfile.TemporaryDirectory() as home:
            result = run_first_setup(
                home=home,
                registry=self.registry,
                input_stream=io.StringIO(""),
                output_stream=io.StringIO(),
                is_tty=False,
                environ={"AGENT_BOX_FACILITIES": "alcf,nersc"},
            )
            self.assertTrue(result.resolved["alcf"]["enabled"])
            self.assertTrue(result.resolved["nersc"]["enabled"])

    def test_explicit_interactive_selection_skips_wizard_and_is_honored(self):
        with tempfile.TemporaryDirectory() as home:
            result = run_first_setup(
                home=home,
                registry=self.registry,
                input_stream=io.StringIO("must not be read"),
                output_stream=io.StringIO(),
                is_tty=True,
                environ={"AGENT_BOX_FACILITIES": "nersc"},
            )
            self.assertFalse(result.resolved["alcf"]["enabled"])
            self.assertTrue(result.resolved["nersc"]["enabled"])
            self.assertFalse(result.configuration["facilities"]["alcf"]["enabled"])
            self.assertTrue(result.configuration["facilities"]["nersc"]["enabled"])

    def test_noninteractive_esnet_without_safe_token_fails_77(self):
        with tempfile.TemporaryDirectory() as home:
            with self.assertRaises(ConfigError) as raised:
                run_first_setup(
                    home=home,
                    registry=self.registry,
                    input_stream=io.StringIO(""),
                    output_stream=io.StringIO(),
                    is_tty=False,
                    environ={"AGENT_BOX_FACILITIES": "esnet-east"},
                )
            self.assertEqual(raised.exception.exit_code, 77)

    def test_existing_alcf_volume_migrates_without_prompt(self):
        with tempfile.TemporaryDirectory() as home:
            globus_dir = os.path.join(home, ".globus", "app")
            os.makedirs(globus_dir)
            open(os.path.join(globus_dir, "tokens.json"), "w", encoding="utf-8").close()
            unreadable_input = io.StringIO("")
            result = run_first_setup(
                home=home,
                registry=self.registry,
                input_stream=unreadable_input,
                output_stream=io.StringIO(),
                is_tty=True,
                environ={},
            )
            self.assertEqual(result.status, "migrated")
            self.assertTrue(result.resolved["alcf"]["enabled"])


if __name__ == "__main__":
    unittest.main()


class RegistryPathRegressionTests(unittest.TestCase):
    """Registry default resolves deterministically via temp dirs, not test-file path."""
    def test_packaged_sibling_preferred(self):
        from agent_box_setup import _default_registry_path
        with tempfile.TemporaryDirectory() as td:
            script_dir = os.path.join(td, "bin")
            config_dir = os.path.join(td, "config")
            os.makedirs(script_dir)
            os.makedirs(config_dir)
            open(os.path.join(config_dir, "facilities.yaml"), "w").close()
            script_path = os.path.join(script_dir, "agent_box_setup.py")
            result = _default_registry_path(script_path)
            self.assertEqual(result, os.path.join(config_dir, "facilities.yaml"))

    def test_repo_parent_fallback(self):
        from agent_box_setup import _default_registry_path
        with tempfile.TemporaryDirectory() as td:
            repo_config = os.path.join(td, "config")
            scripts_dir = os.path.join(td, "scripts")
            os.makedirs(repo_config)
            os.makedirs(scripts_dir)
            open(os.path.join(repo_config, "facilities.yaml"), "w").close()
            script_path = os.path.join(scripts_dir, "agent_box_setup.py")
            result = _default_registry_path(script_path)
            self.assertEqual(result, os.path.join(repo_config, "facilities.yaml"))

    def test_missing_returns_packaged_target(self):
        from agent_box_setup import _default_registry_path
        with tempfile.TemporaryDirectory() as td:
            script_dir = os.path.join(td, "scripts")
            os.makedirs(script_dir)
            script_path = os.path.join(script_dir, "agent_box_setup.py")
            result = _default_registry_path(script_path)
            expected = os.path.join(script_dir, "config", "facilities.yaml")
            self.assertEqual(result, expected)
