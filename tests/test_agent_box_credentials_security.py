import io
import os
import stat
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from agent_box_config import ConfigError, load_registry
from agent_box_credentials import (
    CredentialError,
    amsc_token_path,
    read_amsc_token,
    remove_amsc_token,
    write_amsc_token,
)
from agent_box_setup import LEGACY_CONFIG_RELATIVE_PATH, run_first_setup
import agent_box


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(ROOT, "config", "facilities.yaml")
SECRET = "security-$-candidate-never-leak"


class AmscSecurityRegressionTests(unittest.TestCase):
    def setUp(self):
        self.registry, _ = load_registry(REGISTRY_PATH)

    def _assert_parent_symlink_rejected(self, component):
        with tempfile.TemporaryDirectory() as root:
            home = os.path.join(root, "home")
            outside = os.path.join(root, "outside")
            os.mkdir(home, 0o700)
            os.mkdir(outside, 0o700)
            parts = ["agent-box", "credentials", "esnet"]
            current = home
            for part in parts:
                candidate = os.path.join(current, part)
                if part == component:
                    os.symlink(outside, candidate)
                    break
                os.mkdir(candidate, 0o700)
                current = candidate
            with self.assertRaises(CredentialError):
                write_amsc_token(home, SECRET)
            self.assertFalse(os.path.exists(os.path.join(outside, "amsc-token")))

    def test_symlinked_parent_components_are_rejected(self):
        for component in ("agent-box", "credentials", "esnet"):
            with self.subTest(component=component):
                self._assert_parent_symlink_rejected(component)

    def test_symlinked_home_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            real_home = os.path.join(root, "real")
            link_home = os.path.join(root, "link")
            os.mkdir(real_home, 0o700)
            os.symlink(real_home, link_home)
            with self.assertRaises(CredentialError):
                write_amsc_token(link_home, SECRET)
            self.assertFalse(os.path.exists(os.path.join(real_home, "agent-box")))

    def test_non_directory_or_unsafe_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as home:
            agent_box = os.path.join(home, "agent-box")
            with open(agent_box, "w", encoding="utf-8") as handle:
                handle.write("not-dir")
            with self.assertRaises(CredentialError):
                write_amsc_token(home, SECRET)

        with tempfile.TemporaryDirectory() as home:
            agent_box = os.path.join(home, "agent-box")
            os.mkdir(agent_box, 0o755)
            with self.assertRaises(CredentialError):
                write_amsc_token(home, SECRET)

    def test_existing_unsafe_leaf_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as home:
            path = amsc_token_path(home)
            os.makedirs(os.path.dirname(path), mode=0o700)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("prior")
            os.chmod(path, 0o644)
            with self.assertRaises(CredentialError):
                write_amsc_token(home, SECRET)
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "prior")

    def test_read_validates_open_descriptor_not_only_path(self):
        with tempfile.TemporaryDirectory() as home:
            path = write_amsc_token(home, "trusted")
            attacker = os.path.join(home, "attacker")
            with open(attacker, "w", encoding="utf-8") as handle:
                handle.write("ATTACKER_CONTROLLED")
            os.chmod(attacker, 0o600)
            real_open = os.open
            token_opened = False

            def swapped_open(candidate, flags, *args, **kwargs):
                nonlocal token_opened
                if candidate == path:
                    token_opened = True
                    return real_open(attacker, flags, *args, **kwargs)
                return real_open(candidate, flags, *args, **kwargs)

            with mock.patch("agent_box_credentials.os.open", side_effect=swapped_open):
                with self.assertRaises(CredentialError):
                    read_amsc_token(home)
            self.assertTrue(token_opened)

    def test_existing_persisted_token_is_validated_for_each_selected_profile(self):
        with tempfile.TemporaryDirectory() as home:
            write_amsc_token(home, "existing-safe")
            calls = []

            def validator(facility, version, token):
                calls.append((facility, version, token))
                return {"valid": True}

            result = run_first_setup(
                home=home,
                registry=self.registry,
                input_stream=io.StringIO("3,4\n\ny\n"),
                output_stream=io.StringIO(),
                is_tty=True,
                environ={},
                hidden_reader=lambda _: self.fail("must not prompt"),
                amsc_validator=validator,
            )
            self.assertEqual(result.status, "configured")
            self.assertEqual(
                calls,
                [
                    ("esnet-east", "v2", "existing-safe"),
                    ("esnet-west", "v2", "existing-safe"),
                ],
            )

    def test_external_safe_token_is_validated_for_each_selected_profile(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as external:
            token_file = os.path.join(external, "token")
            with open(token_file, "w", encoding="utf-8") as handle:
                handle.write("external-safe")
            os.chmod(token_file, 0o600)
            calls = []

            run_first_setup(
                home=home,
                registry=self.registry,
                input_stream=io.StringIO("3\n\ny\n"),
                output_stream=io.StringIO(),
                is_tty=True,
                environ={"ESNET_AMSC_TOKEN_FILE": token_file},
                amsc_validator=lambda facility, version, token: (
                    calls.append((facility, version, token)) or {"valid": True}
                ),
            )
            self.assertEqual(calls, [("esnet-east", "v2", "external-safe")])

    def test_existing_token_partial_validation_failure_preserves_token_and_config(self):
        with tempfile.TemporaryDirectory() as home:
            write_amsc_token(home, "existing-safe")
            config_path = os.path.join(home, LEGACY_CONFIG_RELATIVE_PATH)
            self.assertFalse(os.path.exists(config_path))
            calls = []

            def validator(facility, version, token):
                calls.append(facility)
                return {"valid": facility == "esnet-east"}

            with self.assertRaises(ConfigError) as raised:
                run_first_setup(
                    home=home,
                    registry=self.registry,
                    input_stream=io.StringIO("3,4\n\ny\n"),
                    output_stream=io.StringIO(),
                    is_tty=True,
                    environ={},
                    amsc_validator=validator,
                )
            self.assertEqual(raised.exception.exit_code, 77)
            self.assertEqual(calls, ["esnet-east", "esnet-west"])
            self.assertEqual(read_amsc_token(home), "existing-safe")
            self.assertFalse(os.path.exists(config_path))

    def test_hidden_reader_exception_is_redacted_without_chaining(self):
        with tempfile.TemporaryDirectory() as home:
            try:
                run_first_setup(
                    home=home,
                    registry=self.registry,
                    input_stream=io.StringIO("3\n\ny\n"),
                    output_stream=io.StringIO(),
                    is_tty=True,
                    environ={},
                    hidden_reader=lambda _: (_ for _ in ()).throw(RuntimeError(SECRET)),
                    amsc_validator=lambda *args: {"valid": True},
                )
            except ConfigError as exc:
                self.assertEqual(exc.exit_code, 77)
                self.assertNotIn(SECRET, str(exc))
                self.assertIsNone(exc.__cause__)
                self.assertTrue(exc.__suppress_context__)
            else:
                self.fail("expected ConfigError")

    def test_remove_rejects_unsafe_leaf(self):
        with tempfile.TemporaryDirectory() as home:
            path = write_amsc_token(home, SECRET)
            os.chmod(path, 0o644)
            with self.assertRaises(CredentialError):
                remove_amsc_token(home)
            self.assertTrue(os.path.exists(path))

    def test_management_cli_exposes_credential_reconfiguration(self):
        with tempfile.TemporaryDirectory() as home:
            write_amsc_token(home, "old-token")
            output = io.StringIO()
            errors = io.StringIO()
            rc = agent_box.main(
                ["credentials", "esnet", "configure"],
                home=home,
                registry_path=REGISTRY_PATH,
                input_stream=io.StringIO("keep\n"),
                output_stream=output,
                error_stream=errors,
                is_tty=True,
                environ={},
            )
            self.assertEqual(rc, 0)
            self.assertEqual(read_amsc_token(home), "old-token")
            self.assertNotIn("old-token", output.getvalue())
            self.assertEqual(errors.getvalue(), "")

    def test_management_cli_reconfiguration_requires_tty(self):
        with tempfile.TemporaryDirectory() as home:
            rc = agent_box.main(
                ["credentials", "esnet", "configure"],
                home=home,
                registry_path=REGISTRY_PATH,
                input_stream=io.StringIO(""),
                output_stream=io.StringIO(),
                error_stream=io.StringIO(),
                is_tty=False,
                environ={},
            )
            self.assertNotEqual(rc, 0)

    def test_credential_operations_do_not_leak_file_descriptors(self):
        fd_root = "/dev/fd" if os.path.isdir("/dev/fd") else "/proc/self/fd"
        if not os.path.isdir(fd_root):
            self.skipTest("no process fd directory")
        with tempfile.TemporaryDirectory() as home:
            baseline = len(os.listdir(fd_root))
            for index in range(40):
                write_amsc_token(home, "token-%d" % index)
                self.assertEqual(read_amsc_token(home), "token-%d" % index)
                from agent_box_credentials import token_status
                self.assertEqual(token_status(home), "present")
                self.assertTrue(remove_amsc_token(home))
                self.assertEqual(token_status(home), "absent")
            observed = len(os.listdir(fd_root))
            self.assertLessEqual(observed, baseline + 2)

    def test_remove_does_not_unlink_replacement_leaf(self):
        with tempfile.TemporaryDirectory() as home:
            path = write_amsc_token(home, "original-token")
            replacement = os.path.join(os.path.dirname(path), "replacement")
            with open(replacement, "w", encoding="utf-8") as handle:
                handle.write("replacement-token")
            os.chmod(replacement, 0o600)
            real_unlink = os.unlink
            swapped = False

            def swap_before_unlink(candidate, *args, **kwargs):
                nonlocal swapped
                if candidate == "amsc-token" and kwargs.get("dir_fd") is not None:
                    swapped = True
                    os.replace(replacement, path)
                return real_unlink(candidate, *args, **kwargs)

            with mock.patch("agent_box_credentials.os.unlink", side_effect=swap_before_unlink):
                try:
                    result = remove_amsc_token(home)
                except CredentialError:
                    result = False
            self.assertTrue(swapped)
            self.assertFalse(result)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "replacement-token")


if __name__ == "__main__":
    unittest.main()
