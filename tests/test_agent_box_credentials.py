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
    capture_amsc_token,
    read_amsc_token,
    remove_amsc_token,
    token_status,
    write_amsc_token,
)
from agent_box_setup import run_first_setup


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(ROOT, "config", "facilities.yaml")
SECRET = "amsc-test-$-never-print"


class AmscCredentialTests(unittest.TestCase):
    def setUp(self):
        self.registry, _ = load_registry(REGISTRY_PATH)

    def test_write_creates_private_directory_and_file(self):
        with tempfile.TemporaryDirectory() as home:
            path = write_amsc_token(home, SECRET)
            self.assertEqual(path, amsc_token_path(home))
            self.assertEqual(stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            self.assertEqual(read_amsc_token(home), SECRET)

    def test_write_rejects_empty_and_preserves_non_newline_content(self):
        with tempfile.TemporaryDirectory() as home:
            for empty in ("", "\n", "\r\n"):
                with self.subTest(empty=repr(empty)):
                    with self.assertRaises(CredentialError):
                        write_amsc_token(home, empty)
            value = "  content with spaces  \n"
            write_amsc_token(home, value)
            self.assertEqual(read_amsc_token(home), "  content with spaces  ")

    def test_read_rejects_symlink_nonregular_wrong_owner_and_unsafe_mode(self):
        with tempfile.TemporaryDirectory() as home:
            path = write_amsc_token(home, SECRET)
            os.chmod(path, 0o644)
            with self.assertRaises(CredentialError):
                read_amsc_token(home)

            os.unlink(path)
            os.mkfifo(path, 0o600)
            with self.assertRaises(CredentialError):
                read_amsc_token(home)
            os.unlink(path)

            target = os.path.join(home, "target")
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(SECRET)
            os.chmod(target, 0o600)
            os.symlink(target, path)
            with self.assertRaises(CredentialError):
                read_amsc_token(home)
            os.unlink(path)

            write_amsc_token(home, SECRET)
            real_lstat = os.lstat

            def wrong_owner(candidate):
                result = real_lstat(candidate)
                values = list(result)
                values[4] = result.st_uid + 1
                return os.stat_result(values)

            with mock.patch("agent_box_credentials.os.lstat", side_effect=wrong_owner):
                with self.assertRaises(CredentialError):
                    read_amsc_token(home)

    def test_failed_atomic_replace_preserves_existing_token_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as home:
            path = write_amsc_token(home, "old-token")
            before = set(os.listdir(os.path.dirname(path)))
            with mock.patch(
                "agent_box_credentials.os.replace", side_effect=OSError("injected")
            ):
                with self.assertRaises(OSError):
                    write_amsc_token(home, "new-token")
            self.assertEqual(read_amsc_token(home), "old-token")
            self.assertEqual(set(os.listdir(os.path.dirname(path))), before)

    def test_capture_uses_hidden_reader_and_never_prints_secret(self):
        with tempfile.TemporaryDirectory() as home:
            output = io.StringIO()
            prompts = []

            def hidden_reader(prompt):
                prompts.append(prompt)
                return SECRET

            path = capture_amsc_token(home, hidden_reader=hidden_reader, output_stream=output)
            self.assertEqual(path, amsc_token_path(home))
            self.assertEqual(len(prompts), 1)
            self.assertNotIn(SECRET, output.getvalue())
            self.assertNotIn(SECRET, prompts[0])

    def test_status_keep_replace_remove_never_displays_token(self):
        with tempfile.TemporaryDirectory() as home:
            self.assertEqual(token_status(home), "absent")
            write_amsc_token(home, SECRET)
            self.assertEqual(token_status(home), "present")
            self.assertTrue(remove_amsc_token(home))
            self.assertEqual(token_status(home), "absent")
            self.assertFalse(remove_amsc_token(home))

    def test_reconfiguration_wizard_keep_replace_remove(self):
        from agent_box_setup import reconfigure_amsc_credential_interactive

        with tempfile.TemporaryDirectory() as home:
            write_amsc_token(home, "old-token")
            output = io.StringIO()
            status = reconfigure_amsc_credential_interactive(
                home,
                input_stream=io.StringIO("keep\n"),
                output_stream=output,
                hidden_reader=lambda _: self.fail("keep must not prompt"),
            )
            self.assertEqual(status, "present")
            self.assertEqual(read_amsc_token(home), "old-token")
            self.assertNotIn("old-token", output.getvalue())

            output = io.StringIO()
            status = reconfigure_amsc_credential_interactive(
                home,
                input_stream=io.StringIO("replace\n"),
                output_stream=output,
                hidden_reader=lambda _: "new-token",
            )
            self.assertEqual(status, "present")
            self.assertEqual(read_amsc_token(home), "new-token")
            self.assertNotIn("new-token", output.getvalue())

            status = reconfigure_amsc_credential_interactive(
                home,
                input_stream=io.StringIO("remove\ny\n"),
                output_stream=io.StringIO(),
                hidden_reader=lambda _: self.fail("remove must not prompt for token"),
            )
            self.assertEqual(status, "absent")

    def test_reconfiguration_remove_requires_confirmation(self):
        from agent_box_setup import reconfigure_amsc_credential_interactive

        with tempfile.TemporaryDirectory() as home:
            write_amsc_token(home, SECRET)
            status = reconfigure_amsc_credential_interactive(
                home,
                input_stream=io.StringIO("remove\nn\n"),
                output_stream=io.StringIO(),
            )
            self.assertEqual(status, "present")
            self.assertEqual(read_amsc_token(home), SECRET)

    def test_esnet_selection_prompts_once_for_east_and_west(self):
        with tempfile.TemporaryDirectory() as home:
            output = io.StringIO()
            calls = []

            def hidden_reader(prompt):
                calls.append(prompt)
                return SECRET

            def validator(facility, version, token):
                self.assertEqual(token, SECRET)
                return {"facility": facility, "version": version, "valid": True}

            result = run_first_setup(
                home=home,
                registry=self.registry,
                input_stream=io.StringIO("3,4\n\ny\n"),
                output_stream=output,
                is_tty=True,
                environ={},
                hidden_reader=hidden_reader,
                amsc_validator=validator,
            )
            self.assertEqual(result.status, "configured")
            self.assertEqual(len(calls), 1)
            self.assertNotIn(SECRET, output.getvalue())
            self.assertEqual(read_amsc_token(home), SECRET)

    def test_failed_validation_does_not_persist_candidate(self):
        with tempfile.TemporaryDirectory() as home:
            with self.assertRaises(ConfigError) as raised:
                run_first_setup(
                    home=home,
                    registry=self.registry,
                    input_stream=io.StringIO("3\n\ny\n"),
                    output_stream=io.StringIO(),
                    is_tty=True,
                    environ={},
                    hidden_reader=lambda _: SECRET,
                    amsc_validator=lambda facility, version, token: {
                        "facility": facility,
                        "version": version,
                        "valid": False,
                    },
                )
            self.assertEqual(raised.exception.exit_code, 77)
            self.assertEqual(token_status(home), "absent")

    def test_each_selected_site_is_validated_independently(self):
        with tempfile.TemporaryDirectory() as home:
            calls = []

            def validator(facility, version, token):
                calls.append((facility, version))
                return {"valid": True}

            run_first_setup(
                home=home,
                registry=self.registry,
                input_stream=io.StringIO("3,4\n\ny\n"),
                output_stream=io.StringIO(),
                is_tty=True,
                environ={},
                hidden_reader=lambda _: SECRET,
                amsc_validator=validator,
            )
            self.assertEqual(calls, [("esnet-east", "v2"), ("esnet-west", "v2")])

    def test_candidate_never_appears_in_errors_or_configuration(self):
        with tempfile.TemporaryDirectory() as home:
            output = io.StringIO()
            try:
                run_first_setup(
                    home=home,
                    registry=self.registry,
                    input_stream=io.StringIO("3\n\ny\n"),
                    output_stream=output,
                    is_tty=True,
                    environ={},
                    hidden_reader=lambda _: SECRET,
                    amsc_validator=lambda *args: (_ for _ in ()).throw(
                        RuntimeError("validator failed")
                    ),
                )
            except Exception as exc:
                self.assertNotIn(SECRET, str(exc))
            self.assertNotIn(SECRET, output.getvalue())
            config_path = os.path.join(home, "agent-box", "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, encoding="utf-8") as handle:
                    self.assertNotIn(SECRET, handle.read())


if __name__ == "__main__":
    unittest.main()
