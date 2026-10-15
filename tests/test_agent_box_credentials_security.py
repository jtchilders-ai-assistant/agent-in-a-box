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

    def test_remove_reports_failure_truthfully_in_reconfigure_ui(self):
        from agent_box_setup import reconfigure_amsc_credential_interactive

        with tempfile.TemporaryDirectory() as home:
            write_amsc_token(home, "token")
            output = io.StringIO()
            with mock.patch("agent_box_setup.remove_amsc_token", return_value=False):
                status = reconfigure_amsc_credential_interactive(
                    home,
                    input_stream=io.StringIO("remove\ny\n"),
                    output_stream=output,
                )
            self.assertIn("was not removed", output.getvalue())
            self.assertNotIn("token removed.", output.getvalue())
            self.assertEqual(status, "present")

    def test_remove_never_overwrites_preexisting_tombstone_destination(self):
        import agent_box_credentials as credentials

        with tempfile.TemporaryDirectory() as home:
            token_path = write_amsc_token(home, "verified-token")
            directory = os.path.dirname(token_path)
            suffix = "fixed-collision"
            tombstone = os.path.join(directory, ".amsc-token.%s" % suffix)
            with open(tombstone, "w", encoding="utf-8") as handle:
                handle.write("unverified-destination")
            os.chmod(tombstone, 0o600)

            with mock.patch.object(credentials, "_random_tombstone_suffix", return_value=suffix):
                try:
                    result = remove_amsc_token(home)
                except CredentialError:
                    result = False

            self.assertFalse(result)
            with open(tombstone, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "unverified-destination")
            with open(token_path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "verified-token")

    def test_remove_never_restores_over_reoccupied_public_path_on_fsync_failure(self):
        import agent_box_credentials as credentials

        with tempfile.TemporaryDirectory() as home:
            token_path = write_amsc_token(home, "verified-token")
            real_fsync = os.fsync
            state = {"fsyncs": 0, "planted": False}

            def fail_first_fsync(fd):
                state["fsyncs"] += 1
                if state["fsyncs"] == 1:
                    raise OSError("injected token fsync failure")
                return real_fsync(fd)

            real_rename = os.rename

            def plant_before_restore(source, destination, *args, **kwargs):
                if (
                    isinstance(source, str)
                    and source.startswith(".amsc-token.")
                    and destination == "amsc-token"
                    and not state["planted"]
                ):
                    state["planted"] = True
                    with open(token_path, "w", encoding="utf-8") as handle:
                        handle.write("unverified-public")
                    os.chmod(token_path, 0o600)
                return real_rename(source, destination, *args, **kwargs)

            with mock.patch.object(credentials.os, "fsync", side_effect=fail_first_fsync), mock.patch.object(
                credentials.os, "rename", side_effect=plant_before_restore
            ):
                with self.assertRaises(OSError):
                    remove_amsc_token(home)

            # A secure implementation never attempts restoration onto the
            # mutable public path. If a legacy restore seam was reached, the
            # planted object must still survive unchanged.
            if state["planted"]:
                with open(token_path, encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), "unverified-public")

    def test_write_closes_temp_fd_when_fchmod_fails(self):
        fd_root = "/dev/fd" if os.path.isdir("/dev/fd") else "/proc/self/fd"
        if not os.path.isdir(fd_root):
            self.skipTest("no process fd directory")
        with tempfile.TemporaryDirectory() as home:
            baseline = len(os.listdir(fd_root))
            with mock.patch(
                "agent_box_credentials.os.fchmod",
                side_effect=OSError("injected fchmod failure"),
            ):
                with self.assertRaises(OSError):
                    write_amsc_token(home, "token")
            observed = len(os.listdir(fd_root))
            self.assertLessEqual(observed, baseline + 1)

    def test_remove_fails_closed_when_tombstone_name_raced_after_generation(self):
        import agent_box_credentials as credentials

        with tempfile.TemporaryDirectory() as home:
            token_path = write_amsc_token(home, "verified-token")
            directory = os.path.dirname(token_path)
            suffix = "raced-into"
            racer_path = os.path.join(directory, ".amsc-token.%s" % suffix)

            def plant_racer_after_suffix_generation():
                # Deterministically simulate a same-UID racer that
                # creates an object under the tombstone name after the
                # suffix is generated but before the exclusive claim.
                with open(racer_path, "w", encoding="utf-8") as handle:
                    handle.write("racer-object")
                os.chmod(racer_path, 0o600)
                return suffix

            with mock.patch.object(
                credentials,
                "_random_tombstone_suffix",
                side_effect=plant_racer_after_suffix_generation,
            ) as suffix_calls:
                with self.assertRaises(CredentialError):
                    remove_amsc_token(home)
            # The exclusive claim was genuinely attempted under the
            # generated name (name was generated, then raced).
            suffix_calls.assert_called_once_with()
            with open(racer_path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "racer-object")
            with open(token_path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "verified-token")

    def test_remove_mismatch_restore_never_links_over_public_reoccupant(self):
        import agent_box_credentials as credentials

        with tempfile.TemporaryDirectory() as home:
            token_path = write_amsc_token(home, "verified-token")
            directory = os.path.dirname(token_path)
            swapped = {"done": False}
            real_rename = os.rename

            def race_before_move(source, destination, *args, **kwargs):
                # Deterministic namespace race: after the implementation
                # has verified the leaf, a same-UID racer swaps a
                # foreign object under the public name so the rename
                # moves the wrong inode, and once the rename has
                # completed the racer reoccupies the public path with
                # an unrelated object the restore must never displace.
                if source == "amsc-token" and kwargs.get("src_dir_fd") is not None:
                    original = os.path.join(directory, "original-inode")
                    os.rename(token_path, original)
                    foreign = os.path.join(directory, "foreign-object")
                    with open(foreign, "w", encoding="utf-8") as handle:
                        handle.write("foreign-token")
                    os.chmod(foreign, 0o600)
                    os.rename(foreign, token_path)
                    result = real_rename(source, destination, *args, **kwargs)
                    with open(token_path + ".reoccupant", "w",
                              encoding="utf-8") as handle:
                        handle.write("unverified-reoccupant")
                    os.chmod(token_path + ".reoccupant", 0o600)
                    os.rename(token_path + ".reoccupant", token_path)
                    swapped["done"] = True
                    return result
                return real_rename(source, destination, *args, **kwargs)

            with mock.patch.object(
                credentials.os, "rename", side_effect=race_before_move
            ):
                with self.assertRaises(CredentialError):
                    remove_amsc_token(home)
            self.assertTrue(swapped["done"])
            # The no-replace hard-link restore must fail EEXIST and
            # must never displace the public reoccupant.
            with open(token_path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "unverified-reoccupant")
            # The moved foreign object is never destroyed: it survives
            # intact under the retained tombstone name and nowhere
            # else.
            directory = os.path.dirname(token_path)
            survivors = [
                name for name in os.listdir(directory)
                if name.startswith(".amsc-token.")
            ]
            self.assertEqual(len(survivors), 1)
            with open(os.path.join(directory, survivors[0]),
                      encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "foreign-token")
            # The implementation never destroys any object: the
            # original credential inode also survives untouched.
            with open(os.path.join(directory, "original-inode"),
                      encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "verified-token")

    def test_remove_ftruncate_failure_never_restores_over_public_path(self):
        with tempfile.TemporaryDirectory() as home:
            token_path = write_amsc_token(home, "verified-token")
            directory = os.path.dirname(token_path)
            with mock.patch(
                "agent_box_credentials.os.ftruncate",
                side_effect=OSError("injected ftruncate failure"),
            ):
                with self.assertRaises(OSError):
                    remove_amsc_token(home)
            # Destruction had already been attempted through the
            # verified descriptor: no restore of any kind is attempted
            # onto the public path afterwards.
            self.assertFalse(os.path.lexists(token_path))
            survivors = [
                name for name in os.listdir(directory)
                if name.startswith(".amsc-token.")
            ]
            self.assertTrue(survivors)

    def test_remove_token_fsync_failure_never_restores_over_public_path(self):
        with tempfile.TemporaryDirectory() as home:
            token_path = write_amsc_token(home, "verified-token")
            directory = os.path.dirname(token_path)
            real_fsync = os.fsync
            state = {"fsyncs": 0}

            def fail_first_fsync(fd):
                state["fsyncs"] += 1
                if state["fsyncs"] == 1:
                    raise OSError("injected token fsync failure")
                return real_fsync(fd)

            with mock.patch("agent_box_credentials.os.fsync", side_effect=fail_first_fsync):
                with self.assertRaises(OSError):
                    remove_amsc_token(home)
            self.assertEqual(state["fsyncs"], 1)
            # No rename-back onto the public path is attempted after
            # destruction began: the public name stays absent.
            self.assertFalse(os.path.lexists(token_path))
            survivors = [
                name for name in os.listdir(directory)
                if name.startswith(".amsc-token.")
            ]
            self.assertTrue(survivors)

    def test_remove_dir_fsync_failure_never_restores_over_public_path(self):
        with tempfile.TemporaryDirectory() as home:
            token_path = write_amsc_token(home, "verified-token")
            directory = os.path.dirname(token_path)
            real_fsync = os.fsync
            state = {"token_fsynced": False}

            def fail_dir_fsync(fd):
                if state["token_fsynced"]:
                    raise OSError("injected directory fsync failure")
                state["token_fsynced"] = True
                return real_fsync(fd)

            with mock.patch("agent_box_credentials.os.fsync", side_effect=fail_dir_fsync):
                with self.assertRaises(OSError):
                    remove_amsc_token(home)
            self.assertTrue(state["token_fsynced"])
            # The credential inode was destroyed; the directory entry
            # update may not be durable, but no restore is attempted
            # onto the public path.
            self.assertFalse(os.path.lexists(token_path))
            survivors = [
                name for name in os.listdir(directory)
                if name.startswith(".amsc-token.")
            ]
            self.assertTrue(survivors)

    def test_removal_procedure_uses_no_unlink_or_restore_rename(self):
        # Structural guard for the Aegis IMP-01/IMP-02 invariants: the
        # removal procedure must contain no unlink call at all, and no
        # rename call that targets the mutable public leaf name as its
        # destination (an ordinary rename could clobber a reoccupant).
        import ast
        import inspect

        import agent_box_credentials as credentials

        tree = ast.parse(inspect.getsource(credentials.remove_amsc_token))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)):
                continue
            fname = getattr(node.func, "value", None)
            if not (isinstance(fname, ast.Name) and fname.id == "os"):
                continue
            self.assertNotEqual(
                node.func.attr, "unlink",
                "remove_amsc_token must never call os.unlink "
                "(Aegis IMP-01)")
            if node.func.attr == "rename":
                self.assertGreaterEqual(len(node.args), 2)
                destination = ast.dump(node.args[1])
                self.assertNotIn(
                    "leaf_name", destination,
                    "remove_amsc_token must never rename onto the "
                    "public leaf name (Aegis IMP-02), line %d"
                    % node.lineno)
                for keyword in node.keywords:
                    if keyword.arg == "dst_dir_fd":
                        self.assertNotIn(
                            "leaf_name", ast.dump(keyword.value))

    def test_remove_never_unlinks_a_post_verification_substitute(self):
        with tempfile.TemporaryDirectory() as home:
            token_path = write_amsc_token(home, "original-token")
            directory = os.path.dirname(token_path)
            replacement = os.path.join(directory, "replacement-after-check")
            with open(replacement, "w", encoding="utf-8") as handle:
                handle.write("replacement-token")
            os.chmod(replacement, 0o600)
            real_unlink = os.unlink
            swapped = False

            def substitute_before_pending_unlink(candidate, *args, **kwargs):
                nonlocal swapped
                if (
                    isinstance(candidate, str)
                    and candidate.startswith(".amsc-token.pending.")
                    and kwargs.get("dir_fd") is not None
                ):
                    swapped = True
                    # Resolve the trusted directory only for deterministic
                    # attack injection; production code must not depend on it.
                    pending_path = os.path.join(directory, candidate)
                    os.replace(replacement, pending_path)
                return real_unlink(candidate, *args, **kwargs)

            with mock.patch(
                "agent_box_credentials.os.unlink",
                side_effect=substitute_before_pending_unlink,
            ):
                result = remove_amsc_token(home)

            # A secure implementation may avoid the vulnerable unlink
            # entirely; otherwise an injected substitute must survive and
            # the operation must fail closed.
            if swapped:
                self.assertFalse(result)
                self.assertTrue(os.path.exists(replacement))
                with open(replacement, encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), "replacement-token")
            self.assertFalse(os.path.exists(token_path))

    def test_remove_does_not_unlink_replacement_leaf(self):
        with tempfile.TemporaryDirectory() as home:
            path = write_amsc_token(home, "original-token")
            replacement = os.path.join(os.path.dirname(path), "replacement")
            with open(replacement, "w", encoding="utf-8") as handle:
                handle.write("replacement-token")
            os.chmod(replacement, 0o600)
            real_rename = os.rename
            swapped = False

            def swap_before_rename(source, destination, *args, **kwargs):
                nonlocal swapped
                if source == "amsc-token" and kwargs.get("src_dir_fd") is not None:
                    swapped = True
                    os.replace(replacement, path)
                return real_rename(source, destination, *args, **kwargs)

            with mock.patch("agent_box_credentials.os.rename", side_effect=swap_before_rename):
                try:
                    result = remove_amsc_token(home)
                except CredentialError:
                    result = False
            self.assertTrue(swapped)
            self.assertFalse(result)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "replacement-token")

    def test_remove_never_deletes_same_named_files_outside_esnet_directory(self):
        with tempfile.TemporaryDirectory() as home:
            write_amsc_token(home, "real-token")
            agent_box_dir = os.path.join(home, "agent-box")
            credentials_dir = os.path.join(agent_box_dir, "credentials")
            unrelated = [
                os.path.join(agent_box_dir, "amsc-token"),
                os.path.join(credentials_dir, "amsc-token"),
            ]
            for index, path in enumerate(unrelated):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("unrelated-%d" % index)
                os.chmod(path, 0o600)

            self.assertTrue(remove_amsc_token(home))
            for index, path in enumerate(unrelated):
                self.assertTrue(os.path.exists(path))
                with open(path, encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), "unrelated-%d" % index)

    def test_management_cli_redacts_credential_errors(self):
        with tempfile.TemporaryDirectory() as home:
            errors = io.StringIO()
            with mock.patch(
                "agent_box_setup.reconfigure_amsc_credential_interactive",
                side_effect=CredentialError(SECRET),
            ):
                rc = agent_box.main(
                    ["credentials", "esnet", "configure"],
                    home=home,
                    registry_path=REGISTRY_PATH,
                    input_stream=io.StringIO(""),
                    output_stream=io.StringIO(),
                    error_stream=errors,
                    is_tty=True,
                    environ={},
                )
            self.assertEqual(rc, 77)
            self.assertNotIn(SECRET, errors.getvalue())


if __name__ == "__main__":
    unittest.main()
