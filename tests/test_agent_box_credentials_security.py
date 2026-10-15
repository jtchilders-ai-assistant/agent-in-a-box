import errno
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

    def test_operations_fail_closed_without_dir_fd_capabilities(self):
        import agent_box_credentials as credentials

        with tempfile.TemporaryDirectory() as home:
            with mock.patch.object(credentials.os, "supports_dir_fd", frozenset()):
                with self.assertRaises(CredentialError):
                    write_amsc_token(home, "token")
                with self.assertRaises(CredentialError):
                    read_amsc_token(home)
                with self.assertRaises(CredentialError):
                    remove_amsc_token(home)

    def test_backup_move_failures_clean_candidate_without_mutating_predecessor(self):
        import agent_box_credentials as credentials

        for kind in ("collision", "oserror"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as home:
                path = write_amsc_token(home, "old-token")
                directory = os.path.dirname(path)
                collision = os.path.join(directory, ".amsc-token.fixed")
                if kind == "collision":
                    with open(collision, "w", encoding="utf-8") as handle:
                        handle.write("foreign-backup")
                    os.chmod(collision, 0o600)
                    suffix_patch = mock.patch.object(
                        credentials, "_random_tombstone_suffix", return_value="fixed"
                    )
                    move_patch = mock.patch.object(
                        credentials, "_rename_noreplace_at", wraps=credentials._rename_noreplace_at
                    )
                else:
                    suffix_patch = mock.patch.object(
                        credentials, "_random_tombstone_suffix", return_value="move-failure"
                    )
                    real_move = credentials._rename_noreplace_at

                    def fail_backup(src_fd, src, dst_fd, dst):
                        if src == "amsc-token" and dst.startswith(".amsc-token."):
                            raise OSError("injected backup failure")
                        return real_move(src_fd, src, dst_fd, dst)

                    move_patch = mock.patch.object(
                        credentials, "_rename_noreplace_at", side_effect=fail_backup
                    )
                with suffix_patch, move_patch:
                    with self.assertRaises(CredentialError) as raised:
                        write_amsc_token(home, "rejected-candidate")
                self.assertNotIn("rejected-candidate", str(raised.exception))
                with open(path, encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), "old-token")
                if kind == "collision":
                    with open(collision, encoding="utf-8") as handle:
                        self.assertEqual(handle.read(), "foreign-backup")
                for name in os.listdir(directory):
                    with open(os.path.join(directory, name), encoding="utf-8") as handle:
                        self.assertNotEqual(handle.read(), "rejected-candidate")

    def test_write_never_overwrites_post_validation_substitute(self):
        import agent_box_credentials as credentials

        with tempfile.TemporaryDirectory() as home:
            write_amsc_token(home, "trusted-original")
            path = amsc_token_path(home)
            directory = os.path.dirname(path)
            foreign = os.path.join(directory, "foreign")
            with open(foreign, "w", encoding="utf-8") as handle:
                handle.write("foreign-data")
            os.chmod(foreign, 0o600)
            real_publish = credentials._rename_noreplace_at
            injected = {"done": False}

            def swap_before_publish(src_fd, src, dst_fd, dst):
                if src.startswith(".amsc-token.") and dst == "amsc-token":
                    injected["done"] = True
                    os.replace(foreign, path)
                return real_publish(src_fd, src, dst_fd, dst)

            with mock.patch.object(
                credentials, "_rename_noreplace_at", side_effect=swap_before_publish
            ):
                with self.assertRaises(CredentialError):
                    write_amsc_token(home, "new-secret")

            self.assertTrue(injected["done"])
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "foreign-data")

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

    def test_remove_final_namespace_io_error_never_reports_success(self):
        import agent_box_credentials as credentials

        with tempfile.TemporaryDirectory() as home:
            write_amsc_token(home, "verified-token")
            real_lstat = os.lstat
            calls = {"leaf": 0}

            def fail_final_leaf_check(path, *args, **kwargs):
                if path == "amsc-token" and kwargs.get("dir_fd") is not None:
                    calls["leaf"] += 1
                    if calls["leaf"] == 2:
                        raise OSError(5, "injected final verification failure")
                return real_lstat(path, *args, **kwargs)

            with mock.patch.object(
                credentials.os, "lstat", side_effect=fail_final_leaf_check
            ):
                self.assertFalse(remove_amsc_token(home))
            self.assertEqual(calls["leaf"], 2)

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
                    # Deterministic racer: the instant token destruction
                    # hits an fsync failure, an unverified object takes
                    # the public path. A restore attempt would then have
                    # to survive without displacing it.
                    with open(token_path, "w", encoding="utf-8") as handle:
                        handle.write("unverified-public")
                    os.chmod(token_path, 0o600)
                    state["planted"] = True
                    raise OSError("injected token fsync failure")
                return real_fsync(fd)

            real_noreplace = credentials._rename_noreplace_at
            restore_calls = []

            def track_restores(src_dir_fd, src, dst_dir_fd, dst):
                # Any restore seam (tombstone -> public) is recorded so
                # the assertion can prove the planted object survived
                # even if such a seam were ever reached.
                if isinstance(src, str) and src.startswith(".amsc-token.") \
                        and dst == "amsc-token":
                    restore_calls.append((src, dst))
                    with open(os.path.join(os.path.dirname(token_path), src),
                              encoding="utf-8") as handle:
                        moved_content = handle.read()
                    try:
                        return real_noreplace(src_dir_fd, src,
                                             dst_dir_fd, dst)
                    finally:
                        if os.path.lexists(token_path):
                            with open(token_path, encoding="utf-8") as handle:
                                if handle.read() == "unverified-public":
                                    # Reoccupant present: restore cannot
                                    # displace it; replant for the
                                    # assertion below.
                                    with open(os.path.join(
                                            os.path.dirname(token_path), src),
                                            "w", encoding="utf-8") as handle:
                                        handle.write(moved_content)
                return real_noreplace(src_dir_fd, src, dst_dir_fd, dst)

            with mock.patch.object(credentials.os, "fsync",
                                   side_effect=fail_first_fsync), \
                    mock.patch.object(credentials, "_rename_noreplace_at",
                                      side_effect=track_restores):
                with self.assertRaises(OSError):
                    remove_amsc_token(home)

            # A secure implementation never attempts restoration onto the
            # mutable public path once destruction has begun. If a legacy
            # restore seam was reached, the planted object must still
            # survive unchanged.
            self.assertTrue(state["planted"])
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

    def test_remove_mismatch_restore_never_displaces_public_reoccupant(self):
        import agent_box_credentials as credentials

        with tempfile.TemporaryDirectory() as home:
            token_path = write_amsc_token(home, "verified-token")
            directory = os.path.dirname(token_path)
            swapped = {"done": False}
            real_noreplace = credentials._rename_noreplace_at

            def race_before_move(src_dir_fd, src, dst_dir_fd, dst):
                # Deterministic namespace race: after the implementation
                # has verified the leaf, a same-UID racer swaps a
                # foreign object under the public name so the move
                # transports the wrong inode, and once the move has
                # completed the racer reoccupies the public path with
                # an unrelated object the restore must never displace.
                if src == "amsc-token":
                    original = os.path.join(directory, "original-inode")
                    os.rename(token_path, original)
                    foreign = os.path.join(directory, "foreign-object")
                    with open(foreign, "w", encoding="utf-8") as handle:
                        handle.write("foreign-token")
                    os.chmod(foreign, 0o600)
                    os.rename(foreign, token_path)
                    result = real_noreplace(src_dir_fd, src,
                                           dst_dir_fd, dst)
                    with open(token_path + ".reoccupant", "w",
                              encoding="utf-8") as handle:
                        handle.write("unverified-reoccupant")
                    os.chmod(token_path + ".reoccupant", 0o600)
                    os.rename(token_path + ".reoccupant", token_path)
                    swapped["done"] = True
                    return result
                # The tombstone -> public restore attempt: honour real
                # no-replace semantics (EEXIST if the public name is
                # occupied) via the platform helper.
                return real_noreplace(src_dir_fd, src, dst_dir_fd, dst)

            with mock.patch.object(
                credentials, "_rename_noreplace_at",
                side_effect=race_before_move
            ):
                with self.assertRaises(CredentialError):
                    remove_amsc_token(home)
            self.assertTrue(swapped["done"])
            # The atomic no-replace restore must fail EEXIST and must
            # never displace the public reoccupant.
            with open(token_path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "unverified-reoccupant")
            # The moved foreign object is never destroyed: it survives
            # intact under the retained tombstone name and nowhere
            # else.
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

    def test_removal_procedure_uses_no_unlink_or_overwriting_rename(self):
        # Structural guard for the Aegis IMP-01/IMP-02 invariants: the
        # removal procedure must contain no unlink call at all, no
        # os.rename/os.link call at all (every namespace move goes
        # through the atomic no-replace helper), and it must actually
        # call the no-replace helper.
        import ast
        import inspect

        import agent_box_credentials as credentials

        tree = ast.parse(inspect.getsource(credentials.remove_amsc_token))
        helper_calls = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"):
                self.assertNotEqual(
                    node.func.attr, "unlink",
                    "remove_amsc_token must never call os.unlink "
                    "(Aegis IMP-01)")
                self.assertNotIn(
                    node.func.attr, ("rename", "replace", "link"),
                    "remove_amsc_token must never perform an ordinary "
                    "rename/replace/link move (Aegis IMP-02), line %d"
                    % node.lineno)
            if (isinstance(node.func, ast.Name)
                    and node.func.id == "_rename_noreplace_at"):
                helper_calls += 1
        self.assertGreaterEqual(
            helper_calls, 1,
            "remove_amsc_token must move through the atomic no-replace "
            "primitive (Aegis IMP-02)")

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

    def test_remove_does_not_move_replacement_leaf(self):
        with tempfile.TemporaryDirectory() as home:
            path = write_amsc_token(home, "original-token")
            replacement = os.path.join(os.path.dirname(path), "replacement")
            with open(replacement, "w", encoding="utf-8") as handle:
                handle.write("replacement-token")
            os.chmod(replacement, 0o600)
            import agent_box_credentials as credentials
            real_noreplace = credentials._rename_noreplace_at
            swapped = False

            def swap_before_move(src_dir_fd, src, dst_dir_fd, dst):
                nonlocal swapped
                if src == "amsc-token":
                    swapped = True
                    os.replace(replacement, path)
                return real_noreplace(src_dir_fd, src, dst_dir_fd, dst)

            with mock.patch.object(credentials, "_rename_noreplace_at",
                                   side_effect=swap_before_move):
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


class RenameNoReplacePrimitiveTests(unittest.TestCase):
    """Direct coverage of the atomic no-replace rename primitive.

    These exercise the REAL libc call (renameatx_np RENAME_EXCL on
    Darwin, renameat2 RENAME_NOREPLACE on Linux) — no mocks.
    """

    def setUp(self):
        import agent_box_credentials as credentials

        if credentials._rename_noreplace_libc() is None:
            self.skipTest("platform lacks renameatx_np/renameat2")
        self.credentials = credentials
        self.tmp = tempfile.mkdtemp()
        self.dfd = os.open(self.tmp, os.O_RDONLY)

    def tearDown(self):
        os.close(self.dfd)
        for name in os.listdir(self.tmp):
            os.unlink(os.path.join(self.tmp, name))
        os.rmdir(self.tmp)

    def _write(self, name, content="x"):
        with open(os.path.join(self.tmp, name), "w", encoding="utf-8") as h:
            h.write(content)

    def test_primitive_symbol_matches_platform_header_value(self):
        # Darwin flag bits are pinned to sys/stdio.h; Linux bit to the
        # renameat2 uapi. A wrong constant silently disables the
        # no-replace guarantee, so pin them here.
        if sys.platform == "darwin":
            self.assertEqual(self.credentials._DARWIN_RENAME_EXCL, 0x4)
            self.assertEqual(
                self.credentials._DARWIN_RENAME_NOFOLLOW_ANY, 0x10)
            self.assertEqual(
                self.credentials._DARWIN_RENAME_RESOLVE_BENEATH, 0x20)
            for attempt in self.credentials._DARWIN_RENAME_FLAG_ATTEMPTS:
                self.assertTrue(attempt
                                & self.credentials._DARWIN_RENAME_EXCL)
        self.assertEqual(self.credentials._LINUX_RENAME_NOREPLACE, 0x1)

    def test_primitive_moves_when_destination_absent(self):
        self._write("src", "payload")
        self.credentials._rename_noreplace_at(self.dfd, "src",
                                              self.dfd, "dst")
        self.assertFalse(os.path.lexists(os.path.join(self.tmp, "src")))
        with open(os.path.join(self.tmp, "dst"), encoding="utf-8") as h:
            self.assertEqual(h.read(), "payload")

    def test_primitive_refuses_existing_destination_both_sides_survive(
            self):
        self._write("src", "source-content")
        self._write("dst", "destination-content")
        with self.assertRaises(FileExistsError):
            self.credentials._rename_noreplace_at(self.dfd, "src",
                                                  self.dfd, "dst")
        # BOTH endpoints untouched — this is the atomic guarantee an
        # ordinary rename cannot offer.
        with open(os.path.join(self.tmp, "src"), encoding="utf-8") as h:
            self.assertEqual(h.read(), "source-content")
        with open(os.path.join(self.tmp, "dst"), encoding="utf-8") as h:
            self.assertEqual(h.read(), "destination-content")

    def test_primitive_missing_source_raises_enoent(self):
        with self.assertRaises(FileNotFoundError):
            self.credentials._rename_noreplace_at(self.dfd, "absent",
                                                  self.dfd, "dst")

    def test_primitive_source_symlink_moves_link_not_target(self):
        # The public leaf is verified before the move, but the
        # primitive itself must never follow a source symlink: the
        # link entry travels, the (unverified) target stays put.
        outside_dir = tempfile.mkdtemp()
        outside = os.path.join(outside_dir, "target")
        try:
            with open(outside, "w", encoding="utf-8") as h:
                h.write("outside-target")
            os.symlink(outside, os.path.join(self.tmp, "linksrc"))
            self.credentials._rename_noreplace_at(self.dfd, "linksrc",
                                                  self.dfd, "moved")
            moved = os.readlink(os.path.join(self.tmp, "moved"))
            self.assertEqual(moved, outside)
            # Target untouched.
            with open(outside, encoding="utf-8") as h:
                self.assertEqual(h.read(), "outside-target")
        finally:
            link = os.path.join(self.tmp, "moved")
            if os.path.lexists(link):
                os.unlink(link)
            os.unlink(outside)
            os.rmdir(outside_dir)

    def test_primitive_destination_symlink_is_not_replaced_via_target(
            self):
        # A dangling destination symlink answers the name: RENAME_EXCL
        # must refuse (EEXIST) on Darwin; on Linux RENAME_NOREPLACE
        # replaces the symlink entry itself, never its target.  Assert
        # only the invariant that holds on both: the symlink's target
        # object (if any) is never overwritten.
        target_dir = tempfile.mkdtemp()
        try:
            target = os.path.join(target_dir, "victim")
            with open(target, "w", encoding="utf-8") as h:
                h.write("victim")
            os.symlink(target, os.path.join(self.tmp, "dstlink"))
            self._write("src", "payload")
            try:
                self.credentials._rename_noreplace_at(self.dfd, "src",
                                                      self.dfd,
                                                      "dstlink")
                replaced_dst_entry = True
            except FileExistsError:
                replaced_dst_entry = False
            with open(target, encoding="utf-8") as h:
                self.assertEqual(h.read(), "victim")
            if not replaced_dst_entry:
                self.assertTrue(os.path.lexists(
                    os.path.join(self.tmp, "src")))
        finally:
            for name in os.listdir(self.tmp):
                path = os.path.join(self.tmp, name)
                if os.path.islink(path):
                    os.unlink(path)
            victim = os.path.join(target_dir, "victim")
            if os.path.lexists(victim):
                os.unlink(victim)
            os.rmdir(target_dir)

    def test_remove_fails_closed_without_primitive(self):
        # ENOTSUP path: with no platform primitive discoverable, the
        # removal must raise CredentialError and touch nothing — it
        # must never fall back to an overwriting rename.
        import agent_box_credentials as credentials

        with tempfile.TemporaryDirectory() as home:
            token_path = write_amsc_token(home, "verified-token")
            with mock.patch.object(credentials, "_rename_noreplace_libc",
                                   return_value=None):
                with self.assertRaises(CredentialError):
                    remove_amsc_token(home)
            with open(token_path, encoding="utf-8") as h:
                self.assertEqual(h.read(), "verified-token")

    def test_remove_fails_closed_when_helper_reports_enotsup(self):
        import agent_box_credentials as credentials

        with tempfile.TemporaryDirectory() as home:
            token_path = write_amsc_token(home, "verified-token")

            def fake(src_dir_fd, src, dst_dir_fd, dst):
                raise OSError(errno.ENOTSUP, "Operation not supported")

            with mock.patch.object(credentials, "_rename_noreplace_at",
                                   side_effect=fake):
                with self.assertRaises(CredentialError):
                    remove_amsc_token(home)
            with open(token_path, encoding="utf-8") as h:
                self.assertEqual(h.read(), "verified-token")


class WritePublicationSecurityTests(unittest.TestCase):
    """Aegis I-1 write-publication semantics: identity-preserving
    publication via the atomic no-replace primitive, never an ordinary
    os.replace onto the public leaf."""

    def test_write_fails_closed_without_noreplace_primitive(self):
        import agent_box_credentials as credentials

        with tempfile.TemporaryDirectory() as home:
            with mock.patch.object(credentials, "_rename_noreplace_libc",
                                   return_value=None):
                with self.assertRaises(CredentialError):
                    write_amsc_token(home, "token")

    def test_write_publish_procedure_uses_no_overwriting_rename(self):
        # Structural guard for the Aegis I-1 write-publication fix:
        # the write procedure must contain no os.replace/os.rename/
        # os.link call at all (every namespace move goes through the
        # atomic no-replace helper) and it must actually call the
        # no-replace helper.
        import ast
        import inspect

        import agent_box_credentials as credentials

        tree = ast.parse(inspect.getsource(credentials.write_amsc_token))
        helper_calls = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"):
                self.assertNotIn(
                    node.func.attr, ("rename", "replace", "link"),
                    "write_amsc_token must never perform an ordinary "
                    "rename/replace/link move on the public leaf "
                    "(Aegis I-1), line %d" % node.lineno)
            if (isinstance(node.func, ast.Name)
                    and node.func.id == "_rename_noreplace_at"):
                helper_calls += 1
        self.assertGreaterEqual(
            helper_calls, 1,
            "write_amsc_token must publish through the atomic no-replace "
            "primitive (Aegis I-1)")

    def test_absent_destination_publish_is_exact_mode_and_clean(self):
        with tempfile.TemporaryDirectory() as home:
            path = write_amsc_token(home, "first-token")
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "first-token")
            st = os.stat(path)
            self.assertEqual(stat.S_IMODE(st.st_mode), 0o600)
            # No temp or backup debris after a clean absent publish.
            leftovers = [
                name for name in os.listdir(os.path.dirname(path))
                if name != "amsc-token"
            ]
            self.assertEqual(leftovers, [])

    def test_existing_publish_replaces_token_and_empties_backup(self):
        with tempfile.TemporaryDirectory() as home:
            path = write_amsc_token(home, "old-secret")
            old_stat = os.stat(path)
            write_amsc_token(home, "new-secret")
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "new-secret")
            st = os.stat(path)
            self.assertEqual(stat.S_IMODE(st.st_mode), 0o600)
            self.assertNotEqual(st.st_ino, old_stat.st_ino)
            # The predecessor lives on only as a RETAINED empty backup
            # name — its secret bytes were destroyed through the
            # verified descriptor after the new publication was durable.
            backups = [
                name for name in os.listdir(os.path.dirname(path))
                if name != "amsc-token"
            ]
            self.assertEqual(len(backups), 1)
            with open(os.path.join(os.path.dirname(path), backups[0]),
                      encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "")

    def test_write_publication_never_displaces_public_reoccupant(self):
        import agent_box_credentials as credentials

        with tempfile.TemporaryDirectory() as home:
            path = amsc_token_path(home)
            directory = os.path.dirname(path)
            # Build the credential chain through a real write+remove so
            # every component carries the required private mode, then
            # start from an absent leaf.
            write_amsc_token(home, "seed")
            self.assertTrue(remove_amsc_token(home))
            before = set(os.listdir(directory))
            foreign = os.path.join(directory, "foreign")
            with open(foreign, "w", encoding="utf-8") as handle:
                handle.write("reoccupant-data")
            os.chmod(foreign, 0o600)
            real_publish = credentials._rename_noreplace_at
            injected = {"done": False}

            def occupy_before_publish(src_fd, src, dst_fd, dst):
                # Absent-destination flow: a racer occupies the public
                # name right before the temp -> public publish.
                if src.startswith(".amsc-token.") and dst == "amsc-token":
                    injected["done"] = True
                    os.link(foreign, path)
                return real_publish(src_fd, src, dst_fd, dst)

            with mock.patch.object(
                credentials, "_rename_noreplace_at",
                side_effect=occupy_before_publish
            ):
                with self.assertRaises(CredentialError):
                    write_amsc_token(home, "new-secret")
            self.assertTrue(injected["done"])
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "reoccupant-data")
            # The writer's temp was released by identity; the
            # reoccupant and its sibling survive untouched, and the
            # namespace gained nothing but the attacker's own objects.
            self.assertEqual(
                set(os.listdir(directory)), before | {"foreign", "amsc-token"})
            with open(foreign, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "reoccupant-data")

    def test_source_swap_at_backup_move_never_mutates_foreign_object(self):
        import agent_box_credentials as credentials

        with tempfile.TemporaryDirectory() as home:
            path = write_amsc_token(home, "trusted-original")
            directory = os.path.dirname(path)
            foreign = os.path.join(directory, "foreign")
            with open(foreign, "w", encoding="utf-8") as handle:
                handle.write("foreign-object")
            os.chmod(foreign, 0o600)
            real_move = credentials._rename_noreplace_at
            swapped = {"done": False}

            def swap_before_backup_move(src_fd, src, dst_fd, dst):
                # Source-swap race: at the public -> backup move the
                # public name answers a FOREIGN object, not the
                # verified predecessor.
                if src == "amsc-token" and dst.startswith(".amsc-token."):
                    swapped["done"] = True
                    os.replace(foreign, path)
                return real_move(src_fd, src, dst_fd, dst)

            with mock.patch.object(
                credentials, "_rename_noreplace_at",
                side_effect=swap_before_backup_move
            ):
                with self.assertRaises(CredentialError):
                    write_amsc_token(home, "new-secret")
            self.assertTrue(swapped["done"])
            # The foreign object (moved into the backup name) must be
            # restored to the public name via no-replace and survive
            # byte-identical; nothing was destroyed.
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "foreign-object")

    def test_dir_fsync_failure_after_publish_preserves_backup_bytes(self):
        import agent_box_credentials as credentials

        with tempfile.TemporaryDirectory() as home:
            path = write_amsc_token(home, "old-secret")
            directory = os.path.dirname(path)
            real_fsync = os.fsync
            state = {"calls": 0}

            def fail_dir_fsync(fd):
                state["calls"] += 1
                if state["calls"] == 2:
                    # Second fsync in the existing-leaf flow is the
                    # directory fsync that makes publication durable.
                    # A failure there must NOT destroy the predecessor
                    # bytes (destruction happens only after durability).
                    raise OSError("injected directory fsync failure")
                return real_fsync(fd)

            with mock.patch.object(credentials.os, "fsync",
                                   side_effect=fail_dir_fsync):
                with self.assertRaises(OSError):
                    write_amsc_token(home, "new-secret")
            self.assertEqual(state["calls"], 2)
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "new-secret")
            backups = [
                name for name in os.listdir(directory)
                if name != "amsc-token"
            ]
            self.assertEqual(len(backups), 1)
            with open(os.path.join(directory, backups[0]),
                      encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "old-secret")

    def test_failed_write_does_not_leak_file_descriptors(self):
        fd_root = "/dev/fd" if os.path.isdir("/dev/fd") else "/proc/self/fd"
        if not os.path.isdir(fd_root):
            self.skipTest("no process fd directory")
        import agent_box_credentials as credentials

        with tempfile.TemporaryDirectory() as home:
            write_amsc_token(home, "old-token")
            baseline = len(os.listdir(fd_root))
            real_publish = credentials._rename_noreplace_at
            state = {"publish": 0}

            def fail_publish(src_fd, src, dst_fd, dst):
                if src.startswith(".amsc-token.") and dst == "amsc-token":
                    state["publish"] += 1
                    if state["publish"] == 1:
                        # Only the temp -> public publication fails;
                        # the backup -> public restore must run for real.
                        raise OSError("injected publication failure")
                return real_publish(src_fd, src, dst_fd, dst)

            with mock.patch.object(
                credentials, "_rename_noreplace_at", side_effect=fail_publish
            ):
                with self.assertRaises(CredentialError):
                    write_amsc_token(home, "new-token")
            observed = len(os.listdir(fd_root))
            self.assertLessEqual(observed, baseline + 1)
            # Rollback preserved the previous token (no reoccupation).
            self.assertEqual(read_amsc_token(home), "old-token")


if __name__ == "__main__":
    unittest.main()
