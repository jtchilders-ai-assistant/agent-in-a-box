import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from agent_box_credentials import (
    CredentialError,
    amsc_token_path,
    read_amsc_token,
    write_amsc_token,
)


class ParentNamespaceSwapTests(unittest.TestCase):
    def _swap_esnet_parent(self, home):
        credentials = os.path.join(home, "agent-box", "credentials")
        trusted = os.path.join(credentials, "esnet")
        moved = os.path.join(credentials, "esnet-trusted-moved")
        attacker = os.path.join(credentials, "esnet-attacker")
        os.mkdir(attacker, 0o700)
        with open(os.path.join(attacker, "amsc-token"), "w", encoding="utf-8") as handle:
            handle.write("attacker-token")
        os.chmod(os.path.join(attacker, "amsc-token"), 0o600)
        os.rename(trusted, moved)
        os.symlink(attacker, trusted)
        return moved, attacker

    def test_read_cannot_be_redirected_after_trusted_directory_open(self):
        with tempfile.TemporaryDirectory() as home:
            write_amsc_token(home, "trusted-token")
            import agent_box_credentials as credentials

            real_open_token = credentials._open_token_fd_by_path

            def swap_then_open(path):
                self._swap_esnet_parent(home)
                return real_open_token(path)

            with mock.patch(
                "agent_box_credentials._open_token_fd_by_path",
                side_effect=swap_then_open,
            ):
                try:
                    observed = read_amsc_token(home)
                except CredentialError:
                    return
            self.assertEqual(observed, "trusted-token")

    def test_write_cannot_be_redirected_after_trusted_directory_open(self):
        with tempfile.TemporaryDirectory() as home:
            write_amsc_token(home, "old-token")
            real_mkstemp = tempfile.mkstemp
            state = {}

            def swap_then_create_temp(*args, **kwargs):
                moved, attacker = self._swap_esnet_parent(home)
                state.update(moved=moved, attacker=attacker)
                return real_mkstemp(*args, **kwargs)

            with mock.patch(
                "agent_box_credentials.tempfile.mkstemp",
                side_effect=swap_then_create_temp,
            ):
                try:
                    write_amsc_token(home, "new-token")
                except CredentialError:
                    pass

            with open(os.path.join(state["attacker"], "amsc-token"), encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "attacker-token")
            with open(os.path.join(state["moved"], "amsc-token"), encoding="utf-8") as handle:
                self.assertIn(handle.read(), ("old-token", "new-token"))


if __name__ == "__main__":
    unittest.main()
