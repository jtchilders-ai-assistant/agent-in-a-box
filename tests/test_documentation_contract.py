#!/usr/bin/env python3
"""Robust documentation contract tests (context-managed reads, no handle leaks)."""
from __future__ import annotations
import os, unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DocContractTests(unittest.TestCase):
    @staticmethod
    def _read(path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, encoding="utf-8") as f:
            return f.read()

    def test_nersc_skill_exact_contract(self):
        path = os.path.join(REPO, "skills", "nersc-iri", "SKILL.md")
        content = self._read(path)
        self.assertIn("https://api.iri.nersc.gov", content)
        self.assertIn("e4f48665-38b5-4833-a89e-849c71f5b3e3", content)
        self.assertIn("ed3e577d-f7f3-4639-b96e-ff5a8445d699", content)
        self.assertIn("nersc-tokens login", content)
        self.assertIn("fail closed", content)
        self.assertIn("Never claim", content)
        self.assertIn("distinct from the ALCF combined", content)
        self.assertIn("enabled: false", content)
        for ln in content.splitlines():
            self.assertNotIn("validated route", ln)

    def test_esnet_skill_exact_contract(self):
        path = os.path.join(REPO, "skills", "esnet-iri", "SKILL.md")
        content = self._read(path)
        self.assertIn("esnet-east", content)
        self.assertIn("esnet-west", content)
        self.assertIn("ESNET_AMSC_TOKEN_FILE", content)
        for ln in content.splitlines():
            if "AMSC_TOKEN" in ln and "ESNET_AMSC_TOKEN_FILE" not in ln:
                # Allow description/frontmatter lines mentioning AMSC_TOKEN as reference/denial
                if ln.startswith("description:") or ln.startswith("name:"):
                    continue
                self.fail("ESnet references AMSC_TOKEN outside safe file: " + ln)
        self.assertIn("validated route", content)
        self.assertIn("not yet active", content)
        self.assertNotIn("set AMSC_TOKEN", content)
        self.assertNotIn("||", content)
        table_lines = [line for line in content.splitlines() if line.startswith("|")]
        self.assertEqual([line.count("|") for line in table_lines], [4, 4, 4, 4])

    def test_readme_contract(self):
        path = os.path.join(REPO, "README.md")
        content = self._read(path)
        self.assertIn("--pull=always", content)
        self.assertIn("127.0.0.1", content)
        self.assertIn("named volume", content.lower())
        self.assertIn("AGENT_BOX_FACILITIES", content)
        self.assertIn("Default ALCF only", content)
        for ln in content.splitlines():
            if ln.startswith("|| `"):
                self.fail("README broken table row: " + ln)
        self.assertIn("ESnet has registry metadata only", content)
        # Exact user-facing configuration details required
        self.assertIn("facility configuration", content.lower())
        self.assertIn("product config", content.lower())
        self.assertIn("config show", content)
        self.assertIn("config validate", content)
        self.assertIn("AGENT_BOX_FACILITIES=alcf,nersc", content)
        self.assertIn("loopback", content.lower())
        self.assertIn("nersc-tokens", content)

    def test_memory_contract(self):
        path = os.path.join(REPO, "memory", "MEMORY.md")
        content = self._read(path)
        self.assertIn("ALCF user-support agent", content)
        self.assertIn("fail closed", content)
        self.assertIn("NERSC", content)
        self.assertIn("ESnet", content)
        self.assertNotIn("submit there", content.lower())
        # Must reference registry and fail-closed boundary explicitly
        self.assertIn("registry", content.lower())

    def test_soul_contract(self):
        path = os.path.join(REPO, "docs", "DESIGN.md")
        content = self._read(path)
        self.assertIn("ESnet", content)
        self.assertIn("fail closed", content)
        # Design clearly states ESnet unverified state and fail-closed boundary.
        self.assertIn("fail-closed", content)

if __name__ == "__main__":
    unittest.main(verbosity=2)
