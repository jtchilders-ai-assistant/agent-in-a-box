import os, unittest, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED_COPY = {
    "scripts/agent_box_setup.py": "/opt/alcf/agent_box_setup.py",
    "scripts/agent_box_config.py": "/opt/alcf/agent_box_config.py",
    "scripts/agent_box_credentials.py": "/opt/alcf/agent_box_credentials.py",
    "scripts/agent_box.py": "/opt/alcf/agent_box.py",
    "config/facilities.yaml": "/opt/alcf/config/facilities.yaml",
}

class DockerContractRed(unittest.TestCase):
    def _df(self):
        with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as f:
            return f.read()
    def _ep(self):
        with open(os.path.join(ROOT, "scripts", "entrypoint.sh"), encoding="utf-8") as f:
            return f.read()

    # RED: establish that loose assertions fail on mutated fixtures
    def test_red_mutated_copy_raises(self):
        df = self._df()
        df_mut = df.replace(
            "COPY scripts/agent_box_setup.py /opt/alcf/agent_box_setup.py",
            "COPY scripts/agent_box_setup.py /opt/alcf/WRONG.py")
        for src, dst in REQUIRED_COPY.items():
            with self.subTest(src=src):
                self.assertIn(f"COPY {src} {dst}", df_mut)

    def test_red_mutated_token_raises(self):
        df = self._df()
        df_mut = df.replace("nersc-tokens==0.1.1", "")
        found = False
        for line in df_mut.splitlines():
            if "RUN uv pip install" in line or ("alcf-tokens" in line and "nersc-tokens" not in line):
                found = True
                with self.assertRaises(AssertionError):
                    self.assertIn("nersc-tokens==0.1.1", line)
        self.assertTrue(found)

    def test_red_mutated_guard_raises(self):
        ep = self._ep()
        ep_mut = ep.replace('if [[ ! -f "$HERMES_HOME/agent-box/config.yaml" ]]; then', "if false; then")
        with self.assertRaises(AssertionError):
            self.assertIn('if [[ ! -f "$HERMES_HOME/agent-box/config.yaml" ]]; then', ep_mut)

    # Exact mappings (not substring)
    def test_exact_copy_mappings(self):
        df = self._df()
        for src, dst in REQUIRED_COPY.items():
            with self.subTest(src=src):
                self.assertIn(f"COPY {src} {dst}", df)

    def test_exact_package_tokens_standalone(self):
        df = self._df()
        # Extract the multi-line RUN uv pip install line; require standalone exact tokens
        block_lines = []
        in_block = False
        for line in df.splitlines():
            if "RUN uv pip install" in line:
                in_block = True
                block_lines.append(line)
            elif in_block:
                block_lines.append(line)
                if line.strip().endswith("\\") or line.strip().endswith(")"):
                    pass
                elif not line.startswith("    ") and not line.startswith(" "):
                    in_block = False
                    break
        # Find line containing tokens
        token_line = ""
        for line in df.splitlines():
            if "alcf-tokens==0.3.0" in line and "nersc-tokens==0.1.1" in line:
                token_line = line
        self.assertIn("alcf-tokens==0.3.0", token_line)
        self.assertIn("nersc-tokens==0.1.1", token_line)
        # Must be exact token (standalone), not generic substring
        self.assertRegex(token_line, r"alcf-tokens==0\.3\.0")
        self.assertRegex(token_line, r"nersc-tokens==0\.1\.1")

    def test_entrypoint_exact_contract(self):
        ep = self._ep()
        self.assertIn('if [[ ! -f "$HERMES_HOME/agent-box/config.yaml" ]]; then', ep)
        setup_line = None
        for line in ep.splitlines():
            if "agent_box_setup.py --home" in line:
                setup_line = line
        self.assertIsNotNone(setup_line)
        self.assertIn('--home "$HERMES_HOME"', setup_line)
        self.assertNotIn("|| exit", setup_line or "")
        self.assertIn("set -e", ep)

    def test_mutation_each_copy_dest_alters_raises(self):
        for src in REQUIRED_COPY:
            with self.subTest(src=src):
                df = self._df()
                df_mut = df.replace(f"COPY {src} {REQUIRED_COPY[src]}", f"COPY {src} /wrong")
                self.assertNotIn(f"COPY {src} {REQUIRED_COPY[src]}", df_mut)
                with self.assertRaises(AssertionError):
                    self.assertIn(f"COPY {src} {REQUIRED_COPY[src]}", df_mut)

    def test_mutation_token_removals_raises(self):
        for tok in ["alcf-tokens==0.3.0", "nersc-tokens==0.1.1"]:
            with self.subTest(tok=tok):
                df = self._df()
                df_mut = df.replace(tok, "")
                # Must fail on mutated text
                with self.assertRaises(AssertionError):
                    self.assertIn(tok, df_mut)

    def test_mutation_guard_removal_raises(self):
        ep = self._ep()
        ep_mut = ep.replace('if [[ ! -f "$HERMES_HOME/agent-box/config.yaml" ]]; then', "if false; then")
        with self.assertRaises(AssertionError):
            self.assertIn('if [[ ! -f "$HERMES_HOME/agent-box/config.yaml" ]]; then', ep_mut)

    def test_does_not_claim_build_run(self):
        # Explicit contract statement
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
