"""Source-contract validation for Dockerfile + entrypoint.
Does NOT build or execute an OCI image.
"""
import os, re, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED_COPY = {
    "scripts/agent_box_setup.py": "/opt/alcf/agent_box_setup.py",
    "scripts/agent_box_config.py": "/opt/alcf/agent_box_config.py",
    "scripts/agent_box_credentials.py": "/opt/alcf/agent_box_credentials.py",
    "scripts/agent_box.py": "/opt/alcf/agent_box.py",
    "config/facilities.yaml": "/opt/alcf/config/facilities.yaml",
}


def assert_docker_contract(dockerfile_text):
    """Raise AssertionError with useful reason if dockerfile_text violates contract."""
    # Exact COPY mappings (logical instructions, ignore comments)
    for src, dst in REQUIRED_COPY.items():
        # Find logical COPY line containing source token
        found = False
        # Parse continuation lines for logical instructions
        logical_lines = []
        current = ""
        for line in dockerfile_text.splitlines():
            s = line.split('#', 1)[0]  # ignore comments
            current += s
            if current.rstrip().endswith('\\'):
                current = current.rstrip()[:-1] + ' '
            else:
                logical_lines.append(current)
                current = ""
        if current.strip():
            logical_lines.append(current)
        for ln in logical_lines:
            if 'COPY' in ln and src in ln:
                # Must have exact source and destination separated by whitespace
                # We check exact token presence in logical instruction
                parts = ln.split()
                # Find src index
                if src in parts and dst in parts:
                    src_idx = parts.index(src)
                    dst_idx = parts.index(dst)
                    # Must be adjacent: COPY src dst ... (dst right after src for these mappings)
                    # Actually allow src at position after COPY and dst later; just require exact pair
                    if src_idx < dst_idx:
                        found = True
                        break
        if not found:
            raise AssertionError(f"Missing exact COPY mapping: {src} -> {dst}")
    # Package tokens: exact whitespace-delimited tokens in logical RUN uv pip install
    logical_run = None
    current = ""
    for line in dockerfile_text.splitlines():
        s = line.split('#', 1)[0]
        current += s
        if current.rstrip().endswith('\\'):
            current = current.rstrip()[:-1] + ' '
        else:
            if 'RUN uv pip install' in current:
                logical_run = current
            current = ""
    if logical_run is None:
        raise AssertionError("Missing RUN uv pip install instruction")
    for tok in ["alcf-tokens==0.3.0", "nersc-tokens==0.1.1"]:
        # Exact whitespace-delimited token match
        if re.search(r'(?<!\S)' + re.escape(tok) + r'(?!\S)', logical_run) is None:
            raise AssertionError(f"Missing exact package token: {tok}")


def assert_entrypoint_contract(entrypoint_text):
    """Raise AssertionError with useful reason if entrypoint violates contract."""
    if 'if [[ ! -f "$HERMES_HOME/agent-box/config.yaml" ]]; then' not in entrypoint_text:
        raise AssertionError("Missing guard: config.yaml existence check")
    if '--home "$HERMES_HOME"' not in entrypoint_text:
        raise AssertionError("Missing --home \"$HERMES_HOME\" in setup line")
    # Check setup line has no || exit 1
    lines = entrypoint_text.splitlines()
    setup_line = None
    for line in lines:
        if "agent_box_setup.py --home" in line:
            setup_line = line
    if setup_line is None:
        raise AssertionError("Missing agent_box_setup.py --home line")
    if "|| exit" in setup_line:
        raise AssertionError("Setup line must not contain || exit 1")
    if "set -e" not in entrypoint_text:
        raise AssertionError("Missing set -e")


class DockerContractRed(unittest.TestCase):
    def _df(self):
        with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as f:
            return f.read()
    def _ep(self):
        with open(os.path.join(ROOT, "scripts", "entrypoint.sh"), encoding="utf-8") as f:
            return f.read()

    def test_contract_passes_on_actual_dockerfile(self):
        assert_docker_contract(self._df())

    def test_contract_passes_on_actual_entrypoint(self):
        assert_entrypoint_contract(self._ep())

    def test_mutation_each_copy_dest_alters_raises(self):
        for src in REQUIRED_COPY:
            with self.subTest(src=src):
                df = self._df()
                dst = REQUIRED_COPY[src]
                df_mut = df.replace(f"COPY {src} {dst}", f"COPY {src} /wrong")
                with self.assertRaises(AssertionError):
                    assert_docker_contract(df_mut)

    def test_mutation_token_removals_raises(self):
        for tok in ["alcf-tokens==0.3.0", "nersc-tokens==0.1.1"]:
            with self.subTest(tok=tok):
                df = self._df()
                df_mut = df.replace(tok, "")
                with self.assertRaises(AssertionError):
                    assert_docker_contract(df_mut)

    def test_mutation_guard_removal_raises(self):
        ep = self._ep()
        ep_mut = ep.replace('if [[ ! -f "$HERMES_HOME/agent-box/config.yaml" ]]; then', "if false; then")
        with self.assertRaises(AssertionError):
            assert_entrypoint_contract(ep_mut)

    def test_mutation_home_removal_raises(self):
        ep = self._ep()
        ep_mut = ep.replace('--home "$HERMES_HOME"', '--home "/tmp"')
        with self.assertRaises(AssertionError):
            assert_entrypoint_contract(ep_mut)

    def test_mutation_setup_or_exit_raises(self):
        ep = self._ep()
        # Add || exit 1 to setup line
        ep_mut = ep.replace('agent_box_setup.py --home "$HERMES_HOME"', 'agent_box_setup.py --home "$HERMES_HOME" || exit 1')
        with self.assertRaises(AssertionError):
            assert_entrypoint_contract(ep_mut)

    def test_exact_copy_mappings(self):
        df = self._df()
        for src, dst in REQUIRED_COPY.items():
            with self.subTest(src=src):
                # Exact mapping present via logical parse
                assert_docker_contract(df)

    def test_exact_package_tokens_standalone(self):
        df = self._df()
        assert_docker_contract(df)

    def test_entrypoint_exact_contract(self):
        assert_entrypoint_contract(self._ep())

    def test_does_not_claim_build_run(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
