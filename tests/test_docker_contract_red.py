import os, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DockerContractRed(unittest.TestCase):
    def test_required_semantic_strings_present(self):
        with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as f:
            df = f.read()
        self.assertIn("alcf-tokens==0.3.0", df)
        self.assertIn("nersc-tokens==0.1.1", df)
        for src in ["agent_box_setup.py", "agent_box_config.py",
                    "agent_box_credentials.py", "agent_box.py", "config/facilities.yaml"]:
            self.assertIn(src, df)

    def test_entrypoint_contracts(self):
        with open(os.path.join(ROOT, "scripts", "entrypoint.sh"), encoding="utf-8") as f:
            ep = f.read()
        self.assertIn("agent_box_setup.py", ep)
        self.assertIn('--home "$HERMES_HOME"', ep)
        self.assertIn("set -e", ep)
        self.assertNotIn("|| exit 1", ep)
        self.assertIn('if [[ ! -f "$HERMES_HOME/agent-box/config.yaml" ]]; then', ep)

if __name__ == '__main__':
    unittest.main()
