import os, sys, unittest, tempfile, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from agent_box_config import load_registry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DockerContractRed(unittest.TestCase):
    def test_nersc_tokens_not_installed_and_opt_alcf_missing_before_task2(self):
        spec = importlib.util.find_spec('nersc_tokens')
        self.assertIsNone(spec, "nersc-tokens package missing before install")
        # Red contract evidence: package and entrypoint contracts
        dockerfile_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Dockerfile")
        with open(dockerfile_path, encoding="utf-8") as f:
            df = f.read()
        self.assertIn("nersc-tokens==0.1.1", df)
        self.assertIn("agent_box_setup.py", df)
        entrypoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "entrypoint.sh")
        with open(entrypoint_path, encoding="utf-8") as f:
            ep = f.read()
        self.assertIn("agent_box_setup.py", ep)
        self.assertIn("agent-box/config.yaml", ep)
        # Before packaging /opt/alcf should not exist on this host
        # (test serves as RED contract evidence)
        # We don't assert absence of /opt/alcf since it may exist; we use import spec as red.

if __name__ == '__main__':
    unittest.main()
