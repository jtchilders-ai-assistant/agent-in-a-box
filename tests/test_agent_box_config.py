import os, sys, unittest, tempfile, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from agent_box_config import (
    REGISTRY, SCHEMA_VERSION, DEFAULTS, ConfigError,
    resolve_defaults, parse_env, apply_version_env, atomic_yaml_write,
)

class ConfigTests(unittest.TestCase):
    def test_registry_single_source_from_yaml(self):
        self.assertIn("alcf", REGISTRY)
        self.assertEqual(REGISTRY["alcf"]["default_version"], "v1")
        self.assertIn("v1", REGISTRY["alcf"]["versions"])
        self.assertEqual(
            REGISTRY["alcf"]["versions"]["v1"]["base_url"],
            "https://api.alcf.anl.gov/api/v1",
        )
        self.assertEqual(
            REGISTRY["alcf"]["versions"]["v1"]["auth_provider"],
            "alcf-tokens",
        )
        self.assertTrue(DEFAULTS["alcf"]["enabled"])
        self.assertFalse(DEFAULTS["nersc"]["enabled"])

    def test_schema_version(self):
        self.assertEqual(SCHEMA_VERSION, 1)

    def test_parse_env_rejects_unknown(self):
        with self.assertRaises(ConfigError) as cm:
            parse_env("fake")
        self.assertEqual(cm.exception.exit_code, 78)

    def test_parse_env_rejects_duplicate(self):
        with self.assertRaises(ConfigError) as cm:
            parse_env("alcf,alcf")
        self.assertEqual(cm.exception.exit_code, 78)

    def test_version_env_unsupported_fails_closed(self):
        old = os.environ.get("AGENT_BOX_IRI_VERSION_ALCF")
        os.environ["AGENT_BOX_IRI_VERSION_ALCF"] = "v99"
        try:
            with self.assertRaises(ConfigError) as cm:
                apply_version_env()
            self.assertEqual(cm.exception.exit_code, 78)
        finally:
            if old is None:
                os.environ.pop("AGENT_BOX_IRI_VERSION_ALCF", None)
            else:
                os.environ["AGENT_BOX_IRI_VERSION_ALCF"] = old

    def test_atomic_yaml_mode_0600(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            atomic_yaml_write(tmp_path, {"a": 1})
            mode = os.stat(tmp_path).st_mode & 0o777
            self.assertEqual(mode, 0o600)
        finally:
            os.unlink(tmp_path)

    def test_config_error_exit_code_78(self):
        err = ConfigError("bad", 78)
        self.assertEqual(err.exit_code, 78)

    def test_defaults_and_auto_resolution(self):
        conf = {"alcf": {"enabled": "auto", "version": "auto"}}
        resolved = resolve_defaults(conf)
        self.assertTrue(resolved["alcf"]["enabled"])
        self.assertEqual(resolved["alcf"]["version"], "v1")

if __name__ == '__main__':
    unittest.main()
