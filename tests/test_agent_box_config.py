import os, sys, unittest, tempfile, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from agent_box_config import (
    ConfigError, SCHEMA_VERSION,
    load_registry, resolve_configuration, parse_env,
    parse_product_config, atomic_yaml_write,
)

WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(WORK_DIR, "config", "facilities.yaml")

class ConfigTests(unittest.TestCase):
    def test_registry_load_explicit(self):
        reg, facs = load_registry(REGISTRY_PATH)
        self.assertIn("alcf", reg)
        self.assertEqual(reg["alcf"]["default_version"], "v1")
        self.assertEqual(facs["alcf"]["iri"]["default_version"], "v1")

    def test_registry_empty_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump({"schema_version": 1, "facilities": {}}, f)
            p = f.name
        try:
            with self.assertRaises(ConfigError) as cm:
                load_registry(p)
            self.assertEqual(cm.exception.exit_code, 78)
        finally:
            os.unlink(p)

    def test_parse_env_rejects_unknown(self):
        reg, _ = load_registry(REGISTRY_PATH)
        with self.assertRaises(ConfigError) as cm:
            parse_env("fake", registry=reg)
        self.assertEqual(cm.exception.exit_code, 78)

    def test_parse_env_duplicate(self):
        with self.assertRaises(ConfigError) as cm:
            parse_env("alcf,alcf")
        self.assertEqual(cm.exception.exit_code, 78)

    def test_version_env_unsupported_fails(self):
        old = os.environ.get("AGENT_BOX_IRI_VERSION_ALCF")
        os.environ["AGENT_BOX_IRI_VERSION_ALCF"] = "v99"
        reg, _ = load_registry(REGISTRY_PATH)
        try:
            with self.assertRaises(ConfigError) as cm:
                resolve_configuration(reg, environ=os.environ)
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

    def test_resolve_precedence(self):
        reg, _ = load_registry(REGISTRY_PATH)
        # Full precedence: shipped defaults -> persisted -> env facilities
        res = resolve_configuration(reg, persisted={"nersc": {"enabled": True, "version": "v1"}})
        self.assertTrue(res["nersc"]["enabled"])
        self.assertEqual(res["nersc"]["version"], "v1")

    def test_no_boolean_coercion(self):
        reg, _ = load_registry(REGISTRY_PATH)
        with self.assertRaises(ConfigError):
            resolve_configuration(reg, persisted={"alcf": {"enabled": "yes", "version": "v1"}})

    def test_parse_product_config_valid(self):
        yaml_text = "schema_version: 1\nfacilities:\n  alcf:\n    enabled: true\n    version: v1\n"
        data = parse_product_config(yaml_text)
        self.assertEqual(data["schema_version"], 1)

    def test_parse_product_config_bad_version(self):
        with self.assertRaises(ConfigError):
            parse_product_config("schema_version: 2\nfacilities: {}")

    def test_parse_product_config_bad_enabled_type(self):
        with self.assertRaises(ConfigError):
            parse_product_config("schema_version: 1\nfacilities:\n  alcf:\n    enabled: 1\n")

if __name__ == '__main__':
    unittest.main()
