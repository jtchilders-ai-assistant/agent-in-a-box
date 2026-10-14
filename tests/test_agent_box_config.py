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

    def test_registry_requires_default_version_and_nonempty_versions(self):
        invalid_documents = (
            {"schema_version": 1, "facilities": {"x": {"iri": {"versions": {"v1": {"base_url": "https://example.test/api/v1", "auth_provider": "test"}}}}}},
            {"schema_version": 1, "facilities": {"x": {"iri": {"default_version": "v1", "versions": {}}}}},
        )
        for document in invalid_documents:
            with self.subTest(document=document):
                with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
                    yaml.safe_dump(document, handle)
                    path = handle.name
                try:
                    with self.assertRaises(ConfigError):
                        load_registry(path)
                finally:
                    os.unlink(path)

    def test_product_config_rejects_unknown_facility_and_nested_iri_shape(self):
        registry, _ = load_registry(REGISTRY_PATH)
        with self.assertRaises(ConfigError):
            parse_product_config(
                "schema_version: 1\nfacilities:\n  unknown:\n    enabled: true\n",
                registry=registry,
            )
        with self.assertRaises(ConfigError):
            parse_product_config(
                "schema_version: 1\nfacilities:\n  alcf:\n    enabled: true\n    iri: not-a-mapping\n",
                registry=registry,
            )

    def test_environment_facilities_are_an_exact_enabled_set_and_reject_duplicates(self):
        registry, _ = load_registry(REGISTRY_PATH)
        resolved = resolve_configuration(
            registry,
            persisted={"alcf": {"enabled": True}, "nersc": {"enabled": True}},
            environ={"AGENT_BOX_FACILITIES": "nersc"},
        )
        self.assertFalse(resolved["alcf"]["enabled"])
        self.assertTrue(resolved["nersc"]["enabled"])
        with self.assertRaises(ConfigError):
            resolve_configuration(registry, environ={"AGENT_BOX_FACILITIES": "alcf,alcf"})

    def test_persisted_auto_and_unsupported_versions(self):
        registry, _ = load_registry(REGISTRY_PATH)
        resolved = resolve_configuration(
            registry,
            persisted={"nersc": {"enabled": True, "version": "auto"}},
            environ={},
        )
        self.assertEqual(resolved["nersc"]["version"], "v2")
        with self.assertRaises(ConfigError):
            resolve_configuration(
                registry,
                persisted={"nersc": {"enabled": True, "version": "v99"}},
                environ={},
            )

    def test_atomic_write_preserves_target_and_cleans_temp_on_replace_failure(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "config.yaml")
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("old: true\n")
            before = set(os.listdir(directory))
            with mock.patch("agent_box_config.os.replace", side_effect=OSError("injected")):
                with self.assertRaises(OSError):
                    atomic_yaml_write(target, {"new": True})
            with open(target, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "old: true\n")
            self.assertEqual(set(os.listdir(directory)), before)

    def test_atomic_write_fsyncs_directory_after_replace_and_closes_descriptor(self):
        from unittest import mock

        events = []
        real_replace = os.replace
        real_fsync = os.fsync

        def recording_replace(source, target):
            events.append("replace")
            return real_replace(source, target)

        def recording_fsync(fd):
            events.append("fsync")
            return real_fsync(fd)

        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "config.yaml")
            with mock.patch("agent_box_config.os.replace", side_effect=recording_replace), mock.patch(
                "agent_box_config.os.fsync", side_effect=recording_fsync
            ):
                atomic_yaml_write(target, {"ok": True})
            self.assertIn("replace", events)
            self.assertIn("fsync", events[events.index("replace") + 1 :])

    def test_resolve_reads_documented_nested_iri_version(self):
        registry, _ = load_registry(REGISTRY_PATH)
        persisted = parse_product_config(
            "schema_version: 1\n"
            "facilities:\n"
            "  nersc:\n"
            "    enabled: true\n"
            "    iri:\n"
            "      version: v1\n",
            registry=registry,
        )["facilities"]
        resolved = resolve_configuration(registry, persisted=persisted, environ={})
        self.assertEqual(resolved["nersc"]["version"], "v1")

    def test_resolve_rejects_non_string_nested_version(self):
        registry, _ = load_registry(REGISTRY_PATH)
        with self.assertRaises(ConfigError) as raised:
            resolve_configuration(
                registry,
                persisted={"nersc": {"enabled": True, "iri": {"version": 2}}},
                environ={},
            )
        self.assertEqual(raised.exception.exit_code, 78)

    def test_resolve_rejects_non_string_flat_legacy_version(self):
        registry, _ = load_registry(REGISTRY_PATH)
        with self.assertRaises(ConfigError) as raised:
            resolve_configuration(
                registry,
                persisted={"nersc": {"enabled": True, "version": 2}},
                environ={},
            )
        self.assertEqual(raised.exception.exit_code, 78)

    def test_resolve_rejects_non_mapping_facility_entry(self):
        registry, _ = load_registry(REGISTRY_PATH)
        with self.assertRaises(ConfigError) as raised:
            resolve_configuration(
                registry,
                persisted={"nersc": "enabled"},
                environ={},
            )
        self.assertEqual(raised.exception.exit_code, 78)

    def test_registry_rejects_non_string_facility_key(self):
        document = {
            "schema_version": 1,
            "facilities": {
                7: {
                    "iri": {
                        "default_version": "v1",
                        "versions": {
                            "v1": {
                                "base_url": "https://example.test/api/v1",
                                "auth_provider": "test",
                            }
                        },
                    }
                }
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(document, handle)
            path = handle.name
        try:
            with self.assertRaises(ConfigError):
                load_registry(path)
        finally:
            os.unlink(path)

if __name__ == '__main__':
    unittest.main()
