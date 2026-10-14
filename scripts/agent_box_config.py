"""Persistent facility configuration — single source of truth from config."""
import os
import tempfile
import yaml

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "facilities.yaml",
)
SCHEMA_VERSION = 1


class ConfigError(Exception):
    def __init__(self, message, exit_code=78):
        super().__init__(message)
        self.exit_code = exit_code


def _load_registry():
    if not os.path.isfile(CONFIG_PATH):
        raise ConfigError(f"facility config missing: {CONFIG_PATH}", 78)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConfigError(f"malformed YAML: {exc}", 78) from exc
    if not isinstance(data, dict):
        raise ConfigError("facility config must be a mapping", 78)
    registry = {}
    for facility in ("alcf", "nersc", "esnet-east", "esnet-west"):
        if facility not in data:
            raise ConfigError(f"missing facility: {facility}", 78)
        entry = data[facility]
        if not isinstance(entry, dict) or "iri" not in entry:
            raise ConfigError(f"bad shape for {facility}", 78)
        iri = entry["iri"]
        if not isinstance(iri, dict) or "versions" not in iri:
            raise ConfigError(f"bad iri shape for {facility}", 78)
        versions = iri.get("versions", {})
        default_version = iri.get("default_version")
        registry[facility] = {
            "default_version": default_version,
            "versions": versions,
        }
    return registry


REGISTRY = _load_registry()
DEFAULTS = {
    "alcf": {"enabled": True, "version": "v1"},
    "nersc": {"enabled": False, "version": "v2"},
    "esnet-east": {"enabled": False, "version": "v2"},
    "esnet-west": {"enabled": False, "version": "v2"},
}


def resolve_defaults(config):
    result = {}
    for fac, spec in config.items():
        enabled = spec.get("enabled", "auto")
        if enabled == "auto":
            enabled = DEFAULTS.get(fac, {}).get("enabled", False)
        version = spec.get("version", "auto")
        if version == "auto":
            version = DEFAULTS.get(fac, {}).get("version")
        reg = REGISTRY.get(fac)
        if reg and version is not None:
            if version not in reg["versions"]:
                raise ConfigError(f"unsupported version {version} for {fac}", 78)
        result[fac] = {"enabled": bool(enabled), "version": version}
    return result


def parse_env(env_str):
    if env_str is None or env_str.strip() == "":
        return {}
    parts = [p.strip() for p in env_str.split(",") if p.strip()]
    seen = set()
    result = {}
    for p in parts:
        if p in seen:
            raise ConfigError(f"duplicate facility: {p}", 78)
        seen.add(p)
        if p not in REGISTRY:
            raise ConfigError(f"unknown facility: {p}", 78)
        result[p] = {}
    return result


def apply_version_env():
    versions = {
        "AGENT_BOX_IRI_VERSION_NERSC": "nersc",
        "AGENT_BOX_IRI_VERSION_ALCF": "alcf",
        "AGENT_BOX_IRI_VERSION_ESNET_EAST": "esnet-east",
        "AGENT_BOX_IRI_VERSION_ESNET_WEST": "esnet-west",
    }
    for env_key, fac in versions.items():
        val = os.environ.get(env_key)
        if val is not None:
            val = val.strip()
            if val not in REGISTRY.get(fac, {}).get("versions", {}):
                raise ConfigError(f"unsupported version {val} for {fac}", 78)


apply_version_env()


def atomic_yaml_write(path, data):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, default_flow_style=False)
        os.chmod(tmp_path, 0o600)
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
