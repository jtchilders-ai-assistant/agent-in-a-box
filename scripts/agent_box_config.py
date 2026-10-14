"""Persistent facility configuration — single source of truth from config."""
import os
import tempfile
import yaml

SCHEMA_VERSION = 1


class ConfigError(Exception):
    def __init__(self, message, exit_code=78):
        super().__init__(message)
        self.exit_code = exit_code


def load_registry(path):
    if not isinstance(path, str) or not path:
        raise ConfigError("registry path must be a non-empty string", 78)
    if not os.path.isfile(path):
        raise ConfigError(f"facility registry missing: {path}", 78)
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConfigError(f"malformed YAML: {exc}", 78) from exc
    if not isinstance(data, dict):
        raise ConfigError("registry must be a mapping", 78)
    sv = data.get("schema_version")
    if sv is None:
        raise ConfigError("missing schema_version in registry", 78)
    if sv != 1:
        raise ConfigError(f"unknown schema_version: {sv}", 78)
    facilities = data.get("facilities")
    if not isinstance(facilities, dict) or not facilities:
        raise ConfigError("registry facilities must be a non-empty mapping", 78)
    registry = {}
    for fac, entry in facilities.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"bad registry entry shape for {fac}", 78)
        iri = entry.get("iri")
        if not isinstance(iri, dict):
            raise ConfigError(f"bad iri for {fac}", 78)
        versions = iri.get("versions")
        if not isinstance(versions, dict) or not versions:
            raise ConfigError(f"versions must be mapping for {fac}", 78)
        default_version = iri.get("default_version")
        if default_version is None:
            raise ConfigError(f"missing default_version for {fac}", 78)
        if not isinstance(default_version, str):
            raise ConfigError(f"default_version must be string for {fac}", 78)
        if default_version is not None and default_version not in versions:
            raise ConfigError(f"default_version {default_version} not in versions for {fac}", 78)
        for vname, vspec in versions.items():
            if not isinstance(vname, str):
                raise ConfigError(f"version name must be string in {fac}", 78)
            if not isinstance(vspec, dict):
                raise ConfigError(f"version spec must be mapping for {vname} in {fac}", 78)
            endpoint = vspec.get("base_url")
            auth = vspec.get("auth_provider")
            if not isinstance(endpoint, str) or not endpoint:
                raise ConfigError(f"invalid endpoint for {fac}/{vname}", 78)
            if not isinstance(auth, str) or not auth:
                raise ConfigError(f"invalid auth_provider for {fac}/{vname}", 78)
        registry[fac] = {
            "default_version": default_version,
            "versions": versions,
        }
    return registry, data.get("facilities", {})


def resolve_configuration(registry, persisted=None, environ=None):
    if environ is None:
        environ = os.environ
    facilities = list(registry.keys())
    defaults = {
        "alcf": {"enabled": True, "version": (registry.get("alcf") or {}).get("default_version")},
        "nersc": {"enabled": False, "version": (registry.get("nersc") or {}).get("default_version")},
        "esnet-east": {"enabled": False, "version": (registry.get("esnet-east") or {}).get("default_version")},
        "esnet-west": {"enabled": False, "version": (registry.get("esnet-west") or {}).get("default_version")},
    }
    persisted = persisted or {}
    env_facilities = environ.get("AGENT_BOX_FACILITIES")
    enabled_set = set()
    if env_facilities is not None:
        # Use parse_env for duplicate awareness and unknown rejection
        parsed_env = parse_env(env_facilities, registry=registry)
        enabled_set = set(parsed_env.keys())
        for p in enabled_set:
            if p not in registry:
                raise ConfigError(f"unknown facility in AGENT_BOX_FACILITIES: {p}", 78)
    result = {}
    for fac in facilities:
        reg = registry[fac]
        persisted_spec = persisted.get(fac) if isinstance(persisted, dict) else {}
        if persisted_spec is None:
            persisted_spec = {}
        # No boolean coercion: only accept actual bool
        enabled_raw = persisted_spec.get("enabled")
        if isinstance(enabled_raw, bool):
            enabled = enabled_raw
        elif enabled_raw is None:
            enabled = defaults.get(fac, {}).get("enabled", False)
        else:
            raise ConfigError(f"enabled must be bool for {fac}, got {type(enabled_raw).__name__}", 78)
        # AGENT_BOX_FACILITIES defines exact enabled set
        if env_facilities is not None:
            enabled = fac in enabled_set
        else:
            if fac in enabled_set:
                enabled = True
        version = persisted_spec.get("version")
        if version == "auto":
            version = reg.get("default_version")
        elif isinstance(version, str):
            if version not in reg.get("versions", {}):
                raise ConfigError(f"unsupported version {version} for {fac}", 78)
        if not isinstance(version, str):
            version = defaults.get(fac, {}).get("version")
        # Version env overrides
        version_env_key_map = {
            "AGENT_BOX_IRI_VERSION_NERSC": "nersc",
            "AGENT_BOX_IRI_VERSION_ALCF": "alcf",
            "AGENT_BOX_IRI_VERSION_ESNET_EAST": "esnet-east",
            "AGENT_BOX_IRI_VERSION_ESNET_WEST": "esnet-west",
        }
        for env_key, ftarget in version_env_key_map.items():
            val = environ.get(env_key)
            if val is not None and ftarget == fac:
                val = val.strip()
                if val not in reg.get("versions", {}):
                    raise ConfigError(f"unsupported version {val} for {fac}", 78)
                version = val
        if version is None:
            version = reg.get("default_version")
        if not isinstance(enabled, bool):
            raise ConfigError(f"enabled must be bool for {fac}, got {type(enabled).__name__}", 78)
        result[fac] = {"enabled": enabled, "version": version}
    return result


def parse_env(env_str, registry=None):
    if registry is None:
        registry = {}
    if env_str is None or (isinstance(env_str, str) and env_str.strip() == ""):
        return {}
    if not isinstance(env_str, str):
        raise ConfigError("env_str must be string", 78)
    parts = [p.strip() for p in env_str.split(",") if p.strip()]
    seen = set()
    result = {}
    for p in parts:
        if p in seen:
            raise ConfigError(f"duplicate facility: {p}", 78)
        seen.add(p)
        if registry and p not in registry:
            raise ConfigError(f"unknown facility: {p}", 78)
        result[p] = {}
    return result


def parse_product_config(yaml_text, registry=None, schema_version=1):
    if schema_version != 1:
        raise ConfigError(f"unsupported schema_version: {schema_version}", 78)
    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"malformed YAML: {exc}", 78) from exc
    if not isinstance(data, dict):
        raise ConfigError("product config must be mapping", 78)
    sv = data.get("schema_version")
    if sv is None:
        raise ConfigError("missing schema_version", 78)
    if sv != 1:
        raise ConfigError(f"unknown schema_version: {sv}", 78)
    facilities = data.get("facilities")
    if facilities is None:
        raise ConfigError("missing facilities", 78)
    if not isinstance(facilities, dict):
        raise ConfigError("facilities must be mapping", 78)
    for fac, spec in facilities.items():
        if not isinstance(spec, dict):
            raise ConfigError(f"bad spec for {fac}", 78)
        enabled = spec.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            raise ConfigError(f"enabled must be bool for {fac}", 78)
        version = spec.get("version")
        if version is not None and not isinstance(version, str):
            raise ConfigError(f"version must be string for {fac}", 78)
        # Legacy top-level version support
        top_version = spec.get("version") if not isinstance(spec.get("iri"), dict) else None
        # Optional iri mapping validation
        iri_spec = spec.get("iri")
        if iri_spec is not None:
            if not isinstance(iri_spec, dict):
                raise ConfigError(f"iri must be mapping for {fac}", 78)
            if "version" in iri_spec and not isinstance(iri_spec["version"], str):
                raise ConfigError(f"iri.version must be string for {fac}", 78)
        if registry is not None and fac not in registry:
            raise ConfigError(f"unknown facility: {fac}", 78)
    return data


def atomic_yaml_write(path, data):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory)
    try:
        os.fchmod(fd, 0o600)
        handle = open(fd, "w", encoding="utf-8", closefd=False)
        try:
            yaml.safe_dump(data, handle, default_flow_style=False)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
            # Also close the underlying fd since closefd=False
            try:
                os.close(fd)
            except OSError:
                pass
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    try:
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
        dir_fd = None
        if hasattr(os, "O_DIRECTORY"):
            dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        else:
            dir_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            if dir_fd is not None:
                os.close(dir_fd)
        mode = os.stat(path).st_mode & 0o777
        if mode != 0o600:
            os.chmod(path, 0o600)
    except OSError:
        try:
            if os.path.exists(tmp_path) and os.path.abspath(tmp_path) != os.path.abspath(path):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise
