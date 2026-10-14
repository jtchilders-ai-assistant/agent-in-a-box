"""First-run setup wizard and conservative legacy-ALCF migration.

Stream-injected (no direct stdin/stdout coupling) so the wizard is testable
and safe in non-interactive containers.  Rules enforced here:

- ALCF is the default selection when the user just presses Enter, and the
  only default in a no-TTY environment (no override unless requested).
- Numbered, comma-separated facility selection with validation and retry.
- Recommended (registry default) versions are used unless the operator
  opts into advanced version selection; unsupported versions are rejected
  and re-prompted.
- Nothing is written until the operator confirms the summary; cancel
  writes nothing at all.
- The product configuration is written atomically (Task 1
  ``atomic_yaml_write``) at ``<home>/agent-box/config.yaml`` with the
  parent directory held at mode 0700 and the file at 0600.  It contains
  no secrets, and this module never prompts for or stores any token
  (ESnet AMSC credential capture is Task 3).
- Selecting an ESnet (amsc-pat) facility without a *safe*
  ``ESNET_AMSC_TOKEN_FILE`` or a *safe* persisted token path fails closed
  with ConfigError exit code 77 instead of prompting.
- Legacy ALCF detection is existence-only (``os.path.exists`` on known
  candidate paths).  It never opens or reads credential content.

Product config full schema: ``facilities.<name>.enabled`` (bool) and
``facilities.<name>.iri.version`` (string), matching
``agent_box_config.resolve_configuration``.
"""
import dataclasses
import os
import sys

from agent_box_config import (
    ConfigError,
    atomic_yaml_write,
    load_registry,
    parse_env,
    resolve_configuration,
)

# Relative path of the persistent product configuration under the supplied home.
LEGACY_CONFIG_RELATIVE_PATH = os.path.join("agent-box", "config.yaml")

# Conventional location where a future task may persist an ESnet AMSC token
# file.  This module only ever checks its existence and permission bits — it
# never opens or reads it.
PERSISTED_AMSC_TOKEN_RELATIVE_PATH = os.path.join("agent-box", "esnet-amsc-token")

# Environment variable naming an operator-supplied AMSC token file.
AMSC_TOKEN_FILE_ENV = "ESNET_AMSC_TOKEN_FILE"

# Auth provider (per registry) whose facilities need an AMSC token file.
AMSC_AUTH_PROVIDER = "amsc-pat"

# Candidate paths whose mere *existence* indicates a legacy ALCF install.
# Existence-only checks; contents are never read.
_LEGACY_CANDIDATE_RELATIVE_PATHS = (
    os.path.join(".globus", "app", "tokens.json"),
    os.path.join(".globus", "access_tokens"),
    os.path.join(".globus", "refresh_tokens"),
)

# Candidate paths for the exact auth provider that requires the AMSC token.
_DISPLAY_NAMES = {
    "alcf": "ALCF",
    "nersc": "NERSC",
    "esnet-east": "ESnet East",
    "esnet-west": "ESnet West",
}


@dataclasses.dataclass
class SetupResult:
    """Outcome of a first-run setup attempt.

    status: "configured" | "cancelled" | "migrated"
    configuration: the product config mapping that was written, or None
    resolved: result of resolve_configuration for the produced/persisted state
    """

    status: str
    configuration: "dict | None" = None
    resolved: "dict | None" = None


def _display_name(facility):
    return _DISPLAY_NAMES.get(facility, facility.replace("-", " ").title())


def _read_line(stream):
    """Read one line; None on EOF so callers fall back to defaults, never hang."""
    line = stream.readline()
    if line == "":
        return None
    return line.strip()


def _facility_requires_amsc(registry, facility):
    versions = registry.get(facility, {}).get("versions", {})
    return any(
        isinstance(spec, dict) and spec.get("auth_provider") == AMSC_AUTH_PROVIDER
        for spec in versions.values()
    )


def _is_safe_token_path(path):
    """Existence/stat-only safety check.  The file is never opened or read.

    Safe means: absolute, regular file, owned by the current user, and not
    readable or writable by group or others.
    """
    if not isinstance(path, str) or not path:
        return False
    if not os.path.isabs(path):
        return False
    try:
        st = os.stat(path)
    except OSError:
        return False
    import stat as _stat

    if not _stat.S_ISREG(st.st_mode):
        return False
    if st.st_uid != os.geteuid():
        return False
    if st.st_mode & 0o077:
        return False
    return True


def _require_amsc_credentials(registry, selected, environ, home):
    """Fail closed (exit 77) if an amsc-pat facility lacks a safe token path.

    Never prompts: ESnet credential capture is deliberately out of scope
    (Task 3).
    """
    esnet_selected = [f for f in selected if _facility_requires_amsc(registry, f)]
    if not esnet_selected:
        return
    env_path = environ.get(AMSC_TOKEN_FILE_ENV)
    if env_path is not None and _is_safe_token_path(env_path.strip()):
        return
    persisted = os.path.join(home, PERSISTED_AMSC_TOKEN_RELATIVE_PATH)
    if _is_safe_token_path(persisted):
        return
    raise ConfigError(
        "ESnet facility selected (%s) but no safe AMSC token is available: set "
        "%s to an existing absolute token file owned by you with mode 0600 "
        "(group/world access not permitted), or persist a token at %s. "
        "Interactive token capture is not yet available."
        % (", ".join(esnet_selected), AMSC_TOKEN_FILE_ENV, persisted),
        77,
    )


def select_facilities(registry, input_stream, output_stream):
    """Interactive numbered, comma-separated facility selection.

    Enter alone selects ALCF only.  Invalid numbers are reported and
    re-prompted.  EOF falls back to the ALCF default rather than hanging.
    Returns selected registry keys in registry order.
    """
    keys = list(registry)
    if "alcf" not in registry:
        raise ConfigError("registry has no alcf facility to default to", 78)
    output_stream.write("Select facilities to enable (numbers, comma-separated):\n")
    for index, facility in enumerate(keys, start=1):
        output_stream.write("  %d) %s\n" % (index, _display_name(facility)))
    output_stream.write("Press Enter for the default (ALCF only).\n")
    while True:
        line = _read_line(input_stream)
        if line is None or line == "":
            selected_names = {"alcf"}
            break
        tokens = [token.strip() for token in line.split(",") if token.strip()]
        if not tokens:
            selected_names = {"alcf"}
            break
        picked = []
        invalid = False
        seen = set()
        for token in tokens:
            if not token.isdigit():
                invalid = True
                break
            number = int(token)
            if number < 1 or number > len(keys) or number in seen:
                invalid = True
                break
            seen.add(number)
            picked.append(keys[number - 1])
        if invalid:
            output_stream.write(
                "Invalid selection. Enter numbers between 1 and %d, "
                "comma-separated, without duplicates.\n" % len(keys)
            )
            continue
        selected_names = set(picked)
        break
    for facility in keys:
        mark = "x" if facility in selected_names else " "
        output_stream.write("[%s] %s\n" % (mark, _display_name(facility)))
    return [facility for facility in keys if facility in selected_names]


def select_versions(registry, selected, input_stream, output_stream):
    """Version selection: recommended defaults, or advanced per-facility.

    A single gate question asks whether to accept the recommended
    (registry default) versions for every selected facility.  Answering
    "n" prompts for each facility, rejecting unsupported versions with a
    retry.  EOF always falls back to the recommended version.
    """
    recommended = {f: registry[f]["default_version"] for f in selected}
    output_stream.write("Use the recommended IRI versions for the selected "
                        "facilities? [Y/n] ")
    while True:
        answer = _read_line(input_stream)
        if answer is None or answer.lower() in ("", "y", "yes"):
            return dict(recommended)
        if answer.lower() in ("n", "no"):
            break
        output_stream.write("Invalid answer. Enter y or n.\n")
    versions = {}
    for facility in selected:
        supported = sorted(registry[facility]["versions"])
        output_stream.write(
            "IRI version for %s (supported: %s; Enter for %s): "
            % (_display_name(facility), ", ".join(supported),
               registry[facility]["default_version"])
        )
        while True:
            answer = _read_line(input_stream)
            if answer is None or answer == "":
                versions[facility] = registry[facility]["default_version"]
                break
            if answer in registry[facility]["versions"]:
                versions[facility] = answer
                break
            output_stream.write(
                "Unsupported version %s for %s (choose from %s): "
                % (answer, _display_name(facility), ", ".join(supported))
            )
    return versions


def _build_product_config(registry, selected, versions):
    facilities = {}
    for facility in registry:
        version = versions.get(facility, registry[facility]["default_version"])
        facilities[facility] = {
            "enabled": facility in selected,
            "iri": {"version": version},
        }
    return {"schema_version": 1, "facilities": facilities}


def _write_product_config(home, config):
    config_path = os.path.join(home, LEGACY_CONFIG_RELATIVE_PATH)
    parent = os.path.dirname(config_path)
    os.makedirs(parent, mode=0o700, exist_ok=True)
    # Conservative: force the parent directory private even if it pre-existed.
    os.chmod(parent, 0o700)
    atomic_yaml_write(config_path, config)
    return config_path


def _legacy_alcf_detected(home):
    """Existence-only detection of a prior ALCF install. Never reads content."""
    for relative in _LEGACY_CANDIDATE_RELATIVE_PATHS:
        if os.path.exists(os.path.join(home, relative)):
            return True
    if os.path.exists(os.path.join(home, LEGACY_CONFIG_RELATIVE_PATH)):
        return True
    return False


def run_first_setup(home, registry, input_stream, output_stream, is_tty, environ):
    """Run first-run setup: legacy migration, non-interactive default, or wizard.

    Writes nothing unless the outcome is an applied configuration.
    """
    if is_tty and _legacy_alcf_detected(home):
        # Conservative migration: a legacy ALCF install is present.  Keep the
        # runtime's own resolution (ALCF enabled by default) and touch
        # nothing — no prompting, no rewriting of legacy state.
        output_stream.write(
            "Existing ALCF configuration detected; keeping current settings.\n"
        )
        return SetupResult(
            status="migrated",
            configuration=None,
            resolved=resolve_configuration(registry, persisted=None, environ=environ),
        )

    if not is_tty:
        # Non-interactive: ALCF-only default, no override unless
        # AGENT_BOX_FACILITIES is explicitly provided.
        env_facilities = environ.get("AGENT_BOX_FACILITIES")
        if env_facilities is not None:
            selected = list(parse_env(env_facilities, registry=registry))
        else:
            selected = ["alcf"]
        versions = {f: registry[f]["default_version"] for f in selected}
        _require_amsc_credentials(registry, selected, environ, home)
        config = _build_product_config(registry, selected, versions)
        _write_product_config(home, config)
        return SetupResult(
            status="configured",
            configuration=config,
            resolved=resolve_configuration(
                registry, persisted=config["facilities"], environ=environ
            ),
        )

    output_stream.write("Welcome to ALCF Agent in a Box first-run setup.\n")
    selected = select_facilities(registry, input_stream, output_stream)
    versions = select_versions(registry, selected, input_stream, output_stream)
    output_stream.write("\nSummary:\n")
    for facility in registry:
        if facility in selected:
            output_stream.write(
                "  %s: enabled, IRI version %s\n"
                % (_display_name(facility), versions[facility])
            )
        else:
            output_stream.write("  %s: disabled\n" % _display_name(facility))
    output_stream.write("No credentials are collected or stored by setup.\n")
    output_stream.write("Apply this configuration? [y/N] ")
    answer = _read_line(input_stream)
    if answer is None or answer.lower() not in ("y", "yes"):
        output_stream.write("Setup cancelled. Nothing was written.\n")
        return SetupResult(status="cancelled", configuration=None, resolved=None)
    # Post-confirm guard: still fail closed rather than capture credentials.
    _require_amsc_credentials(registry, selected, environ, home)
    config = _build_product_config(registry, selected, versions)
    _write_product_config(home, config)
    output_stream.write("Configuration written to %s\n"
                        % os.path.join(home, LEGACY_CONFIG_RELATIVE_PATH))
    return SetupResult(
        status="configured",
        configuration=config,
        resolved=resolve_configuration(
            registry, persisted=config["facilities"], environ=environ
        ),
    )


def main(argv=None):
    """CLI entry point: run first-run setup exactly once.

    Exits 0 when configured/cancelled/migrated, and with the ConfigError
    exit code (77 for missing ESnet credentials, 78 otherwise) on failure.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    home = os.environ.get("HOME", "")
    if argv and argv[0] == "--home":
        if len(argv) < 2:
            print("--home requires a path", file=sys.stderr)
            return 78
        home = argv[1]
    registry_path = os.environ.get(
        "AGENT_BOX_REGISTRY",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "config", "facilities.yaml"),
    )
    try:
        registry, _ = load_registry(registry_path)
        if os.path.exists(os.path.join(home, LEGACY_CONFIG_RELATIVE_PATH)):
            print("Product configuration already present; nothing to do.")
            return 0
        result = run_first_setup(
            home=home,
            registry=registry,
            input_stream=sys.stdin,
            output_stream=sys.stdout,
            is_tty=sys.stdin.isatty(),
            environ=os.environ,
        )
        print("first-run setup status: %s" % result.status)
        return 0
    except ConfigError as exc:
        print("setup error: %s" % exc, file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
