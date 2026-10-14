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
  no secrets.
- Interactive ESnet (amsc-pat) selection with no safe token already
  available prompts exactly once through a hidden reader (Task 3
  ``agent_box_credentials``), validates the candidate independently for
  every selected site/version via an injected validator *before*
  persisting anything, and fails closed with ConfigError exit code 77
  — persisting nothing, surfacing no secret — when validation fails, the
  validator raises, or no validator callback is supplied.
- Non-interactive ESnet selection keeps the Task 2 behavior: it requires
  a *safe* ``ESNET_AMSC_TOKEN_FILE`` or a *safe* persisted token path
  and otherwise fails closed with ConfigError exit code 77.  It never
  prompts.
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
    load_product_config,
    load_registry,
    parse_env,
    resolve_configuration,
)
from agent_box_credentials import (
    CredentialError,
    prompt_amsc_token,
    reconfigure_amsc_token,
    write_amsc_token,
)

# Relative path of the persistent product configuration under the supplied home.
LEGACY_CONFIG_RELATIVE_PATH = os.path.join("agent-box", "config.yaml")

# Approved-design location for a persisted ESnet AMSC token file, relative to
# the supplied home.  The wizard only ever stats this path via
# _is_safe_token_path; reading and writing the token goes through
# agent_box_credentials (Task 3).
PERSISTED_AMSC_TOKEN_RELATIVE_PATH = os.path.join(
    "agent-box", "credentials", "esnet", "amsc-token"
)

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
    """Existence/stat-only safety check via lstat — never follows symlinks.

    Safe means: absolute, an *actual* regular file (lstat, so a symlink —
    even to a safe-looking target — is rejected), owned by the current
    user, and not readable or writable by group or others.  The file is
    never opened or read.
    """
    if not isinstance(path, str) or not path:
        return False
    if not os.path.isabs(path):
        return False
    try:
        st = os.lstat(path)
    except OSError:
        return False
    import stat as _stat

    if _stat.S_ISLNK(st.st_mode) or not _stat.S_ISREG(st.st_mode):
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


def _esnet_selections(registry, selected, versions):
    """(facility, version) pairs for selected amsc-pat facilities."""
    return [
        (facility, versions[facility])
        for facility in selected
        if _facility_requires_amsc(registry, facility)
    ]


def _capture_and_validate_amsc_token(home, selections, hidden_reader,
                                     amsc_validator, output_stream):
    """Interactive ESnet credential flow (Task 3).

    Prompts exactly once through the hidden reader, then validates the
    candidate *independently* for every selected (facility, version)
    BEFORE persisting anything.  A failed validation, a validator
    exception, or a missing validator fails closed with ConfigError
    exit code 77, persists nothing, and surfaces no secret.
    """
    if amsc_validator is None:
        # No network validation client yet and none injected: fail
        # closed rather than persist an unvalidated credential.
        raise ConfigError(
            "ESnet facility selected but no AMSC token validator is "
            "available; refusing to store an unvalidated token.",
            77,
        )
    output_stream.write(
        "An ESnet facility was selected. You will be asked for your "
        "AmSC token once; it is validated before it is stored.\n"
    )
    try:
        token = prompt_amsc_token(hidden_reader=hidden_reader,
                                  output_stream=output_stream)
    except (CredentialError, EOFError, KeyboardInterrupt):
        # Never quote the candidate: the message stays secret-free.
        raise ConfigError(
            "ESnet AmSC token capture was cancelled or produced an empty "
            "token; nothing was stored.",
            77,
        )
    for facility, version in selections:
        try:
            result = amsc_validator(facility, version, token)
        except Exception:
            # Suppress the validator's exception entirely: it could echo
            # call arguments.  Surface a secret-free failure instead.
            raise ConfigError(
                "ESnet AmSC token validation failed for %s (IRI version "
                "%s): the validator raised; nothing was stored."
                % (facility, version),
                77,
            )
        if not (isinstance(result, dict) and result.get("valid") is True):
            raise ConfigError(
                "ESnet AmSC token validation failed for %s (IRI version "
                "%s); nothing was stored." % (facility, version),
                77,
            )
    write_amsc_token(home, token)
    output_stream.write("ESnet AmSC token validated and stored privately "
                        "at %s\n"
                        % os.path.join(home, PERSISTED_AMSC_TOKEN_RELATIVE_PATH))


def _ensure_amsc_credentials_interactive(home, registry, selected, versions,
                                         environ, hidden_reader,
                                         amsc_validator, output_stream):
    """Resolve ESnet credentials for the interactive wizard path.

    Honors an existing *safe* env/persisted token without re-prompting;
    otherwise runs the one-prompt validate-then-persist flow.
    """
    selections = _esnet_selections(registry, selected, versions)
    if not selections:
        return
    env_path = environ.get(AMSC_TOKEN_FILE_ENV)
    if env_path is not None and _is_safe_token_path(env_path.strip()):
        return
    persisted = os.path.join(home, PERSISTED_AMSC_TOKEN_RELATIVE_PATH)
    if _is_safe_token_path(persisted):
        return
    _capture_and_validate_amsc_token(home, selections, hidden_reader,
                                     amsc_validator, output_stream)


def reconfigure_amsc_credential(home, choice, hidden_reader=None,
                                output_stream=None):
    """Expose keep/replace/remove reconfiguration of a persisted token.

    Thin delegation to ``agent_box_credentials.reconfigure_amsc_token``
    so a configure flow can drive credential changes without importing
    the credential module directly.  Returns the resulting
    ``token_status`` value ('absent' | 'present' | 'unsafe').
    """
    return reconfigure_amsc_token(home, choice, hidden_reader=hidden_reader,
                                  output_stream=output_stream)


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


def _legacy_token_evidence(home):
    """Existence-only detection of a legacy ALCF install. Never reads content."""
    for relative in _LEGACY_CANDIDATE_RELATIVE_PATHS:
        if os.path.exists(os.path.join(home, relative)):
            return True
    return False


def run_first_setup(home, registry, input_stream, output_stream, is_tty,
                    environ, hidden_reader=None, amsc_validator=None):
    """Run first-run setup: legacy migration, non-interactive default, or wizard.

    Facility iteration and wizard numbering follow the shipped registry's
    insertion order (config/facilities.yaml), which is the documented
    wizard display order.

    ``hidden_reader`` and ``amsc_validator`` drive the interactive ESnet
    credential flow (see ``_capture_and_validate_amsc_token``); the
    wizard fails closed with exit 77 without a validator.  They are
    unused on the migration and non-interactive paths, which keep the
    safe-token-file requirement.

    Writes nothing unless the outcome is an applied configuration
    (plus, for a validated interactive ESnet selection, the private
    AMSC token file).
    """
    config_path = os.path.join(home, LEGACY_CONFIG_RELATIVE_PATH)
    if os.path.exists(config_path) or _legacy_token_evidence(home):
        # Migration path — no prompting in either sub-case.
        persisted_facilities = None
        configuration = None
        status = "migrated"
        if os.path.exists(config_path):
            # Product config already exists: read and parse it so the
            # operator's persisted enabled/version state survives migration.
            parsed = load_product_config(config_path, registry=registry)
            persisted_facilities = parsed["facilities"]
            configuration = parsed
            output_stream.write(
                "Existing product configuration detected; preserving "
                "persisted facility state.\n"
            )
        else:
            # Only legacy token evidence exists: durably seed the
            # ALCF-default product config (atomic write, private paths).
            output_stream.write(
                "Existing ALCF install detected; writing ALCF-default "
                "product configuration.\n"
            )
            default_selected = ["alcf"]
            versions = {
                f: registry[f]["default_version"] for f in default_selected
            }
            configuration = _build_product_config(
                registry, default_selected, versions
            )
            _write_product_config(home, configuration)
            persisted_facilities = configuration["facilities"]
        # Environment overrides stay highest precedence during migration:
        # resolve_configuration applies AGENT_BOX_* on top of persisted state.
        return SetupResult(
            status=status,
            configuration=configuration,
            resolved=resolve_configuration(
                registry, persisted=persisted_facilities, environ=environ
            ),
        )

    env_facilities = environ.get("AGENT_BOX_FACILITIES")
    if env_facilities is not None or not is_tty:
        # Explicit selection (AGENT_BOX_FACILITIES) is honored with or
        # without a TTY and skips the wizard without reading input.
        # Non-interactive with no explicit selection: ALCF-only default.
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
    output_stream.write("Apply this configuration? [y/N] ")
    answer = _read_line(input_stream)
    if answer is None or answer.lower() not in ("y", "yes"):
        output_stream.write("Setup cancelled. Nothing was written.\n")
        return SetupResult(status="cancelled", configuration=None, resolved=None)
    # Post-confirm credential step: prompts once (hidden), validates per
    # selected site/version, and fails closed (exit 77) without
    # persisting anything on failure.  No-op when no ESnet facility is
    # selected or a safe token is already available.
    _ensure_amsc_credentials_interactive(
        home, registry, selected, versions, environ, hidden_reader,
        amsc_validator, output_stream,
    )
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
    home = None
    if argv and argv[0] == "--home":
        if len(argv) < 2:
            print("--home requires a path", file=sys.stderr)
            return 78
        home = argv[1]
    else:
        # Fail closed: no HOME and no --home means nowhere safe to read or
        # write persistent configuration.
        home = os.environ.get("HOME")
        if not home:
            print(
                "setup error: HOME is not set and no --home was supplied; "
                "refusing to run",
                file=sys.stderr,
            )
            return 78
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
