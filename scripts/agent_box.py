"""agent-box management CLI — inspect and validate the product configuration.

Current commands (first-run configuration logic lives in the separate
first-run-setup entry point, agent_box_setup, which `configure` delegates to):

    agent-box configure         — run the stream-injected first-run wizard
    agent-box config show       — print the persisted facility configuration
    agent-box config validate   — validate the persisted configuration

Design constraints:
- All I/O is stream-injected (output_stream/error_stream) and the home
  directory and registry path are explicit parameters, so the CLI is fully
  testable without touching the real user home.
- Output is secret-free by construction: it renders only facility names,
  enabled booleans, and IRI versions from the product config, which itself
  is validated to carry no credential material.  This module never opens,
  reads, or references credential files.
- Facility listing follows the shipped registry's insertion order (the
  documented wizard order).
"""
import os
import sys

from agent_box_config import (
    ConfigError,
    load_product_config,
    load_registry,
)

USAGE = (
    "usage: agent-box configure | agent-box config show "
    "| agent-box config validate "
    "| agent-box credentials esnet configure"
)


def _config_show(home, registry_path, output_stream):
    registry, _ = load_registry(registry_path)
    config_path = os.path.join(home, "agent-box", "config.yaml")
    data = load_product_config(config_path, registry=registry)
    facilities = data["facilities"]
    output_stream.write("Product configuration: %s\n" % config_path)
    output_stream.write("schema_version: %s\n" % data["schema_version"])
    output_stream.write("facilities (registry order):\n")
    for facility in registry:
        spec = facilities.get(facility)
        if spec is None:
            output_stream.write("  %s: not configured\n" % facility)
            continue
        enabled = spec.get("enabled", False)
        iri_spec = spec.get("iri")
        version = None
        if isinstance(iri_spec, dict):
            version = iri_spec.get("version")
        if version is None:
            version = spec.get("version")
        if version is None:
            version = registry[facility]["default_version"]
        output_stream.write(
            "  %s: enabled=%s iri_version=%s\n" % (facility, enabled, version)
        )
    return 0


def _config_validate(home, registry_path, output_stream):
    registry, _ = load_registry(registry_path)
    config_path = os.path.join(home, "agent-box", "config.yaml")
    load_product_config(config_path, registry=registry)
    output_stream.write("product configuration valid: %s\n" % config_path)
    return 0


def main(argv, home, registry_path, output_stream, error_stream,
         input_stream=None, is_tty=None, environ=None):
    """Entry point for the agent-box management CLI.

    Returns a process exit code (0 on success; ConfigError's exit_code —
    77/78 — on failure; 2 on usage errors).  Never raises ConfigError to
    callers.

    `configure` delegates to the first-run flow in agent_box_setup.  Its
    interactivity is fully injectable (input_stream/is_tty/environ); when
    not injected it falls back to sys.stdin, its isatty() state, and
    os.environ.  `config show`/`config validate` keep their existing API.
    """
    argv = list(argv)
    if input_stream is None:
        input_stream = sys.stdin
    if is_tty is None:
        is_tty = input_stream.isatty()
    if environ is None:
        environ = os.environ
    if not argv or argv[0] in ("-h", "--help"):
        output_stream.write(USAGE + "\n")
        return 0 if argv else 2
    try:
        if argv[0] == "configure":
            # First-run wizard (wizard / migration / non-interactive default)
            # lives in agent_box_setup; import lazily so config subcommands
            # keep a light import surface.
            from agent_box_setup import run_first_setup

            registry, _ = load_registry(registry_path)
            result = run_first_setup(
                home=home,
                registry=registry,
                input_stream=input_stream,
                output_stream=output_stream,
                is_tty=is_tty,
                environ=environ,
            )
            output_stream.write("configure status: %s\n" % result.status)
            return 0
        if argv[0] == "credentials":
            # ESnet AmSC credential reconfiguration.  Interactive-only:
            # the hidden-reader capture and confirmation flow require a
            # TTY, and it fails closed (77) without one.  Streams stay
            # injected; nothing secret is ever rendered.
            if len(argv) < 3 or argv[1] != "esnet" or argv[2] != "configure":
                error_stream.write(USAGE + "\n")
                return 2
            if not is_tty:
                error_stream.write(
                    "agent-box: credentials esnet configure requires an "
                    "interactive terminal\n"
                )
                return 77
            from agent_box_setup import reconfigure_amsc_credential_interactive

            status = reconfigure_amsc_credential_interactive(
                home,
                input_stream=input_stream,
                output_stream=output_stream,
            )
            output_stream.write("credential reconfiguration status: %s\n"
                                % status)
            return 0
        if argv[0] == "config":
            if len(argv) < 2:
                error_stream.write(USAGE + "\n")
                return 2
            if argv[1] == "show":
                return _config_show(home, registry_path, output_stream)
            if argv[1] == "validate":
                return _config_validate(home, registry_path, output_stream)
            error_stream.write("unknown config subcommand: %s\n%s\n"
                               % (argv[1], USAGE))
            return 2
        error_stream.write("unknown command: %s\n%s\n" % (argv[0], USAGE))
        return 2
    except ConfigError as exc:
        error_stream.write("agent-box: %s\n" % exc)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main(
        sys.argv[1:],
        home=os.environ.get("HOME", ""),
        registry_path=os.environ.get(
            "AGENT_BOX_REGISTRY",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "config", "facilities.yaml")),
        output_stream=sys.stdout,
        error_stream=sys.stderr,
    ))
