"""agent-box management CLI — inspect and validate the product configuration.

Current commands (first-run configuration lives in the separate
first-run-setup entry point, agent_box_setup):

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

USAGE = "usage: agent-box config show | agent-box config validate"


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


def main(argv, home, registry_path, output_stream, error_stream):
    """Entry point for the agent-box management CLI.

    Returns a process exit code (0 on success; ConfigError's exit_code —
    77/78 — on failure).  Never raises ConfigError to callers.
    """
    argv = list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        output_stream.write(USAGE + "\n")
        return 0 if argv else 2
    try:
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
