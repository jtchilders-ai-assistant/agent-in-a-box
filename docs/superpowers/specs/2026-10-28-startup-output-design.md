# Startup Output UX Design

## Goal

Make `agent-in-a-box` startup output immediately actionable: users should reliably find the browser URL and login instructions, while diagnostics remain available on demand.

## Scope

The image entrypoint (`scripts/entrypoint.sh`) accepts optional `-v` / `--verbose` arguments. The documented container command may append this flag after the image name, for example:

```bash
docker run … ghcr.io/jtchilders-ai-assistant/alcf-agent:latest --verbose
```

Unknown entrypoint arguments fail before startup with a clear usage message and a nonzero exit code.

## Output Modes

### Default

Default output contains only concise user-facing milestones:

- a short product/version line;
- any interactive Globus-authentication prompts required to proceed;
- a short startup milestone; and
- the final READY card.

Errors, including token refresh failure, remain visible regardless of mode.

Routine implementation diagnostics are suppressed from the terminal: config file paths, provider/model rendering detail, catalog/availability reports, seed/refresh detail, and internal listener addresses.

These diagnostics are appended to a durable startup log under `$HERMES_HOME` so they can be inspected after launch.

### Verbose

`-v` and `--verbose` preserve the existing detailed startup information while also writing it to the startup log. The final READY card remains present and visually distinct.

## Ready Card

After Caddy is ready, print a bounded, high-contrast card containing:

- `READY — OPEN THIS URL` with `http://localhost:<public-port>`;
- dashboard username; and
- an instruction to use the dashboard password supplied/generated during startup.

Do not print a user-supplied password in this card. Existing first-run generated-password handling remains unchanged.

## Implementation

1. Parse entrypoint arguments before service startup and establish a `VERBOSE` boolean.
2. Replace the current single `log` behavior with separate user-milestone and diagnostic log helpers. Diagnostics append to `$HERMES_HOME/startup.log`; verbose mode additionally mirrors them to the terminal.
3. Redirect/route the model availability report and existing routine diagnostic calls through the diagnostic helper.
4. Preserve direct interactive commands and error output; they must remain visible in default mode.
5. Replace the current URL lines with the READY card.
6. Document default and verbose behavior in the README.

## Tests

- Entry-point argument parser accepts `-v` and `--verbose` and rejects unknown arguments.
- Default mode keeps the READY card and hides diagnostic lines.
- Verbose mode emits diagnostic lines and retains the READY card.
- Errors remain terminal-visible with verbose disabled.
- README includes the verbose invocation and identifies the startup log.

## Acceptance Criteria

A default user can identify the exact browser URL without scanning diagnostic output. A user who needs diagnostics can append `--verbose` or inspect the persisted startup log. Existing authentication, error, and dashboard behavior remains intact.
