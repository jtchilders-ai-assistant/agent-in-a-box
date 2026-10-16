# Multi-Facility IRI Activation Design

## Status

Approved by Taylor Childers on 2026-10-16. This document records the implementation boundary for activating the already-developed multi-facility configuration work in Agent in a Box.

## Goal

Enable a laptop user to configure ALCF, NERSC, and ESnet IRI access through one persistent, secure product configuration while retaining ALCF-only as the zero-input default.

## Scope

The existing `feat/multi-facility-iri` branch already implements a facility registry, non-secret product configuration, configuration precedence, an interactive selection wizard, and secure ESnet credential-storage primitives. This increment activates those pieces in the packaged OCI image and closes the integration gap in the setup CLI.

## User experience

1. A fresh noninteractive volume with no facility override configures ALCF only.
2. A fresh interactive volume opens a numbered wizard. Enter selects ALCF only. Users may select NERSC and either ESnet site.
3. The wizard persists only non-secret settings in `/opt/data/agent-box/config.yaml`, mode 0600.
4. NERSC uses `nersc-tokens`; NERSC authentication is separate from ALCF authentication.
5. ESnet token capture remains hidden. It is accepted only after every selected ESnet profile validates it on a protected non-mutating route. It is stored only at `/opt/data/agent-box/credentials/esnet/amsc-token` using the existing descriptor-relative credential layer.
6. In noninteractive automation, ESnet requires `ESNET_AMSC_TOKEN_FILE`; plaintext `AMSC_TOKEN` is never consumed.

## Facility source of truth

`config/facilities.yaml` is the sole shipped authority for supported facilities, fixed HTTPS origins, API versions, defaults, and credential-provider names. `auto` resolves to the shipped tested default. It does not probe for the largest numbered live API version.

Current reviewed defaults:

- ALCF: `https://api.alcf.anl.gov/api/v1`, v1
- NERSC: `https://api.iri.nersc.gov/api/v2`, v2; v1 remains selectable
- ESnet East: `https://iri-dev.ppg.es.net/api/v2`, v2; v1 remains selectable
- ESnet West: `https://esnet-west.sdn-sense.net/api/v2`, v2; v1 remains selectable

The ESnet origins are provisional registry data until a protected read-only validation route and PAT-header contract are confirmed with ESnet. This blocks release activation of ESnet but does not weaken the fail-closed implementation.

## Activation architecture

The Docker image must install `nersc-tokens==0.1.1` alongside the existing pinned ALCF package, copy the registry/configuration/wizard/credential scripts into `/opt/alcf`, and run product setup before the existing service-specific startup flow.

`agent-box configure` must inject production adapters, not merely test hooks:

- a hidden-token reader that uses `getpass.getpass` only when a TTY exists;
- an ESnet validator adapter selected from the fixed registry profile;
- secret-free exit-77 reporting on all credential/validation failure paths.

No direct network request may accept arbitrary origin, redirect, scheme, path, query, fragment, or userinfo supplied by an environment variable or configuration file. The validator must construct its URL from the registry base URL plus a reviewed relative protected read-only route, refuse redirects, enforce HTTPS, use bounded timeouts, and never render authorization material or response bodies.

## Skills

Ship facility skills that explain how to use enabled facilities and their supported authentication paths. The skills must treat the resolved product configuration as an authorization boundary: disabled facilities are not callable. NERSC skill material must use the confirmed NERSC Globus resource-server scope, client ID, and `nersc-tokens` contract. ESnet skill material must not assert a PAT header or validator route until confirmed.

## Test and release criteria

- Test-first coverage proves setup invokes its production validator/hidden-reader adapters and fails closed if the ESnet contract is unavailable.
- Image-content tests assert new registry/setup scripts and required Python dependency are actually copied/installed.
- Mocked local HTTPS-free transport tests cover URL confinement, redirect refusal, non-2xx failure, and authorization redaction.
- Full existing Python suite passes on supported host runtimes.
- A NERSC smoke uses a user-initiated `nersc-tokens login` and a protected non-mutating endpoint; no NERSC credential is committed or logged.
- ESnet activation requires a separately documented read-only authenticated validation result from both selected profile/version combinations. Until then its wizard selection fails closed with exit 77.
- Build and test a multi-architecture OCI image before merge.

## Non-goals

- Do not discover or guess new endpoints dynamically.
- Do not embed tokens in product/Hermes YAML, child-process environment, Docker arguments, or logs.
- Do not move ALCF away from being the default.
- Do not merge the feature branch or publish a release before NERSC and ESnet acceptance evidence exists.
