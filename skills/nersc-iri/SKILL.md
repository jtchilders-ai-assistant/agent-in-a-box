---
name: nersc-iri
description: NERSC IRI Facility API interaction — service base, token setup with nersc-tokens, registry-selected endpoint, fail-closed boundary if disabled or unconfigured.
category: research
---

# NERSC IRI Facility API

Configuration boundary: the registry `config/facilities.yaml` is the source of truth. NERSC `iri` default version is `v2`. The selected registry version fixes the endpoint; do not invent endpoints.

- Service base: `https://api.iri.nersc.gov`
- Registry versions: `v1` (`/api/v1`), `v2` (`/api/v2`) — default `v2`
- Auth provider: `nersc-tokens`

Credential flow (separate from ALCF):

```
nersc-tokens login
```

The NERSC package (`nersc-tokens==0.1.1`, confirmed by https://github.com/NERSC/iri-api-get-globus-token) uses the command-line login flow distinct from the ALCF combined `alcf-tokens` flow. It must be completed separately.

Documented Globus parameters (from user's configuration):

- Client ID: `e4f48665-38b5-4833-a89e-849c71f5b3e3`
- Scope includes: `https://auth.globus.org/scopes/ed3e577d-f7f3-4639-b96e-ff5a8445d699/iri_api`
- Resource server: `ed3e577d-f7f3-4639-b96e-ff5a8445d699`
- Auth params include `prompt=login`

Boundary rules:
- Select endpoint only from the registry (`v1` or `v2`); do not override the base URL manually.
- If NERSC is disabled (`enabled: false` in the resolved product config), fail closed — do not call NERSC endpoints.
- Never claim the user's personal access token (PAT) is validated or authorized by this release; authentication state is handled by the external `nersc-tokens` package.
- Non-mutating verification only; no mutating operations without explicit confirmation.
