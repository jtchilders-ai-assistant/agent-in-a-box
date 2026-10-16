---
name: esnet-iri
description: ESnet IRI Facility API interaction — east/west selection, version endpoint table, PAT file/input boundary, no AMSC_TOKEN direct reference, unverified state acknowledged.
category: research
---

# ESnet IRI Facility API

Facility selection boundary: choose either `esnet-east`, `esnet-west`, or both via the registry/config. Each has versions `v1` and `v2` (default `v2`). The registry endpoint table:

| Facility | v1 base | v2 base |
|---|---|---|---|
| East | `esnet-east` | `https://iri-dev.ppg.es.net/api/v1` | `https://iri-dev.ppg.es.net/api/v2` |
| West | `esnet-west` | `https://esnet-west.sdn-sense.net/api/v1` | `https://esnet-west.sdn-sense.net/api/v2` |

Auth provider: `amsc-pat` (AmSC PAT based). Boundary rules:

- Token must come from `ESNET_AMSC_TOKEN_FILE` (safe file path) or from secure interactive capture — never from `AMSC_TOKEN` plaintext environment variable.
- Do not claim the user's direct PAT is validated or authorized at release; validation depends on a reviewed validator route that is not yet active.
- Fail closed if ESnet is selected but no safe token file exists and no validated route is configured.
- Facility disabled (`enabled: false`) = fail closed for ESnet endpoints.
- Interactive flow securely prompts/captures per existing `agent_box_setup.py` code; do not expose token in output, exceptions, or configuration.
