# Multi-Facility IRI Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate the existing secure multi-facility configuration implementation in the Agent in a Box image while preserving ALCF-only default behavior and refusing unverified ESnet credential use.

**Architecture:** Retain `config/facilities.yaml` as the reviewed source of facility endpoints/version/auth metadata. Package the existing configuration and credential modules into the OCI image; wire the setup CLI to production-safe hidden input and a bounded ESnet validator adapter. NERSC uses the separately pinned `nersc-tokens` package. ESnet remains fail-closed until its PAT contract and protected read-only validation route are confirmed.

**Tech Stack:** Python 3, PyYAML, urllib standard library, Dockerfile, unittest/pytest, shell entrypoint.

---

## File structure

- Modify `Dockerfile`: install NERSC helper and package multi-facility assets.
- Modify `scripts/entrypoint.sh`: invoke first-run product configuration without breaking existing ALCF startup.
- Modify `scripts/agent_box.py`: inject production-only setup dependencies.
- Create `scripts/esnet_amsc_validator.py`: fixed-registry, redirect-refusing, secret-safe ESnet validation adapter.
- Modify `scripts/agent_box_setup.py`: use the adapter only when a reviewed route is configured; reject unavailable ESnet validation.
- Create `skills/nersc-iri/SKILL.md`: NERSC token and API guidance.
- Create `skills/esnet-iri/SKILL.md`: fail-closed ESnet guidance.
- Create `tests/test_esnet_amsc_validator.py`: URL confinement, redirect/refusal, response/redaction behavior.
- Modify `tests/test_agent_box_setup_review.py`: integration behavior through production adapters.
- Modify `tests/test_container_contract.py` or create it if absent: Dockerfile/entrypoint packaging contract.
- Modify `README.md` and `docs/DESIGN.md`: one-command configuration documentation and explicit ESnet qualification.

### Task 1: Establish the current test baseline and package contract

**Files:**
- Test: existing `tests/test_agent_box_*.py`
- Inspect: `Dockerfile`, `scripts/entrypoint.sh`, `tests/`

- [ ] **Step 1: Run the existing multi-facility test group**

Run:
```bash
python -m pytest -q tests/test_agent_box_config.py tests/test_agent_box_setup.py tests/test_agent_box_setup_review.py tests/test_agent_box_credentials.py tests/test_agent_box_credentials_security.py tests/test_agent_box_credentials_parent_swap.py
```

Expected: all tests pass before activation edits.

- [ ] **Step 2: Identify the actual entrypoint and current container-test conventions**

Run:
```bash
python -c "from pathlib import Path; print(Path('scripts/entrypoint.sh').exists())"
python -m pytest --collect-only -q
```

Expected: identify the package test convention before adding activation tests.

- [ ] **Step 3: Commit the approved design document**

```bash
git add docs/superpowers/specs/2026-10-16-multi-facility-activation-design.md
git commit -m "docs: specify multi-facility activation"
```

### Task 2: Define a test-first ESnet validator boundary

**Files:**
- Create: `tests/test_esnet_amsc_validator.py`
- Create: `scripts/esnet_amsc_validator.py`

- [ ] **Step 1: Write failing tests for a fixed-profile validator**

The tests must call a `validate(facility, version, token, registry)` function with an injected opener. Assert it:

```python
assert validate("esnet-east", "v2", "candidate", registry, opener=fake) == {"valid": True}
assert fake.request.full_url == "https://iri-dev.ppg.es.net/api/v2/<reviewed-read-only-route>"
assert fake.request.get_header("Authorization") == "Bearer candidate"
```

Also assert rejection of a registry origin containing HTTP, userinfo, query, fragment, or a path outside the exact `/api/vN` base; reject redirects and never include `candidate` in an exception/result.

- [ ] **Step 2: Verify RED**

Run:
```bash
python -m pytest -q tests/test_esnet_amsc_validator.py
```

Expected: FAIL because module/function does not exist.

- [ ] **Step 3: Implement the minimal adapter**

Implement only the tested function in `scripts/esnet_amsc_validator.py`: strict HTTPS origin/base-path parser, fixed relative validator route supplied by reviewed registry metadata, request with bounded timeout and no redirect handler, no response body rendering, and `{ "valid": True }` only on the expected non-mutating response.

- [ ] **Step 4: Verify GREEN**

Run:
```bash
python -m pytest -q tests/test_esnet_amsc_validator.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/esnet_amsc_validator.py tests/test_esnet_amsc_validator.py
git commit -m "feat: add confined ESnet token validator"
```

### Task 3: Make unsupported ESnet validation unavailable by policy

**Files:**
- Modify: `config/facilities.yaml`
- Modify: `scripts/agent_box.py`
- Modify: `scripts/agent_box_setup.py`
- Test: `tests/test_agent_box_setup_review.py`

- [ ] **Step 1: Write failing tests for production setup dependency injection**

Add tests demonstrating that `agent_box.main(["configure"], ...)`:

```python
assert exit_code == 77
assert "AMSC_TOKEN" not in output_and_error
assert not config_path.exists()
```

when ESnet is selected but the registry lacks a reviewed validator route. Add a separate test proving ALCF+NERSC configuration succeeds without requiring an ESnet adapter.

- [ ] **Step 2: Verify RED**

Run:
```bash
python -m pytest -q tests/test_agent_box_setup_review.py
```

Expected: FAIL because the current production CLI has no policy-aware adapter integration.

- [ ] **Step 3: Implement minimum policy wiring**

Add optional reviewed validator-route metadata to ESnet profiles. `agent_box.py` supplies `getpass.getpass` as hidden reader only in interactive mode and constructs a validator from the selected registry profile. If a selected ESnet profile has no reviewed route, setup raises fixed secret-free `ConfigError(..., 77)` before capture/persistence. Do not invent a route from the user-supplied ESnet URL.

- [ ] **Step 4: Verify GREEN**

Run:
```bash
python -m pytest -q tests/test_agent_box_setup_review.py tests/test_esnet_amsc_validator.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config/facilities.yaml scripts/agent_box.py scripts/agent_box_setup.py tests/test_agent_box_setup_review.py
git commit -m "feat: wire fail-closed ESnet setup validation"
```

### Task 4: Package facility configuration and NERSC authentication

**Files:**
- Modify: `Dockerfile`
- Modify: `scripts/entrypoint.sh`
- Modify/Create: container contract test

- [ ] **Step 1: Write failing package-contract tests**

Assert the Dockerfile installs the exact `nersc-tokens==0.1.1` release and copies `config/facilities.yaml`, `agent_box.py`, `agent_box_config.py`, `agent_box_setup.py`, `agent_box_credentials.py`, and `esnet_amsc_validator.py` to `/opt/alcf`. Assert entrypoint invokes product setup before the dashboard and remains noninteractive-safe.

- [ ] **Step 2: Verify RED**

Run the targeted package test. Expected: FAIL because these assets are not copied and the package is not installed.

- [ ] **Step 3: Make the minimal Dockerfile/entrypoint changes**

Install the pinned NERSC package in the Hermes venv. Copy the registry and helper closure. Invoke `agent-box` setup with durable `$HERMES_HOME` before service-specific authentication. Preserve ALCF-only behavior on fresh no-TTY volumes and ensure an ESnet exit-77 failure stops startup without launching descendants.

- [ ] **Step 4: Verify GREEN**

Run the package test and shell syntax check:

```bash
bash -n scripts/entrypoint.sh
python -m pytest -q <container-contract-test>
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile scripts/entrypoint.sh tests/<container-contract-test>
git commit -m "feat: package multi-facility setup"
```

### Task 5: Add facility skills and user documentation

**Files:**
- Create: `skills/nersc-iri/SKILL.md`
- Create: `skills/esnet-iri/SKILL.md`
- Modify: `README.md`
- Modify: `docs/DESIGN.md`

- [ ] **Step 1: Write documentation contract tests or assertions**

Assert README commands retain `--pull=always`, loopback-only publication, volume persistence, ALCF-only default, `AGENT_BOX_FACILITIES`, and state that ESnet is unavailable until a verified PAT validation contract is shipped. Assert no documentation tells users to pass `AMSC_TOKEN` directly.

- [ ] **Step 2: Verify RED**

Run the targeted documentation test. Expected: FAIL until onboarding is updated.

- [ ] **Step 3: Write skills and docs**

NERSC skill: document `nersc-tokens login`, NERSC IRI scope/resource server, configuration selection, and non-mutating verification. ESnet skill: document the facility selection boundary, file-based token policy, explicit unavailability while validator metadata is absent, and no PAT header claim. Update README and DESIGN accordingly.

- [ ] **Step 4: Verify GREEN**

Run targeted documentation tests. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add skills/nersc-iri/SKILL.md skills/esnet-iri/SKILL.md README.md docs/DESIGN.md tests/<documentation-contract-test>
git commit -m "docs: document multi-facility configuration"
```

### Task 6: Full verification and independent review

**Files:**
- Verify: all changed files

- [ ] **Step 1: Run the full Python suite and static checks**

```bash
python -m pytest -q
python -m compileall -q scripts
bash -n scripts/entrypoint.sh
git diff --check main...HEAD
```

Expected: all pass.

- [ ] **Step 2: Run a local no-TTY first-run smoke**

Use a temporary `HOME` and invoke the packaged setup CLI without facility overrides. Assert a config exists, ALCF is enabled, NERSC/ESnet are disabled, and no credential file exists.

- [ ] **Step 3: Run NERSC acceptance only with user-initiated authentication**

After the user completes `nersc-tokens login`, invoke a protected non-mutating NERSC endpoint. Record only status/method/version, never tokens.

- [ ] **Step 4: Do not claim ESnet complete without acceptance evidence**

Require the ESnet team to provide the PAT header semantics and protected read-only validator route. Run it against both selected profiles and versions before changing registry metadata to activate ESnet.

- [ ] **Step 5: Request independent review and remediate Critical/Important findings**

Review `main...HEAD` against this plan. Fix all Critical/Important findings, rerun the full suite, and repeat review when code changes materially.
