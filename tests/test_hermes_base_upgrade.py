#!/usr/bin/env python3
"""Contract tests for the Hermes v0.23.2 base-image upgrade (plan v2 Phase B).

Static checks (Dockerfile/README/template/populate_models/entrypoint) run with
the product interpreter. The runtime checks run upstream Hermes' own config
validation, named-provider resolution and Chat Completions request builder in a
SUBPROCESS with the upstream checkout's interpreter, so the product test venv
never imports Hermes. Set HERMES_UPSTREAM_CHECKOUT to the v0.23.2 checkout;
without it those tests skip with an explicit reason.

All tokens here are fake. Nothing touches the network: the outgoing request is
captured by an httpx.MockTransport inside the upstream harness.

Run:  python3 -m unittest tests.test_hermes_base_upgrade -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock
import urllib.error

import yaml

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
SCRIPTS = ROOT / "scripts"
for _p in (str(SCRIPTS), str(TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import populate_models as pm  # noqa: E402
import test_generic_inference as tgi  # noqa: E402

EXPECTED_BASE = "nousresearch/hermes-agent:v0.23.2"
FLOATING_TAG = re.compile(r"(?:^|[-.])(latest|main|canary|nightly|dev|edge|rc\d*)(?:$|[-.])", re.I)
PATCH = ROOT / "patches" / "0001-strip-tool-message-name.patch"
FAKE_TOKEN = "fake-alcf-test-token"
UPSTREAM_ENV = "HERMES_UPSTREAM_CHECKOUT"
SENTINEL_RE = re.compile(r"^#__ALCF_[A-Z_]+__", re.M)

# Phrases that mark a README mention of the retired patch as history, not as the
# current state of the image. Matched as whole words, within the SAME sentence
# as the mention (or in the enclosing heading), never anywhere on the line.
HISTORY_MARKERS = ("no longer", "removed", "retired", "previously", "formerly", "obsolete",
                   "historical", "history")
PATCH_MENTION = re.compile(r"strip_tool_message_name|0001-strip-tool-message-name")
_HISTORY_RE = re.compile(r"\b(?:" + "|".join(re.escape(m) for m in HISTORY_MARKERS) + r")\b", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# --- helpers -------------------------------------------------------------------

def dockerfile_base_arg() -> str:
    """Default value of `ARG HERMES_BASE=...` (parsed, not substring-matched)."""
    values = []
    for raw in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*ARG\s+HERMES_BASE\s*=\s*(\S+)\s*$", raw)
        if m:
            values.append(m.group(1).strip("\"'"))
    if len(values) != 1:
        raise AssertionError(f"expected exactly one ARG HERMES_BASE default, found {values}")
    return values[0]


def readme_component_base() -> str:
    """Source cell of the 'Hermes Agent' row of the README components table."""
    rows = []
    in_table = False
    for raw in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in raw.strip().strip("|").split("|")] if raw.startswith("|") else None
        if cells and cells[:3] == ["Piece", "Source", "Purpose"]:
            in_table = True
            continue
        if in_table:
            if cells is None:
                break
            if set("".join(cells)) <= set("-: "):
                continue
            rows.append(cells)
    hermes = [r for r in rows if r[0].lower().startswith("hermes agent")]
    if len(hermes) != 1:
        raise AssertionError(f"expected one 'Hermes Agent' row in README components table, got {hermes}")
    return hermes[0][1].strip("`")


def readme_current_patch_mentions(text: str) -> list:
    """README mentions of the retired patch that are NOT framed as history.

    A mention is history only if a history marker appears as a whole word in
    the same sentence, or in the heading of the section it sits in. A marker
    elsewhere on the line does not count, and a table row always describes the
    current image, so a mention there is never history.
    """
    current, heading = [], ""
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            heading = line
        if not PATCH_MENTION.search(line):
            continue
        if line.startswith("|"):
            current.append(line)
            continue
        if _HISTORY_RE.search(heading) and not line.startswith("#"):
            continue
        current += [s for s in _SENTENCE_SPLIT.split(line)
                    if PATCH_MENTION.search(s) and not _HISTORY_RE.search(s)]
    return current


def assert_provider_schema(tc: unittest.TestCase, doc, *, where: str) -> dict:
    """The v0.23.2 keyed `providers:` contract shared by every emitter."""
    tc.assertIsInstance(doc, dict, f"{where}: not a YAML mapping")
    tc.assertFalse("custom_providers" in doc,
                   f"{where}: still emits the legacy custom_providers list")
    tc.assertTrue("providers" in doc, f"{where}: no root `providers` key")
    return assert_provider_entries(tc, doc["providers"], where=where)


def assert_provider_entries(tc: unittest.TestCase, providers, *, where: str) -> dict:
    """The keyed entries that live under `providers:` ({key: entry})."""
    tc.assertIsInstance(providers, dict, f"{where}: `providers` must be a mapping, not a list")
    tc.assertTrue(providers, f"{where}: `providers` is empty")
    for key, entry in providers.items():
        ctx = f"{where}: providers.{key}"
        tc.assertIsInstance(entry, dict, ctx)
        tc.assertEqual(entry.get("name"), key, f"{ctx}: name must equal the provider key")
        tc.assertIn("api", entry, f"{ctx}: endpoint must use the `api` key")
        tc.assertNotIn("base_url", entry, f"{ctx}: use `api`, not base_url")
        tc.assertIs(entry.get("discover_models"), False, f"{ctx}: discover_models must be false")
        for banned in ("max_tokens", "max_output_tokens"):
            tc.assertNotIn(banned, entry,
                           f"{ctx}: provider-level {banned} is ignored by Hermes v0.23.2")
        extra = entry.get("extra_body")
        tc.assertIsInstance(extra, dict, f"{ctx}: output cap must live in extra_body")
        tc.assertIsInstance(extra.get("max_tokens"), int, f"{ctx}: extra_body.max_tokens")
        tc.assertGreater(extra["max_tokens"], 0, ctx)
        models = entry.get("models")
        tc.assertIsInstance(models, dict, f"{ctx}: models must be a non-empty mapping")
        tc.assertTrue(models, f"{ctx}: empty class must be omitted, not emitted")
        for mid, meta in models.items():
            tc.assertIsInstance(meta, dict, f"{ctx}.models.{mid}")
            tc.assertIsInstance(meta.get("context_length"), int, f"{ctx}.models.{mid}.context_length")
    return providers


def assert_cap_split(tc: unittest.TestCase, providers: dict, *, where: str) -> None:
    """Baseline providers carry BASELINE_MAX_TOKENS, -reasoning ones REASONING_MAX_TOKENS."""
    for key, entry in providers.items():
        want = pm.REASONING_MAX_TOKENS if key.endswith("-reasoning") else pm.BASELINE_MAX_TOKENS
        tc.assertEqual(entry["extra_body"]["max_tokens"], want, f"{where}: providers.{key}")


def fallback_as_providers(fallbacks) -> dict:
    """Expected {provider: {model: ctx}} for populate_models' static fallbacks."""
    out: dict = {}
    for _cluster, provider, _base, _sel, fallback in fallbacks:
        for mid, (ctx, is_reasoning) in fallback.items():
            out.setdefault(provider + ("-reasoning" if is_reasoning else ""), {})[mid] = ctx
    return out


def image_model_env(**overrides) -> dict:
    """The image's ALCF_CLUSTER / ALCF_MODEL ENV defaults as `docker run` sees
    them, plus `-e` overrides; ALCF_MODEL is empty (unset) unless one sets it."""
    env = {"ALCF_MODEL": ""}
    for instr in dockerfile_instructions():
        if instr.startswith("ENV "):
            for tok in instr.split()[1:]:
                key, _, val = tok.partition("=")
                if key in ("ALCF_CLUSTER", "ALCF_MODEL"):
                    env[key] = val
    env.update(overrides)
    return env


def template_fallback_doc() -> dict:
    """The committed fallback: the template from its sentinel line to EOF."""
    tmpl = (ROOT / "config" / "config.template.yaml").read_text(encoding="utf-8")
    m = SENTINEL_RE.search(tmpl)
    if not m:
        raise AssertionError("config.template.yaml has no #__ALCF_..._ sentinel line")
    return yaml.safe_load(tmpl[m.start():])


def entrypoint_render_program() -> str:
    """The Python render program embedded in entrypoint.sh's render_config."""
    body = (SCRIPTS / "entrypoint.sh").read_text(encoding="utf-8")
    start = body.index('"$ALCF_DIR/config.template.yaml" "$CONFIG_OUT" <<\'PYEOF\'\n')
    start = body.index("\n", start) + 1
    return body[start:body.index("\nPYEOF\n", start)] + "\n"


def render_alcf_config(providers_block: str, *, launch_provider: str, model: str) -> str:
    """Render the template exactly as entrypoint.sh does (splice + substitute).
    ALCF mode renders model.api_key EMPTY: providers mint their token via key_cmd."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "config.yaml"
        env = dict(os.environ,
                   ALCF_ACCESS_TOKEN="",
                   ALCF_BASE_URL="https://h/v1",
                   ALCF_MODEL=model,
                   ALCF_LAUNCH_PROVIDER=launch_provider,
                   ALCF_MAX_TOKENS=str(pm.BASELINE_MAX_TOKENS),
                   ALCF_CONTEXT_LENGTH="128000",
                   ALCF_DASHBOARD_USER="alcf",
                   ALCF_DASHBOARD_PASSWORD_HASH="test-password-hash",
                   ALCF_PROVIDERS_BLOCK=providers_block,
                   ALCF_EXTRA_CONFIG_BLOCK="")
        proc = subprocess.run(
            [sys.executable, "-", str(ROOT / "config" / "config.template.yaml"), str(out)],
            input=entrypoint_render_program(), text=True, capture_output=True, env=env, timeout=30)
        if proc.returncode != 0:
            raise AssertionError(f"entrypoint render program failed: {proc.stderr}")
        return out.read_text(encoding="utf-8")


TEST_CLUSTER = [("x", "alcf-x", "https://h/v1", None,
                 {"chatty": (128000, False), "thinky": (262144, True)})]


def build_test_block() -> str:
    """populate_models.build_block for one known cluster, fully offline, with
    the default key_cmd binary."""
    with mock.patch.dict(os.environ), \
            mock.patch.object(pm, "_get_token", side_effect=RuntimeError("offline test")), \
            mock.patch.object(pm, "_clusters", return_value=TEST_CLUSTER), \
            contextlib.redirect_stderr(io.StringIO()):
        os.environ.pop("ALCF_TOKENS_BIN", None)
        return pm.build_block()


# --- upstream runtime harness ---------------------------------------------------

UPSTREAM_HARNESS = textwrap.dedent(r'''
    """Run upstream Hermes against a rendered config; print one JSON evidence blob."""
    import json, logging, os, socket, sys, traceback

    # Hard network deny: models.dev lookups / endpoint probes inside upstream
    # (e.g. switch_model's capability fetch) must fail fast, never leave the box.
    def _no_network(*_a, **_k):
        raise OSError("network disabled in contract harness")
    socket.socket.connect = _no_network
    socket.socket.connect_ex = _no_network
    socket.create_connection = _no_network
    socket.getaddrinfo = _no_network

    config_path, checks = sys.argv[1], json.loads(sys.argv[2])
    switches = json.loads(sys.argv[3]) if len(sys.argv) > 3 else []
    ev = {"harness_error": None, "validate_issues": [], "unknown_key_warnings": [],
          "providers": {}, "switches": [], "tool_message": None}

    def fail(stage):
        ev["harness_error"] = f"{stage}: {traceback.format_exc(limit=3)}"
        print(json.dumps(ev, default=str))
        sys.exit(0)

    try:
        import hermes_yaml
        import httpx, openai
        from hermes_cli.config import validate_config_structure
        from hermes_cli.config_providers import get_compatible_custom_providers
        from hermes_cli.runtime_provider import frozen_runtime_config, resolve_runtime_provider
        from agent.transports.chat_completions import ChatCompletionsTransport
    except Exception:
        fail("import")

    try:
        with open(config_path, encoding="utf-8") as fh:
            cfg = hermes_yaml.safe_load(fh)
    except Exception:
        fail("load_config")
    ev["root_keys"] = sorted(cfg) if isinstance(cfg, dict) else None

    class _Grab(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if "unknown config keys" in msg:
                ev["unknown_key_warnings"].append(msg)
            if "config migration" in msg:
                ev["migration_warnings"].append(msg)
    ev["migration_warnings"] = []
    logging.getLogger().addHandler(_Grab(level=logging.WARNING))

    # The boot-time schema check upstream runs (post_update.step_migrate_config):
    # a current stamp is a no-op "up-to-date"; a missing one reads as version 0,
    # below the support floor, and logs "config migration skipped" every boot.
    try:
        from hermes_cli.config_version_stamp import read_config_version_stamp
        from hermes_cli.post_update import step_migrate_config
        stamp, latest = read_config_version_stamp()
        ev["config_version"] = {"stamp": stamp, "latest": latest,
                                "migrate": step_migrate_config()}
    except Exception:
        fail("config_version")

    try:
        ev["validate_issues"] = [{"severity": str(getattr(i, "severity", "")),
                                  "message": str(getattr(i, "message", i))}
                                 for i in validate_config_structure(cfg)]
        get_compatible_custom_providers(cfg)
    except Exception:
        fail("validate_config_structure")

    transport = ChatCompletionsTransport()
    seen = []

    # A key_cmd provider resolves to a CALLABLE api_key (CommandTokenSource)
    # that both wire clients invoke per request. Call it the way they do and
    # compare the minted value; record whether it was callable. Mint errors are
    # reduced to their class name (never a token).
    def _credential(out, value, fake_token):
        out["api_key_callable"] = callable(value)
        if callable(value):
            try:
                value = value()
            except Exception as e:
                out["api_key_mint_error"] = type(e).__name__
                value = None
        out["api_key_is_fake"] = value == fake_token

    def _handler(request):
        seen.append(json.loads(request.content or b"{}"))
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}]})

    def _wire(out, model, base_url, provider_name, request_overrides, stage):
        try:
            kwargs = transport.build_kwargs(
                model, [{"role": "user", "content": "hi"}],
                base_url=base_url, provider_name=provider_name,
                request_overrides=request_overrides)
            out["api_kwargs_extra_body"] = kwargs.get("extra_body")
            seen.clear()
            client = openai.OpenAI(api_key="fake-key", base_url=base_url or "http://x/v1",
                                   http_client=httpx.Client(transport=httpx.MockTransport(_handler)))
            client.chat.completions.create(**kwargs)
            body = seen[-1]
            out["wire_max_tokens"] = body.get("max_tokens")
            out["wire_keys"] = sorted(body)
        except Exception:
            fail(stage)

    # The main agent loop's request with agent.reasoning_effort UNSET, built the
    # way chat_completion_helpers._build_chat_completions_kwargs does for a
    # provider with a registered profile: reasoning_config is
    # reasoning_params.unset_reasoning_default(agent) (_reasoning_config_for_wire)
    # and the profile shapes the reasoning fields. supports_reasoning=False is
    # what _supports_reasoning_extra_body() gives a non-OpenRouter/Nous host;
    # CustomProfile ignores it either way.
    def _agent_wire(out, model, rt, stage):
        try:
            from agent.reasoning_params import unset_reasoning_default
            from providers import get_provider_profile

            class _Agent:
                api_mode = rt.get("api_mode")
                provider = rt.get("provider")
                _ollama_num_ctx = None
            _Agent.model = model
            default = unset_reasoning_default(_Agent())
            profile = get_provider_profile(rt.get("provider"))
            kwargs = transport.build_kwargs(
                model, [{"role": "user", "content": "hi"}],
                base_url=rt.get("base_url"), provider_profile=profile,
                reasoning_config=default, supports_reasoning=False,
                request_overrides=rt.get("request_overrides"))
            seen.clear()
            client = openai.OpenAI(api_key="fake-key", base_url=rt.get("base_url") or "http://x/v1",
                                   http_client=httpx.Client(transport=httpx.MockTransport(_handler)))
            client.chat.completions.create(**kwargs)
            body = seen[-1]
            out["agent_path"] = {
                "profile": type(profile).__name__ if profile is not None else None,
                "unset_reasoning_default": default,
                "wire_max_tokens": body.get("max_tokens"),
                "wire_reasoning_effort": body.get("reasoning_effort"),
                "wire_keys": sorted(body)}
        except Exception:
            fail(stage)

    for check in checks:
        name, model = check["provider"], check["model"]
        out = ev["providers"][name] = {}
        try:
            with frozen_runtime_config(cfg):
                rt = resolve_runtime_provider(requested=name, target_model=model)
        except Exception as e:
            out["resolve_error"] = f"{type(e).__name__}: {e}"
            continue
        out["provider"] = rt.get("provider")
        out["api_mode"] = rt.get("api_mode")
        out["base_url"] = rt.get("base_url")
        _credential(out, rt.get("api_key"), check.get("fake_token"))
        out["request_overrides"] = rt.get("request_overrides")
        _wire(out, model, rt.get("base_url"), rt.get("provider"),
              rt.get("request_overrides"), f"build_request[{name}]")
        if check.get("agent_path"):
            _agent_wire(out, model, rt, f"agent_request[{name}]")

    # In-session /model switch through upstream's ONE shared pipeline
    # (hermes_cli.model_switch.switch_model, used by the CLI, gateway and ACP),
    # called the way acp_adapter/server.py does: providers dict + compatible
    # custom-provider list from the loaded config. The outgoing request is then
    # built with the switch RESULT's request_overrides, as the agent does.
    for sw in switches:
        out = {"from_provider": sw["from_provider"], "to_model": sw["to_model"]}
        ev["switches"].append(out)
        try:
            from hermes_cli.config import load_config
            from hermes_cli.model_switch import switch_model
            live = load_config()
            with frozen_runtime_config(live):
                res = switch_model(
                    raw_input=sw["to_model"], current_provider=sw["from_provider"],
                    current_model=sw["from_model"], current_base_url=sw["from_base_url"],
                    current_api_key=sw["fake_token"],
                    explicit_provider=sw.get("explicit_provider", ""),
                    user_providers=live.get("providers") if isinstance(live.get("providers"), dict) else {},
                    custom_providers=get_compatible_custom_providers(live))
        except Exception:
            fail(f"switch_model[{sw['to_model']}]")
        out["success"] = bool(res.success)
        out["error_message"] = res.error_message
        out["new_model"] = res.new_model
        out["target_provider"] = res.target_provider
        out["provider_changed"] = res.provider_changed
        out["base_url"] = res.base_url
        out["api_mode"] = res.api_mode
        _credential(out, res.api_key, sw["fake_token"])
        out["request_overrides"] = res.request_overrides
        if res.success:
            _wire(out, res.new_model, res.base_url, res.target_provider,
                  res.request_overrides or None, f"build_request[switch:{sw['to_model']}]")

    try:
        msgs = [{"role": "tool", "tool_call_id": "c1", "name": "terminal", "content": "ok"}]
        conv = transport.convert_messages(msgs, model="m", base_url="https://h/v1", provider="custom")
        ev["tool_message"] = conv[0]
    except Exception:
        fail("convert_messages")

    print(json.dumps(ev, default=str))
''')


def run_populate_cli(*args, stdin: str) -> subprocess.CompletedProcess:
    """Run scripts/populate_models.py with no caller credentials; it mints the
    (fake) token itself from a fake alcf-tokens CLI named by ALCF_TOKENS_BIN."""
    env = {k: v for k, v in os.environ.items() if not k.endswith(("_TOKEN", "_API_KEY"))}
    with tempfile.TemporaryDirectory() as tmp:
        cli = Path(tmp) / "alcf-tokens"
        cli.write_text(fake_tokens_cli_script(FAKE_TOKEN), encoding="utf-8")
        cli.chmod(0o755)
        env["ALCF_TOKENS_BIN"] = str(cli)
        return subprocess.run([sys.executable, str(SCRIPTS / "populate_models.py"), *args],
                              input=stdin, text=True, capture_output=True, env=env, timeout=30)


def fake_tokens_cli_script(token: str) -> str:
    """A fake alcf-tokens whose `get-token inference` prints *token*."""
    return ("#!/bin/sh\n"
            '[ "$1" = get-token ] || exit 2\n'
            f"printf '%s\\n' '{token}'\n")


UPSTREAM_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "TMPDIR")


def upstream_subprocess_env(root: Path, *, home: Path, hermes_home: Path) -> dict:
    """Allowlisted env for the upstream harness: nothing from the caller's
    environment leaks in (no credentials, no proxies, no HERMES_* settings)
    except the few variables needed to run Python at all."""
    env = {k: os.environ[k] for k in UPSTREAM_ENV_ALLOWLIST if k in os.environ}
    env.setdefault("LANG", "C.UTF-8")
    env.update(HOME=str(home), HERMES_HOME=str(hermes_home), PYTHONPATH=str(root),
               PYTHONDONTWRITEBYTECODE="1", NO_COLOR="1")
    return env


class UpstreamHarnessMixin:
    """Runs UPSTREAM_HARNESS with the upstream interpreter in a subprocess."""

    def upstream(self) -> Path:
        root = os.environ.get(UPSTREAM_ENV, "").strip()
        if not root:
            self.skipTest(f"{UPSTREAM_ENV} is unset; upstream Hermes v0.23.2 runtime "
                          "contract cannot be checked")
        root_path = Path(root)
        py = root_path / ".venv" / "bin" / "python"
        # A set-but-broken checkout is a harness failure, never a skip.
        self.assertTrue(py.exists(), f"{UPSTREAM_ENV}={root} has no .venv/bin/python")
        return root_path

    def run_upstream(self, config_text: str, checks: list, switches: list | None = None) -> dict:
        root = self.upstream()
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            hermes_home = home / ".hermes"
            hermes_home.mkdir(parents=True)
            # Upstream's load_config() (used by the switch pipeline) reads
            # $HERMES_HOME/config.yaml, so the rendered config lives there.
            cfg = hermes_home / "config.yaml"
            # ALCF providers mint their token through key_cmd
            # (`/opt/hermes/.venv/bin/alcf-tokens get-token inference`); point
            # that at a fake CLI here that prints FAKE_TOKEN.
            fake_cli = Path(tmp) / "bin" / "alcf-tokens"
            fake_cli.parent.mkdir()
            fake_cli.write_text(fake_tokens_cli_script(FAKE_TOKEN), encoding="utf-8")
            fake_cli.chmod(0o755)
            config_text = config_text.replace(pm.DEFAULT_ALCF_TOKENS_BIN, str(fake_cli))
            cfg.write_text(config_text, encoding="utf-8")
            env = upstream_subprocess_env(root, home=home, hermes_home=hermes_home)
            proc = subprocess.run(
                [str(root / ".venv" / "bin" / "python"), "-c", UPSTREAM_HARNESS,
                 str(cfg), json.dumps(checks), json.dumps(switches or [])],
                cwd=str(root), env=env, text=True, capture_output=True, timeout=120)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        self.assertEqual(proc.returncode, 0, f"upstream harness crashed:\n{proc.stderr[-4000:]}")
        self.assertTrue(lines, f"upstream harness printed no JSON:\n{proc.stdout}\n{proc.stderr[-4000:]}")
        ev = json.loads(lines[-1])
        print(f"\n[upstream evidence] {json.dumps(ev, sort_keys=True)}", file=sys.stderr)
        self.assertIsNone(ev["harness_error"], "upstream harness error (not a contract failure)")
        return ev

    def assert_config_version_current(self, ev: dict) -> None:
        """The rendered config carries upstream's current schema stamp, so the boot
        migration check is a silent no-op (not 'below-support-floor')."""
        cv = ev["config_version"]
        self.assertEqual(cv["stamp"], cv["latest"],
                         "rendered config's _config_version != upstream "
                         "DEFAULT_CONFIG['_config_version']")
        self.assertEqual(cv["migrate"], {"ok": True, "skipped": "up-to-date"})
        self.assertEqual(ev["migration_warnings"], [])

    def assert_runtime_contract(self, ev: dict, checks: list) -> None:
        self.assertIn("providers", ev["root_keys"], "rendered config has no `providers` root key")
        self.assertNotIn("custom_providers", ev["root_keys"],
                         "rendered config still uses the legacy custom_providers list")
        errors = [i for i in ev["validate_issues"] if "error" in i["severity"].lower()]
        self.assertEqual(errors, [], "validate_config_structure reported errors")
        self.assertEqual(ev["unknown_key_warnings"], [],
                         "Hermes ignored provider keys (e.g. max_tokens) in the rendered config")
        for check in checks:
            out = ev["providers"][check["provider"]]
            with self.subTest(provider=check["provider"]):
                self.assertNotIn("resolve_error", out)
                self.assertEqual(out["provider"], "custom")
                self.assertEqual(out["api_mode"], "chat_completions")
                self.assertEqual(out["base_url"], check["base_url"])
                if check.get("key_cmd"):
                    self.assertTrue(out["api_key_callable"],
                                    "ALCF provider credential is not the per-request key_cmd callable")
                self.assertNotIn("api_key_mint_error", out)
                self.assertTrue(out["api_key_is_fake"],
                                "resolved credential does not yield the fake token")
                overrides = out["request_overrides"] or {}
                self.assertEqual((overrides.get("extra_body") or {}).get("max_tokens"),
                                 check["max_tokens"],
                                 "request_overrides.extra_body.max_tokens does not carry the cap")
                self.assertEqual(out["wire_max_tokens"], check["max_tokens"],
                                 "outgoing Chat Completions body does not carry the cap")
        tool = ev["tool_message"]
        self.assertNotIn("name", tool, "upstream no longer strips `name` from role:tool messages")
        self.assertEqual(tool["tool_call_id"], "c1")

    def assert_switch_contract(self, ev: dict, switches: list) -> None:
        """A /model switch must carry the TARGET provider's cap onto the wire."""
        self.assertEqual(len(ev["switches"]), len(switches))
        for sw, out in zip(switches, ev["switches"]):
            with self.subTest(switch=f"{sw['from_model']} -> {sw['to_model']}"):
                self.assertTrue(out["success"], f"switch_model failed: {out['error_message']}")
                self.assertEqual(out["new_model"], sw["to_model"])
                # Upstream reports a user `providers:` entry either as its bare key
                # or as custom:<key> (runtime_provider_custom.persisted_provider_id);
                # both name the same endpoint.
                self.assertEqual(out["target_provider"].removeprefix("custom:"),
                                 sw["expect_provider"].removeprefix("custom:"))
                self.assertTrue(out["provider_changed"])
                self.assertEqual(out["base_url"], sw["from_base_url"])
                self.assertEqual(out["api_mode"], "chat_completions")
                if sw.get("key_cmd"):
                    self.assertTrue(out["api_key_callable"],
                                    "switched ALCF credential is not the per-request key_cmd callable")
                self.assertNotIn("api_key_mint_error", out)
                self.assertTrue(out["api_key_is_fake"],
                                "switched credential does not yield the fake token")
                self.assertEqual(((out["request_overrides"] or {}).get("extra_body") or {})
                                 .get("max_tokens"), sw["max_tokens"],
                                 "switch result request_overrides lacks the target provider's cap")
                self.assertEqual(out["wire_max_tokens"], sw["max_tokens"],
                                 "outgoing body after the switch does not carry the target cap")


# "key_cmd": the credential must be the per-request callable (run_upstream
# points the rendered key_cmd at a fake CLI printing FAKE_TOKEN).
ALCF_CHECKS = [
    {"provider": "custom:alcf-x", "model": "chatty", "base_url": "https://h/v1",
     "max_tokens": pm.BASELINE_MAX_TOKENS, "fake_token": FAKE_TOKEN, "key_cmd": True},
    {"provider": "custom:alcf-x-reasoning", "model": "thinky", "base_url": "https://h/v1",
     "max_tokens": pm.REASONING_MAX_TOKENS, "fake_token": FAKE_TOKEN, "key_cmd": True},
]

# Launch on the plain model, then /model-switch to the reasoning one: the turn
# after the switch must use alcf-x-reasoning's 12288 cap, not the launch 2048.
ALCF_SWITCHES = [
    {"from_provider": "custom:alcf-x", "from_model": "chatty", "from_base_url": "https://h/v1",
     "to_model": "thinky", "expect_provider": "custom:alcf-x-reasoning",
     "max_tokens": pm.REASONING_MAX_TOKENS, "fake_token": FAKE_TOKEN, "key_cmd": True},
]

# Hand-written in the shipped credential shape (key_cmd, no api_key);
# run_upstream rewrites the alcf-tokens path to its FAKE_TOKEN CLI.
POSITIVE_CONTROL_CONFIG = f"""\
model:
  provider: "custom:alcf-x"
  model: "chatty"
  default: "chatty"
  context_length: 128000
providers:
  alcf-x:
    name: alcf-x
    api: "https://h/v1"
    key_cmd: "{pm.DEFAULT_ALCF_TOKENS_BIN} get-token inference"
    discover_models: false
    extra_body:
      max_tokens: {pm.BASELINE_MAX_TOKENS}
    models:
      chatty:
        context_length: 128000
  alcf-x-reasoning:
    name: alcf-x-reasoning
    api: "https://h/v1"
    key_cmd: "{pm.DEFAULT_ALCF_TOKENS_BIN} get-token inference"
    discover_models: false
    extra_body:
      max_tokens: {pm.REASONING_MAX_TOKENS}
    models:
      thinky:
        context_length: 262144
"""


# --- tests ------------------------------------------------------------------------

class BaseTagContract(unittest.TestCase):
    def test_dockerfile_pins_exact_v0_23_2_base(self):
        base = dockerfile_base_arg()
        self.assertEqual(base, EXPECTED_BASE)

    def test_dockerfile_base_is_never_a_floating_tag(self):
        base = dockerfile_base_arg()
        self.assertIn(":", base, "HERMES_BASE must carry an explicit tag")
        tag = base.rsplit(":", 1)[1]
        self.assertIsNone(FLOATING_TAG.search(tag), f"floating/pre-release tag: {tag}")
        self.assertRegex(tag, r"^v\d+\.\d+\.\d+$", "base must be a final release tag")

    def test_readme_components_table_states_dockerfile_base(self):
        self.assertEqual(readme_component_base(), dockerfile_base_arg())

    def test_readme_prose_names_only_the_dockerfile_base(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        tags = set(re.findall(r"nousresearch/hermes-agent:[\w.\-]+", text))
        self.assertEqual(tags, {dockerfile_base_arg()})


def dockerfile_instructions() -> list:
    """Dockerfile instructions with comments dropped, `\\` continuations joined
    and whitespace runs collapsed to one space."""
    out, cur = [], ""
    for raw in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            cur += line[:-1] + " "
            continue
        out.append(cur + line)
        cur = ""
    if cur:
        out.append(cur)
    return [" ".join(i.split()) for i in out]


# `uv` in command position (`uv pip ...`, `&& uv ...`, `uvx ...`); not "$uv" or a path.
BARE_UV = re.compile(r"(?<![\w$/.\-\"'])uvx?\s+[\w-]")
UV_GLOB = "/opt/hermes/tools/uv-*/uv"
DEPS = "alcf-tokens==0.3.0 globus-compute-sdk requests"


class DependencyInstallContract(unittest.TestCase):
    """The base venv is PM-built and sealed; uv is internal tooling, not on PATH
    (CI run 45195114759: `uv: not found`, exit 127). The ALCF deps must go in
    through the base's own pinned uv, found by an exactly-one glob, behind a
    guard that fails the build if any pre-existing package changes version."""

    def install_run(self) -> str:
        runs = [i for i in dockerfile_instructions() if i.startswith("RUN ") and DEPS in i]
        self.assertEqual(len(runs), 1, "expected exactly one RUN installing the ALCF deps")
        return runs[0]

    def test_bare_uv_detector_catches_the_broken_line(self):
        self.assertRegex("RUN uv pip install --python /opt/hermes/.venv/bin/python x", BARE_UV)
        self.assertRegex("RUN set -e && uvx foo run", BARE_UV)
        self.assertNotRegex('"$uv" pip install --python "$py" x', BARE_UV)
        self.assertNotRegex("/opt/hermes/tools/uv-0.13.5-linux-x64/uv pip freeze", BARE_UV)

    def test_dockerfile_never_invokes_bare_uv(self):
        for inst in dockerfile_instructions():
            if inst.startswith("RUN "):
                self.assertNotRegex(inst, BARE_UV, f"bare uv on PATH does not exist: {inst[:80]}")

    def test_install_runs_as_root_under_set_eu(self):
        insts = dockerfile_instructions()
        run = self.install_run()
        users = [(n, i) for n, i in enumerate(insts) if i.startswith("USER ")]
        before = [i for n, i in users if n < insts.index(run)]
        self.assertTrue(before and before[-1] == "USER root", "install layer must run as root")
        self.assertTrue(run.startswith("RUN set -eu;"), run[:40])

    def test_uv_is_the_single_pinned_binary_from_the_base(self):
        run = self.install_run()
        self.assertIn(f"for c in {UV_GLOB}; do", run)
        self.assertIn('[ -x "$c" ]', run)
        self.assertRegex(run, r'if \[ "\$n" -ne 1 \]; then .*exactly one.*exit 1; fi')
        self.assertIn('"$uv" pip install --python "$py" --no-cache ' + DEPS, run)
        self.assertIn("py=/opt/hermes/.venv/bin/python", run)

    def test_drift_guard_compares_freezes_around_the_install(self):
        run = self.install_run()
        before = run.index('"$uv" pip freeze --python "$py" > /tmp/venv-before.txt')
        install = run.index('"$uv" pip install')
        after = run.index('"$uv" pip freeze --python "$py" > /tmp/venv-after.txt')
        self.assertLess(before, install)
        self.assertLess(install, after)
        self.assertRegex(run[before:install], r"\[ ! -s /tmp/venv-before.txt \].*exit 1")
        guard = run[after:]
        self.assertIn("grep -vxFf /tmp/venv-after.txt /tmp/venv-before.txt", guard)
        self.assertRegex(guard, r'if \[ -n "\$changed" \]; then .*diff /tmp/venv-before.txt '
                                r'/tmp/venv-after.txt.*exit 1; fi')

    def test_install_run_is_valid_posix_sh(self):
        res = subprocess.run(["sh", "-n", "-c", self.install_run()[len("RUN "):]],
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_drift_guard_semantics(self):
        """Run the Dockerfile's own guard command over fake freezes: additions
        pass; a version change or a removal is reported."""
        run = self.install_run()
        cmd = re.search(r'changed="\$\((grep -vxFf [^)]*)\)"', run).group(1)
        base = "hermes-agent @ file:///opt/hermes\nrequests==2.32.5\nruamel-yaml==0.18.16\n"
        cases = {
            "addition only": (base + "alcf-tokens==0.3.0\n", ""),
            "version change": (base.replace("2.32.5", "2.33.0") + "alcf-tokens==0.3.0\n",
                               "requests==2.32.5"),
            "removal": (base.replace("ruamel-yaml==0.18.16\n", ""), "ruamel-yaml==0.18.16"),
        }
        for name, (after, expected) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                script = cmd.replace("/tmp/", tmp + "/")
                Path(tmp, "venv-before.txt").write_text(base, encoding="utf-8")
                Path(tmp, "venv-after.txt").write_text(after, encoding="utf-8")
                res = subprocess.run(["sh", "-c", script], capture_output=True, text=True,
                                     timeout=10)
                self.assertEqual(res.stdout.strip(), expected, res.stderr)

    def test_smoke_imports_after_install(self):
        run = self.install_run()
        smoke = run.index('"$py" -c "import hermes_cli, alcf_tokens.auth, globus_compute_sdk, '
                          'requests, ruamel.yaml"')
        self.assertLess(run.index('"$uv" pip install'), smoke)
        self.assertIn("/opt/hermes/.venv/bin/alcf-tokens --help >/dev/null", run[smoke:])


class PatchRetirementContract(unittest.TestCase):
    def test_local_tool_message_patch_is_removed(self):
        self.assertFalse(PATCH.exists(), f"{PATCH.relative_to(ROOT)} still present")

    def test_dockerfile_does_not_reference_the_patch(self):
        text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertFalse("strip_tool_message_name" in text,
                         "Dockerfile still references strip_tool_message_name")
        self.assertFalse("0001-strip-tool-message-name" in text,
                         "Dockerfile still applies the tool-message patch")

    def test_template_does_not_set_strip_tool_message_name(self):
        text = (ROOT / "config" / "config.template.yaml").read_text(encoding="utf-8")
        model = yaml.safe_load(text).get("model") or {}
        self.assertNotIn("strip_tool_message_name", model, "template still sets the patch flag")
        self.assertFalse("strip_tool_message_name" in text,
                         "config.template.yaml still mentions strip_tool_message_name")

    def test_readme_mentions_patch_only_as_history(self):
        current = readme_current_patch_mentions((ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertEqual(current, [], "README still describes the patch in current tense")

    def test_history_heuristic_is_sentence_scoped(self):
        """Guards the README check itself against the brittle any-word-on-the-line rule."""
        cases = {
            # current tense, a marker word elsewhere on the line: still current
            "Hermes was removed from PATH. We apply `strip_tool_message_name` at build.": 1,
            "| Tool-message patch (retired) | `patches/0001-strip-tool-message-name.patch` | x |": 1,
            "The patch `0001-strip-tool-message-name` is applied; unremoved hunks fail.": 1,
            # genuine history
            "The `strip_tool_message_name` patch was retired in v0.23.2.": 0,
            "Previously the image applied `0001-strip-tool-message-name.patch`.": 0,
            "### History\n\nThe image applied `0001-strip-tool-message-name.patch`.": 0,
        }
        for text, want in cases.items():
            with self.subTest(text=text):
                self.assertEqual(len(readme_current_patch_mentions(text)), want)


class PopulateModelsSchemaContract(unittest.TestCase):
    def test_emit_cluster_output_matches_providers_schema(self):
        """_emit_cluster CONTRACT: it returns the lines of one cluster's keyed
        provider ENTRIES, every line indented two spaces so build_block can nest
        them directly under its single `providers:` root. On their own the lines
        parse to the {provider_key: entry} mapping itself (no list, no root key)."""
        mapping = {"chatty": (128000, False), "thinky": (262144, True)}
        lines = pm._emit_cluster("alcf-x", "https://h/v1", mapping)
        self.assertIsInstance(lines, list)
        self.assertTrue(lines)
        for line in lines:
            self.assertTrue(line.startswith("  ") and not line.startswith("  - "),
                            f"not a 2-space keyed-entry line: {line!r}")
        doc = yaml.safe_load("\n".join(lines))
        providers = assert_provider_entries(self, doc, where="_emit_cluster")
        self.assertEqual(set(providers), {"alcf-x", "alcf-x-reasoning"})
        self.assertEqual(providers["alcf-x"]["api"], "https://h/v1")
        self.assertEqual(providers["alcf-x"]["extra_body"]["max_tokens"], pm.BASELINE_MAX_TOKENS)
        self.assertEqual(providers["alcf-x-reasoning"]["extra_body"]["max_tokens"],
                         pm.REASONING_MAX_TOKENS)
        self.assertEqual(providers["alcf-x"]["models"], {"chatty": {"context_length": 128000}})
        self.assertEqual(providers["alcf-x-reasoning"]["models"],
                         {"thinky": {"context_length": 262144}})

    def test_build_block_is_providers_root_over_emit_cluster(self):
        """build_block = one `providers:` root + each cluster's _emit_cluster lines."""
        _cluster, provider, base, _sel, fallback = TEST_CLUSTER[0]
        self.assertEqual(build_test_block(),
                         "providers:\n" + "\n".join(pm._emit_cluster(provider, base, fallback)) + "\n")

    def test_build_block_matches_providers_schema(self):
        providers = assert_provider_schema(self, yaml.safe_load(build_test_block()),
                                           where="build_block")
        self.assertEqual(set(providers), {"alcf-x", "alcf-x-reasoning"})
        assert_cap_split(self, providers, where="build_block")
        self.assertEqual(providers["alcf-x-reasoning"]["models"]["thinky"]["context_length"], 262144)

    def test_empty_class_is_omitted(self):
        with mock.patch.object(pm, "_clusters", return_value=[
                ("x", "alcf-x", "https://h/v1", None, {"thinky": (262144, True)})]), \
                mock.patch.object(pm, "_get_token", side_effect=RuntimeError("offline test")), \
                contextlib.redirect_stderr(io.StringIO()):
            doc = yaml.safe_load(pm.build_block())
        providers = assert_provider_schema(self, doc, where="build_block(reasoning only)")
        self.assertEqual(set(providers), {"alcf-x-reasoning"})

    def test_populate_models_static_fallback_matches_schema(self):
        with mock.patch.object(pm, "_get_token", side_effect=RuntimeError("offline test")), \
                contextlib.redirect_stderr(io.StringIO()):
            doc = yaml.safe_load(pm.build_block())
        providers = assert_provider_schema(self, doc, where="populate_models fallback")
        assert_cap_split(self, providers, where="populate_models fallback")
        got = {k: {m: v["context_length"] for m, v in e["models"].items()}
               for k, e in providers.items()}
        self.assertEqual(got, fallback_as_providers(pm._clusters()))


class TemplateFallbackContract(unittest.TestCase):
    def test_committed_template_fallback_matches_schema(self):
        providers = assert_provider_schema(self, template_fallback_doc(),
                                           where="config.template.yaml fallback")
        assert_cap_split(self, providers, where="config.template.yaml fallback")
        for entry in providers.values():
            # The credential is the per-request key_cmd, never a token literal.
            self.assertNotIn("api_key", entry)
            self.assertEqual(entry["key_cmd"], "/opt/hermes/.venv/bin/alcf-tokens get-token inference")
            for mid, meta in entry["models"].items():
                self.assertGreaterEqual(meta["context_length"], pm.MIN_CONTEXT, mid)

    def test_template_and_populate_models_fallbacks_agree(self):
        providers = assert_provider_schema(self, template_fallback_doc(),
                                           where="config.template.yaml fallback")
        got = {k: {m: v["context_length"] for m, v in e["models"].items()}
               for k, e in providers.items()}
        self.assertEqual(got, fallback_as_providers(pm._clusters()))

    def test_template_and_populate_models_key_cmd_agree(self):
        """One alcf-tokens path: the template fallback's key_cmd is the one
        populate_models emits by default."""
        with mock.patch.dict(os.environ):
            os.environ.pop("ALCF_TOKENS_BIN", None)
            default_cmd = pm.alcf_key_cmd()
        self.assertEqual(default_cmd, f"{pm.DEFAULT_ALCF_TOKENS_BIN} get-token inference")
        for entry in template_fallback_doc()["providers"].values():
            self.assertEqual(entry["key_cmd"], default_cmd)

    def test_whole_template_parses_with_providers_mapping(self):
        doc = yaml.safe_load((ROOT / "config" / "config.template.yaml").read_text(encoding="utf-8"))
        assert_provider_schema(self, doc, where="config.template.yaml")


class ProviderParsingContract(UpstreamHarnessMixin, unittest.TestCase):
    """pm.parse_providers is the structured reader entrypoint.sh uses: a real YAML
    parser (ruamel.yaml, which the Hermes runtime venv ships, else PyYAML) and
    {key: {"api", "models": {id: ctx}}}. Anything unreadable fails closed."""

    @staticmethod
    def via_yaml(text: str) -> dict:
        doc = yaml.safe_load(text)
        return {k: {"api": e["api"],
                    "models": {m: v["context_length"] for m, v in e["models"].items()}}
                for k, e in doc["providers"].items()}

    def test_parse_matches_yaml_on_generated_block(self):
        block = build_test_block()
        self.assertEqual(pm.parse_providers(block), self.via_yaml(block))

    def test_parse_matches_yaml_on_whole_template(self):
        text = (ROOT / "config" / "config.template.yaml").read_text(encoding="utf-8")
        self.assertEqual(pm.parse_providers(text), self.via_yaml(text))

    def test_parse_handles_quoted_ids_and_comments(self):
        text = textwrap.dedent('''\
            model:
              provider: "custom:x"
            providers:
              # a comment
              generic-inference:
                name: generic-inference
                api: "http://127.0.0.1:9/v1"
                api_key: "${ALCF_ACCESS_TOKEN}"
                discover_models: false
                extra_body:
                  max_tokens: 4096
                models:
                  "selected/model":
                    context_length: 131072
                  'other:tag':
                    context_length: 70000
            delegation:
              provider: "custom:x"
            ''')
        self.assertEqual(pm.parse_providers(text), self.via_yaml(text))

    def test_parse_rejects_legacy_or_missing_providers(self):
        for text in ("custom_providers:\n  - name: a\n", "model:\n  provider: x\n", ""):
            with self.subTest(text=text):
                self.assertEqual(pm.parse_providers(text), {})

    def test_malformed_yaml_fails_closed(self):
        """Invalid YAML never yields a partial map: the first provider below is
        well formed, so a line-oriented reader would happily return it."""
        good = textwrap.dedent('''\
            providers:
              alcf-x:
                api: "https://h/v1"
                models:
                  chatty:
                    context_length: 131072
            ''')
        for bad in (good + '  alcf-y: [unclosed\n',
                    good + '  alcf-y:\n    api: "https://h/v1\n',
                    good + '\tbad-tab: 1\n',
                    good + '  alcf-x:\n    api: "https://dup/v1"\n    models: {}\n'):
            with self.subTest(bad=bad[len(good):]):
                with self.assertRaises(pm.ProviderParseError):
                    pm.parse_providers(bad)
                with self.assertRaises(pm.LaunchProviderError):
                    pm.launch_provider("chatty", bad, "https://h/v1")

    def test_malformed_structure_fails_closed(self):
        for bad in ("- providers\n",
                    "providers:\n",
                    "providers: [a, b]\n",
                    "providers:\n  alcf-x: just-a-string\n",
                    "providers:\n  alcf-x:\n    models: {m: {context_length: 70000}}\n",
                    "providers:\n  alcf-x:\n    api: https://h/v1\n",
                    "providers:\n  alcf-x:\n    api: https://h/v1\n    models: [m]\n",
                    "providers:\n  alcf-x:\n    api: https://h/v1\n    models: {m: {}}\n",
                    "providers:\n  alcf-x:\n    api: https://h/v1\n"
                    "    models: {m: {context_length: big}}\n"):
            with self.subTest(bad=bad):
                with self.assertRaises(pm.ProviderParseError):
                    pm.parse_providers(bad)

    def test_context_length_must_be_a_positive_integer(self):
        """B7c-R4: 0, negative, bool, float, string and null are all rejected;
        so is the same block once it stops being valid YAML."""
        head = "providers:\n  alcf-x:\n    api: https://h/v1\n    models:\n      m:\n"
        self.assertEqual(pm.parse_providers(head + "        context_length: 1\n"),
                         {"alcf-x": {"api": "https://h/v1", "models": {"m": 1}}})
        for value in ("0", "-1", "-131072", "true", "false", "131072.0", '"131072"', "null"):
            with self.subTest(context_length=value):
                bad = head + f"        context_length: {value}\n"
                with self.assertRaises(pm.ProviderParseError):
                    pm.parse_providers(bad)
                with self.assertRaises(pm.LaunchProviderError):
                    pm.launch_provider("m", bad, "https://h/v1")
        with self.assertRaises(pm.ProviderParseError):
            pm.parse_providers(head + "        context_length: [131072\n")

    def test_no_yaml_parser_fails_closed(self):
        block = build_test_block()
        with mock.patch.dict(sys.modules, {"ruamel": None, "ruamel.yaml": None, "yaml": None}):
            with self.assertRaises(pm.ProviderParseError) as cm:
                pm.parse_providers(block)
        self.assertIn("no YAML parser", str(cm.exception))

    def test_cli_fails_closed_on_malformed_yaml(self):
        bad = CROSS_CLUSTER_BLOCK + "  alcf-c: [unclosed\n"
        for args in (("--launch-provider", "plain-b", "--base-url", "https://h/b/v1"),
                     ("--list-models", "--base-url", "https://h/b/v1")):
            with self.subTest(args=args):
                res = run_populate_cli(*args, stdin=bad)
                self.assertEqual(res.returncode, 1, res.stderr)
                self.assertEqual(res.stdout.strip(), "")
                self.assertIn("not valid YAML", res.stderr)
                self.assertNotIn(FAKE_TOKEN, res.stderr)

    def test_ruamel_path_under_upstream_runtime(self):
        """Under the Hermes runtime interpreter (ruamel.yaml; PyYAML blocked so it
        cannot be the one used) the reader and CLI agree with PyYAML here."""
        root = self.upstream()
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp) / "alcf_tokens"
            stub.mkdir()
            (stub / "__init__.py").write_text("", encoding="utf-8")
            (stub / "auth.py").write_text(
                "def get_access_token(*a, **k):\n    raise RuntimeError('offline test')\n",
                encoding="utf-8")
            code = textwrap.dedent(f'''\
                import io, json, sys
                sys.modules["yaml"] = None
                sys.path[:0] = [{str(SCRIPTS)!r}, {tmp!r}]
                import populate_models as pm
                block = sys.stdin.read()
                print(json.dumps(pm.parse_providers(block), sort_keys=True))
                rejected = 0
                for v in ("0", "-1", "true", "131072.0"):
                    try:
                        pm.parse_providers("providers:\\n  x:\\n    api: u\\n    models:\\n"
                                           "      m:\\n        context_length: " + v + "\\n")
                    except pm.ProviderParseError:
                        rejected += 1
                print(rejected)
                sys.stdin = io.StringIO(block)
                sys.argv = ["populate_models.py", "--launch-provider", "shared-model",
                            "--base-url", "https://h/b/v1"]
                sys.exit(pm.main())
                ''')
            env = upstream_subprocess_env(root, home=Path(tmp), hermes_home=Path(tmp))
            env.pop("PYTHONPATH")
            res = subprocess.run([str(root / ".venv" / "bin" / "python"), "-c", code],
                                 input=CROSS_CLUSTER_BLOCK, text=True, capture_output=True,
                                 env=env, timeout=60)
        self.assertEqual(res.returncode, 0, res.stderr)
        parsed, rejected, provider = res.stdout.strip().splitlines()
        self.assertEqual(json.loads(parsed), self.via_yaml(CROSS_CLUSTER_BLOCK))
        self.assertEqual(rejected, "4", "ruamel path must reject non-positive/bool/float context")
        self.assertEqual(provider, "custom:alcf-b-reasoning")


CROSS_CLUSTER_BLOCK = textwrap.dedent(f'''\
    providers:
      alcf-a-reasoning:
        name: alcf-a-reasoning
        api: "https://h/a/v1"
        api_key: "${{ALCF_ACCESS_TOKEN}}"
        discover_models: false
        extra_body:
          max_tokens: {pm.REASONING_MAX_TOKENS}
        models:
          shared-model:
            context_length: 131072
          only-a:
            context_length: 131072
      alcf-b:
        name: alcf-b
        api: "https://h/b/v1"
        api_key: "${{ALCF_ACCESS_TOKEN}}"
        discover_models: false
        extra_body:
          max_tokens: {pm.BASELINE_MAX_TOKENS}
        models:
          plain-b:
            context_length: 128000
      alcf-b-reasoning:
        name: alcf-b-reasoning
        api: "https://h/b/v1/"
        api_key: "${{ALCF_ACCESS_TOKEN}}"
        discover_models: false
        extra_body:
          max_tokens: {pm.REASONING_MAX_TOKENS}
        models:
          shared-model:
            context_length: 131072
          dup-on-b:
            context_length: 131072
      alcf-b-extra:
        name: alcf-b-extra
        api: "https://h/b/v1"
        api_key: "${{ALCF_ACCESS_TOKEN}}"
        discover_models: false
        extra_body:
          max_tokens: {pm.BASELINE_MAX_TOKENS}
        models:
          dup-on-b:
            context_length: 128000
    ''')


class LaunchProviderContract(unittest.TestCase):
    """launch_provider(model, providers_text, base_url=None) CONTRACT.

    The launch model maps to EXACTLY ONE named provider that lists it, among
    the providers whose `api` is the launch endpoint (base_url; trailing slash
    ignored) -- or among all providers when base_url is None (delegation uses
    delegation_provider instead). Zero or several matches raise LaunchProviderError
    with a non-secret message; there is no heuristic or default fallback. It
    never touches the network (catalog fetchers are patched to explode).
    """

    def setUp(self):
        for name in ("_get_token", "_fetch_models"):
            p = mock.patch.object(pm, name, side_effect=AssertionError(f"{name}: network used"))
            p.start()
            self.addCleanup(p.stop)

    def test_exactly_one_provider_on_the_endpoint_is_selected(self):
        self.assertEqual(pm.launch_provider("plain-b", CROSS_CLUSTER_BLOCK, "https://h/b/v1"),
                         "custom:alcf-b")
        self.assertEqual(pm.launch_provider("shared-model", CROSS_CLUSTER_BLOCK, "https://h/b/v1"),
                         "custom:alcf-b-reasoning")
        self.assertEqual(pm.launch_provider("shared-model", CROSS_CLUSTER_BLOCK, "https://h/a/v1/"),
                         "custom:alcf-a-reasoning")
        self.assertEqual(pm.launch_provider("only-a", CROSS_CLUSTER_BLOCK), "custom:alcf-a-reasoning")

    def test_generated_block_selection_follows_reasoning_class(self):
        block = build_test_block()
        self.assertEqual(pm.launch_provider("chatty", block, "https://h/v1"), "custom:alcf-x")
        self.assertEqual(pm.launch_provider("thinky", block, "https://h/v1"),
                         "custom:alcf-x-reasoning")

    def test_zero_providers_fails_closed(self):
        for model, base in (("unknown/model", "https://h/b/v1"),
                            ("only-a", "https://h/b/v1"),     # served, but not on this endpoint
                            ("plain-b", "https://elsewhere/v1")):
            with self.subTest(model=model, base=base):
                with self.assertRaises(pm.LaunchProviderError) as cm:
                    pm.launch_provider(model, CROSS_CLUSTER_BLOCK, base)
                self.assertIn(model, str(cm.exception))
        with self.assertRaises(pm.LaunchProviderError):
            pm.launch_provider("chatty", "", "https://h/v1")

    def test_duplicate_providers_fail_closed(self):
        with self.assertRaises(pm.LaunchProviderError) as cm:
            pm.launch_provider("dup-on-b", CROSS_CLUSTER_BLOCK, "https://h/b/v1")
        self.assertIn("alcf-b-extra", str(cm.exception))
        self.assertIn("alcf-b-reasoning", str(cm.exception))
        # Without an endpoint, a model served by two clusters is ambiguous too.
        with self.assertRaises(pm.LaunchProviderError):
            pm.launch_provider("shared-model", CROSS_CLUSTER_BLOCK)

    def run_cli(self, *args, stdin=CROSS_CLUSTER_BLOCK):
        return run_populate_cli(*args, stdin=stdin)

    def test_cli_selects_exactly_one_provider(self):
        res = self.run_cli("--launch-provider", "plain-b", "--base-url", "https://h/b/v1")
        self.assertEqual((res.returncode, res.stdout.strip()), (0, "custom:alcf-b"), res.stderr)

    def test_cli_zero_or_duplicate_exits_non_zero_without_secrets(self):
        for model in ("unknown/model", "dup-on-b"):
            with self.subTest(model=model):
                res = self.run_cli("--launch-provider", model, "--base-url", "https://h/b/v1")
                self.assertNotEqual(res.returncode, 0)
                self.assertEqual(res.stdout.strip(), "", "no provider may be printed on failure")
                self.assertIn(model, res.stderr)
                self.assertFalse(FAKE_TOKEN in res.stderr + res.stdout)

    def test_cli_lists_models_on_the_endpoint(self):
        res = self.run_cli("--list-models", "--base-url", "https://h/b/v1")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout.strip(), "dup-on-b, plain-b, shared-model")
        res = self.run_cli("--list-models", "--base-url", "https://nowhere/v1")
        self.assertNotEqual(res.returncode, 0)
        self.assertEqual(res.stdout.strip(), "")


class DelegationProviderContract(unittest.TestCase):
    """delegation_provider(model, providers_text, launch_base_url) CONTRACT (B7c-R1).

    Endpoint-pinned like the launch model: if any provider on the LAUNCH
    endpoint lists the model, the launch_provider rule applies there (exactly
    one, else raise). Otherwise exactly one provider on another endpoint must
    list it. Zero, or several non-launch providers, raise LaunchProviderError
    naming the candidates. Never touches the network.
    """

    def setUp(self):
        for name in ("_get_token", "_fetch_models"):
            p = mock.patch.object(pm, name, side_effect=AssertionError(f"{name}: network used"))
            p.start()
            self.addCleanup(p.stop)

    def test_model_on_several_clusters_prefers_the_launch_endpoint(self):
        self.assertEqual(pm.delegation_provider("shared-model", CROSS_CLUSTER_BLOCK, "https://h/a/v1"),
                         "custom:alcf-a-reasoning")
        self.assertEqual(pm.delegation_provider("shared-model", CROSS_CLUSTER_BLOCK, "https://h/b/v1/"),
                         "custom:alcf-b-reasoning")

    def test_model_unique_elsewhere_selects_that_provider(self):
        self.assertEqual(pm.delegation_provider("only-a", CROSS_CLUSTER_BLOCK, "https://h/b/v1"),
                         "custom:alcf-a-reasoning")
        self.assertEqual(pm.delegation_provider("plain-b", CROSS_CLUSTER_BLOCK, "https://h/a/v1"),
                         "custom:alcf-b")

    def test_model_on_several_non_launch_providers_fails_closed(self):
        for model, launch, names in (("shared-model", "https://h/c/v1",
                                      ("alcf-a-reasoning", "alcf-b-reasoning")),
                                     ("dup-on-b", "https://h/a/v1",
                                      ("alcf-b-extra", "alcf-b-reasoning"))):
            with self.subTest(model=model, launch=launch):
                with self.assertRaises(pm.LaunchProviderError) as cm:
                    pm.delegation_provider(model, CROSS_CLUSTER_BLOCK, launch)
                for name in (model, *names):
                    self.assertIn(name, str(cm.exception))

    def test_duplicate_on_the_launch_endpoint_fails_closed(self):
        # Pinned to the launch endpoint, where two providers list it: no guess,
        # and no escape to another endpoint either.
        with self.assertRaises(pm.LaunchProviderError) as cm:
            pm.delegation_provider("dup-on-b", CROSS_CLUSTER_BLOCK, "https://h/b/v1")
        self.assertIn("alcf-b-extra", str(cm.exception))

    def test_unlisted_model_fails_closed(self):
        with self.assertRaises(pm.LaunchProviderError) as cm:
            pm.delegation_provider("unknown/model", CROSS_CLUSTER_BLOCK, "https://h/a/v1")
        self.assertIn("unknown/model", str(cm.exception))

    def test_cli(self):
        res = run_populate_cli("--delegation-provider", "shared-model", "--base-url", "https://h/a/v1",
                               stdin=CROSS_CLUSTER_BLOCK)
        self.assertEqual((res.returncode, res.stdout.strip()), (0, "custom:alcf-a-reasoning"),
                         res.stderr)
        res = run_populate_cli("--delegation-provider", "shared-model", "--base-url", "https://h/c/v1",
                               stdin=CROSS_CLUSTER_BLOCK)
        self.assertNotEqual(res.returncode, 0)
        self.assertEqual(res.stdout.strip(), "")
        self.assertIn("alcf-a-reasoning", res.stderr)
        self.assertIn("alcf-b-reasoning", res.stderr)


class ExceptionLogHygieneContract(unittest.TestCase):
    """B7c-R6: catalog/context failures log the exception class and HTTP status,
    never str(e) (which can carry a URL, response body or credential)."""

    LEAKY = f"https://h/v1?token={FAKE_TOKEN}"

    def errors(self):
        import resolve_context_length as rcl
        http = urllib.error.HTTPError(self.LEAKY, 401, self.LEAKY, None, None)
        self.addCleanup(http.close)
        with_response = RuntimeError(self.LEAKY)
        with_response.response = mock.Mock(status_code=503)
        return rcl, ((http, "HTTPError, HTTP 401"), (with_response, "RuntimeError, HTTP 503"),
                     (ValueError(self.LEAKY), "ValueError"))

    def test_summary_is_class_and_status_only(self):
        rcl, cases = self.errors()
        for summary in (pm._exc_summary, rcl._exc_summary):
            for exc, want in cases:
                with self.subTest(fn=summary.__module__, exc=type(exc).__name__):
                    self.assertEqual(summary(exc), want)

    def test_failure_lines_do_not_echo_the_exception_text(self):
        rcl, cases = self.errors()
        exc = cases[0][0]
        with contextlib.redirect_stderr(io.StringIO()) as err:
            with mock.patch.object(pm, "_fetch_models", side_effect=exc):
                pm._resolve_cluster("x", "tok", lambda m: m, {"m": (131072, False)})
            with mock.patch.object(pm, "_get_token", side_effect=exc), \
                    mock.patch.object(pm, "_clusters", return_value=[]):
                pm.build_block()
            with mock.patch.object(rcl, "_fetch_max_lens", side_effect=exc), \
                    contextlib.redirect_stdout(io.StringIO()):
                rcl._mode_mapping("https://h/resource_server/x/vllm/v1", ["m"])
        self.assertNotIn(FAKE_TOKEN, err.getvalue())
        self.assertEqual(err.getvalue().count("HTTPError, HTTP 401"), 3, err.getvalue())


# Offline stand-in for /opt/alcf/populate_models.py inside the bounded
# entrypoint: runs the REAL module with every catalog/token fetcher disabled (so
# it emits its committed static fallback), optionally over a test cluster list.
# ALCF_TEST_LIVE=c1,c2 makes those clusters "discovered LIVE" instead: the real
# _resolve_cluster sees a successful (offline) fetch whose selection is the
# cluster's fallback mapping, so the block is identical but the source is live.
# ALCF_TEST_FORCE_LAUNCH_CONTEXT overrides the launch context the entrypoint
# passes, to reach the sub-floor injection refusal past the earlier floor guard.
OFFLINE_POPULATE = textwrap.dedent('''\
    import importlib.util, json, os, sys
    spec = importlib.util.spec_from_file_location("populate_models_real", {real!r})
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    live = set(filter(None, os.environ.get("ALCF_TEST_LIVE", "").split(",")))
    def _offline(*_a, **_k):
        raise RuntimeError("offline contract test")
    def _fetch(cluster, _token):
        if cluster in live:
            return []
        raise RuntimeError("offline contract test")
    pm._get_token = (lambda: "fake-catalog-token") if live else _offline
    pm._fetch_models = _fetch
    _real_clusters = pm._clusters
    def _clusters(include_metis=True, include_minerva=True, include_tara=True):
        if os.environ.get("ALCF_TEST_CLUSTERS"):
            out = [(c, p, b, None, {{m: tuple(v) for m, v in fb.items()}})
                   for c, p, b, fb in json.loads(os.environ["ALCF_TEST_CLUSTERS"])]
        else:
            out = _real_clusters(include_metis=include_metis, include_minerva=include_minerva,
                                 include_tara=include_tara)
        return [(c, p, b, (lambda _models, fb=fb: dict(fb)) if c in live else sel, fb)
                for c, p, b, sel, fb in out]
    pm._clusters = _clusters
    if os.environ.get("ALCF_TEST_FORCE_LAUNCH_CONTEXT"):
        os.environ["ALCF_LAUNCH_CONTEXT"] = os.environ["ALCF_TEST_FORCE_LAUNCH_CONTEXT"]
    rc = pm.main()
    state_file = os.environ.get("ALCF_LAUNCH_SOURCE_FILE")
    if state_file and "ALCF_TEST_FORCE_STATE" in os.environ:
        with open(state_file, "w", encoding="utf-8") as f:
            f.write(os.environ["ALCF_TEST_FORCE_STATE"])
    sys.exit(rc)
    ''')
CRASHING_POPULATE = "import sys\nsys.exit(1)\n"
ALL_LIVE = "sophia,metis,minerva,tara"

STUB_CTXLEN = 'import os\nprint(os.environ.get("TEST_CTXLEN", "128000"))\n'
ALCF_TOKEN_IN_HARNESS = "test-alcf-token"  # what the harness's fake alcf-tokens prints
HINT_LIST = ("openai/gpt-oss-120b, openai/gpt-oss-20b, google/gemma-4-31B-it, "
             "google/gemma-4-26B-A4B-it, nvidia/nemotron-3-super-120b, argonne/AuroraGPT-IT-v4-0125")
_ERR_PREFIX = re.compile(r"^\x1b\[31m\[agent-in-a-box\]\x1b\[0m ?")


class AlcfEntrypointMixin:
    """ALCF-mode run of the bounded entrypoint from tests/test_generic_inference.py."""

    def setUp(self):
        self.harness = tgi.EntrypointAndWorkflowContractTests(
            "test_rendered_credential_bearing_config_is_forced_to_mode_0600")
        self.harness.setUp()
        self.addCleanup(self.harness.doCleanups)
        alcf = self.harness.alcf_dir
        (alcf / "populate_models.py").write_text(
            OFFLINE_POPULATE.format(real=str(SCRIPTS / "populate_models.py")), encoding="utf-8")
        (alcf / "resolve_context_length.py").write_text(STUB_CTXLEN, encoding="utf-8")

    def run_alcf(self, **env):
        res = self.harness.run_entrypoint(**env)
        self.assertFalse(ALCF_TOKEN_IN_HARNESS in res.stdout + res.stderr,
                         "the inference token leaked into startup output")
        return res

    def config(self) -> dict:
        return yaml.safe_load((self.harness.home / "config.yaml").read_text(encoding="utf-8"))

    def startup_log(self) -> str:
        path = self.harness.home / "startup.log"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def assert_refused(self, res):
        self.assertEqual(res.returncode, 78, res.stderr)
        self.assertFalse((self.harness.home / "config.yaml").exists(),
                         "a refused launch must not leave a rendered config")

    @staticmethod
    def printed_valid_list(stderr: str) -> list:
        lines = [_ERR_PREFIX.sub("", ln) for ln in stderr.splitlines()]
        for i, line in enumerate(lines):
            if "Set ALCF_MODEL to one of these" in line:
                return sorted(x.strip() for x in lines[i + 1].split(",") if x.strip())
        raise AssertionError(f"no valid-model list in stderr:\n{stderr}")


class EntrypointFloorGuardContract(AlcfEntrypointMixin, unittest.TestCase):
    """I4: sub-floor launch model -> exit 78 + the GENERATED valid-model list."""

    def test_sub_floor_model_lists_generated_launch_endpoint_models(self):
        res = self.run_alcf(ALCF_MODEL="meta-llama/Llama-3.3-70B-Instruct", TEST_CTXLEN="16384")
        self.assert_refused(res)
        self.assertIn("REFUSING TO START", res.stderr)
        self.assertEqual(self.printed_valid_list(res.stderr), sorted(pm.SOPHIA_FALLBACK))

    def test_sub_floor_on_metis_lists_metis_models(self):
        res = self.run_alcf(ALCF_CLUSTER="metis", ALCF_MODEL="Mistral-Large-3-675B",
                            TEST_CTXLEN="8192")
        self.assert_refused(res)
        self.assertEqual(self.printed_valid_list(res.stderr), sorted(pm.METIS_FALLBACK))

    def test_hint_fallback_when_no_generated_model_on_endpoint(self):
        clusters = [["x", "alcf-x", "https://elsewhere/v1", {"m": [131072, False]}]]
        res = self.run_alcf(ALCF_MODEL="tiny/model", TEST_CTXLEN="16384",
                            ALCF_TEST_CLUSTERS=json.dumps(clusters))
        self.assert_refused(res)
        self.assertEqual(self.printed_valid_list(res.stderr),
                         sorted(x.strip() for x in HINT_LIST.split(",")))


class EntrypointLaunchProviderContract(AlcfEntrypointMixin, unittest.TestCase):
    """I1: the entrypoint maps the launch model to exactly one provider or exits 78."""

    def test_unique_model_selects_its_provider(self):
        for model, want in (("google/gemma-4-31B-it", "custom:alcf-sophia-reasoning"),
                            ("argonne/AuroraGPT-IT-v4-0125", "custom:alcf-sophia")):
            with self.subTest(model=model):
                res = self.run_alcf(ALCF_MODEL=model)
                self.assertEqual(res.returncode, 0, res.stderr)
                self.assertEqual(self.config()["model"]["provider"], want)

    def test_model_on_several_clusters_resolves_on_the_launch_endpoint(self):
        # muse-spark-bf16 is served by minerva AND tara (ALCF models doc,
        # 2026-11-23); ALCF_CLUSTER pins it to the launch endpoint's provider.
        for cluster, want in (("minerva", "custom:alcf-minerva-reasoning"),
                              ("tara", "custom:alcf-tara-reasoning")):
            with self.subTest(cluster=cluster):
                res = self.run_alcf(ALCF_CLUSTER=cluster, ALCF_MODEL="muse-spark-bf16")
                self.assertEqual(res.returncode, 0, res.stderr)
                self.assertEqual(self.config()["model"]["provider"], want)
                self.assertNotIn("not verified against the catalog", res.stderr)

    def test_unlisted_model_fails_closed(self):
        # With a LIVE catalog an absent model is really not served (or was
        # excluded), so there is nothing to inject: refuse. (The static-fallback
        # case is EntrypointCatalogOutageContract.)
        res = self.run_alcf(ALCF_MODEL="not-a-served/model", ALCF_TEST_LIVE=ALL_LIVE)
        self.assert_refused(res)
        self.assertIn("not-a-served/model", res.stderr)
        self.assertNotIn("catalog unavailable", res.stderr)
        self.assertEqual(self.printed_valid_list(res.stderr), sorted(pm.SOPHIA_FALLBACK))

    def test_model_in_two_providers_on_the_endpoint_fails_closed(self):
        clusters = [["sophia", "alcf-sophia", pm.SOPHIA_BASE, {"dup/model": [131072, False]}],
                    ["sophia", "alcf-shadow", pm.SOPHIA_BASE, {"dup/model": [131072, True]}]]
        res = self.run_alcf(ALCF_MODEL="dup/model", ALCF_TEST_CLUSTERS=json.dumps(clusters))
        self.assert_refused(res)
        self.assertIn("dup/model", res.stderr)
        self.assertIn("alcf-sophia", res.stderr)
        self.assertIn("alcf-shadow-reasoning", res.stderr)

    def test_delegation_model_resolves_to_exactly_one_provider(self):
        # Per the ALCF models doc (2026-11-23) each is served by one cluster only,
        # and not on the sophia launch endpoint: inkling-bf16 by tara (Minerva
        # serves inkling-2-small), gpt-oss-120b (no "openai/" prefix) by metis.
        for model, want in (("inkling-bf16", "custom:alcf-tara-reasoning"),
                            ("gpt-oss-120b", "custom:alcf-metis-reasoning")):
            with self.subTest(model=model):
                res = self.run_alcf(ALCF_MODEL="google/gemma-4-31B-it",
                                    ALCF_DELEGATION_MODEL=model)
                self.assertEqual(res.returncode, 0, res.stderr)
                self.assertEqual(self.config()["delegation"], {"provider": want, "model": model})

    def test_ambiguous_delegation_model_fails_closed(self):
        # muse-spark-bf16 is listed by minerva AND tara (ALCF models doc,
        # 2026-11-23), neither on the sophia launch endpoint: refuse, naming both.
        res = self.run_alcf(ALCF_MODEL="google/gemma-4-31B-it",
                            ALCF_DELEGATION_MODEL="muse-spark-bf16")
        self.assert_refused(res)
        self.assertIn("muse-spark-bf16", res.stderr)
        for name in ("alcf-minerva-reasoning", "alcf-tara-reasoning"):
            self.assertIn(name, res.stderr)

    def test_default_launch_delegating_muse_spark_stays_on_tara(self):
        # The image defaults (Tara, muse-spark-bf16) delegating muse-spark-bf16:
        # Minerva serves it too, but the launch endpoint wins.
        res = self.run_alcf(**image_model_env(ALCF_DELEGATION_MODEL="muse-spark-bf16"))
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(self.config()["model"]["provider"], "custom:alcf-tara-reasoning")
        self.assertEqual(self.config()["delegation"],
                         {"provider": "custom:alcf-tara-reasoning", "model": "muse-spark-bf16"})


class EntrypointDelegationContract(AlcfEntrypointMixin, unittest.TestCase):
    """B7c-R1: ALCF_DELEGATION_MODEL is endpoint-pinned like the launch model.

    Offered on the launch endpoint -> that provider; else exactly one provider
    elsewhere -> that one; else (none, or several elsewhere) exit 78 naming the
    candidates.
    """

    CLUSTERS = json.dumps([
        ["sophia", "alcf-sophia", pm.SOPHIA_BASE, {"launch/model": [131072, False],
                                                   "dup/model": [131072, False]}],
        ["x", "alcf-x", "https://h/x/v1", {"dup/model": [131072, True],
                                           "elsewhere/model": [131072, False],
                                           "unique/model": [131072, False]}],
        ["y", "alcf-y", "https://h/y/v1", {"dup/model": [131072, False],
                                           "elsewhere/model": [131072, True]}],
    ])

    def run_delegation(self, model, **env):
        return self.run_alcf(ALCF_MODEL="launch/model", ALCF_DELEGATION_MODEL=model,
                             ALCF_TEST_CLUSTERS=self.CLUSTERS, **env)

    def test_model_on_several_clusters_uses_the_launch_provider(self):
        res = self.run_delegation("dup/model")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(self.config()["delegation"],
                         {"provider": "custom:alcf-sophia", "model": "dup/model"})

    def test_real_fallback_gpt_oss_on_metis_launch_uses_metis(self):
        # gpt-oss-120b is on metis AND minerva; launching on metis pins it there.
        res = self.run_alcf(ALCF_CLUSTER="metis", ALCF_MODEL="gemma-4-31B-it",
                            ALCF_DELEGATION_MODEL="gpt-oss-120b")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(self.config()["delegation"],
                         {"provider": "custom:alcf-metis-reasoning", "model": "gpt-oss-120b"})

    def test_model_unique_elsewhere_uses_that_provider(self):
        res = self.run_delegation("unique/model")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(self.config()["delegation"],
                         {"provider": "custom:alcf-x", "model": "unique/model"})

    def test_model_on_several_non_launch_clusters_fails_closed(self):
        res = self.run_delegation("elsewhere/model")
        self.assert_refused(res)
        self.assertIn("REFUSING TO START", res.stderr)
        for name in ("elsewhere/model", "alcf-x", "alcf-y-reasoning"):
            self.assertIn(name, res.stderr)

    def test_unlisted_model_fails_closed(self):
        res = self.run_delegation("nowhere/model")
        self.assert_refused(res)
        self.assertIn("nowhere/model", res.stderr)


class EntrypointLaunchClusterContract(AlcfEntrypointMixin, unittest.TestCase):
    """B4b-I1: the launch cluster's providers are always generated, even when its
    ALCF_ENABLE_<CLUSTER> flag is 0 (the flag only trims the OTHER clusters)."""

    def test_disabled_launch_cluster_is_still_generated(self):
        res = self.run_alcf(ALCF_CLUSTER="metis", ALCF_ENABLE_METIS="0", ALCF_MODEL="gemma-4-31B-it")
        self.assertEqual(res.returncode, 0, res.stderr)
        cfg = self.config()
        self.assertEqual(cfg["model"]["provider"], "custom:alcf-metis-reasoning")
        self.assertIn("gemma-4-31B-it", cfg["providers"]["alcf-metis-reasoning"]["models"])
        overrides = [ln for ln in self.startup_log().splitlines()
                     if "ALCF_ENABLE_METIS=0" in ln and "launch cluster" in ln]
        self.assertEqual(len(overrides), 1, self.startup_log())

    def test_disabled_launch_cluster_is_generated_when_discovered_live(self):
        res = self.run_alcf(ALCF_CLUSTER="metis", ALCF_ENABLE_METIS="0", ALCF_MODEL="gpt-oss-120b",
                            ALCF_TEST_LIVE=ALL_LIVE)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(self.config()["model"]["provider"], "custom:alcf-metis-reasoning")

    def test_disabled_non_launch_cluster_is_still_dropped(self):
        res = self.run_alcf(ALCF_CLUSTER="sophia", ALCF_ENABLE_METIS="0",
                            ALCF_MODEL="google/gemma-4-31B-it")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertFalse([k for k in self.config()["providers"] if k.startswith("alcf-metis")])
        self.assertNotIn("launch cluster", self.startup_log())


class EntrypointClusterContract(AlcfEntrypointMixin, unittest.TestCase):
    """T1: all four serving clusters are launchable; Tara + muse-spark-bf16 is
    the default launch; an unknown cluster refuses with EX_CONFIG."""

    def test_default_launch_is_muse_spark_on_tara(self):
        # ALCF_CLUSTER / ALCF_MODEL unset (empty) -> the built-in defaults.
        res = self.run_alcf(ALCF_CLUSTER="", ALCF_MODEL="")
        self.assertEqual(res.returncode, 0, res.stderr)
        model = self.config()["model"]
        self.assertEqual(model["provider"], "custom:alcf-tara-reasoning")
        self.assertEqual(model["model"], "muse-spark-bf16")
        self.assertEqual(model["base_url"], pm.TARA_BASE)
        self.assertEqual(self.config()["providers"]["alcf-tara-reasoning"]["extra_body"],
                         {"max_tokens": pm.REASONING_MAX_TOKENS})

    def test_dockerfile_sets_cluster_but_not_model(self):
        # An image-wide ALCF_MODEL would survive `-e ALCF_CLUSTER=sophia` and aim
        # Muse Spark at Sophia; the per-cluster default lives in the entrypoint.
        env = " ".join(i for i in dockerfile_instructions() if i.startswith("ENV ")).split()
        self.assertIn("ALCF_CLUSTER=tara", env)
        self.assertFalse([t for t in env
                          if t.startswith("ALCF_CLUSTER=") and t != "ALCF_CLUSTER=tara"])
        self.assertFalse([t for t in env if t.startswith("ALCF_MODEL")], env)

    def test_entrypoint_has_one_default_model_table(self):
        body = (SCRIPTS / "entrypoint.sh").read_text(encoding="utf-8")
        code = [ln for ln in body.splitlines() if not ln.lstrip().startswith("#")]
        self.assertEqual(len([ln for ln in code if 'ALCF_MODEL="${ALCF_MODEL:-' in ln]), 1)
        defaults = re.findall(r'^\s*(\w+)\).*?default_model="([^"]+)"', "\n".join(code), re.M | re.S)
        self.assertEqual(dict(defaults), {"tara": "muse-spark-bf16", "sophia": "google/gemma-4-31B-it",
                                          "metis": "gemma-4-31B-it", "minerva": "muse-spark-bf16"})

    def test_image_defaults_launch_muse_spark_on_tara(self):
        res = self.run_alcf(**image_model_env())
        self.assertEqual(res.returncode, 0, res.stderr)
        model = self.config()["model"]
        self.assertEqual(model["model"], "muse-spark-bf16")
        self.assertEqual(model["provider"], "custom:alcf-tara-reasoning")
        self.assertEqual(model["base_url"], pm.TARA_BASE)

    def test_image_with_sophia_cluster_launches_gemma_on_sophia(self):
        # `docker run -e ALCF_CLUSTER=sophia` with ALCF_MODEL unset.
        res = self.run_alcf(**image_model_env(ALCF_CLUSTER="sophia"))
        self.assertEqual(res.returncode, 0, res.stderr)
        model = self.config()["model"]
        self.assertEqual(model["model"], "google/gemma-4-31B-it")
        self.assertEqual(model["provider"], "custom:alcf-sophia-reasoning")
        self.assertEqual(model["base_url"], pm.SOPHIA_BASE)
        self.assertNotIn("muse-spark-bf16", res.stderr)

    def test_explicit_model_wins_over_cluster_default(self):
        for cluster, want in (("sophia", "custom:alcf-sophia-reasoning"),
                              ("tara", "custom:alcf-tara-reasoning")):
            with self.subTest(cluster=cluster):
                res = self.run_alcf(**image_model_env(ALCF_CLUSTER=cluster,
                                                      ALCF_MODEL="openai/gpt-oss-120b"))
                self.assertEqual(res.returncode, 0, res.stderr)
                self.assertEqual(self.config()["model"]["model"], "openai/gpt-oss-120b")
                self.assertEqual(self.config()["model"]["provider"], want)

    def test_tara_and_minerva_resolve_their_endpoints(self):
        for cluster, model, base, provider in (
                ("tara", "muse-spark-bf16", pm.TARA_BASE, "custom:alcf-tara-reasoning"),
                ("minerva", "inkling-2-small", pm.MINERVA_BASE, "custom:alcf-minerva-reasoning")):
            with self.subTest(cluster=cluster):
                res = self.run_alcf(ALCF_CLUSTER=cluster, ALCF_MODEL=model)
                self.assertEqual(res.returncode, 0, res.stderr)
                # In the static fallback list, so nothing had to be injected.
                self.assertNotIn("not verified against the catalog", res.stderr)
                self.assertEqual(self.config()["model"]["base_url"], base)
                self.assertEqual(self.config()["model"]["provider"], provider)

    def test_metis_and_minerva_defaults_are_served_ids(self):
        # Live catalogs 2026-11-23: Metis serves gemma-4-31B-it (no google/
        # prefix); Minerva serves no gemma. Both defaults resolve without
        # catalog-outage injection.
        for cluster, model, provider in (("metis", "gemma-4-31B-it", "custom:alcf-metis-reasoning"),
                                         ("minerva", "muse-spark-bf16", "custom:alcf-minerva-reasoning")):
            with self.subTest(cluster=cluster):
                res = self.run_alcf(**image_model_env(ALCF_CLUSTER=cluster))
                self.assertEqual(res.returncode, 0, res.stderr)
                self.assertNotIn("not verified against the catalog", res.stderr)
                self.assertEqual(self.config()["model"]["model"], model)
                self.assertEqual(self.config()["model"]["provider"], provider)

    def test_sophia_default_model_is_unchanged(self):
        # ALCF_CLUSTER=sophia with ALCF_MODEL unset keeps the old Sophia default.
        res = self.run_alcf(ALCF_CLUSTER="sophia", ALCF_MODEL="")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(self.config()["model"]["model"], "google/gemma-4-31B-it")
        self.assertEqual(self.config()["model"]["provider"], "custom:alcf-sophia-reasoning")

    def test_unknown_cluster_exits_78_listing_all_clusters(self):
        res = self.run_alcf(ALCF_CLUSTER="polaris", ALCF_MODEL="muse-spark-bf16")
        self.assert_refused(res)
        self.assertIn("polaris", res.stderr)
        for cluster in ("sophia", "metis", "minerva", "tara"):
            self.assertIn(cluster, res.stderr)

    def test_delegation_gpt_oss_120b_on_tara_launch_pins_to_tara(self):
        # openai/gpt-oss-120b is listed by sophia AND tara; a tara launch pins it.
        res = self.run_alcf(ALCF_CLUSTER="tara", ALCF_MODEL="muse-spark-bf16",
                            ALCF_DELEGATION_MODEL="openai/gpt-oss-120b")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(self.config()["delegation"],
                         {"provider": "custom:alcf-tara-reasoning", "model": "openai/gpt-oss-120b"})

    def test_disabled_tara_launch_cluster_is_still_generated(self):
        res = self.run_alcf(ALCF_CLUSTER="tara", ALCF_ENABLE_TARA="0", ALCF_MODEL="muse-spark-bf16")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(self.config()["model"]["provider"], "custom:alcf-tara-reasoning")
        overrides = [ln for ln in self.startup_log().splitlines()
                     if "ALCF_ENABLE_TARA=0" in ln and "launch cluster" in ln]
        self.assertEqual(len(overrides), 1, self.startup_log())

    def test_disabled_minerva_launch_cluster_is_diagnosed(self):
        res = self.run_alcf(ALCF_CLUSTER="minerva", ALCF_ENABLE_MINERVA="0",
                            ALCF_MODEL="inkling-2-small")
        self.assertEqual(res.returncode, 0, res.stderr)
        overrides = [ln for ln in self.startup_log().splitlines()
                     if "ALCF_ENABLE_MINERVA=0" in ln and "launch cluster" in ln]
        self.assertEqual(len(overrides), 1, self.startup_log())

    def test_disabled_tara_non_launch_cluster_is_dropped(self):
        res = self.run_alcf(ALCF_CLUSTER="sophia", ALCF_ENABLE_TARA="0",
                            ALCF_MODEL="google/gemma-4-31B-it")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertFalse([k for k in self.config()["providers"] if k.startswith("alcf-tara")])

    def test_entrypoint_passes_the_tara_flag_to_both_populate_calls(self):
        body = (SCRIPTS / "entrypoint.sh").read_text(encoding="utf-8")
        self.assertEqual(body.count('ALCF_ENABLE_TARA="${ALCF_ENABLE_TARA:-1}"'), 2)


class EntrypointCatalogOutageContract(AlcfEntrypointMixin, unittest.TestCase):
    """B4b-I2: a launch model absent from the generated block.

    - launch cluster discovered LIVE  -> exit 78 (EntrypointLaunchProviderContract
      .test_unlisted_model_fails_closed);
    - launch cluster on STATIC FALLBACK -> the launch model is injected into the
      launch cluster's plain or -reasoning provider with the resolved launch
      context (only if >= the 64k floor), with a loud outage notice;
    - otherwise exit 78 with a distinct "catalog unavailable" message.
    """

    def assert_outage_notice(self, res, model):
        notice = [ln for ln in res.stderr.splitlines() if "catalog" in ln.lower() and model in ln]
        self.assertTrue(notice, f"no outage notice naming {model} on stderr:\n{res.stderr}")
        self.assertIn(model, self.startup_log())

    def test_fallback_injects_plain_launch_model(self):
        model = "acme/new-chat-model"
        res = self.run_alcf(ALCF_MODEL=model, TEST_CTXLEN="200000")
        self.assertEqual(res.returncode, 0, res.stderr)
        cfg = self.config()
        self.assertEqual(cfg["model"]["provider"], "custom:alcf-sophia")
        self.assertEqual(cfg["providers"]["alcf-sophia"]["models"][model], {"context_length": 200000})
        self.assertEqual(cfg["providers"]["alcf-sophia"]["api"], pm.SOPHIA_BASE)
        self.assert_outage_notice(res, model)

    def test_fallback_injects_reasoning_launch_model_on_the_launch_cluster(self):
        model = "gpt-oss-999b"
        res = self.run_alcf(ALCF_CLUSTER="metis", ALCF_MODEL=model, TEST_CTXLEN="131072")
        self.assertEqual(res.returncode, 0, res.stderr)
        cfg = self.config()
        self.assertEqual(cfg["model"]["provider"], "custom:alcf-metis-reasoning")
        self.assertEqual(cfg["providers"]["alcf-metis-reasoning"]["models"][model],
                         {"context_length": 131072})
        for key, entry in cfg["providers"].items():
            if not key.startswith("alcf-metis"):
                self.assertNotIn(model, entry["models"], f"injected into non-launch cluster {key}")
        self.assert_outage_notice(res, model)

    def test_fallback_model_already_listed_is_not_injected_or_announced(self):
        res = self.run_alcf(ALCF_MODEL="google/gemma-4-31B-it")
        self.assertEqual(res.returncode, 0, res.stderr)
        cfg = self.config()
        self.assertEqual(cfg["providers"]["alcf-sophia-reasoning"]["models"]["google/gemma-4-31B-it"],
                         {"context_length": pm.SOPHIA_FALLBACK["google/gemma-4-31B-it"][0]})
        self.assertNotIn("catalog unavailable", res.stderr)

    def test_fallback_sub_floor_launch_context_refuses_distinctly(self):
        model = "acme/new-chat-model"
        res = self.run_alcf(ALCF_MODEL=model, ALCF_TEST_FORCE_LAUNCH_CONTEXT="32768")
        self.assert_refused(res)
        self.assertIn("catalog unavailable", res.stderr)
        self.assertIn(f"'{model}' is not in the static fallback list", res.stderr)
        self.assertEqual(self.printed_valid_list(res.stderr), sorted(pm.SOPHIA_FALLBACK))

    def test_generation_failure_refuses_distinctly_for_unlisted_model(self):
        (self.harness.alcf_dir / "populate_models.py").write_text(CRASHING_POPULATE, encoding="utf-8")
        model = "acme/new-chat-model"
        res = self.run_alcf(ALCF_MODEL=model)
        self.assert_refused(res)
        self.assertIn(f"'{model}' is not in the static fallback list", res.stderr)

    def test_live_catalog_never_injects(self):
        res = self.run_alcf(ALCF_MODEL="acme/new-chat-model", ALCF_TEST_LIVE=ALL_LIVE,
                            TEST_CTXLEN="200000")
        self.assert_refused(res)
        self.assertNotIn("catalog unavailable", res.stderr)
        self.assertIn("does not map to exactly", res.stderr)


RUN_FILES = {"ctxlen.log", "populate_models.log", "launch_source", "launch_provider.log",
             "generic_inference.log"}


def assert_private_run_dir(test, tmpdir: Path) -> Path:
    """Exactly one mktemp run dir (mode 0700) whose files are all mode 0600."""
    dirs = sorted(tmpdir.glob("aib-run.*"))
    test.assertEqual(len(dirs), 1, dirs)
    rundir = dirs[0]
    test.assertEqual(rundir.stat().st_mode & 0o777, 0o700)
    test.assertEqual({p.name for p in rundir.iterdir()}, RUN_FILES)
    for path in rundir.iterdir():
        test.assertEqual(path.stat().st_mode & 0o777, 0o600, path.name)
    return rundir


class EntrypointRunDirContract(AlcfEntrypointMixin, unittest.TestCase):
    """B7c-R2: every entrypoint diagnostic/state file lives in one private
    mktemp run dir under $TMPDIR, never at a fixed /tmp path."""

    def test_no_fixed_tmp_paths(self):
        body = (SCRIPTS / "entrypoint.sh").read_text(encoding="utf-8")
        self.assertNotIn("/tmp/", body)
        self.assertIn('RUNDIR="$(mktemp -d "${TMPDIR:-/tmp}/aib-run.XXXXXX")"', body)
        self.assertRegex(body, r"\(umask 077 && : > \"\$CTXLEN_LOG\"")

    def test_alcf_run_writes_only_into_the_private_run_dir(self):
        res = self.run_alcf(ALCF_MODEL="google/gemma-4-31B-it")
        self.assertEqual(res.returncode, 0, res.stderr)
        rundir = assert_private_run_dir(self, self.harness.tmpdir)
        self.assertEqual((rundir / "launch_source").read_text(encoding="utf-8").strip(), "fallback")
        self.assertIn("[populate_models]", (rundir / "populate_models.log").read_text(encoding="utf-8"))
        self.assertIn(f"Run diagnostics directory: {rundir}", self.startup_log())
        for path in rundir.iterdir():
            self.assertNotIn(ALCF_TOKEN_IN_HARNESS, path.read_text(encoding="utf-8"), path.name)

    def test_refusal_diagnostic_is_read_from_the_run_dir(self):
        clusters = [["sophia", "alcf-sophia", pm.SOPHIA_BASE, {"dup/model": [131072, False]}],
                    ["sophia", "alcf-shadow", pm.SOPHIA_BASE, {"dup/model": [131072, True]}]]
        res = self.run_alcf(ALCF_MODEL="dup/model", ALCF_TEST_CLUSTERS=json.dumps(clusters))
        self.assert_refused(res)
        rundir = assert_private_run_dir(self, self.harness.tmpdir)
        logged = (rundir / "launch_provider.log").read_text(encoding="utf-8")
        self.assertIn("alcf-shadow-reasoning", logged)
        self.assertIn("alcf-shadow-reasoning", res.stderr)


EXPIRY_WORDING = ("access tokens auto-refresh for up to 30 days, after which, or after a "
                  "high-assurance timeout, a new login is required")


class EntrypointInferenceLoginContract(AlcfEntrypointMixin, unittest.TestCase):
    """ALCF mode gates startup on a LIVE `alcf-tokens test-token inference`
    (via inference_token_check.py), never starts a login without a TTY, and
    keeps $HERMES_HOME/.inference_token_status honest."""

    MODEL = "google/gemma-4-31B-it"

    def status_file(self) -> str:
        path = self.harness.home / ".inference_token_status"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def assert_no_login(self):
        calls = self.harness.token_cli_calls()
        self.assertFalse([c for c in calls if c.split()[:1] == ["login"]],
                         f"alcf-tokens login was invoked: {calls}")

    def test_ok_starts_after_live_check(self):
        res = self.run_alcf(ALCF_MODEL=self.MODEL)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("ALCF inference login verified live (whoami OK)", self.startup_log())
        self.assertIn("test-token inference", self.harness.token_cli_calls())
        self.assertEqual(self.status_file(), "")
        self.assert_no_login()

    def test_rejected_login_without_tty_exits_77_without_login(self):
        res = self.run_alcf(ALCF_MODEL=self.MODEL, FAKE_TEST_TOKEN="auth")
        self.assertEqual(res.returncode, 77, res.stderr)
        self.assertIn("HTTP 401", res.stderr)
        stderr = "\n".join(_ERR_PREFIX.sub("", ln) for ln in res.stderr.splitlines())
        restart = stderr.find("restart interactively: docker run -it")
        oneshot = stderr.find("docker run -it --rm -v <your volume>:/opt/data "
                              f"--entrypoint {self.harness.venv_bin}/alcf-tokens <image> login")
        self.assertGreaterEqual(restart, 0, stderr)
        self.assertGreater(oneshot, restart, "the one-shot login line must follow the -it line")
        # The container has exited: `docker exec` into it is impossible.
        self.assertNotIn("docker exec", stderr)
        self.assertNotIn("bearer", res.stdout + res.stderr, "the response body was echoed")
        self.assert_no_login()
        self.assertFalse((self.harness.home / "config.yaml").exists())

    def test_failing_get_token_is_auth(self):
        """get-token AuthError (no usable stored login) -> auth -> exit 77."""
        res = self.run_alcf(ALCF_MODEL=self.MODEL, FAKE_GET_TOKEN="auth")
        self.assertEqual(res.returncode, 77, res.stderr)
        self.assertIn("get-token inference failed: no valid stored login", res.stderr)
        self.assertNotIn("test-token inference", self.harness.token_cli_calls())
        self.assert_no_login()

    def test_get_token_network_failure_without_tty_is_unreachable_not_77(self):
        """A refresh that cannot reach auth.globus.org is an uncaught traceback
        from get-token, not a rejected login: continue with UNREACHABLE."""
        res = self.run_alcf(ALCF_MODEL=self.MODEL, FAKE_GET_TOKEN="network")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(self.status_file().startswith("UNREACHABLE\n"), self.status_file())
        self.assertIn("get-token network error", res.stderr)
        self.assertNotIn("Traceback", res.stdout + res.stderr)
        self.assertNotIn("test-token inference", self.harness.token_cli_calls())
        self.assert_no_login()

    def test_unreachable_gateway_continues_with_status(self):
        res = self.run_alcf(ALCF_MODEL=self.MODEL, FAKE_TEST_TOKEN="unavailable")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue((self.harness.home / "config.yaml").exists())
        self.assertIn("gateway is unreachable (HTTP 503)", res.stderr)
        self.assertTrue(self.status_file().startswith("UNREACHABLE\n"), self.status_file())
        self.assert_no_login()

    def test_unrecognised_check_output_continues_as_unknown(self):
        res = self.run_alcf(ALCF_MODEL=self.MODEL, FAKE_TEST_TOKEN="garbage")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(self.status_file().startswith("UNKNOWN\n"), self.status_file())
        self.assertNotIn("not json at all", res.stdout + res.stderr + self.status_file())

    def test_token_check_0_uses_legacy_local_gate(self):
        res = self.run_alcf(ALCF_MODEL=self.MODEL, ALCF_TOKEN_CHECK="0", FAKE_TEST_TOKEN="auth")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("ALCF_TOKEN_CHECK=0", self.startup_log())
        calls = self.harness.token_cli_calls()
        self.assertIn("get-token inference", calls)
        self.assertNotIn("test-token inference", calls)
        self.assertEqual(self.status_file(), "")

    def test_generic_mode_never_runs_the_live_check(self):
        tgi._CatalogHandler.payload = {"data": [{"id": "selected/model", "context_length": 131072}]}
        res = self.harness.run_entrypoint(
            INFERENCE_URL=self.harness.base_url, INFERENCE_TOKEN=tgi.SECRET,
            INFERENCE_MODEL="selected/model", FAKE_TEST_TOKEN="auth")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertNotIn("test-token inference", self.harness.token_cli_calls())

    def test_rendered_config_holds_no_token(self):
        res = self.run_alcf(ALCF_MODEL=self.MODEL)
        self.assertEqual(res.returncode, 0, res.stderr)
        text = (self.harness.home / "config.yaml").read_text(encoding="utf-8")
        self.assertNotIn(ALCF_TOKEN_IN_HARNESS, text)
        cfg = yaml.safe_load(text)
        self.assertEqual(cfg["model"]["api_key"], "")
        key_cmd = f"{self.harness.venv_bin}/alcf-tokens get-token inference"
        for name, entry in cfg["providers"].items():
            with self.subTest(provider=name):
                self.assertNotIn("api_key", entry)
                self.assertEqual(entry["key_cmd"], key_cmd)

    # --- the background refresh tick (step 6 runs token_refresh_tick) --------

    def tick(self, before: str, **env):
        self.harness.stop_with(f"{before}\ntoken_refresh_tick")
        return self.run_alcf(ALCF_MODEL=self.MODEL, **env)

    def test_tick_rejected_login_writes_expired(self):
        res = self.tick(f"printf auth > '{self.harness.token_mode}'")
        self.assertEqual(res.returncode, 0, res.stderr)
        status = self.status_file()
        self.assertTrue(status.startswith("EXPIRED\n"), status)
        self.assertIn(f"ALCF login expired or rejected (HTTP 401); {EXPIRY_WORDING}", status)
        self.assertIn("ALCF INFERENCE LOGIN EXPIRED OR REJECTED", res.stderr)
        self.assert_no_login()

    def test_tick_failing_get_token_writes_expired(self):
        res = self.tick(f"printf auth > '{self.harness.get_token_mode}'")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(self.status_file().startswith("EXPIRED\n"), self.status_file())
        self.assertIn(EXPIRY_WORDING, self.status_file())

    def test_tick_get_token_network_failure_is_unreachable_not_expired(self):
        res = self.tick(f"printf network > '{self.harness.get_token_mode}'")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(self.status_file().startswith("UNREACHABLE\n"), self.status_file())
        self.assertNotIn("EXPIRED", res.stderr + self.status_file())
        self.assert_no_login()

    def test_tick_unreachable_is_not_expired(self):
        res = self.tick(f"printf unavailable > '{self.harness.token_mode}'")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(self.status_file().startswith("UNREACHABLE\n"), self.status_file())
        self.assertNotIn("EXPIRED", res.stderr + self.status_file())

    def test_tick_unknown_is_not_expired(self):
        res = self.tick(f"printf garbage > '{self.harness.token_mode}'")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(self.status_file().startswith("UNKNOWN\n"), self.status_file())
        self.assertNotIn("EXPIRED", res.stderr + self.status_file())

    def test_tick_ok_rerenders_and_clears_status(self):
        # Start degraded (UNREACHABLE), then the gateway recovers.
        res = self.tick(f"printf ok > '{self.harness.token_mode}'",
                        FAKE_TEST_TOKEN="unavailable")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("Inference login verified; config re-rendered.", self.startup_log())
        self.assertEqual(self.status_file(), "")
        self.assertNotIn(ALCF_TOKEN_IN_HARNESS,
                         (self.harness.home / "config.yaml").read_text(encoding="utf-8"))


# Recording stand-ins for the two token-consuming helpers: each records whether
# the entrypoint put the inference token in its environment, then mints the
# token the way the real helper does (the real module's _get_token, i.e. the
# alcf-tokens CLI) and records whether that produced the harness's fake token.
_HELPER_RECORD_PRELUDE = textwrap.dedent('''\
    import importlib.util, json, os, sys
    def record(helper, got):
        leaked = sorted(k for k, v in os.environ.items() if {fake!r} in v)
        with open({record!r}, "a", encoding="utf-8") as f:
            f.write(json.dumps({{"helper": helper, "leaked": leaked, "got_fake": got,
                                 "env_names": sorted(k for k in os.environ
                                                     if k in ("ALCF_INFER_TOKEN",
                                                              "ALCF_ACCESS_TOKEN"))}}) + "\\n")
    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    ''')
RECORDING_POPULATE = _HELPER_RECORD_PRELUDE + textwrap.dedent('''\
    pm = load({real!r}, "populate_models_real")
    def _fetch(_cluster, token):
        record("populate_models", token == {fake!r})
        raise RuntimeError("offline contract test")
    pm._fetch_models = _fetch
    sys.exit(pm.main())
    ''')
RECORDING_CTXLEN = _HELPER_RECORD_PRELUDE + textwrap.dedent('''\
    rcl = load({real!r}, "resolve_context_length_real")
    record("resolve_context_length", rcl._get_token() == {fake!r})
    print(128000)
    ''')


class EntrypointHelperTokenContract(AlcfEntrypointMixin, unittest.TestCase):
    """I1: in ALCF mode the entrypoint passes NO bearer token to any helper via
    the environment; the helpers mint it themselves through the alcf-tokens CLI
    (ALCF_TOKENS_BIN) and keep it in memory."""

    def setUp(self):
        super().setUp()
        self.record = self.harness.runtime / "helper-env.jsonl"
        alcf = self.harness.alcf_dir
        fmt = dict(fake=ALCF_TOKEN_IN_HARNESS, record=str(self.record))
        (alcf / "populate_models.py").write_text(
            RECORDING_POPULATE.format(real=str(SCRIPTS / "populate_models.py"), **fmt),
            encoding="utf-8")
        (alcf / "resolve_context_length.py").write_text(
            RECORDING_CTXLEN.format(real=str(SCRIPTS / "resolve_context_length.py"), **fmt),
            encoding="utf-8")

    def test_helpers_get_no_token_env_but_mint_the_token_via_the_cli(self):
        res = self.run_alcf(ALCF_MODEL="google/gemma-4-31B-it")
        self.assertEqual(res.returncode, 0, res.stderr)
        rows = [json.loads(ln) for ln in self.record.read_text(encoding="utf-8").splitlines()]
        self.assertEqual({r["helper"] for r in rows}, {"populate_models", "resolve_context_length"},
                         rows)
        for row in rows:
            with self.subTest(helper=row["helper"]):
                self.assertEqual(row["leaked"], [], "a helper env carried the inference token")
                self.assertEqual(row["env_names"], [])
                self.assertTrue(row["got_fake"], "the helper did not obtain the token via the CLI")
        self.assertIn("get-token inference", self.harness.token_cli_calls())

    def test_no_token_env_override_remains(self):
        for name in ("entrypoint.sh", "populate_models.py", "resolve_context_length.py"):
            with self.subTest(file=name):
                self.assertNotIn("ALCF_INFER_TOKEN",
                                 (SCRIPTS / name).read_text(encoding="utf-8"))


class HelperGetTokenContract(unittest.TestCase):
    """The helpers' _get_token runs ALCF_TOKENS_BIN and ignores any token env."""

    def _cli(self, tmp: str, body: str) -> str:
        cli = Path(tmp) / "alcf-tokens"
        cli.write_text("#!/bin/sh\n" + body, encoding="utf-8")
        cli.chmod(0o755)
        return str(cli)

    def _modules(self):
        sys.path.insert(0, str(SCRIPTS))
        self.addCleanup(sys.path.remove, str(SCRIPTS))
        import resolve_context_length as rcl
        return (("populate_models", pm), ("resolve_context_length", rcl))

    def test_cli_token_is_returned_and_env_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            cli = self._cli(tmp, '[ "$1 $2" = "get-token inference" ] || exit 2\n'
                                 f"printf '%s\\n' '{FAKE_TOKEN}'\n")
            for name, mod in self._modules():
                with self.subTest(helper=name), mock.patch.dict(
                        os.environ, {"ALCF_TOKENS_BIN": cli, "ALCF_INFER_TOKEN": "env-token"}):
                    self.assertEqual(mod._get_token(), FAKE_TOKEN)

    def test_cli_failure_raises_without_echoing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            cli = self._cli(tmp, f"printf '%s\\n' '{FAKE_TOKEN}'\n"
                                 f"printf '%s\\n' 'refresh {FAKE_TOKEN}' >&2\nexit 1\n")
            for name, mod in self._modules():
                with self.subTest(helper=name), mock.patch.dict(os.environ, {"ALCF_TOKENS_BIN": cli}):
                    with self.assertRaises(RuntimeError) as ctx:
                        mod._get_token()
                    self.assertNotIn(FAKE_TOKEN, str(ctx.exception))


class ResolveContextLengthContract(unittest.TestCase):
    """D1: the launch context window is read from max_model_len (Sophia) or,
    when that is absent, capabilities.context_window_tokens (Minerva/Tara),
    the same precedence populate_models.py applies per cluster."""

    TARA = "https://inference-api.alcf.anl.gov/resource_server/tara/api/v1"
    SOPHIA = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
    TARA_MODELS = {"data": [
        {"id": "muse-spark-bf16", "object": "model", "cluster": "tara", "framework": "vllm",
         "display_name": "Muse Spark", "capabilities": {"context_window_tokens": 1048576,
                                                       "reasoning": {"supported": True}}},
        {"id": "mistral-large-4", "object": "model", "cluster": "tara", "framework": "vllm",
         "display_name": "Mistral", "capabilities": {"context_window_tokens": 262144}},
    ]}
    SOPHIA_MODELS = {"data": [
        {"id": "google/gemma-4-31B-it", "framework": "vllm", "max_model_len": 128000},
        # both present: max_model_len wins
        {"id": "both", "max_model_len": 65536, "capabilities": {"context_window_tokens": 999999}},
        {"id": "neither", "framework": "vllm"},
        {"id": "bad", "max_model_len": 0, "capabilities": {"context_window_tokens": "big"}},
    ]}

    def setUp(self):
        sys.path.insert(0, str(SCRIPTS))
        self.addCleanup(sys.path.remove, str(SCRIPTS))
        import resolve_context_length as rcl
        self.rcl = rcl

    def run_main(self, doc: dict, *argv: str) -> tuple:
        body = json.dumps(doc).encode()
        out, err = io.StringIO(), io.StringIO()
        env = {k: v for k, v in os.environ.items() if k != "ALCF_CONTEXT_LENGTH"}
        with mock.patch.object(self.rcl, "_get_token", return_value=FAKE_TOKEN), \
                mock.patch.object(self.rcl.urllib.request, "urlopen",
                                  side_effect=lambda *_a, **_k: io.BytesIO(body)), \
                mock.patch.object(sys, "argv", ["resolve_context_length.py", *argv]), \
                mock.patch.dict(os.environ, env, clear=True), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(self.rcl.main(), 0)
        self.assertNotIn(FAKE_TOKEN, out.getvalue() + err.getvalue())
        return out.getvalue(), err.getvalue()

    def test_tara_shaped_value_reads_capabilities_window(self):
        out, err = self.run_main(self.TARA_MODELS, self.TARA, "muse-spark-bf16")
        self.assertEqual(out.strip(), "1048576", err)
        self.assertNotIn("fallback", err)

    def test_tara_shaped_mapping_reads_capabilities_window(self):
        out, _ = self.run_main(self.TARA_MODELS, self.TARA, "--mode", "mapping",
                               "--models", "muse-spark-bf16,mistral-large-4")
        self.assertEqual(yaml.safe_load(out), {"models": {
            "muse-spark-bf16": {"context_length": 1048576},
            "mistral-large-4": {"context_length": 262144}}})

    def test_sophia_shaped_unchanged_and_max_model_len_preferred(self):
        out, _ = self.run_main(self.SOPHIA_MODELS, self.SOPHIA, "google/gemma-4-31B-it")
        self.assertEqual(out.strip(), "128000")
        out, _ = self.run_main(self.SOPHIA_MODELS, self.SOPHIA, "both")
        self.assertEqual(out.strip(), "65536")

    def test_neither_field_or_invalid_values_fall_back(self):
        for model in ("neither", "bad"):
            with self.subTest(model=model):
                out, err = self.run_main(self.SOPHIA_MODELS, self.SOPHIA, model)
                self.assertEqual(out.strip(), str(self.rcl.DEFAULT_CONTEXT))
                self.assertIn("fallback", err)


class EntrypointLaunchSourceValidationContract(AlcfEntrypointMixin, unittest.TestCase):
    """B7c-R3: the launch-cluster state file must hold one of the values
    populate_models writes; anything else is a fixed refusal (exit 78)."""

    def test_unrecognised_state_refuses_without_echoing_it(self):
        for state in ("bogus-state-xyz", "generation-failed", "live\nfallback",
                      "live; touch pwned", "LIVE"):
            with self.subTest(state=state):
                res = self.run_alcf(ALCF_MODEL="google/gemma-4-31B-it", ALCF_TEST_FORCE_STATE=state)
                self.assert_refused(res)
                self.assertIn("unrecognised launch-cluster state", res.stderr)
                self.assertNotIn("bogus-state-xyz", res.stderr)
                self.assertNotIn("pwned", res.stderr)

    def test_recognised_states_are_accepted(self):
        for state in ("live", "fallback", "fallback\n", ""):
            with self.subTest(state=state):
                res = self.run_alcf(ALCF_MODEL="google/gemma-4-31B-it", ALCF_TEST_FORCE_STATE=state)
                self.assertEqual(res.returncode, 0, res.stderr)


class EntrypointOutputCapValidationContract(AlcfEntrypointMixin, unittest.TestCase):
    """B7c-R5: ALCF_MAX_TOKENS / ALCF_REASONING_MAX_TOKENS, when set, must be
    positive integers; a bad value exits 78 instead of silently using a default."""

    BAD = ("0", "-5", "abc", "12.5", "1e3", " 2048", "2048 ", "02048", "2048\nabc", "bogus-cap-xyz")

    def test_bad_caps_refuse(self):
        for var in ("ALCF_MAX_TOKENS", "ALCF_REASONING_MAX_TOKENS"):
            for value in self.BAD:
                with self.subTest(var=var, value=value):
                    res = self.run_alcf(ALCF_MODEL="google/gemma-4-31B-it", **{var: value})
                    self.assert_refused(res)
                    self.assertIn(f"{var} must be a positive integer", res.stderr)
                    self.assertNotIn("bogus-cap-xyz", res.stderr)

    def test_valid_caps_reach_the_providers(self):
        res = self.run_alcf(ALCF_MODEL="google/gemma-4-31B-it", ALCF_MAX_TOKENS="4096",
                            ALCF_REASONING_MAX_TOKENS="16384")
        self.assertEqual(res.returncode, 0, res.stderr)
        providers = self.config()["providers"]
        self.assertEqual(providers["alcf-sophia"]["extra_body"]["max_tokens"], 4096)
        self.assertEqual(providers["alcf-sophia-reasoning"]["extra_body"]["max_tokens"], 16384)

    def test_empty_caps_mean_unset(self):
        res = self.run_alcf(ALCF_MODEL="google/gemma-4-31B-it", ALCF_MAX_TOKENS="",
                            ALCF_REASONING_MAX_TOKENS="")
        self.assertEqual(res.returncode, 0, res.stderr)
        providers = self.config()["providers"]
        self.assertEqual(providers["alcf-sophia"]["extra_body"]["max_tokens"], 2048)
        self.assertEqual(providers["alcf-sophia-reasoning"]["extra_body"]["max_tokens"], 12288)


class PopulateLaunchClusterContract(unittest.TestCase):
    """populate_models side of B4b-I1/I2, driven through build_block() offline.

    Inputs: ALCF_LAUNCH_CLUSTER / ALCF_LAUNCH_MODEL / ALCF_LAUNCH_CONTEXT. The
    launch cluster's source is reported to ALCF_LAUNCH_SOURCE_FILE as one of
    live | fallback | fallback-injected | fallback-missing.
    """

    CLUSTERS = [("x", "alcf-x", "https://h/x/v1", None, {"chatty": (128000, False)}),
                ("y", "alcf-y", "https://h/y/v1", None, {"thinky": (131072, True)})]

    def build(self, *, live=(), **env):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        source = Path(tmp.name) / "launch-source"
        clusters = [(c, p, b, (lambda _m, fb=fb: dict(fb)) if c in live else None, fb)
                    for c, p, b, _s, fb in self.CLUSTERS]

        def fetch(cluster, _token):
            if cluster in live:
                return []
            raise RuntimeError("offline test")

        full_env = {"ALCF_LAUNCH_SOURCE_FILE": str(source), **env}
        with mock.patch.dict(os.environ, full_env), \
                mock.patch.object(pm, "_get_token", return_value="fake-catalog-token"), \
                mock.patch.object(pm, "_fetch_models", side_effect=fetch), \
                mock.patch.object(pm, "_clusters", return_value=clusters), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            block = pm.build_block()
        state = source.read_text(encoding="utf-8").strip() if source.exists() else None
        return yaml.safe_load(block)["providers"], state, err.getvalue()

    def test_fallback_injects_absent_launch_model_by_reasoning_class(self):
        for model, key in (("acme/plain", "alcf-y"), ("acme/gpt-oss-x", "alcf-y-reasoning")):
            with self.subTest(model=model):
                providers, state, err = self.build(ALCF_LAUNCH_CLUSTER="y", ALCF_LAUNCH_MODEL=model,
                                                   ALCF_LAUNCH_CONTEXT="100000")
                self.assertEqual(providers[key]["models"][model], {"context_length": 100000})
                self.assertEqual(state, "fallback-injected")
                self.assertIn(model, err)
                self.assertNotIn(model, providers["alcf-x"]["models"])

    def test_live_launch_cluster_never_injects(self):
        providers, state, _err = self.build(live=("x", "y"), ALCF_LAUNCH_CLUSTER="y",
                                            ALCF_LAUNCH_MODEL="acme/plain", ALCF_LAUNCH_CONTEXT="100000")
        self.assertNotIn("alcf-y", providers)
        self.assertEqual(state, "live")

    def test_fallback_with_sub_floor_or_unknown_context_does_not_inject(self):
        for ctx in (str(pm.MIN_CONTEXT - 1), "", "not-a-number", "0"):
            with self.subTest(ctx=ctx):
                providers, state, _err = self.build(ALCF_LAUNCH_CLUSTER="y", ALCF_LAUNCH_MODEL="acme/plain",
                                                    ALCF_LAUNCH_CONTEXT=ctx)
                self.assertNotIn("alcf-y", providers)
                self.assertEqual(state, "fallback-missing")

    def test_fallback_with_listed_model_reports_plain_fallback(self):
        providers, state, _err = self.build(ALCF_LAUNCH_CLUSTER="y", ALCF_LAUNCH_MODEL="thinky",
                                            ALCF_LAUNCH_CONTEXT="100000")
        self.assertEqual(providers["alcf-y-reasoning"]["models"]["thinky"], {"context_length": 131072})
        self.assertEqual(state, "fallback")

    def test_without_launch_inputs_nothing_changes(self):
        providers, state, _err = self.build()
        self.assertEqual(set(providers), {"alcf-x", "alcf-y-reasoning"})
        self.assertIsNone(state)

    def test_disabled_launch_cluster_is_included_with_a_diagnostic(self):
        calls = []
        real = pm._clusters

        def spy(include_metis=True, include_minerva=True, include_tara=True):
            calls.append((include_metis, include_minerva, include_tara))
            return real(include_metis=include_metis, include_minerva=include_minerva,
                        include_tara=include_tara)

        off = (False, False, False)
        for launch, want in (("metis", (True, False, False)),
                             ("minerva", (False, True, False)),
                             ("tara", (False, False, True)),
                             ("sophia", (False, False, False))):
            with self.subTest(launch=launch):
                calls.clear()
                with mock.patch.dict(os.environ, {"ALCF_LAUNCH_CLUSTER": launch}), \
                        mock.patch.object(pm, "_get_token", side_effect=RuntimeError("offline test")), \
                        mock.patch.object(pm, "_clusters", side_effect=spy), \
                        contextlib.redirect_stderr(io.StringIO()) as err:
                    block = pm.build_block(include_metis=off[0], include_minerva=off[1],
                                           include_tara=off[2])
                self.assertEqual(calls, [want])
                names = set(yaml.safe_load(block)["providers"])
                self.assertEqual(any(n.startswith("alcf-metis") for n in names), want[0])
                self.assertEqual(any(n.startswith("alcf-minerva") for n in names), want[1])
                self.assertEqual(any(n.startswith("alcf-tara") for n in names), want[2])
                overridden = launch in ("metis", "minerva", "tara")
                self.assertEqual("launch cluster" in err.getvalue(), overridden, err.getvalue())


class GenericRenderMixin:
    """Reuses the bounded-entrypoint harness from tests/test_generic_inference.py."""

    def setUp(self):
        self.harness = tgi.EntrypointAndWorkflowContractTests(
            "test_rendered_credential_bearing_config_is_forced_to_mode_0600")
        self.harness.setUp()
        self.addCleanup(self.harness.doCleanups)
        tgi._CatalogHandler.payload = {"data": [
            {"id": "selected/model", "context_length": 131072},
            {"id": "other-model", "context_length": 70000},
        ]}

    def render(self) -> str:
        result = self.harness.run_entrypoint(
            INFERENCE_URL=self.harness.base_url,
            INFERENCE_TOKEN=tgi.SECRET,
            INFERENCE_MODEL="selected/model",
            INFERENCE_MAX_TOKENS="4096",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return (self.harness.home / "config.yaml").read_text(encoding="utf-8")


class GenericRunDirAndCapContract(GenericRenderMixin, unittest.TestCase):
    """B7c-R2/R5 in generic mode: the generic_inference log lives in the private
    run dir; a bad ALCF cap refuses before the endpoint is ever contacted."""

    def test_generic_log_lives_in_the_private_run_dir(self):
        self.render()
        rundir = assert_private_run_dir(self, self.harness.tmpdir)
        log = (rundir / "generic_inference.log").read_text(encoding="utf-8")
        self.assertIn("selected/model", log)
        self.assertNotIn(tgi.SECRET, log)

    def test_bad_cap_refuses_before_contacting_the_endpoint(self):
        res = self.harness.run_entrypoint(
            INFERENCE_URL=self.harness.base_url, INFERENCE_TOKEN=tgi.SECRET,
            INFERENCE_MODEL="selected/model", ALCF_MAX_TOKENS="0")
        self.assertEqual(res.returncode, 78, res.stderr)
        self.assertIn("ALCF_MAX_TOKENS must be a positive integer", res.stderr)
        self.assertEqual(tgi._CatalogHandler.seen, [])
        self.assertFalse((self.harness.home / "config.yaml").exists())


class GenericInferenceRenderContract(GenericRenderMixin, unittest.TestCase):
    def test_generic_block_matches_providers_schema(self):
        doc = yaml.safe_load(self.render())
        providers = assert_provider_schema(self, doc, where="generic-inference render")
        self.assertEqual(set(providers), {"generic-inference"})
        entry = providers["generic-inference"]
        self.assertEqual(entry["api"], self.harness.base_url)
        self.assertEqual(entry["extra_body"]["max_tokens"], 4096)
        self.assertEqual(entry["models"]["selected/model"]["context_length"], 131072)
        self.assertEqual(doc["model"]["provider"], "custom:generic-inference")


HERMES_VERSION_LINE = "Hermes Agent v0.23.2 (2026.9.24) · upstream ae2a81cc"


class HermesVersionRecordContract(GenericRenderMixin, unittest.TestCase):
    """v0.23.2 ships no /opt/hermes/.hermes_build_sha (banner said 'unknown').
    The image build records `hermes --version`'s first line in
    /opt/alcf/.hermes_version; the entrypoint shows it and never runs hermes."""

    def version_run(self) -> str:
        runs = [i for i in dockerfile_instructions()
                if i.startswith("RUN ") and "/opt/alcf/.hermes_version" in i]
        self.assertEqual(len(runs), 1, "expected exactly one RUN recording the Hermes version")
        return runs[0]

    def test_dockerfile_records_version_after_deps_as_root(self):
        insts = dockerfile_instructions()
        run = self.version_run()
        at = insts.index(run)
        deps_at = next(n for n, i in enumerate(insts) if i.startswith("RUN ") and DEPS in i)
        self.assertGreater(at, deps_at, "version must be recorded after the deps install")
        users = [i for i in insts[:at] if i.startswith("USER ")]
        self.assertEqual(users[-1], "USER root")
        self.assertTrue(run.startswith("RUN set -eu;"), run[:40])
        self.assertIn('/opt/hermes/.venv/bin/hermes --version)"', run)
        self.assertIn('HOME="$tmp" HERMES_HOME="$tmp"', run, "no build-time cache in the image")
        self.assertRegex(run, r'if \[ -z "\$line" \]; then .*exit 1; fi')
        self.assertIn("printf '%s\\n' \"$line\" > /opt/alcf/.hermes_version", run)
        res = subprocess.run(["sh", "-n", "-c", run[len("RUN "):]],
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_entrypoint_reads_recorded_version_with_fallbacks_and_never_runs_hermes(self):
        body = (ROOT / "scripts" / "entrypoint.sh").read_text(encoding="utf-8")
        code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
        self.assertNotRegex(code, r"hermes\S*\s+--version")
        recorded = code.index('"$ALCF_DIR/.hermes_version"')
        legacy = code.index("/opt/hermes/.hermes_build_sha")
        banner = code.index('hermes  : %s')
        self.assertLess(recorded, legacy)
        self.assertLess(legacy, banner)
        self.assertIn('"${_hermes_sha:-unknown}"', code[banner:])

    def test_banner_and_version_file_show_recorded_version(self):
        (self.harness.alcf_dir / ".hermes_version").write_text(HERMES_VERSION_LINE + "\n",
                                                                encoding="utf-8")
        result = self.harness.run_entrypoint(
            INFERENCE_URL=self.harness.base_url, INFERENCE_TOKEN=tgi.SECRET,
            INFERENCE_MODEL="selected/model", INFERENCE_MAX_TOKENS="4096")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"hermes  : {HERMES_VERSION_LINE}", result.stdout)
        txt = (self.harness.home / "ALCF_VERSION.txt").read_text(encoding="utf-8")
        self.assertIn(f"hermes: {HERMES_VERSION_LINE}\n", txt)

    def test_banner_without_any_record_says_unknown(self):
        if Path("/opt/hermes/.hermes_build_sha").exists():
            self.fail("host has /opt/hermes/.hermes_build_sha; cannot test the 'unknown' fallback")
        result = self.harness.run_entrypoint(
            INFERENCE_URL=self.harness.base_url, INFERENCE_TOKEN=tgi.SECRET,
            INFERENCE_MODEL="selected/model", INFERENCE_MAX_TOKENS="4096")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("hermes  : unknown", result.stdout)


class UpstreamRuntimeContract(UpstreamHarnessMixin, unittest.TestCase):
    def test_positive_control_hand_written_v0_23_2_config(self):
        """Proves the harness itself: a correct providers: config must pass today."""
        ev = self.run_upstream(POSITIVE_CONTROL_CONFIG, ALCF_CHECKS)
        self.assert_runtime_contract(ev, ALCF_CHECKS)

    def test_rendered_alcf_config_resolves_caps_in_upstream(self):
        config = render_alcf_config(build_test_block(), launch_provider="custom:alcf-x",
                                    model="chatty")
        self.assertNotIn("${ALCF_ACCESS_TOKEN}", config)
        self.assertNotIn(FAKE_TOKEN, config)
        doc = yaml.safe_load(config)
        self.assertEqual(doc["model"]["api_key"], "")
        for entry in doc["providers"].values():
            self.assertNotIn("api_key", entry)
            self.assertEqual(entry["key_cmd"], f"{pm.DEFAULT_ALCF_TOKENS_BIN} get-token inference")
        ev = self.run_upstream(config, ALCF_CHECKS)
        self.assert_runtime_contract(ev, ALCF_CHECKS)
        self.assert_config_version_current(ev)

    def test_unstamped_config_trips_the_support_floor_warning(self):
        """Negative control for the stamp check: a config without _config_version
        (as rendered before the fix) is what upstream refuses to migrate."""
        self.assertNotIn("_config_version", POSITIVE_CONTROL_CONFIG)
        ev = self.run_upstream(POSITIVE_CONTROL_CONFIG, [])
        self.assertIsNone(ev["config_version"]["stamp"])
        self.assertEqual(ev["config_version"]["migrate"],
                         {"ok": True, "skipped": "below-support-floor"})
        self.assertTrue(any("predates version 12" in w for w in ev["migration_warnings"]),
                        ev["migration_warnings"])

    def test_template_stamp_is_upstream_current_schema_version(self):
        template = yaml.safe_load((ROOT / "config" / "config.template.yaml").read_text(encoding="utf-8"))
        ev = self.run_upstream(render_alcf_config("", launch_provider="custom:alcf-sophia-reasoning",
                                                  model="google/gemma-4-31B-it"), [])
        self.assertEqual(template["_config_version"], ev["config_version"]["latest"])

    def test_positive_control_model_switch_uses_target_provider_cap(self):
        """Proves the switch harness: a correct config switches to the 12288 cap today."""
        ev = self.run_upstream(POSITIVE_CONTROL_CONFIG, [], ALCF_SWITCHES)
        self.assert_switch_contract(ev, ALCF_SWITCHES)

    def test_rendered_alcf_config_model_switch_uses_target_provider_cap(self):
        config = render_alcf_config(build_test_block(), launch_provider="custom:alcf-x",
                                    model="chatty")
        ev = self.run_upstream(config, [], ALCF_SWITCHES)
        self.assert_switch_contract(ev, ALCF_SWITCHES)
        self.assertEqual(ev["unknown_key_warnings"], [],
                         "Hermes ignored provider keys (e.g. max_tokens) in the rendered config")

    def test_rendered_template_fallback_resolves_caps_in_upstream(self):
        """No generated block: the committed template fallback is what Hermes loads."""
        config = render_alcf_config("", launch_provider="custom:alcf-sophia-reasoning",
                                    model="google/gemma-4-31B-it")
        checks = [{"provider": "custom:alcf-sophia-reasoning", "model": "google/gemma-4-31B-it",
                   "base_url": f"{pm.INFER_HOST}/sophia/vllm/v1",
                   "max_tokens": pm.REASONING_MAX_TOKENS, "fake_token": FAKE_TOKEN,
                   "key_cmd": True}]
        ev = self.run_upstream(config, checks)
        self.assert_runtime_contract(ev, checks)
        self.assert_config_version_current(ev)


class UpstreamTaraDefaultRuntimeContract(UpstreamHarnessMixin, AlcfEntrypointMixin,
                                         unittest.TestCase):
    """T1: the DEFAULT launch config (entrypoint-rendered, ALCF_CLUSTER/ALCF_MODEL
    unset) loaded by upstream Hermes v0.23.2: Tara's 12288 reasoning cap is on
    the wire, and the reasoning effort Hermes sends with agent.reasoning_effort
    unset is pinned.

    Wire behaviour pinned here (upstream v0.23.2):
    plugins/model-providers/custom/__init__.py CustomProfile.default_reasoning_config
    returns {"enabled": True, "effort": "medium"} for any `custom` provider;
    agent/reasoning_params.py unset_reasoning_default applies it unless the
    models.dev catalog marks the model non-reasoning (muse-spark-bf16 is not in
    it); CustomProfile.build_api_kwargs_extras puts it on the wire as top-level
    `reasoning_effort: "medium"` (no extra_body.reasoning). So Hermes overrides
    Muse Spark's server-side default effort (xhigh per the ALCF doc) with medium.
    This test documents that; it does not change the effort or the cap.
    """

    def test_default_launch_carries_tara_cap_and_medium_effort_upstream(self):
        self.upstream()
        res = self.run_alcf(ALCF_CLUSTER="", ALCF_MODEL="")
        self.assertEqual(res.returncode, 0, res.stderr)
        config = (self.harness.home / "config.yaml").read_text(encoding="utf-8")
        # No token in the rendered config: the entrypoint's key_cmd points at
        # the harness's fake alcf-tokens, which mints ALCF_TOKEN_IN_HARNESS.
        self.assertNotIn(ALCF_TOKEN_IN_HARNESS, config)
        checks = [{"provider": "custom:alcf-tara-reasoning", "model": "muse-spark-bf16",
                   "base_url": pm.TARA_BASE, "max_tokens": pm.REASONING_MAX_TOKENS,
                   "fake_token": ALCF_TOKEN_IN_HARNESS, "agent_path": True, "key_cmd": True}]
        ev = self.run_upstream(config, checks)
        self.assert_runtime_contract(ev, checks)
        agent = ev["providers"]["custom:alcf-tara-reasoning"]["agent_path"]
        self.assertEqual(agent["profile"], "CustomProfile")
        self.assertEqual(agent["unset_reasoning_default"], {"enabled": True, "effort": "medium"})
        self.assertEqual(agent["wire_reasoning_effort"], "medium")
        self.assertNotIn("reasoning", agent["wire_keys"])
        self.assertEqual(agent["wire_max_tokens"], 12288)


class UpstreamGenericRuntimeContract(UpstreamHarnessMixin, GenericRenderMixin, unittest.TestCase):
    def test_rendered_generic_config_resolves_cap_in_upstream(self):
        self.upstream()
        config = self.render()
        checks = [{"provider": "custom:generic-inference", "model": "selected/model",
                   "base_url": self.harness.base_url, "max_tokens": 4096,
                   "fake_token": tgi.SECRET}]
        ev = self.run_upstream(config, checks)
        self.assert_runtime_contract(ev, checks)
        self.assert_config_version_current(ev)


# Resolves a key_cmd provider, mints, rotates the token file, forces the
# CommandTokenSource cache past its window and mints again. Prints booleans
# only (never a token).
KEY_CMD_LIFECYCLE_HARNESS = textwrap.dedent(r'''
    import json, socket, sys, time
    from unittest import mock

    def _no_network(*_a, **_k):
        raise OSError("network disabled in contract harness")
    socket.socket.connect = _no_network
    socket.create_connection = _no_network
    socket.getaddrinfo = _no_network

    config_path, token_file, old, new = sys.argv[1:5]
    import hermes_yaml
    import agent.command_token_source as cts
    from hermes_cli.runtime_provider import frozen_runtime_config, resolve_runtime_provider

    with open(config_path, encoding="utf-8") as fh:
        cfg = hermes_yaml.safe_load(fh)
    ev = {}
    with frozen_runtime_config(cfg):
        named = resolve_runtime_provider(requested="custom:alcf-x", target_model="chatty")
        launch = resolve_runtime_provider(requested=None)
    key, launch_key = named.get("api_key"), launch.get("api_key")
    ev["named_callable"] = callable(key)
    ev["launch_callable"] = callable(launch_key)
    ev["launch_base_url"] = launch.get("base_url")
    ev["first_is_old"] = key() == old
    with open(token_file, "w", encoding="utf-8") as fh:
        fh.write(new + "\n")
    ev["cached_is_old"] = key() == old
    real = time.monotonic()
    with mock.patch.object(cts.time, "monotonic",
                           return_value=real + cts._NO_TTL_REFRESH_SECONDS + 1):
        ev["after_expiry_is_new"] = key() == new
    ev["launch_mints_new"] = launch_key() == new
    print(json.dumps(ev))
''')


class UpstreamKeyCmdLifecycleContract(UpstreamHarnessMixin, unittest.TestCase):
    """The rendered key_cmd credential is re-minted by upstream: a token
    rotated on disk (what `alcf-tokens` does when it refreshes) is picked up
    once the CommandTokenSource cache window lapses, with no re-render."""

    OLD, NEW = "fake-alcf-old-token", "fake-alcf-new-token"

    def test_rotated_token_is_reminted_after_cache_window(self):
        root = self.upstream()
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            hermes_home = home / ".hermes"
            hermes_home.mkdir(parents=True)
            token_file = Path(tmp) / "token"
            token_file.write_text(self.OLD + "\n", encoding="utf-8")
            fake_cli = Path(tmp) / "alcf-tokens"
            fake_cli.write_text(
                '#!/bin/sh\n[ "$1" = get-token ] || exit 2\n'
                f"cat '{token_file}'\n", encoding="utf-8")
            fake_cli.chmod(0o755)
            config = render_alcf_config(build_test_block(), launch_provider="custom:alcf-x",
                                        model="chatty")
            self.assertEqual(yaml.safe_load(config)["model"]["api_key"], "")
            config = config.replace(pm.DEFAULT_ALCF_TOKENS_BIN, str(fake_cli))
            cfg = hermes_home / "config.yaml"
            cfg.write_text(config, encoding="utf-8")
            proc = subprocess.run(
                [str(root / ".venv" / "bin" / "python"), "-c", KEY_CMD_LIFECYCLE_HARNESS,
                 str(cfg), str(token_file), self.OLD, self.NEW],
                cwd=str(root), env=upstream_subprocess_env(root, home=home, hermes_home=hermes_home),
                text=True, capture_output=True, timeout=120)
        self.assertEqual(proc.returncode, 0, f"lifecycle harness crashed:\n{proc.stderr[-4000:]}")
        for tok in (self.OLD, self.NEW):
            self.assertNotIn(tok, proc.stdout + proc.stderr)
        ev = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(ev, {
            "named_callable": True, "launch_callable": True,
            "launch_base_url": "https://h/v1",
            "first_is_old": True, "cached_is_old": True,
            "after_expiry_is_new": True, "launch_mints_new": True,
        })


if __name__ == "__main__":
    unittest.main()
