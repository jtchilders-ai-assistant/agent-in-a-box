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

# Words that mark a README mention of the retired patch as history, not as the
# current state of the image.
HISTORY_MARKERS = ("no longer", "removed", "retired", "previously", "formerly", "obsolete")


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


def assert_provider_schema(tc: unittest.TestCase, doc, *, where: str) -> dict:
    """The v0.23.2 keyed `providers:` contract shared by every emitter."""
    tc.assertIsInstance(doc, dict, f"{where}: not a YAML mapping")
    tc.assertFalse("custom_providers" in doc,
                   f"{where}: still emits the legacy custom_providers list")
    tc.assertTrue("providers" in doc, f"{where}: no root `providers` key")
    providers = doc["providers"]
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
    """Render the template exactly as entrypoint.sh does (splice + substitute)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "config.yaml"
        env = dict(os.environ,
                   ALCF_ACCESS_TOKEN=FAKE_TOKEN,
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
    """populate_models.build_block for one known cluster, fully offline."""
    with mock.patch.object(pm, "_get_token", side_effect=RuntimeError("offline test")), \
            mock.patch.object(pm, "_clusters", return_value=TEST_CLUSTER), \
            contextlib.redirect_stderr(io.StringIO()):
        return pm.build_block()


# --- upstream runtime harness ---------------------------------------------------

UPSTREAM_HARNESS = textwrap.dedent(r'''
    """Run upstream Hermes against a rendered config; print one JSON evidence blob."""
    import json, logging, os, sys, traceback

    config_path, checks = sys.argv[1], json.loads(sys.argv[2])
    ev = {"harness_error": None, "validate_issues": [], "unknown_key_warnings": [],
          "providers": {}, "tool_message": None}

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
    logging.getLogger().addHandler(_Grab(level=logging.WARNING))

    try:
        ev["validate_issues"] = [{"severity": str(getattr(i, "severity", "")),
                                  "message": str(getattr(i, "message", i))}
                                 for i in validate_config_structure(cfg)]
        get_compatible_custom_providers(cfg)
    except Exception:
        fail("validate_config_structure")

    transport = ChatCompletionsTransport()
    seen = []

    def _handler(request):
        seen.append(json.loads(request.content or b"{}"))
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}]})

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
        out["api_key_is_fake"] = rt.get("api_key") == check.get("fake_token")
        out["request_overrides"] = rt.get("request_overrides")
        try:
            kwargs = transport.build_kwargs(
                model, [{"role": "user", "content": "hi"}],
                base_url=rt.get("base_url"), provider_name=rt.get("provider"),
                request_overrides=rt.get("request_overrides"))
            out["api_kwargs_extra_body"] = kwargs.get("extra_body")
            seen.clear()
            client = openai.OpenAI(api_key="fake-key", base_url=rt.get("base_url") or "http://x/v1",
                                   http_client=httpx.Client(transport=httpx.MockTransport(_handler)))
            client.chat.completions.create(**kwargs)
            body = seen[-1]
            out["wire_max_tokens"] = body.get("max_tokens")
            out["wire_keys"] = sorted(body)
        except Exception:
            fail(f"build_request[{name}]")

    try:
        msgs = [{"role": "tool", "tool_call_id": "c1", "name": "terminal", "content": "ok"}]
        conv = transport.convert_messages(msgs, model="m", base_url="https://h/v1", provider="custom")
        ev["tool_message"] = conv[0]
    except Exception:
        fail("convert_messages")

    print(json.dumps(ev, default=str))
''')


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

    def run_upstream(self, config_text: str, checks: list) -> dict:
        root = self.upstream()
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home.mkdir()
            cfg = Path(tmp) / "rendered.yaml"
            cfg.write_text(config_text, encoding="utf-8")
            env = {k: v for k, v in os.environ.items()
                   if not k.endswith(("_API_KEY", "_TOKEN")) and k != "HERMES_HOME"}
            env.update(HERMES_HOME=str(home), PYTHONPATH=str(root),
                       PYTHONDONTWRITEBYTECODE="1", NO_COLOR="1")
            proc = subprocess.run(
                [str(root / ".venv" / "bin" / "python"), "-c", UPSTREAM_HARNESS,
                 str(cfg), json.dumps(checks)],
                cwd=str(root), env=env, text=True, capture_output=True, timeout=120)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        self.assertEqual(proc.returncode, 0, f"upstream harness crashed:\n{proc.stderr[-4000:]}")
        self.assertTrue(lines, f"upstream harness printed no JSON:\n{proc.stdout}\n{proc.stderr[-4000:]}")
        ev = json.loads(lines[-1])
        print(f"\n[upstream evidence] {json.dumps(ev, sort_keys=True)}", file=sys.stderr)
        self.assertIsNone(ev["harness_error"], "upstream harness error (not a contract failure)")
        return ev

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
                self.assertTrue(out["api_key_is_fake"], "resolved api_key is not the rendered token")
                overrides = out["request_overrides"] or {}
                self.assertEqual((overrides.get("extra_body") or {}).get("max_tokens"),
                                 check["max_tokens"],
                                 "request_overrides.extra_body.max_tokens does not carry the cap")
                self.assertEqual(out["wire_max_tokens"], check["max_tokens"],
                                 "outgoing Chat Completions body does not carry the cap")
        tool = ev["tool_message"]
        self.assertNotIn("name", tool, "upstream no longer strips `name` from role:tool messages")
        self.assertEqual(tool["tool_call_id"], "c1")


ALCF_CHECKS = [
    {"provider": "custom:alcf-x", "model": "chatty", "base_url": "https://h/v1",
     "max_tokens": pm.BASELINE_MAX_TOKENS, "fake_token": FAKE_TOKEN},
    {"provider": "custom:alcf-x-reasoning", "model": "thinky", "base_url": "https://h/v1",
     "max_tokens": pm.REASONING_MAX_TOKENS, "fake_token": FAKE_TOKEN},
]

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
    api_key: "{FAKE_TOKEN}"
    discover_models: false
    extra_body:
      max_tokens: {pm.BASELINE_MAX_TOKENS}
    models:
      chatty:
        context_length: 128000
  alcf-x-reasoning:
    name: alcf-x-reasoning
    api: "https://h/v1"
    api_key: "{FAKE_TOKEN}"
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
        current = [ln for ln in (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
                   if ("strip_tool_message_name" in ln or "0001-strip-tool-message-name" in ln)
                   and not any(m in ln.lower() for m in HISTORY_MARKERS)]
        self.assertEqual(current, [], "README still describes the patch in current tense")


class PopulateModelsSchemaContract(unittest.TestCase):
    def test_emit_cluster_output_matches_providers_schema(self):
        mapping = {"chatty": (128000, False), "thinky": (262144, True)}
        doc = yaml.safe_load("\n".join(pm._emit_cluster("alcf-x", "https://h/v1", mapping)))
        if isinstance(doc, list):  # legacy list items: still a contract failure below
            doc = {"custom_providers": doc}
        elif isinstance(doc, dict) and "providers" not in doc and "custom_providers" not in doc:
            doc = {"providers": doc}  # bare mapping entries, to be nested under providers:
        providers = assert_provider_schema(self, doc, where="_emit_cluster")
        self.assertEqual(set(providers), {"alcf-x", "alcf-x-reasoning"})
        self.assertEqual(providers["alcf-x"]["api"], "https://h/v1")
        self.assertEqual(providers["alcf-x"]["extra_body"]["max_tokens"], pm.BASELINE_MAX_TOKENS)
        self.assertEqual(providers["alcf-x-reasoning"]["extra_body"]["max_tokens"],
                         pm.REASONING_MAX_TOKENS)
        self.assertEqual(providers["alcf-x"]["models"], {"chatty": {"context_length": 128000}})
        self.assertEqual(providers["alcf-x-reasoning"]["models"],
                         {"thinky": {"context_length": 262144}})

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
            self.assertEqual(entry["api_key"], "${ALCF_ACCESS_TOKEN}")
            for mid, meta in entry["models"].items():
                self.assertGreaterEqual(meta["context_length"], pm.MIN_CONTEXT, mid)

    def test_template_and_populate_models_fallbacks_agree(self):
        providers = assert_provider_schema(self, template_fallback_doc(),
                                           where="config.template.yaml fallback")
        got = {k: {m: v["context_length"] for m, v in e["models"].items()}
               for k, e in providers.items()}
        self.assertEqual(got, fallback_as_providers(pm._clusters()))

    def test_whole_template_parses_with_providers_mapping(self):
        doc = yaml.safe_load((ROOT / "config" / "config.template.yaml").read_text(encoding="utf-8"))
        assert_provider_schema(self, doc, where="config.template.yaml")


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


class UpstreamRuntimeContract(UpstreamHarnessMixin, unittest.TestCase):
    def test_positive_control_hand_written_v0_23_2_config(self):
        """Proves the harness itself: a correct providers: config must pass today."""
        ev = self.run_upstream(POSITIVE_CONTROL_CONFIG, ALCF_CHECKS)
        self.assert_runtime_contract(ev, ALCF_CHECKS)

    def test_rendered_alcf_config_resolves_caps_in_upstream(self):
        config = render_alcf_config(build_test_block(), launch_provider="custom:alcf-x",
                                    model="chatty")
        self.assertNotIn("${ALCF_ACCESS_TOKEN}", config)
        ev = self.run_upstream(config, ALCF_CHECKS)
        self.assert_runtime_contract(ev, ALCF_CHECKS)

    def test_rendered_template_fallback_resolves_caps_in_upstream(self):
        """No generated block: the committed template fallback is what Hermes loads."""
        config = render_alcf_config("", launch_provider="custom:alcf-sophia-reasoning",
                                    model="google/gemma-4-31B-it")
        checks = [{"provider": "custom:alcf-sophia-reasoning", "model": "google/gemma-4-31B-it",
                   "base_url": f"{pm.INFER_HOST}/sophia/vllm/v1",
                   "max_tokens": pm.REASONING_MAX_TOKENS, "fake_token": FAKE_TOKEN}]
        ev = self.run_upstream(config, checks)
        self.assert_runtime_contract(ev, checks)


class UpstreamGenericRuntimeContract(UpstreamHarnessMixin, GenericRenderMixin, unittest.TestCase):
    def test_rendered_generic_config_resolves_cap_in_upstream(self):
        self.upstream()
        config = self.render()
        checks = [{"provider": "custom:generic-inference", "model": "selected/model",
                   "base_url": self.harness.base_url, "max_tokens": 4096,
                   "fake_token": tgi.SECRET}]
        ev = self.run_upstream(config, checks)
        self.assert_runtime_contract(ev, checks)


if __name__ == "__main__":
    unittest.main()
