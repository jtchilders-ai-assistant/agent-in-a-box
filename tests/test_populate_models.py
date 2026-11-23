#!/usr/bin/env python3
"""Unit tests for scripts/populate_models.py model selection + status banner.

Pure/offline: every test feeds the selectors and the /jobs classifier fixture
payloads captured from the LIVE ALCF gateway on 2026-08-24, so no Globus token
and no network are needed.

Run:  python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")
sys.path.insert(0, SCRIPTS)

import populate_models as pm  # noqa: E402


# --- Fixtures: real /models entries, trimmed to the fields we read -----------

SOPHIA_MODELS = [
    {"id": "google/gemma-4-31B-it", "framework": "vllm",
     "max_model_len": 128000, "reasoning_parser": "gemma4"},
    {"id": "openai/gpt-oss-120b", "framework": "vllm", "max_model_len": 65536},
    # below the 64k floor -> excluded, and recorded for the banner
    {"id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct", "framework": "vllm",
     "max_model_len": 16384},
    # no max_model_len but allowlisted -> kept
    {"id": "openai/gpt-oss-20b", "framework": "vllm"},
    # non-chat framework + embedding/science ids -> excluded
    {"id": "sam3", "framework": "sam3service"},
    {"id": "google/embeddinggemma-300m", "framework": "vllm"},
    {"id": "genslm-test/genslm-esmc-300M-codon", "framework": "vllm"},
]

METIS_MODELS = [
    {"id": "gemma-4-31B-it", "framework": "api"},
    {"id": "gpt-oss-120b", "framework": "api"},
    # verified 8192 on Metis -> below floor, excluded
    {"id": "Mistral-Large-3-675B-Instruct-2512", "framework": "api"},
]

MINERVA_MODELS = [
    {"id": "gpt-oss-120b", "framework": "api", "capabilities": {
        "context_window_tokens": 131072, "reasoning": {"supported": True},
        "api_protocols": ["chat_completions", "responses"]}},
    {"id": "inkling-bf16", "framework": "api", "capabilities": {
        "context_window_tokens": 262144, "reasoning": {"supported": True},
        "api_protocols": ["chat_completions", "responses"]}},
    {"id": "nemotron-3-ultra", "framework": "api", "capabilities": {
        "context_window_tokens": 262144, "reasoning": {"supported": True},
        "api_protocols": ["chat_completions", "responses"]}},
]

# Tara publishes the same capabilities schema (v1) as Minerva. Ids/windows from
# the ALCF models doc (plans/evidence/tara/alcf-models-doc-2026-11-23.md); the
# responses-only and sub-floor entries are synthetic negatives.
TARA_MODELS = [
    {"id": "muse-spark-bf16", "framework": "api", "capabilities": {
        "context_window_tokens": 1048576, "reasoning": {"supported": True},
        "api_protocols": ["chat_completions", "responses"]}},
    {"id": "openai/gpt-oss-20b", "framework": "api", "capabilities": {
        "context_window_tokens": 65536, "reasoning": {"supported": True},
        "api_protocols": ["chat_completions"]}},
    # no chat_completions protocol -> not a chat model for Hermes
    {"id": "responses-only-model", "framework": "api", "capabilities": {
        "context_window_tokens": 1048576, "reasoning": {"supported": True},
        "api_protocols": ["responses"]}},
    # below the 64k floor -> excluded and recorded under tara
    {"id": "small-window-model", "framework": "api", "capabilities": {
        "context_window_tokens": 32768, "reasoning": {"supported": False},
        "api_protocols": ["chat_completions"]}},
]

# Sophia was fully down at capture time: nothing running, two queued jobs whose
# estimated start was ~31 hours out.
SOPHIA_JOBS = {
    "running": [],
    "queued": [
        {"Models": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
         "Estimated Start Time": "Tue Aug 25 00:30:00 2026 (Chicago time)",
         "Job Comments": "Not Running: No available resources on nodes"},
        {"Models": "arcee-ai/Trinity-Large-Thinking-W4A16",
         "Estimated Start Time": "Tue Aug 25 00:30:00 2026 (Chicago time)",
         "Job Comments": "Not Running: No available resources on nodes"},
    ],
}

METIS_JOBS = {
    "running": [{"Models": "gemma-4-31B-it,gpt-oss-120b,"
                           "Mistral-Large-3-675B-Instruct-2512",
                 "Model Status": "running"}],
    "queued": [],
}


class TestSelectors(unittest.TestCase):
    def setUp(self):
        pm.EXCLUDED_BY_FLOOR.clear()

    def test_sophia_filters_and_windows(self):
        sel = pm._select_sophia(SOPHIA_MODELS)
        self.assertEqual(sel["google/gemma-4-31B-it"], (128000, True))
        self.assertEqual(sel["openai/gpt-oss-120b"], (65536, True))
        # allowlisted add-back keeps a model with no max_model_len
        self.assertEqual(sel["openai/gpt-oss-20b"][0], 128000)
        # non-chat frameworks, embeddings and science models are dropped
        for mid in ("sam3", "google/embeddinggemma-300m",
                    "genslm-test/genslm-esmc-300M-codon"):
            self.assertNotIn(mid, sel)

    def test_sub_floor_model_excluded_and_recorded(self):
        sel = pm._select_sophia(SOPHIA_MODELS)
        mid = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
        self.assertNotIn(mid, sel)
        self.assertEqual(pm.EXCLUDED_BY_FLOOR["sophia"][mid], 16384)

    def test_metis_uses_verified_windows_and_drops_mistral_large(self):
        sel = pm._select_metis(METIS_MODELS)
        self.assertEqual(sel["gemma-4-31B-it"], (131072, True))
        self.assertEqual(sel["gpt-oss-120b"], (131072, True))
        self.assertNotIn("Mistral-Large-3-675B-Instruct-2512", sel)
        self.assertEqual(
            pm.EXCLUDED_BY_FLOOR["metis"]["Mistral-Large-3-675B-Instruct-2512"], 8192)

    def test_minerva_prefers_capabilities_over_id_heuristic(self):
        sel = pm._select_minerva(MINERVA_MODELS)
        self.assertEqual(sel["gpt-oss-120b"], (131072, True))
        # REGRESSION GUARD: neither id matches REASONING_ID_PATTERNS, so without
        # reading capabilities.reasoning.supported these would be classed as
        # plain chat, land in the 2048-token baseline provider, and return empty
        # content (their reasoning channel eats the whole budget).
        self.assertFalse(pm._is_reasoning_id("inkling-bf16"))
        self.assertFalse(pm._is_reasoning_id("nemotron-3-ultra"))
        self.assertEqual(sel["inkling-bf16"], (262144, True))
        self.assertEqual(sel["nemotron-3-ultra"], (262144, True))

    def test_minerva_falls_back_to_heuristic_without_capabilities(self):
        sel = pm._select_minerva([
            {"id": "some-thinking-model", "framework": "api"},
            {"id": "plain-chat-model", "framework": "api"},
        ])
        self.assertEqual(sel["some-thinking-model"],
                         (pm.MINERVA_DEFAULT_CONTEXT, True))
        self.assertEqual(sel["plain-chat-model"],
                         (pm.MINERVA_DEFAULT_CONTEXT, False))

    def test_tara_reads_capabilities(self):
        sel = pm._select_tara(TARA_MODELS)
        self.assertEqual(sel, {"muse-spark-bf16": (1048576, True),
                               "openai/gpt-oss-20b": (65536, True)})

    def test_tara_skips_model_without_chat_completions(self):
        self.assertNotIn("responses-only-model", pm._select_tara(TARA_MODELS))
        self.assertNotIn("responses-only-model", pm.EXCLUDED_BY_FLOOR.get("tara", {}))

    def test_tara_sub_floor_model_excluded_and_recorded(self):
        sel = pm._select_tara(TARA_MODELS)
        self.assertNotIn("small-window-model", sel)
        self.assertEqual(pm.EXCLUDED_BY_FLOOR["tara"]["small-window-model"], 32768)
        self.assertNotIn("minerva", pm.EXCLUDED_BY_FLOOR)

    def test_tara_falls_back_to_default_window_without_capabilities(self):
        sel = pm._select_tara([{"id": "plain-chat-model", "framework": "api"}])
        self.assertEqual(sel["plain-chat-model"], (pm.TARA_DEFAULT_CONTEXT, False))

    def test_minerva_and_tara_share_one_capabilities_selector(self):
        # One implementation parameterised by cluster + default window, not a copy.
        for cluster, default in (("minerva", pm.MINERVA_DEFAULT_CONTEXT),
                                 ("tara", pm.TARA_DEFAULT_CONTEXT)):
            with self.subTest(cluster=cluster):
                pm.EXCLUDED_BY_FLOOR.clear()
                sel = pm._select_capabilities(cluster, default)(TARA_MODELS + [
                    {"id": "plain-chat-model", "framework": "api"}])
                self.assertEqual(sel["plain-chat-model"], (default, False))
                self.assertEqual(set(pm.EXCLUDED_BY_FLOOR), {cluster})


class TestTaraFallback(unittest.TestCase):
    def test_fallback_is_the_documented_table(self):
        self.assertEqual(pm.TARA_FALLBACK, {
            "google/gemma-4-31B-it-262k": (262144, True),
            "muse-spark-bf16": (1048576, True),
            "mistral-large-4": (1048576, True),
            "nemotron-3-ultra": (1048576, True),
            "nemotron-35-super-vl": (1048576, True),
            "openai/gpt-oss-120b": (131072, True),
            "openai/gpt-oss-20b": (65536, True),
            "beam-1.0-fp8": (1048576, True),
            "inkling-bf16": (1048576, True),
        })

    def test_catalog_failure_uses_fallback(self):
        with mock.patch.object(pm, "_fetch_models", side_effect=RuntimeError("offline test")), \
                contextlib.redirect_stderr(io.StringIO()):
            mapping, src = pm._resolve_cluster("tara", "fake-token", pm._select_tara,
                                               pm.TARA_FALLBACK)
        self.assertEqual(src, "fallback")
        self.assertEqual(mapping, pm.TARA_FALLBACK)

    def test_tara_is_registered_with_its_endpoint(self):
        (entry,) = [c for c in pm._clusters() if c[0] == "tara"]
        _cluster, provider, base, selector, fallback = entry
        self.assertEqual(provider, "alcf-tara")
        self.assertEqual(base, f"{pm.INFER_HOST}/tara/api/v1")
        self.assertEqual(base, pm.TARA_BASE)
        self.assertIs(selector, pm._select_tara)
        self.assertIs(fallback, pm.TARA_FALLBACK)


class TestJobClassification(unittest.TestCase):
    """_fetch_job_states parsing, exercised through the pure helpers."""

    def _states(self, doc):
        live = set()
        for job in doc.get("running", []) or []:
            if str(job.get("Model Status", "")).lower() in ("", "running"):
                live.update(pm._split_models_field(job))
        return live

    def test_comma_joined_models_field_is_split(self):
        live = self._states(METIS_JOBS)
        self.assertEqual(live, {"gemma-4-31B-it", "gpt-oss-120b",
                                "Mistral-Large-3-675B-Instruct-2512"})

    def test_offline_cluster_yields_no_live_models(self):
        # REGRESSION GUARD for the bug this change fixes: Sophia had zero running
        # models, so the OLD code labelled every offered model "cold (~10-15m)"
        # when the queue's estimated start was ~31 hours away.
        self.assertEqual(self._states(SOPHIA_JOBS), set())


class TestFormatting(unittest.TestCase):
    def test_context_window_rendered_per_model(self):
        mapping = {"a": (131072, True), "b": (128000, False), "c": (262144, True)}
        self.assertEqual(pm._fmt_model("a", mapping), "a (131k ctx)")
        self.assertEqual(pm._fmt_model("b", mapping), "b (128k ctx)")
        self.assertEqual(pm._fmt_model("c", mapping), "c (262k ctx)")

    def test_unknown_window_is_labelled_not_faked(self):
        self.assertEqual(pm._fmt_model("missing", {}), "missing (ctx unknown)")

    def test_extra_detail_appended(self):
        out = pm._fmt_model("a", {"a": (131072, True)}, "starts ~Tue Aug 25")
        self.assertEqual(out, "a (131k ctx; starts ~Tue Aug 25)")


class TestClusterRegistry(unittest.TestCase):
    def test_default_registry_has_four_serving_clusters(self):
        names = [c[0] for c in pm._clusters()]
        self.assertEqual(names, ["sophia", "metis", "minerva", "tara"])

    def test_toggles_drop_clusters(self):
        self.assertEqual([c[0] for c in pm._clusters(include_minerva=False)],
                         ["sophia", "metis", "tara"])
        self.assertEqual([c[0] for c in pm._clusters(include_metis=False)],
                         ["sophia", "minerva", "tara"])
        self.assertEqual([c[0] for c in pm._clusters(include_tara=False)],
                         ["sophia", "metis", "minerva"])

    def test_provisioning_cluster_has_no_provider(self):
        # A cluster registered upstream that serves nothing yet must never
        # produce a provider block. None is provisioning today (tara serves).
        self.assertEqual(pm.PROVISIONING_CLUSTERS, ())
        names = [c[0] for c in pm._clusters()]
        for cluster in pm.PROVISIONING_CLUSTERS:
            self.assertNotIn(cluster, names)

    def test_every_cluster_has_a_nonempty_fallback(self):
        for cluster, provider, base, _sel, fallback in pm._clusters():
            self.assertTrue(fallback, f"{cluster} has an empty fallback")
            self.assertTrue(base.startswith(pm.INFER_HOST))
            self.assertTrue(provider.startswith("alcf-"))
            for mid, (ctx, _is_reasoning) in fallback.items():
                self.assertGreaterEqual(
                    ctx, pm.MIN_CONTEXT,
                    f"{cluster} fallback {mid} is below the context floor")


class TestTaraToggle(unittest.TestCase):
    """ALCF_ENABLE_TARA=0 drops Tara unless it is the launch cluster."""

    def providers(self, launch="", **build_kwargs):
        env = {"ALCF_LAUNCH_CLUSTER": launch}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(pm, "_get_token", side_effect=RuntimeError("offline test")), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            block = pm.build_block(**build_kwargs)
        return block, err.getvalue()

    def test_disabled_tara_is_dropped(self):
        block, err = self.providers(launch="sophia", include_tara=False)
        self.assertNotIn("alcf-tara", block)
        self.assertNotIn("launch cluster", err)

    def test_disabled_tara_kept_when_it_is_the_launch_cluster(self):
        block, err = self.providers(launch="tara", include_tara=False)
        self.assertIn("  alcf-tara-reasoning:\n", block)
        self.assertIn("ALCF_ENABLE_TARA=0 overridden: tara is the launch cluster", err)

    def test_main_reads_the_env_flag_and_cli_switch(self):
        for argv, env, want in ((["populate_models.py"], {"ALCF_ENABLE_TARA": "0"}, False),
                                (["populate_models.py"], {"ALCF_ENABLE_TARA": "1"}, True),
                                (["populate_models.py", "--no-tara"], {}, False)):
            with self.subTest(argv=argv, env=env):
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.dict(os.environ, env), \
                        mock.patch.object(pm, "build_block", return_value="providers:\n") as bb, \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(pm.main(), 0)
                self.assertEqual(bb.call_args.kwargs["include_tara"], want)

    def test_status_report_reads_the_env_flag(self):
        seen = []

        def spy(**kwargs):
            seen.append(kwargs["include_tara"])
            return []

        for env, want in (({"ALCF_ENABLE_TARA": "0", "ALCF_LAUNCH_CLUSTER": "sophia"}, False),
                          ({"ALCF_ENABLE_TARA": "0", "ALCF_LAUNCH_CLUSTER": "tara"}, True)):
            with self.subTest(env=env):
                seen.clear()
                with mock.patch.dict(os.environ, env), \
                        mock.patch.object(pm, "_get_token", return_value="fake-token"), \
                        mock.patch.object(pm, "_clusters", side_effect=spy), \
                        contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    pm.status_report()
                self.assertEqual(seen, [want])


class TestEmit(unittest.TestCase):
    def test_reasoning_split_gives_each_class_its_own_max_tokens(self):
        mapping = {"chatty": (128000, False), "thinky": (262144, True)}
        block = "\n".join(pm._emit_cluster("alcf-x", "https://h/v1", mapping))
        self.assertIn("  alcf-x:\n    name: alcf-x\n", block + "\n")
        self.assertIn("  alcf-x-reasoning:\n    name: alcf-x-reasoning\n", block)
        self.assertIn(f"    extra_body:\n      max_tokens: {pm.BASELINE_MAX_TOKENS}\n", block)
        self.assertIn(f"    extra_body:\n      max_tokens: {pm.REASONING_MAX_TOKENS}\n", block)
        self.assertIn("context_length: 262144", block)

    def test_single_class_emits_one_provider(self):
        block = "\n".join(pm._emit_cluster(
            "alcf-x", "https://h/v1", {"thinky": (262144, True)}))
        self.assertIn("  alcf-x-reasoning:\n", block)
        self.assertNotIn("  alcf-x:\n", block + "\n")


if __name__ == "__main__":
    unittest.main()
