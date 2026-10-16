#!/usr/bin/env python3
"""Endpoint-registry contract tests for config/facilities.yaml."""
from __future__ import annotations

import os
import unittest

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(REPO, "config", "facilities.yaml")


class RegistryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(REGISTRY_PATH):
            raise AssertionError("facility registry missing: %s" % REGISTRY_PATH)
        with open(REGISTRY_PATH, encoding="utf-8") as registry_file:
            cls.cfg = yaml.safe_load(registry_file)
        if not isinstance(cls.cfg, dict):
            raise AssertionError("facility registry must be a YAML mapping")

    def test_alcf_base_and_default(self):
        alcf = self.cfg["facilities"]["alcf"]["iri"]
        self.assertEqual(alcf["default_version"], "v1")
        self.assertEqual(alcf["versions"]["v1"]["base_url"], "https://api.alcf.anl.gov/api/v1")

    def test_nersc_base_and_default(self):
        nersc = self.cfg["facilities"]["nersc"]["iri"]
        self.assertEqual(nersc["default_version"], "v2")
        self.assertEqual(nersc["versions"]["v1"]["base_url"], "https://api.iri.nersc.gov/api/v1")
        self.assertEqual(nersc["versions"]["v2"]["base_url"], "https://api.iri.nersc.gov/api/v2")

    def test_esnet_east_and_west(self):
        e = self.cfg["facilities"]["esnet-east"]["iri"]
        w = self.cfg["facilities"]["esnet-west"]["iri"]
        self.assertEqual(e["default_version"], "v2")
        self.assertEqual(w["default_version"], "v2")
        self.assertEqual(e["versions"]["v1"]["base_url"], "https://iri-dev.ppg.es.net/api/v1")
        self.assertEqual(e["versions"]["v2"]["base_url"], "https://iri-dev.ppg.es.net/api/v2")
        self.assertEqual(w["versions"]["v1"]["base_url"], "https://esnet-west.sdn-sense.net/api/v1")
        self.assertEqual(w["versions"]["v2"]["base_url"], "https://esnet-west.sdn-sense.net/api/v2")


if __name__ == "__main__":
    unittest.main()
