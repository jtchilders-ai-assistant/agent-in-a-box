#!/usr/bin/env python3
"""Endpoint-registry contract tests for config/facilities.yaml."""
from __future__ import annotations
import os, sys, unittest, yaml
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RegistryContractTests(unittest.TestCase):
    def test_alcf_base_and_default(self):
        cfg = yaml.safe_load(open(os.path.join(REPO, "config", "facilities.yaml")))
        alcf = cfg["facilities"]["alcf"]["iri"]
        self.assertEqual(alcf["default_version"], "v1")
        self.assertEqual(alcf["versions"]["v1"]["base_url"], "https://api.alcf.anl.gov/api/v1")

    def test_nersc_base_and_default(self):
        cfg = yaml.safe_load(open(os.path.join(REPO, "config", "facilities.yaml")))
        nersc = cfg["facilities"]["nersc"]["iri"]
        self.assertEqual(nersc["default_version"], "v2")
        self.assertEqual(nersc["versions"]["v1"]["base_url"], "https://api.iri.nersc.gov/api/v1")
        self.assertEqual(nersc["versions"]["v2"]["base_url"], "https://api.iri.nersc.gov/api/v2")

    def test_esnet_east_and_west(self):
        cfg = yaml.safe_load(open(os.path.join(REPO, "config", "facilities.yaml")))
        e = cfg["facilities"]["esnet-east"]["iri"]
        w = cfg["facilities"]["esnet-west"]["iri"]
        self.assertEqual(e["default_version"], "v2")
        self.assertEqual(w["default_version"], "v2")
        self.assertEqual(e["versions"]["v1"]["base_url"], "https://iri-dev.ppg.es.net/api/v1")
        self.assertEqual(e["versions"]["v2"]["base_url"], "https://iri-dev.ppg.es.net/api/v2")
        self.assertEqual(w["versions"]["v1"]["base_url"], "https://esnet-west.sdn-sense.net/api/v1")
        self.assertEqual(w["versions"]["v2"]["base_url"], "https://esnet-west.sdn-sense.net/api/v2")


if __name__ == "__main__":
    unittest.main()
