import os, sys, unittest, tempfile, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from agent_box_config import REGISTRY, DEFAULTS

class ConfigTests(unittest.TestCase):
    def test_registry_alcf_v1_only(self):
        self.assertEqual(list(REGISTRY['alcf']['versions'].keys()), ['v1'])
    def test_registry_nersc_default_v2(self):
        self.assertEqual(REGISTRY['nersc']['default_version'], 'v2')
    def test_registry_esnet_east_west_defaults(self):
        self.assertEqual(REGISTRY['esnet-east']['default_version'], 'v2')
        self.assertEqual(REGISTRY['esnet-west']['default_version'], 'v2')
    def test_defaults_alcf_on_others_off(self):
        self.assertTrue(DEFAULTS['alcf']['enabled'])
        self.assertFalse(DEFAULTS['nersc']['enabled'])

if __name__ == '__main__':
    unittest.main()
