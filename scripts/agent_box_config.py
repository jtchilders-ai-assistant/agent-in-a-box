"""Persistent facility configuration."""
REGISTRY = {
    "alcf": {"default_version": "v1", "versions": {"v1": {"base_url": "https://api.alcf.anl.gov/api/v1", "auth_provider": "alcf-tokens"}}},
    "nersc": {"default_version": "v2", "versions": {"v1": {"base_url": "https://api.iri.nersc.gov/api/v1", "auth_provider": "nersc-tokens"}, "v2": {"base_url": "https://api.iri.nersc.gov/api/v2", "auth_provider": "nersc-tokens"}}},
    "esnet-east": {"default_version": "v2", "versions": {"v1": {"base_url": "https://iri-dev.ppg.es.net/api/v1", "auth_provider": "amsc-pat"}, "v2": {"base_url": "https://iri-dev.ppg.es.net/api/v2", "auth_provider": "amsc-pat"}}},
    "esnet-west": {"default_version": "v2", "versions": {"v1": {"base_url": "https://esnet-west.sdn-sense.net/api/v1", "auth_provider": "amsc-pat"}, "v2": {"base_url": "https://esnet-west.sdn-sense.net/api/v2", "auth_provider": "amsc-pat"}}},
}
DEFAULTS = {"alcf": {"enabled": True, "version": "v1"}, "nersc": {"enabled": False, "version": "v2"}, "esnet-east": {"enabled": False, "version": "v2"}, "esnet-west": {"enabled": False, "version": "v2"}}
