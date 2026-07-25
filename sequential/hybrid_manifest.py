"""Immutable experiment identity for CS-HR runs."""

import os

from .core import canonical_digest
from .io import atomic_json, read_json


def build_hybrid_manifest(config_path, output_path, git_commit, dataset_hash,
                          logical_run_ids=None):
    config_bytes = open(config_path, 'rb').read()
    payload = {
        'schema_version': 1, 'experiment': 'cs_hr',
        'config_path': os.path.abspath(config_path),
        'config_sha256': __import__('hashlib').sha256(config_bytes).hexdigest(),
        'dataset_manifest_sha256': str(dataset_hash),
        'git_commit': str(git_commit),
        'logical_run_ids': list(logical_run_ids or []),
        'status': 'planned',
    }
    payload['manifest_hash'] = canonical_digest(payload)
    atomic_json(output_path, payload)
    return payload


def validate_hybrid_manifest(path):
    manifest = read_json(path)
    if manifest.get('schema_version') != 1 or manifest.get('experiment') != 'cs_hr':
        raise ValueError('unsupported hybrid experiment manifest')
    digest = manifest.get('manifest_hash')
    payload = dict(manifest); payload.pop('manifest_hash', None)
    if canonical_digest(payload) != digest:
        raise ValueError('hybrid manifest digest mismatch')
    return {'valid': True, 'manifest_hash': digest,
            'logical_run_count': len(manifest.get('logical_run_ids', []))}

