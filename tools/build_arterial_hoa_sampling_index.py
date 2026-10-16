"""Build a read-only memmap sampling index for the audited frozen HOA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.run_arterial_sequential import load_frozen_hoa

SCENES = ('300_0.3', '300_0.6', '700_0.3', '700_0.6')
INTERSECTIONS = tuple(f'intersection_{index}_1' for index in range(1, 7))
COUNT_PER_POOL = 5 * 400 * 360
DTYPE = np.dtype([
    ('state', '<f4', (12,)), ('next_state', '<f4', (12,)),
    ('phase', 'u1'), ('next_phase', 'u1'), ('action', 'u1'),
    ('reward', '<f4'), ('terminated', '?'), ('truncated', '?'),
    ('episode_id', '<u2'), ('decision_step', '<u2'),
    ('collector_seed', 'u1'), ('training_stage', 'u1'),
    ('policy_version', '<u4'),
])


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
            'w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
        temporary = Path(handle.name)
    temporary.replace(path)


def build(hoa_manifest, hoa_hash, output_dir):
    hoa_path, hoa, scene_indexes = load_frozen_hoa(hoa_manifest, hoa_hash)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'manifest.json'
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    arrays = {}
    offsets = {}
    for scene in SCENES:
        scene_dir = output_dir / scene.replace('.', '_')
        scene_dir.mkdir(parents=True, exist_ok=True)
        for intersection in INTERSECTIONS:
            path = scene_dir / f'{intersection}.npy'
            arrays[(scene, intersection)] = np.lib.format.open_memmap(
                path, mode='w+', dtype=DTYPE, shape=(COUNT_PER_POOL,))
            offsets[(scene, intersection)] = 0
    for scene in SCENES:
        for archive_path in scene_indexes[scene]['history_paths']:
            source_manifest = json.loads(
                (Path(archive_path) / 'manifest.json').read_text(encoding='utf-8'))
            seed = int(source_manifest['training_seed'])
            for name in source_manifest['transition_files']:
                records = torch.load(Path(archive_path) / name, weights_only=False)
                if len(records) != 2160:
                    raise ValueError(f'Unexpected HOA shard size: {archive_path}/{name}')
                grouped = {intersection: [] for intersection in INTERSECTIONS}
                for record in records:
                    metadata = record.metadata
                    if (metadata.scene_id != scene or metadata.source != 'offline_history'
                            or int(metadata.collector_training_seed) != seed):
                        raise ValueError(f'HOA identity mismatch: {archive_path}/{name}')
                    grouped[metadata.intersection_id].append(record)
                for intersection, values in grouped.items():
                    if len(values) != 360:
                        raise ValueError(f'Unbalanced HOA shard: {archive_path}/{name}')
                    start = offsets[(scene, intersection)]
                    target = arrays[(scene, intersection)][start:start + len(values)]
                    target['state'] = np.stack([item.state for item in values])
                    target['next_state'] = np.stack([item.next_state for item in values])
                    target['phase'] = [int(item.phase[0]) for item in values]
                    target['next_phase'] = [int(item.next_phase[0]) for item in values]
                    target['action'] = [item.action for item in values]
                    target['reward'] = [item.reward for item in values]
                    target['terminated'] = [item.terminated for item in values]
                    target['truncated'] = [item.truncated for item in values]
                    target['episode_id'] = [item.metadata.episode_id for item in values]
                    target['decision_step'] = [item.metadata.decision_step for item in values]
                    target['collector_seed'] = [seed] * len(values)
                    target['training_stage'] = [item.metadata.training_stage for item in values]
                    target['policy_version'] = [item.metadata.policy_version for item in values]
                    offsets[(scene, intersection)] += len(values)
    for key, array in arrays.items():
        array.flush()
        if offsets[key] != COUNT_PER_POOL:
            raise ValueError(f'Incomplete HOA sampling pool: {key}={offsets[key]}')
    del arrays
    pools = []
    for scene in SCENES:
        for intersection in INTERSECTIONS:
            path = output_dir / scene.replace('.', '_') / f'{intersection}.npy'
            pools.append({
                'scene_id': scene, 'intersection_id': intersection,
                'count': COUNT_PER_POOL,
                'file': str(path.relative_to(output_dir)),
                'sha256': sha256_file(path),
            })
            os.chmod(path, 0o444)
    payload = {
        'schema_version': 1, 'status': 'completed',
        'source_hoa_manifest_path': str(hoa_path),
        'source_hoa_archive_version': hoa['archive_version'],
        'source_hoa_archive_hash': hoa['archive_hash'],
        'num_transitions': sum(item['count'] for item in pools),
        'dtype': DTYPE.descr, 'pools': pools,
        'sampling_contract': (
            'uniform scene, then uniform intersection, then uniform transition'),
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(manifest_path, payload)
    os.chmod(manifest_path, 0o444)
    return manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hoa-manifest', default=
                        'artifacts/arterial_experiments/historical_archive/manifest.json')
    parser.add_argument('--hoa-hash', required=True)
    parser.add_argument('--output-dir', default=
                        'artifacts/arterial_experiments/historical_archive/sampling_index')
    args = parser.parse_args()
    print(build(args.hoa_manifest, args.hoa_hash, args.output_dir))


if __name__ == '__main__':
    main()
