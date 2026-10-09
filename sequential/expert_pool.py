"""Read-only HITL expert-correction pool for HA-SODQN.

The pool wraps a packaged broadcast-session archive produced by
``dev/hitl_live/package.py``.  Unlike :class:`HistoricalArchive` (the frozen
Plan 1 curriculum), the expert pool is a *single-network, precollected*
correction dataset: every transition it contains was actually executed in a
live session — including gate-triggered expert takeovers — so it is causally
visible from stage 1 and never mutates.

The adapter exposes the minimal surface that
:class:`sequential.historical_archive.VisibleArchiveView` and
:class:`sequential.owp.HistoricalSampler` require: ``manifest`` with an
``archive_digest``, ``datasets`` keyed by network, and a ``view()`` returning a
``VisibleArchiveView`` in which every packaged transition is eligible.
"""

import os

from dataset.offline_trajectory_dataset import OfflineTrajectoryDataset

from .historical_archive import VisibleArchiveView, derive_rng_seed
from .io import read_json, sha256_file


EXPERT_POOL_SCHEMA_VERSION = 1


class ExpertPoolArchive:
    """Load and verify a ``expert_pool_manifest.json`` + its dataset layer."""

    def __init__(self, manifest_path, training_seed, verify_hashes=True):
        self.manifest_path = os.path.abspath(manifest_path)
        manifest = read_json(self.manifest_path)
        if manifest.get('schema_version') != EXPERT_POOL_SCHEMA_VERSION:
            raise ValueError('Unsupported expert pool manifest schema')
        if manifest.get('kind') != 'hitl_expert_pool':
            raise ValueError('Manifest is not a hitl_expert_pool')
        # VisibleArchiveView derives its digest from ``archive_digest`` and
        # ``behavior_seed_rule``; map them onto the pool manifest fields.
        self.manifest = dict(manifest)
        self.manifest['archive_digest'] = manifest['pool_digest']
        self.manifest.setdefault('behavior_seed_rule', {
            'exclude_matching_training_seed': False,
        })
        self.pool_digest = manifest.get('pool_digest')
        self.network = manifest['network']
        dataset_record = manifest['dataset_manifest']
        dataset_path = dataset_record['path']
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(
                os.path.dirname(self.manifest_path), dataset_path)
        dataset_path = os.path.abspath(dataset_path)
        if sha256_file(dataset_path) != dataset_record['sha256']:
            raise ValueError(
                'Expert pool dataset manifest changed after packaging')
        dataset = OfflineTrajectoryDataset(
            dataset_path,
            seed=derive_rng_seed(
                training_seed, f'expert-pool-load:{self.network}'),
            verify_hashes=verify_hashes,
        )
        if dataset.observation_dim != int(manifest['observation_dim']):
            raise ValueError('Expert pool observation_dim mismatch')
        if dataset.action_dim != int(manifest['action_dim']):
            raise ValueError('Expert pool action_dim mismatch')
        if dataset.transition_count != int(manifest['transition_count']):
            raise ValueError('Expert pool transition_count mismatch')
        self.datasets = {self.network: dataset}

    def view(self):
        """All packaged transitions are eligible; packaging already quarantined
        invalid episodes, so no seed/episode filtering is applied."""
        indices = {
            self.network: self.datasets[self.network].eligible_indices(
                self.network),
        }
        visibility = {
            'archive_mode': 'EXPERT',
            'network': self.network,
            'dataset_id': self.manifest['dataset_id'],
            'transition_count': len(indices[self.network]),
            'behavior_mode_counts': dict(
                self.manifest.get('behavior_mode_counts', {})),
        }
        view = VisibleArchiveView(self, indices, visibility)
        return view
