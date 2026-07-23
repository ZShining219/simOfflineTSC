import json
import os
import tempfile

import numpy as np

from .checkpoint import RollingRecoveryManager
from .core import canonical_transition_digest
from .io import atomic_json, sha256_file


class EpisodeTransaction:
    def __init__(self, attempt_dir, agent, identity, journal=None):
        self.attempt_dir = os.path.abspath(attempt_dir)
        self.agent = agent
        self.identity = dict(identity)
        self.journal = journal
        self.trajectory_dir = os.path.join(self.attempt_dir, 'trajectory', 'episodes')
        self.marker_dir = os.path.join(self.attempt_dir, 'trajectory', 'committed')
        self.temporary_dir = os.path.join(self.attempt_dir, 'trajectory', 'temporary')
        self.recovery = RollingRecoveryManager(
            os.path.join(self.attempt_dir, 'checkpoints', 'recovery')
        )
        os.makedirs(self.trajectory_dir, exist_ok=True)
        os.makedirs(self.marker_dir, exist_ok=True)
        os.makedirs(self.temporary_dir, exist_ok=True)
        name = (
            f'stage_{int(self.identity["stage_index"]):02d}_'
            f'episode_{int(self.identity["local_episode"]):04d}'
        )
        self.shard_path = os.path.join(self.trajectory_dir, f'{name}.npz')
        self.marker_path = os.path.join(self.marker_dir, f'{name}.json')
        self.temp_path = os.path.join(self.temporary_dir, f'{name}.npz.tmp')
        self.transitions = []
        self.started = False

    def begin(self, extra_state=None):
        if os.path.exists(self.marker_path):
            raise FileExistsError('Episode is already committed')
        if os.path.exists(self.shard_path):
            os.unlink(self.shard_path)
        self._discard_uncommitted_temporary_files()
        self.recovery.save_episode_start(
            self.agent, self.identity, extra_state=extra_state,
        )
        self.transitions = []
        self.started = True
        return self

    def _discard_uncommitted_temporary_files(self):
        for name in os.listdir(self.temporary_dir):
            path = os.path.join(self.temporary_dir, name)
            if os.path.isfile(path):
                os.unlink(path)

    def append(self, transition):
        if not self.started:
            raise RuntimeError('Episode transaction has not started')
        self.transitions.append(dict(transition))

    def _write_shard(self):
        if not self.transitions:
            raise ValueError('Cannot commit an empty episode trajectory')
        fields = (
            'stage_index', 'local_episode', 'global_episode', 'decision_index',
            'global_decision_step', 'state', 'phase', 'action', 'reward',
            'next_state', 'next_phase', 'terminated', 'truncated',
            'transition_id',
        )
        arrays = {}
        for field in fields:
            values = [transition[field] for transition in self.transitions]
            arrays[field] = np.asarray(values)
        descriptor, temporary = tempfile.mkstemp(
            prefix='.tmp-trajectory-', suffix='.npz', dir=self.temporary_dir,
        )
        os.close(descriptor)
        try:
            with open(temporary, 'wb') as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.shard_path)
        except Exception:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    def prepare_shard(self):
        self._write_shard()
        return self.shard_path

    def commit(self):
        if not os.path.isfile(self.shard_path):
            self._write_shard()
        transition_digests = [
            canonical_transition_digest(transition)
            for transition in self.transitions
        ]
        marker = {
            'schema_version': 1,
            'identity': self.identity,
            'transition_count': len(self.transitions),
            'trajectory_path': self.shard_path,
            'trajectory_sha256': sha256_file(self.shard_path),
            'canonical_transition_digests': transition_digests,
        }
        atomic_json(self.marker_path, marker)
        if self.journal is not None:
            operation_key = (
                f'stage_{self.identity["stage_index"]}:'
                f'episode_{self.identity["local_episode"]}:trajectory'
            )
            self.journal.record_operation(operation_key, self.marker_path)
            self.journal.update_episode(
                self.identity['stage_index'], self.identity['local_episode'],
                self.identity['global_episode'],
                {'trajectory_marker': self.marker_path},
            )
        self.started = False
        return marker

    def rollback(self):
        path, payload = self.recovery.restore_latest(self.agent)
        self._discard_uncommitted_temporary_files()
        self.transitions = []
        self.started = False
        return {'recovery_checkpoint': path, 'identity': payload['identity']}
