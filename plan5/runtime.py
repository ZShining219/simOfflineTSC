"""Auditable single-child runtime for Plan5 anchors and transitions."""
import copy
import os
from pathlib import Path
import time

from sequential.io import atomic_json, read_json, sha256_file

from .checkpoint import restore_checkpoint, restore_rng_state, save_checkpoint
from .checkpoint_index import CheckpointIndex
from .config import HISTORICAL_SCHEDULE, SCENES, load_config
from .evaluator import (
    formal_evaluation_schedule, full_checkpoint_schedule,
)
from .frozen_evaluator import (
    Plan5FrozenEvaluator, save_evaluation_snapshot,
)
from .manifest import validate_run_manifest, validate_ppo_calibration_row
from .probe import fixed_probe_inference, load_probe_rows
from .provenance import collect_source_provenance
from .smoke import _build_agent, _train_episodes
from .sumo_runtime import build_world
from .validation import (
    validate_checkpoint_binding, validate_formal_artifacts,
    validate_launch_row, validate_transition_row,
)
from .schema import validate_json_fields


class Plan5FormalRunner:
    def __init__(self, row, attempt_dir, *, probe_manifest,
                 interface='traci', resume_from=None, ppo_config=None,
                 verify_source=True):
        self.row = copy.deepcopy(row)
        if self.row.get('run_type') == 'CALIBRATION':
            validate_ppo_calibration_row(self.row)
        else:
            validate_launch_row(self.row)
        self.config = load_config()
        if self.row['config_sha256'] != self.config.sha256():
            raise ValueError('Formal row/config SHA mismatch')
        self.source_provenance = collect_source_provenance()
        if verify_source and (
                not self.source_provenance['formal_source_ready']
                or self.source_provenance['head'] != self.row['source_commit']):
            raise ValueError(
                'Formal run source is dirty or does not match source_commit'
            )
        self.attempt_dir = Path(attempt_dir).resolve()
        existing = (
            set(path.name for path in self.attempt_dir.iterdir())
            if self.attempt_dir.exists() else set()
        )
        if existing - {'run_row.json'}:
            raise FileExistsError(
                f'Plan5 attempt directory is not empty: {self.attempt_dir}'
            )
        self.attempt_dir.mkdir(parents=True, exist_ok=True)
        self.probe_manifest = (
            None if probe_manifest is None else Path(probe_manifest).resolve()
        )
        if self.row['run_type'] != 'CALIBRATION' \
                and self.probe_manifest is None:
            raise ValueError('Formal anchor/transition requires frozen probe')
        self.interface = interface
        self.resume_from = None if resume_from is None else Path(
            resume_from
        ).resolve()
        self.ppo_config = (
            None if ppo_config is None else Path(ppo_config).resolve()
        )
        self.ppo_parameters = None
        if self.row['algorithm_id'] == 'PPO' \
                and self.row['run_type'] != 'CALIBRATION':
            if self.ppo_config is None:
                raise ValueError(
                    'Formal PPO run requires a frozen calibration config'
                )
            validate_json_fields(self.ppo_config, 'ppo_formal_config.json')
            ppo_payload = read_json(self.ppo_config)
            if ppo_payload['formal_status'] != 'READY':
                raise ValueError('PPO formal config is not READY')
            expected_sha = self.row.get('ppo_config_sha256')
            if not expected_sha or sha256_file(self.ppo_config) != expected_sha:
                raise ValueError('PPO config SHA is not bound to formal row')
            if str(self.ppo_config) != str(Path(
                    self.row.get('ppo_config_path', '')).resolve()):
                raise ValueError('PPO config path is not bound to formal row')
            if ppo_payload['config_sha256'] != self.config.sha256():
                raise ValueError('PPO calibration used a different Plan5 config')
            self.ppo_parameters = {
                'learning_rate': float(ppo_payload['learning_rate']),
                'entropy_coefficient': float(
                    ppo_payload['entropy_coefficient']
                ),
            }
        self.world = None
        self.agent = None
        self.paired_initialization = None
        self.evaluations = []
        self.probe_records = []
        self.episode_summaries = []
        self.index = CheckpointIndex(
            self.attempt_dir / 'checkpoint_index.json'
        )
        self.evaluator = Plan5FrozenEvaluator()
        self._probe_cache = (
            {} if self.probe_manifest is None else {
                split: load_probe_rows(self.probe_manifest, split)
                for split in ('main', 'heldout')
            }
        )
        self._write_initial_manifest()

    @property
    def algorithm_id(self):
        return self.row['algorithm_id']

    @property
    def seed(self):
        return int(self.row['training_seed'])

    def _identity(self):
        identity = {
            'logical_run_id': self.row['logical_run_id'],
            'algorithm_id': self.algorithm_id,
            'training_seed': self.seed,
            'run_type': self.row['run_type'],
            'config_sha256': self.row['config_sha256'],
            'source_commit': self.row['source_commit'],
        }
        for field in ('scene', 'transition_id', 'source_scene', 'target_scene'):
            if self.row.get(field):
                identity[field] = self.row[field]
        for field in (
                'source_checkpoint_path', 'source_checkpoint_sha256',
                'model_state_digest', 'optimizer_state_digest',
                'algorithm_state_digest', 'rng_state_digest',
                'ppo_config_path', 'ppo_config_sha256', 'attempt_id'):
            if self.row.get(field):
                identity[field] = self.row[field]
        if self.row['run_type'] == 'CALIBRATION':
            identity.update({
                'learning_rate': float(self.row['learning_rate']),
                'entropy_coefficient': float(
                    self.row['entropy_coefficient']
                ),
            })
        return identity

    def _write_initial_manifest(self):
        resolved_config = self.attempt_dir / 'resolved_config.json'
        atomic_json(resolved_config, self.config.as_dict())
        atomic_json(self.attempt_dir / 'training_manifest.json', {
            'schema_version': 1, 'identity': self._identity(),
            'source_provenance': {
                'branch': self.source_provenance['branch'],
                'head': self.source_provenance['head'],
                'source_scope_clean': self.source_provenance[
                    'source_scope_clean'
                ],
                'source_scope_digest': self.source_provenance[
                    'source_scope_digest'
                ],
            },
            'attempt_dir': str(self.attempt_dir),
            'resolved_config': str(resolved_config.resolve()),
            'resolved_config_sha256': sha256_file(resolved_config),
            'probe_manifest': (
                None if self.probe_manifest is None
                else str(self.probe_manifest)
            ),
            'probe_manifest_sha256': (
                None if self.probe_manifest is None
                else sha256_file(self.probe_manifest)
            ),
            'interface': self.interface,
            'resume_from': None if self.resume_from is None else str(
                self.resume_from
            ),
            'ppo_config': (
                None if self.ppo_config is None else str(self.ppo_config)
            ),
            'ppo_config_sha256': (
                None if self.ppo_config is None else sha256_file(self.ppo_config)
            ),
            'status': 'running', 'started_at_unix': time.time(),
        })

    def _write_analysis_indexes(self):
        assets = {}
        for name, records in (
                ('episode_summaries', self.episode_summaries),
                ('evaluation_index', self.evaluations),
                ('probe_index', self.probe_records)):
            path = self.attempt_dir / f'{name}.json'
            atomic_json(path, {
                'schema_version': 1, 'identity': self._identity(),
                'records': records,
            })
            assets[name] = {
                'path': str(path.resolve()), 'sha256': sha256_file(path),
                'record_count': len(records),
            }
        return assets

    def _begin_agent(self):
        run_type = self.row['run_type']
        if run_type in {'ANCHOR', 'CALIBRATION'}:
            scene = self.row['scene']
            self.world, command = build_world(
                SCENES[scene], self.attempt_dir / 'training_sumo' / scene,
                interface=self.interface, context=True,
            )
            ppo_parameters = self.ppo_parameters
            if run_type == 'CALIBRATION':
                ppo_parameters = {
                    'learning_rate': float(self.row['learning_rate']),
                    'entropy_coefficient': float(
                        self.row['entropy_coefficient']
                    ),
                }
            self.agent, self.paired_initialization = _build_agent(
                self.algorithm_id, self.world, self.seed, self.config,
                ppo_parameters,
            )
            if self.algorithm_id in {'DDQN', 'CTXDDQN'}:
                self.agent.begin_stage(1, scene, 'clear')
            else:
                self.agent.begin_stage(1, scene)
        else:
            validate_transition_row(self.row)
            validate_checkpoint_binding(self.row)
            source = self.row['source_scene']
            target = self.row['target_scene']
            source_world, _ = build_world(
                SCENES[source],
                self.attempt_dir / 'source_binding_sumo' / source,
                interface=self.interface, context=True,
            )
            try:
                self.agent, self.paired_initialization = _build_agent(
                    self.algorithm_id, source_world, self.seed, self.config,
                    self.ppo_parameters,
                )
                source_checkpoint = restore_checkpoint(
                    self.agent, self.row['source_checkpoint_path'],
                    {'algorithm_id': self.algorithm_id,
                     'training_seed': self.seed, 'scene': source},
                )
            finally:
                source_world.close()
            self.world, command = build_world(
                SCENES[target], self.attempt_dir / 'training_sumo' / target,
                interface=self.interface, context=True,
            )
            self.agent.rebind_environment(self.world)
            if self.algorithm_id in {'DDQN', 'CTXDDQN'}:
                self.agent.begin_stage(2, target, 'clear')
            else:
                self.agent.begin_stage(2, target)
            # Target-world construction is outside the checkpointed learning
            # state.  Reassert the exact source RNG boundary so H34/L32 start
            # from the same algorithm-internal stream even if a simulator or
            # generator constructor ever consumes a global RNG.
            restore_rng_state(source_checkpoint['rng_state'])
        if self.resume_from is not None:
            resumed = restore_checkpoint(
                self.agent, self.resume_from,
                {'logical_run_id': self.row['logical_run_id'],
                 'algorithm_id': self.algorithm_id,
                 'training_seed': self.seed},
            )
            self._import_resume_checkpoint_index(
                max_episode=int(resumed['episode'])
            )
        return command

    def _import_resume_checkpoint_index(self, *, max_episode):
        """Carry immutable checkpoint evidence into a recovery attempt."""
        if self.resume_from is None:
            return
        prior_index = None
        for parent in self.resume_from.parents:
            candidate = parent / 'checkpoint_index.json'
            if candidate.is_file():
                prior_index = candidate
                break
        if prior_index is None:
            raise ValueError(
                'Formal resume checkpoint has no prior checkpoint index'
            )
        payload = read_json(prior_index)
        for entry in payload.get('checkpoints', []):
            # Artifacts after the committed recovery boundary remain preserved
            # in the failed attempt but cannot enter the resumed run's index.
            if int(entry['episode']) > int(max_episode):
                continue
            if entry.get('logical_run_id') != self.row['logical_run_id']:
                raise ValueError('Formal resume checkpoint index identity mismatch')
            self.index.register(
                logical_run_id=entry['logical_run_id'],
                checkpoint_type=entry['checkpoint_type'],
                episode=int(entry['episode']), path=entry['path'],
                canonical_state_digest=entry['canonical_state_digest'],
            )

    def _schedule(self):
        if self.row['run_type'] in {'ANCHOR', 'CALIBRATION'}:
            return formal_evaluation_schedule(
                'ANCHOR', scene=self.row['scene'],
            )
        return formal_evaluation_schedule(
            'TRANSITION', source_scene=self.row['source_scene'],
            target_scene=self.row['target_scene'],
        )

    def _state_payload(self, completed_episode):
        return {
            'completed_episode': int(completed_episode),
            'episode_summaries': copy.deepcopy(self.episode_summaries),
            'evaluations': copy.deepcopy(self.evaluations),
            'probe_records': copy.deepcopy(self.probe_records),
        }

    def _save_recovery(self, episode):
        directory = self.attempt_dir / 'checkpoints' / 'recovery'
        directory.mkdir(parents=True, exist_ok=True)
        new = directory / 'latest.new.pt'
        latest = directory / 'latest.pt'
        previous = directory / 'previous.pt'
        payload = save_checkpoint(
            self.agent, new, self._identity(), episode=episode,
            stage=1 if self.row['run_type'] in {'ANCHOR', 'CALIBRATION'} else 2,
            extra_state=self._state_payload(episode),
        )
        if latest.exists():
            os.replace(latest, previous)
        os.replace(new, latest)
        atomic_json(self.attempt_dir / 'current_state.json', {
            'schema_version': 1, 'logical_run_id': self.row['logical_run_id'],
            'completed_episode': episode,
            'recovery_checkpoint': str(latest.resolve()),
            'recovery_checkpoint_sha256': sha256_file(latest),
            'canonical_state_digest': payload['canonical_state_digest'],
            'updated_at_unix': time.time(),
        })

    def _save_evaluation_snapshot(self, episode):
        snapshot_path = (
            self.attempt_dir / 'checkpoints' / 'evaluation'
            / f'episode_{episode:04d}.pt'
        )
        snapshot = save_evaluation_snapshot(
            self.agent, snapshot_path,
            {**self._identity(), 'episode': episode},
        )
        self.index.register(
            logical_run_id=self.row['logical_run_id'],
            checkpoint_type='evaluation_snapshot', episode=episode,
            path=snapshot_path,
            canonical_state_digest=snapshot['model_state_digest'],
        )
        return snapshot_path

    def _save_full_checkpoint(self, episode):
        schedule_type = (
            'ANCHOR' if self.row['run_type'] == 'CALIBRATION'
            else self.row['run_type']
        )
        if episode not in full_checkpoint_schedule(schedule_type):
            return None
        full_path = (
            self.attempt_dir / 'checkpoints' / 'resumable'
            / f'episode_{episode:04d}.pt'
        )
        payload = save_checkpoint(
            self.agent, full_path, self._identity(), episode=episode,
            stage=1 if self.row['run_type'] in {'ANCHOR', 'CALIBRATION'} else 2,
            # Full checkpoints are committed only after this episode's frozen
            # evaluations and probe records are complete.  They are therefore
            # directly resumable at episode + 1.
            extra_state=self._state_payload(episode),
        )
        self.index.register(
            logical_run_id=self.row['logical_run_id'],
            checkpoint_type='full_resumable', episode=episode,
            path=full_path,
            canonical_state_digest=payload['canonical_state_digest'],
        )
        return full_path

    def _evaluate(self, episode, snapshot_path, parent_checkpoint):
        cells = [
            cell for cell in self._schedule() if cell['episode'] == episode
        ]
        for cell in cells:
            evaluation_dir = (
                self.attempt_dir / 'evaluations'
                / f'episode_{episode:04d}' / cell['scene'] / 'attempt_1'
            )
            result = self.evaluator.evaluate(
                self.agent, snapshot_path, SCENES[cell['scene']],
                evaluation_dir, interface=self.interface,
                parent_checkpoint=parent_checkpoint,
            )
            self.evaluations.append({
                **cell, 'manifest_path': str(
                    (evaluation_dir / 'evaluation_manifest.json').resolve()
                ),
                'manifest_sha256': sha256_file(
                    evaluation_dir / 'evaluation_manifest.json'
                ),
                'summary': result['summary'],
            })

    def _run_probe(self, episode):
        if self.row['run_type'] == 'CALIBRATION':
            return
        required = (
            episode in {0, 100}
            if self.row['run_type'] == 'ANCHOR'
            else episode in HISTORICAL_SCHEDULE
        )
        if not required:
            return
        scene = (
            self.row['scene'] if self.row['run_type'] == 'ANCHOR'
            else self.row['source_scene']
        )
        for split in ('main', 'heldout'):
            rows = [
                row for row in self._probe_cache[split]
                if row['scene'] == scene
            ]
            result = fixed_probe_inference(self.agent, rows)
            path = (
                self.attempt_dir / 'probe' / f'episode_{episode:04d}'
                / f'{split}.json'
            )
            atomic_json(path, result)
            self.probe_records.append({
                'episode': episode, 'scene': scene, 'split': split,
                'path': str(path.resolve()), 'sha256': sha256_file(path),
                'row_count': result['row_count'],
                'actions_digest': result['actions_digest'],
                'policy_vectors_digest': result['policy_vectors_digest'],
                'target_policy_vectors_digest': result[
                    'target_policy_vectors_digest'
                ],
                'critic_values_digest': result['critic_values_digest'],
            })

    def run(self):
        started = time.perf_counter()
        try:
            command = self._begin_agent()
            start_episode = 0
            if self.resume_from is not None:
                from .checkpoint import load_checkpoint
                recovery = load_checkpoint(self.resume_from)
                extra = recovery.get('extra_state', {})
                if int(extra.get('completed_episode', -1)) != int(
                        recovery['episode']):
                    raise ValueError(
                        'Formal attempt resume requires a committed recovery '
                        'checkpoint, not a pre-evaluation snapshot'
                    )
                self.episode_summaries = copy.deepcopy(
                    extra.get('episode_summaries', [])
                )
                self.evaluations = copy.deepcopy(extra.get('evaluations', []))
                self.probe_records = copy.deepcopy(
                    extra.get('probe_records', [])
                )
                start_episode = int(extra.get(
                    'completed_episode', recovery['episode']
                )) + 1
            if start_episode == 0:
                snapshot = self._save_evaluation_snapshot(0)
                self._evaluate(0, snapshot, snapshot)
                self._run_probe(0)
                self._save_full_checkpoint(0)
                self._save_recovery(0)
                start_episode = 1
            for episode in range(start_episode, 101):
                summary = _train_episodes(
                    self.agent, self.world, self.algorithm_id,
                    episode, episode, self.config,
                )[0]
                self.episode_summaries.append(summary)
                snapshot = self._save_evaluation_snapshot(episode)
                self._evaluate(episode, snapshot, snapshot)
                self._run_probe(episode)
                self._save_full_checkpoint(episode)
                self._save_recovery(episode)
            self.index.validate()
            completion = validate_formal_artifacts(
                self.row, self.episode_summaries, self.evaluations,
                self.probe_records, self.index.path,
                resume_from=self.resume_from,
            )
            atomic_json(
                self.attempt_dir / 'run_validation.json', completion,
            )
            analysis_indexes = self._write_analysis_indexes()
            result = {
                'schema_version': 1, 'identity': self._identity(),
                'resolved_sumo_command': command,
                'episode_count': len(self.episode_summaries),
                'evaluation_count': len(self.evaluations),
                'probe_record_count': len(self.probe_records),
                'analysis_indexes': analysis_indexes,
                'paired_initialization': self.paired_initialization,
                'parameter_ownership': (
                    self.agent.parameter_ownership()
                    if self.algorithm_id == 'PPO' else None
                ),
                'checkpoint_index': str(self.index.path.resolve()),
                'checkpoint_index_sha256': sha256_file(self.index.path),
                'run_validation': str(
                    (self.attempt_dir / 'run_validation.json').resolve()
                ),
                'run_validation_sha256': sha256_file(
                    self.attempt_dir / 'run_validation.json'
                ),
                'wall_time_seconds': time.perf_counter() - started,
                'status': 'completed', 'valid': True,
            }
            atomic_json(self.attempt_dir / 'run_summary.json', result)
            manifest = {
                **read_json(self.attempt_dir / 'training_manifest.json'),
                'status': 'completed', 'completed_at_unix': time.time(),
                'run_summary': str(
                    (self.attempt_dir / 'run_summary.json').resolve()
                ),
            }
            atomic_json(self.attempt_dir / 'training_manifest.json', manifest)
            return result
        except Exception as error:
            manifest_path = self.attempt_dir / 'training_manifest.json'
            if manifest_path.is_file():
                manifest = {
                    **read_json(manifest_path), 'status': 'failed',
                    'failed_at_unix': time.time(),
                    'failure_class': type(error).__name__,
                    'failure_message': str(error),
                }
                atomic_json(manifest_path, manifest)
            raise
        finally:
            if self.world is not None:
                self.world.close()


def run_formal_child(row, attempt_dir, **kwargs):
    return Plan5FormalRunner(row, attempt_dir, **kwargs).run()
