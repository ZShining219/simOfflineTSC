import json
import os
import random
import signal
import time

import numpy as np
import torch

from .agent import SequentialDQNAgent
from .hybrid_agent import HybridDQNAgent
from .checkpoint import (
    atomic_torch_save, build_full_checkpoint, load_full_checkpoint,
    save_stage_checkpoint,
)
from .config import simulator_config_path
from .diagnostics import ReplayDiagnostics
from .evaluator import IndependentEvaluator, save_online_snapshot
from .io import atomic_json, read_json, sha256_file
from .state import SequentialJournal, SequentialRunPhase
from .trainer import SequentialStageTrainer
from .transaction import EpisodeTransaction


class SequentialInterrupted(BaseException):
    def __init__(self, signum):
        super().__init__(f'Sequential child interrupted by signal {signum}')
        self.signum = signum


def _restore_rng(rng):
    random.setstate(rng['python_random_state'])
    np.random.set_state(rng['numpy_random_state'])
    torch.set_rng_state(rng['torch_cpu_rng_state'])
    if torch.cuda.is_available() and rng['torch_cuda_rng_states']:
        torch.cuda.set_rng_state_all(rng['torch_cuda_rng_states'])


class SequentialChildRunner:
    def __init__(self, manifest_path, logical_run_id, attempt_dir,
                 resume_path=None, resume_state_path=None):
        self.manifest_path = os.path.abspath(manifest_path)
        self.plan = read_json(self.manifest_path)
        matches = [
            child for child in self.plan['children']
            if child['logical_run_id'] == logical_run_id
        ]
        if len(matches) != 1:
            raise ValueError(f'Logical run identity is not unique: {logical_run_id}')
        self.child = matches[0]
        self.logical_run_id = logical_run_id
        self.attempt_dir = os.path.abspath(attempt_dir)
        self.attempt_id = os.path.basename(self.attempt_dir)
        self.resume_path = os.path.abspath(resume_path) if resume_path else None
        self.resume_state_path = (
            os.path.abspath(resume_state_path) if resume_state_path else None
        )
        self.networks = list(self.child['networks'])
        self.stage_episodes = list(self.child['stage_episodes'])
        self.policy = self.child['policy']
        self.training_seed = int(self.child['training_seed'])
        self.parent_manifest = read_json(self.child['parent_import_manifest'])
        self._validate_parent_binding()
        self.model_config = self.plan['model']
        self.trainer_config = dict(self.plan['trainer'])
        self.interface = self.child.get('interface', 'libsumo')
        self.world = None
        self.agent = None
        self.trainer = None
        self.resume_payload = None
        os.makedirs(self.attempt_dir, exist_ok=True)
        self.logical_root = os.path.dirname(os.path.dirname(self.attempt_dir))
        self.launch_root = os.path.dirname(self.logical_root)
        self.shared_evaluation_root = os.path.join(
            self.launch_root, '_shared_evaluation'
        )
        self.evaluator = IndependentEvaluator(
            self.shared_evaluation_root, retries=3,
            timeout_seconds=int(self.trainer_config.get('evaluation_timeout_seconds', 300)),
        )
        self.journal = SequentialJournal(
            self.attempt_dir, self.logical_run_id, self.attempt_id,
            initial_stage=1, initial_global_episode=(
                0 if self.child.get('condition') in {'M0', 'M1', 'M2', 'M3'}
                else self.stage_episodes[0]
            ),
            resume_state_path=self.resume_state_path,
        )
        self.diagnostics = ReplayDiagnostics(
            os.path.join(self.attempt_dir, 'replay_diagnostics'),
            trace_samples=bool(self.child.get('trace_replay_samples', False)),
        )
        self._write_run_manifest()

    def _validate_parent_binding(self):
        expected_episode = int(self.child['parent_checkpoint_episode'])
        if self.child.get('budget_id') != f'b{expected_episode}':
            raise ValueError('Child budget does not match parent checkpoint episode')
        if int(self.stage_episodes[0]) != expected_episode:
            raise ValueError('Stage 1 budget does not match parent checkpoint episode')
        if int(self.parent_manifest.get('checkpoint_episode', -1)) != expected_episode:
            raise ValueError('Parent import checkpoint episode does not match child')
        child_checkpoint = os.path.abspath(self.child['parent_checkpoint'])
        imported_checkpoint = os.path.abspath(self.parent_manifest['checkpoint_path'])
        if child_checkpoint != imported_checkpoint:
            raise ValueError('Parent checkpoint path differs from import manifest')
        recorded_hash = self.child['parent_checkpoint_file_sha256']
        if self.parent_manifest.get('checkpoint_file_sha256') != recorded_hash:
            raise ValueError('Parent checkpoint hash differs from import manifest')
        if sha256_file(child_checkpoint) != recorded_hash:
            raise ValueError('Parent checkpoint file SHA-256 mismatch')
        if self.parent_manifest.get('digests') != self.child.get('parent_digests'):
            raise ValueError('Parent state digests differ from import manifest')

    def _write_run_manifest(self):
        atomic_json(os.path.join(self.attempt_dir, 'child_run_manifest.json'), {
            'schema_version': 1,
            'logical_run_id': self.logical_run_id,
            'attempt_id': self.attempt_id,
            'order_id': self.child['order_id'],
            'networks': self.networks,
            'training_seed': self.training_seed,
            'policy': self.policy,
            'condition': self.child.get('condition'),
            'budget_id': self.child['budget_id'],
            'parent_checkpoint_episode': self.child['parent_checkpoint_episode'],
            'stage_episodes': self.stage_episodes,
            'parent_import_manifest': self.child['parent_import_manifest'],
            'parent_checkpoint': self.child['parent_checkpoint'],
            'parent_trajectory_reference': os.path.join(
                self.parent_manifest['source_run_path'], 'trajectory'
            ),
            'parent_trajectory_config_digest': self.parent_manifest['digests'][
                'environment_signature_digest'
            ],
            'resume_checkpoint': self.resume_path,
            'resume_state': self.resume_state_path,
        })

    def _create_world(self, network):
        from common import interface as registry_interface
        from common.registry import Registry
        import world.world_sumo as world_sumo

        registry_interface.Command_Setting_Interface({
            'command': {'sumo_seed': None}
        })
        Registry.mapping['logger_mapping']['path'].path = self.attempt_dir
        return world_sumo.World(
            simulator_config_path(network), 1, interface=self.interface,
        )

    def _stage_global_base(self, stage_index):
        return sum(int(value) for value in self.stage_episodes[:stage_index - 1])

    def _write_resume_pointer(self, checkpoint_path, reason):
        atomic_json(os.path.join(self.attempt_dir, 'resume_pointer.json'), {
            'schema_version': 1,
            'checkpoint_path': os.path.abspath(checkpoint_path),
            'reason': reason,
            'stage_index': self.journal.state['stage_index'],
            'local_episode': self.journal.state['local_episode'],
            'phase': self.journal.state['phase'],
        })

    def _initialise_agent(self):
        stage_index = int(self.journal.state['stage_index'])
        if self.journal.phase == SequentialRunPhase.OLD_WORLD_CLOSED:
            stage_index = min(stage_index + 1, len(self.networks))
        network = self.networks[stage_index - 1]
        self.world = self._create_world(network)
        agent_cls = HybridDQNAgent if self.policy == 'hybrid' else SequentialDQNAgent
        hybrid_kwargs = {}
        if agent_cls is HybridDQNAgent:
            hybrid_kwargs = {
                'online_ratio': float(self.child.get('hybrid_online_ratio', 0.5)),
                'historical_sampling': self.child.get(
                    'historical_sampling', 'stage_balanced_episode_stratified'
                ),
            }
        self.agent = agent_cls.from_parent(
            self.world, 0, self.model_config, self.trainer_config,
            self.parent_manifest, **hybrid_kwargs,
        )
        if self.resume_path:
            self.resume_payload = load_full_checkpoint(self.resume_path)
            self.agent.load_full_state_dict(self.resume_payload['agent_state'])
            extra = self.resume_payload.get('extra_state', {})
            if 'replay_diagnostics' in extra:
                self.diagnostics.load_state_dict(extra['replay_diagnostics'])
        self.trainer = SequentialStageTrainer(
            self.agent, self.world, self.trainer_config,
            decision_hook=self._decision_progress_hook,
            replay_diagnostics=self.diagnostics,
        )
        if self.child.get('condition') in {'M0', 'M1', 'M2', 'M3'} and not self.resume_path:
            # Hybrid conditions deliberately start Stage 1 with an empty pool;
            # the parent checkpoint supplies model/optimizer/RNG only.
            self.agent.begin_stage(1, self.networks[0], self.policy)

    def _decision_progress_hook(self, progress):
        if not (
            self.child.get('variant') == 'fault'
            and self.child.get('fault_point')
            == 'stage3_episode4_simulation_step180'
            and int(progress['stage_index']) == 3
            and int(progress['local_episode']) == 4
        ):
            return
        progress_path = os.path.join(self.attempt_dir, 'training_progress.json')
        atomic_json(progress_path, {'schema_version': 1, **progress})
        if int(progress['simulation_step']) != 180:
            return
        self._await_fault_signal(self.child['fault_point'], {
            'progress_path': progress_path, **progress,
        })

    def _await_fault_signal(self, fault_point, payload=None):
        if not (
            self.child.get('variant') == 'fault'
            and self.child.get('fault_point') == fault_point
        ):
            return
        payload = payload or {}
        expected_stage = {
            'after_REPLAY_POLICY_APPLIED_before_local0': 2,
            'stage2_matrix_half_complete': 2,
            'stage3_episode4_simulation_step180': 3,
        }[fault_point]
        if int(payload.get('stage_index', -1)) != expected_stage:
            return
        operation_key = f'fault_trigger:{fault_point}'
        if self.journal.operation_completed(operation_key):
            return
        trigger_path = os.path.join(
            self.attempt_dir, 'fault_triggers', f'{fault_point}.json'
        )
        atomic_json(trigger_path, {
            'schema_version': 1, 'fault_point': fault_point, **payload,
        })
        self.journal.record_operation(operation_key, trigger_path, payload)
        self.journal.record_event('FAULT_TRIGGER_READY', {
            'fault_point': fault_point, 'trigger_path': trigger_path, **payload,
        })
        # Give the external harness a deterministic observation window.  A
        # real SIGTERM interrupts this wait through the installed handler.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            time.sleep(0.05)

    def _checkpoint_extra_state(self):
        return {'replay_diagnostics': self.diagnostics.state_dict()}

    def _snapshot_path(self, stage_index, local_episode):
        return os.path.join(
            self.attempt_dir, 'checkpoints', 'online',
            f'stage_{stage_index:02d}_episode_{local_episode:04d}.pt',
        )

    def _ensure_snapshot(self, stage_index, local_episode, global_episode):
        path = self._snapshot_path(stage_index, local_episode)
        if os.path.isfile(path):
            payload = torch.load(path, map_location='cpu')
            if payload['online_parameter_digest'] != self.agent.training_state_digests()[
                'online_parameter_digest'
            ]:
                raise ValueError('Existing online snapshot does not match agent')
            return path
        save_online_snapshot(self.agent, path, {
            'logical_run_id': self.logical_run_id,
            'stage_index': stage_index,
            'local_episode': local_episode,
            'global_episode': global_episode,
            'training_network': self.networks[stage_index - 1],
        })
        return path

    def _evaluate_cell(self, stage_index, training_network, evaluation_network,
                       local_episode, global_episode):
        operation_key = (
            f'evaluation:stage_{stage_index}:local_{local_episode}:'
            f'{evaluation_network}'
        )
        snapshot = self._ensure_snapshot(stage_index, local_episode, global_episode)

        def action():
            before = self.agent.training_state_digests()
            rng = self.agent.full_state_dict()['rng_state']
            try:
                result = self.evaluator.evaluate(
                    snapshot, evaluation_network, {
                        'simulator_config': simulator_config_path(evaluation_network),
                        'interface': self.interface,
                        'steps': int(self.trainer_config['test_steps']),
                        'action_interval': int(self.trainer_config['action_interval']),
                        'sumo_seed_mode': 'fixed_default',
                    }, {
                        'stage_index': stage_index,
                        'training_network': training_network,
                        'evaluation_network': evaluation_network,
                        'local_episode': local_episode,
                        'global_episode': global_episode,
                    },
                )
            finally:
                _restore_rng(rng)
            after = self.agent.training_state_digests()
            if before != after:
                raise RuntimeError('Independent evaluation changed training state')
            return result['alias_path']

        artifact, reused = self.journal.perform_once(
            operation_key, action, artifact_validator=os.path.isfile,
        )
        return {'alias_path': artifact, 'reused_logical': reused}

    def _save_stage_checkpoint(self, stage_index, local_episode):
        path = os.path.join(
            self.attempt_dir, 'checkpoints', 'stages',
            f'stage_{stage_index:02d}_episode_{local_episode:04d}.pt',
        )
        operation_key = f'stage_{stage_index}:checkpoint'

        def action():
            save_stage_checkpoint(
                self.agent, path, {
                    'stage_index': stage_index,
                    'local_episode': local_episode,
                    'global_episode': self._stage_global_base(stage_index) + local_episode,
                }, extra_state=self._checkpoint_extra_state(),
            )
            self._write_resume_pointer(path, 'stage_boundary')
            return path

        return self.journal.perform_once(
            operation_key, action, artifact_validator=os.path.isfile,
        )[0]

    def _run_matrix(self, stage_index, local_episode):
        if self.journal.phase == SequentialRunPhase.STAGE_CHECKPOINTED:
            self.journal.transition(SequentialRunPhase.MATRIX_EVALUATION_IN_PROGRESS)
        training_network = self.networks[stage_index - 1]
        global_episode = self._stage_global_base(stage_index) + local_episode
        # CS-HR conditions use the causal lower triangle: future scenes are
        # neither trained on nor evaluated before their stage is reached.
        evaluation_networks = (
            self.networks[:stage_index]
            if self.child.get('condition') in {'M0', 'M1', 'M2', 'M3'}
            else self.networks
        )
        for matrix_position, evaluation_network in enumerate(evaluation_networks, start=1):
            self._evaluate_cell(
                stage_index, training_network, evaluation_network,
                local_episode, global_episode,
            )
            if stage_index == 2 and matrix_position == 2:
                self._await_fault_signal('stage2_matrix_half_complete', {
                    'stage_index': stage_index,
                    'local_episode': local_episode,
                    'matrix_cells_complete': matrix_position,
                    'matrix_cell_count': len(evaluation_networks),
                })
        if self.journal.phase == SequentialRunPhase.MATRIX_EVALUATION_IN_PROGRESS:
            self.journal.transition(SequentialRunPhase.MATRIX_EVALUATION_COMPLETE)

    def _close_old_world(self, stage_index):
        path = os.path.join(
            self.attempt_dir, 'world', f'stage_{stage_index:02d}_close.json'
        )

        def action():
            atomic_json(path, self.world.close())
            return path

        self.journal.perform_once(
            f'stage_{stage_index}:world_close', action, os.path.isfile,
        )

    def _apply_policy(self, stage_index, network):
        operation_key = f'stage_{stage_index}:replay_policy'
        if self.journal.operation_completed(operation_key):
            return
        if self.resume_payload and self.resume_payload.get('identity', {}).get(
            'operation_key'
        ) == operation_key:
            self.journal.record_operation(operation_key, self.resume_path)
            return
        self.agent.begin_stage(stage_index, network, self.policy)
        path = os.path.join(
            self.attempt_dir, 'checkpoints', 'committed',
            f'stage_{stage_index:02d}_policy_applied.pt',
        )
        payload = build_full_checkpoint(
            self.agent, 'operation_committed', {
                'operation_key': operation_key,
                'stage_index': stage_index, 'local_episode': 0,
            }, extra_state=self._checkpoint_extra_state(),
        )
        atomic_torch_save(payload, path)
        self._write_resume_pointer(path, 'replay_policy_applied')
        self.journal.record_operation(operation_key, path)

    def _reconcile_committed_episode(self):
        if not self.resume_payload:
            return
        identity = self.resume_payload.get('identity', {})
        completed = identity.get('completed_local_episode')
        if completed is None:
            return
        stage_index = int(identity['stage_index'])
        completed = int(completed)
        if (
            stage_index != int(self.journal.state['stage_index'])
            or completed <= int(self.journal.state['local_episode'])
        ):
            return
        source_attempt = os.path.dirname(os.path.dirname(os.path.dirname(
            self.resume_path
        )))
        name = f'stage_{stage_index:02d}_episode_{completed:04d}'
        marker = os.path.join(
            source_attempt, 'trajectory', 'committed', f'{name}.json'
        )
        if not os.path.isfile(marker):
            return
        trajectory_key = f'stage_{stage_index}:episode_{completed}:trajectory'
        self.journal.record_operation(trajectory_key, marker)
        diagnostic_path, _ = self.diagnostics.episode_record(
            self.agent, stage_index, completed,
            self._stage_global_base(stage_index) + completed,
        )
        self.journal.record_operation(
            f'stage_{stage_index}:episode_{completed}:diagnostic', diagnostic_path
        )
        self.journal.update_episode(
            stage_index, completed,
            self._stage_global_base(stage_index) + completed,
            {'reconciled_from_checkpoint': self.resume_path},
        )

    def _train_remaining_stage(self, stage_index):
        budget = int(self.stage_episodes[stage_index - 1])
        start = int(self.journal.state['local_episode']) + 1
        training_network = self.networks[stage_index - 1]
        committed_local = int(self.journal.state['local_episode'])
        if committed_local > 0:
            self._evaluate_cell(
                stage_index, training_network, training_network,
                committed_local,
                self._stage_global_base(stage_index) + committed_local,
            )
        for local_episode in range(start, budget + 1):
            global_episode = self._stage_global_base(stage_index) + local_episode
            identity = {
                'stage_index': stage_index,
                'local_episode': local_episode,
                'global_episode': global_episode,
            }
            transaction = EpisodeTransaction(
                self.attempt_dir, self.agent, identity,
            ).begin(extra_state=self._checkpoint_extra_state())
            self._write_resume_pointer(
                transaction.recovery.latest_path, 'episode_start'
            )
            self.trainer.trajectory_sink = transaction.append
            try:
                self.trainer.train_episode(local_episode, global_episode)
                transaction.prepare_shard()
                committed_checkpoint = os.path.join(
                    self.attempt_dir, 'checkpoints', 'committed',
                    f'stage_{stage_index:02d}_episode_{local_episode:04d}.pt',
                )
                payload = build_full_checkpoint(
                    self.agent, 'episode_committed', {
                        **identity, 'completed_local_episode': local_episode,
                    }, extra_state=self._checkpoint_extra_state(),
                )
                atomic_torch_save(payload, committed_checkpoint)
                marker = transaction.commit()
                self._write_resume_pointer(
                    committed_checkpoint, 'episode_committed'
                )
            except BaseException:
                transaction.rollback()
                raise
            trajectory_key = (
                f'stage_{stage_index}:episode_{local_episode}:trajectory'
            )
            self.journal.record_operation(
                trajectory_key, transaction.marker_path,
                {'canonical_transition_digests': marker[
                    'canonical_transition_digests'
                ]},
            )
            diagnostic_path, _ = self.diagnostics.episode_record(
                self.agent, stage_index, local_episode, global_episode,
            )
            self.journal.record_operation(
                f'stage_{stage_index}:episode_{local_episode}:diagnostic',
                diagnostic_path,
            )
            self.journal.update_episode(
                stage_index, local_episode, global_episode,
                {'trajectory_marker': transaction.marker_path},
            )
            self._evaluate_cell(
                stage_index, training_network, training_network,
                local_episode, global_episode,
            )

    def _advance_state_machine(self):
        if self.resume_payload:
            self._reconcile_committed_episode()
        if (
            not self.resume_path
            and self.child.get('condition') not in {'M0', 'M1', 'M2', 'M3'}
            and self.journal.phase == SequentialRunPhase.TRAINING
            and int(self.journal.state['stage_index']) == 1
        ):
            parent_episode = int(self.child['parent_checkpoint_episode'])
            self.journal.set_context(
                1, parent_episode, parent_episode, {'source': 'Plan1 parent'}
            )
            self.journal.transition(SequentialRunPhase.STAGE_TRAINING_COMPLETE)
        while self.journal.phase != SequentialRunPhase.COMPLETED:
            stage_index = int(self.journal.state['stage_index'])
            local_episode = int(self.journal.state['local_episode'])
            phase = self.journal.phase
            if phase == SequentialRunPhase.TRAINING:
                self._train_remaining_stage(stage_index)
                self.journal.transition(
                    SequentialRunPhase.STAGE_TRAINING_COMPLETE
                )
            elif phase == SequentialRunPhase.STAGE_TRAINING_COMPLETE:
                self._save_stage_checkpoint(stage_index, local_episode)
                self.journal.transition(SequentialRunPhase.STAGE_CHECKPOINTED)
            elif phase in (
                SequentialRunPhase.STAGE_CHECKPOINTED,
                SequentialRunPhase.MATRIX_EVALUATION_IN_PROGRESS,
            ):
                self._run_matrix(stage_index, local_episode)
            elif phase == SequentialRunPhase.MATRIX_EVALUATION_COMPLETE:
                if stage_index == len(self.networks):
                    self.journal.transition(SequentialRunPhase.COMPLETED)
                    continue
                self._close_old_world(stage_index)
                self.journal.transition(SequentialRunPhase.OLD_WORLD_CLOSED)
            elif phase == SequentialRunPhase.OLD_WORLD_CLOSED:
                next_stage = stage_index + 1
                next_network = self.networks[next_stage - 1]
                new_world = self._create_world(next_network)
                self.trainer.rebind_environment(new_world)
                self.world = new_world
                self.journal.set_context(
                    next_stage, 0, self._stage_global_base(next_stage),
                    {'network': next_network},
                )
                self.journal.transition(SequentialRunPhase.NEW_WORLD_CREATED)
            elif phase == SequentialRunPhase.NEW_WORLD_CREATED:
                network = self.networks[stage_index - 1]
                self._apply_policy(stage_index, network)
                self.journal.transition(SequentialRunPhase.REPLAY_POLICY_APPLIED)
                self._await_fault_signal(
                    'after_REPLAY_POLICY_APPLIED_before_local0', {
                        'stage_index': stage_index, 'local_episode': 0,
                    },
                )
            elif phase == SequentialRunPhase.REPLAY_POLICY_APPLIED:
                network = self.networks[stage_index - 1]
                global_episode = self._stage_global_base(stage_index)
                self._evaluate_cell(
                    stage_index, network, network, 0, global_episode,
                )
                self.journal.transition(SequentialRunPhase.ZERO_SHOT_EVALUATED)
            elif phase == SequentialRunPhase.ZERO_SHOT_EVALUATED:
                self.journal.transition(SequentialRunPhase.NEXT_STAGE_READY)
            elif phase == SequentialRunPhase.NEXT_STAGE_READY:
                self.journal.transition(SequentialRunPhase.TRAINING)
            else:
                raise AssertionError(phase)

    def run(self):
        previous_handlers = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }

        def interrupt(signum, frame):
            raise SequentialInterrupted(signum)

        for signum in previous_handlers:
            signal.signal(signum, interrupt)
        try:
            self._initialise_agent()
            self._advance_state_machine()
            final = self.agent.training_state_digests()
            atomic_json(os.path.join(self.attempt_dir, 'completed.json'), {
                'schema_version': 1,
                'logical_run_id': self.logical_run_id,
                'attempt_id': self.attempt_id,
                'phase': self.journal.phase.value,
                'final_training_state': final,
            })
            return {'exit_code': 0, 'status': 'completed', 'final': final}
        except SequentialInterrupted as error:
            atomic_json(os.path.join(self.attempt_dir, 'interrupted.json'), {
                'signal': error.signum,
                'phase': self.journal.phase.value,
                'state': self.journal.state,
            })
            return {
                'exit_code': 128 + int(error.signum),
                'status': 'interrupted', 'signal': error.signum,
            }
        finally:
            if self.world is not None:
                report = self.world.close()
                atomic_json(
                    os.path.join(self.attempt_dir, 'final_world_close.json'), report
                )
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def run_child(manifest_path, logical_run_id, attempt_dir,
              resume_path=None, resume_state_path=None):
    return SequentialChildRunner(
        manifest_path, logical_run_id, attempt_dir,
        resume_path=resume_path, resume_state_path=resume_state_path,
    ).run()
