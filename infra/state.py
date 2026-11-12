import enum
import json
import os
import tempfile
import time

from .io import atomic_json, read_json


class SequentialRunPhase(str, enum.Enum):
    TRAINING = 'TRAINING'
    STAGE_TRAINING_COMPLETE = 'STAGE_TRAINING_COMPLETE'
    STAGE_CHECKPOINTED = 'STAGE_CHECKPOINTED'
    MATRIX_EVALUATION_IN_PROGRESS = 'MATRIX_EVALUATION_IN_PROGRESS'
    MATRIX_EVALUATION_COMPLETE = 'MATRIX_EVALUATION_COMPLETE'
    OLD_WORLD_CLOSED = 'OLD_WORLD_CLOSED'
    NEW_WORLD_CREATED = 'NEW_WORLD_CREATED'
    REPLAY_POLICY_APPLIED = 'REPLAY_POLICY_APPLIED'
    ZERO_SHOT_EVALUATED = 'ZERO_SHOT_EVALUATED'
    NEXT_STAGE_READY = 'NEXT_STAGE_READY'
    COMPLETED = 'COMPLETED'


ALLOWED_TRANSITIONS = {
    SequentialRunPhase.TRAINING: {SequentialRunPhase.STAGE_TRAINING_COMPLETE},
    SequentialRunPhase.STAGE_TRAINING_COMPLETE: {SequentialRunPhase.STAGE_CHECKPOINTED},
    SequentialRunPhase.STAGE_CHECKPOINTED: {
        SequentialRunPhase.MATRIX_EVALUATION_IN_PROGRESS,
    },
    SequentialRunPhase.MATRIX_EVALUATION_IN_PROGRESS: {
        SequentialRunPhase.MATRIX_EVALUATION_IN_PROGRESS,
        SequentialRunPhase.MATRIX_EVALUATION_COMPLETE,
    },
    SequentialRunPhase.MATRIX_EVALUATION_COMPLETE: {
        SequentialRunPhase.OLD_WORLD_CLOSED, SequentialRunPhase.COMPLETED,
    },
    SequentialRunPhase.OLD_WORLD_CLOSED: {SequentialRunPhase.NEW_WORLD_CREATED},
    SequentialRunPhase.NEW_WORLD_CREATED: {SequentialRunPhase.REPLAY_POLICY_APPLIED},
    SequentialRunPhase.REPLAY_POLICY_APPLIED: {SequentialRunPhase.ZERO_SHOT_EVALUATED},
    SequentialRunPhase.ZERO_SHOT_EVALUATED: {SequentialRunPhase.NEXT_STAGE_READY},
    SequentialRunPhase.NEXT_STAGE_READY: {SequentialRunPhase.TRAINING},
    SequentialRunPhase.COMPLETED: set(),
}


class SequentialJournal:
    def __init__(self, run_dir, logical_run_id, attempt_id, initial_stage=1,
                 initial_global_episode=400, resume_state_path=None):
        self.run_dir = os.path.abspath(run_dir)
        self.state_path = os.path.join(self.run_dir, 'current_state.json')
        self.events_path = os.path.join(self.run_dir, 'events.jsonl')
        os.makedirs(self.run_dir, exist_ok=True)
        if os.path.exists(self.state_path):
            self.state = read_json(self.state_path)
            if self.state['logical_run_id'] != logical_run_id:
                raise ValueError('Journal logical run ID mismatch')
            if self.state['attempt_id'] != attempt_id:
                raise ValueError('Journal attempt ID mismatch')
            self._reconcile_events()
        elif resume_state_path is not None:
            source = read_json(resume_state_path)
            if source['logical_run_id'] != logical_run_id:
                raise ValueError('Resume journal logical run ID mismatch')
            self.state = dict(source)
            self.state['attempt_id'] = attempt_id
            self.state['event_sequence'] = 0
            self.state['completed_operations'] = dict(
                source.get('completed_operations', {})
            )
            atomic_json(self.state_path, self.state)
            self._append_event('ATTEMPT_RESUMED', {
                'source_state_path': os.path.abspath(resume_state_path),
                'source_attempt_id': source['attempt_id'],
            })
        else:
            self.state = {
                'schema_version': 1,
                'logical_run_id': logical_run_id,
                'attempt_id': attempt_id,
                'phase': SequentialRunPhase.TRAINING.value,
                'stage_index': int(initial_stage),
                'local_episode': 0,
                'global_episode': (
                    int(initial_global_episode) if initial_stage == 1 else 0
                ),
                'completed_operations': {},
                'event_sequence': 0,
            }
            atomic_json(self.state_path, self.state)
            self._append_event('JOURNAL_CREATED', {})

    def _read_complete_events(self):
        if not os.path.isfile(self.events_path):
            return []
        events = []
        with open(self.events_path, encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    # An interrupted final append is not a committed event.
                    break
        return events

    def _reconcile_events(self):
        events = self._read_complete_events()
        pending = [
            event for event in events
            if int(event['sequence']) > int(self.state['event_sequence'])
        ]
        for event in pending:
            event_type = event['event_type']
            payload = event.get('payload', {})
            if event_type == 'PHASE_TRANSITION':
                self.state['phase'] = payload['to']
            elif event_type == 'OPERATION_COMMITTED':
                key = payload['operation_key']
                self.state['completed_operations'][key] = {
                    'artifact': payload['artifact'],
                    'payload': payload.get('payload', {}),
                    'completed_at_unix': payload['completed_at_unix'],
                }
            elif event_type in ('EPISODE_COMMITTED', 'CONTEXT_UPDATED'):
                self.state.update({
                    'stage_index': int(event['stage_index']),
                    'local_episode': int(event['local_episode']),
                    'global_episode': int(event['global_episode']),
                })
            self.state['event_sequence'] = int(event['sequence'])
        if pending:
            atomic_json(self.state_path, self.state)

    @property
    def phase(self):
        return SequentialRunPhase(self.state['phase'])

    def _append_event(self, event_type, payload):
        self.state['event_sequence'] += 1
        event = {
            'schema_version': 1,
            'sequence': self.state['event_sequence'],
            'event_type': event_type,
            'phase': self.state['phase'],
            'stage_index': self.state['stage_index'],
            'local_episode': self.state['local_episode'],
            'global_episode': self.state['global_episode'],
            'attempt_id': self.state['attempt_id'],
            'time_unix': time.time(),
            'payload': payload,
        }
        with open(self.events_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        atomic_json(self.state_path, self.state)
        return event

    def transition(self, next_phase, payload=None):
        next_phase = SequentialRunPhase(next_phase)
        if next_phase not in ALLOWED_TRANSITIONS[self.phase]:
            raise ValueError(f'Invalid phase transition: {self.phase} -> {next_phase}')
        previous = self.phase
        self.state['phase'] = next_phase.value
        return self._append_event('PHASE_TRANSITION', {
            'from': previous.value, 'to': next_phase.value,
            **(payload or {}),
        })

    def update_episode(self, stage_index, local_episode, global_episode, payload=None):
        self.state.update({
            'stage_index': int(stage_index),
            'local_episode': int(local_episode),
            'global_episode': int(global_episode),
        })
        return self._append_event('EPISODE_COMMITTED', payload or {})

    def set_context(self, stage_index, local_episode, global_episode, payload=None):
        self.state.update({
            'stage_index': int(stage_index),
            'local_episode': int(local_episode),
            'global_episode': int(global_episode),
        })
        return self._append_event('CONTEXT_UPDATED', payload or {})

    def record_event(self, event_type, payload=None):
        """Persist an audit-only event without changing run semantics."""
        return self._append_event(str(event_type), payload or {})

    def operation_completed(self, operation_key):
        return operation_key in self.state['completed_operations']

    def record_operation(self, operation_key, artifact, payload=None):
        if self.operation_completed(operation_key):
            existing = self.state['completed_operations'][operation_key]
            if existing['artifact'] != artifact:
                raise ValueError(f'Operation artifact changed: {operation_key}')
            return existing
        record = {
            'artifact': artifact,
            'payload': payload or {},
            'completed_at_unix': time.time(),
        }
        self.state['completed_operations'][operation_key] = record
        self._append_event('OPERATION_COMMITTED', {
            'operation_key': operation_key, **record,
        })
        return record

    def perform_once(self, operation_key, action, artifact_validator=None,
                     payload=None):
        if self.operation_completed(operation_key):
            artifact = self.state['completed_operations'][operation_key]['artifact']
            if artifact_validator is not None and not artifact_validator(artifact):
                raise ValueError(f'Committed artifact is invalid: {operation_key}')
            return artifact, True
        artifact = action()
        if artifact_validator is not None and not artifact_validator(artifact):
            raise ValueError(f'Operation did not produce a valid artifact: {operation_key}')
        self.record_operation(operation_key, artifact, payload=payload)
        return artifact, False

    def recovery_requirements(self):
        phase = self.phase
        recreate_world = phase in {
            SequentialRunPhase.NEW_WORLD_CREATED,
            SequentialRunPhase.REPLAY_POLICY_APPLIED,
            SequentialRunPhase.ZERO_SHOT_EVALUATED,
            SequentialRunPhase.NEXT_STAGE_READY,
            SequentialRunPhase.TRAINING,
        }
        replay_already_applied = self.operation_completed(
            f'stage_{self.state["stage_index"]}:replay_policy'
        )
        return {
            'recreate_world_connection': recreate_world,
            'reapply_replay_policy': recreate_world and not replay_already_applied,
            'resume_phase': phase.value,
        }
