import os
import random
import tempfile
import unittest

import numpy as np
import torch

from sequential.checkpoint import (
    RollingRecoveryManager, load_full_checkpoint,
)
from sequential.core import canonical_digest, capture_rng_state
from sequential.io import atomic_json
from sequential.state import SequentialJournal, SequentialRunPhase
from sequential.transaction import EpisodeTransaction


class StatefulAgent:
    def __init__(self):
        self.value = 0

    def full_state_dict(self):
        return {
            'schema_version': 1,
            'value': self.value,
            'rng_state': capture_rng_state(),
        }

    def load_full_state_dict(self, state):
        self.value = state['value']
        rng = state['rng_state']
        random.setstate(rng['python_random_state'])
        np.random.set_state(rng['numpy_random_state'])
        torch.set_rng_state(rng['torch_cpu_rng_state'])
        return self


def make_transition(agent, decision):
    random_value = random.random()
    numpy_value = float(np.random.random())
    torch_value = float(torch.rand(1).item())
    agent.value += 1
    state = np.array([[random_value, numpy_value]], dtype=np.float32)
    next_state = np.array([[numpy_value, torch_value]], dtype=np.float32)
    return {
        'stage_index': 2,
        'local_episode': 1,
        'global_episode': 401,
        'decision_index': decision,
        'global_decision_step': 144000 + decision,
        'state': state,
        'phase': np.array([decision % 8], dtype=np.int8),
        'action': np.array([decision % 8], dtype=np.int64),
        'reward': np.array(random_value + numpy_value),
        'next_state': next_state,
        'next_phase': np.array([(decision + 1) % 8], dtype=np.int8),
        'terminated': False,
        'truncated': decision == 3,
        'transition_id': f't{decision}',
    }


class SequentialRecoveryTests(unittest.TestCase):
    def setUp(self):
        random.seed(31)
        np.random.seed(31)
        torch.manual_seed(31)

    def test_journal_all_boundaries_are_persistent_and_operations_are_once_only(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = SequentialJournal(directory, 'logical', 'attempt_1')
            calls = {'count': 0}

            def action():
                calls['count'] += 1
                artifact = os.path.join(directory, 'artifact.json')
                atomic_json(artifact, {'valid': True})
                return artifact

            artifact, reused = journal.perform_once(
                'stage_1:trajectory', action, os.path.isfile,
            )
            self.assertFalse(reused)
            reopened = SequentialJournal(directory, 'logical', 'attempt_1')
            same, reused = reopened.perform_once(
                'stage_1:trajectory', action, os.path.isfile,
            )
            self.assertTrue(reused)
            self.assertEqual(artifact, same)
            self.assertEqual(calls['count'], 1)

            transitions = [
                SequentialRunPhase.STAGE_TRAINING_COMPLETE,
                SequentialRunPhase.STAGE_CHECKPOINTED,
                SequentialRunPhase.MATRIX_EVALUATION_IN_PROGRESS,
                SequentialRunPhase.MATRIX_EVALUATION_IN_PROGRESS,
                SequentialRunPhase.MATRIX_EVALUATION_COMPLETE,
                SequentialRunPhase.OLD_WORLD_CLOSED,
                SequentialRunPhase.NEW_WORLD_CREATED,
                SequentialRunPhase.REPLAY_POLICY_APPLIED,
                SequentialRunPhase.ZERO_SHOT_EVALUATED,
                SequentialRunPhase.NEXT_STAGE_READY,
                SequentialRunPhase.TRAINING,
            ]
            for phase in transitions:
                reopened.transition(phase)
                reopened = SequentialJournal(directory, 'logical', 'attempt_1')
                self.assertEqual(reopened.phase, phase)
            with open(reopened.events_path, encoding='utf-8') as handle:
                events = [line for line in handle if line.strip()]
            self.assertEqual(len(events), 13)

    def test_recovery_recreates_world_without_reapplying_committed_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = SequentialJournal(directory, 'logical', 'attempt_1', initial_stage=2)
            journal.transition(SequentialRunPhase.STAGE_TRAINING_COMPLETE)
            journal.transition(SequentialRunPhase.STAGE_CHECKPOINTED)
            journal.transition(SequentialRunPhase.MATRIX_EVALUATION_IN_PROGRESS)
            journal.transition(SequentialRunPhase.MATRIX_EVALUATION_COMPLETE)
            journal.transition(SequentialRunPhase.OLD_WORLD_CLOSED)
            journal.transition(SequentialRunPhase.NEW_WORLD_CREATED)
            before_policy = journal.recovery_requirements()
            self.assertTrue(before_policy['recreate_world_connection'])
            self.assertTrue(before_policy['reapply_replay_policy'])
            journal.record_operation('stage_2:replay_policy', 'clear-applied')
            journal.transition(SequentialRunPhase.REPLAY_POLICY_APPLIED)
            after_policy = journal.recovery_requirements()
            self.assertTrue(after_policy['recreate_world_connection'])
            self.assertFalse(after_policy['reapply_replay_policy'])

    def test_journal_reconciles_complete_event_when_current_state_is_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = SequentialJournal(directory, 'logical', 'attempt_1')
            stale = dict(journal.state)
            stale['completed_operations'] = {}
            artifact = os.path.join(directory, 'cell.json')
            atomic_json(artifact, {'valid': True})
            journal.record_operation('evaluation:cell', artifact)
            atomic_json(journal.state_path, stale)
            recovered = SequentialJournal(directory, 'logical', 'attempt_1')
            self.assertTrue(recovered.operation_completed('evaluation:cell'))
            self.assertEqual(
                recovered.state['completed_operations']['evaluation:cell']['artifact'],
                artifact,
            )

    def test_rolling_recovery_falls_back_to_previous_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = StatefulAgent()
            manager = RollingRecoveryManager(directory)
            manager.save_episode_start(agent, {'episode': 1})
            agent.value = 1
            manager.save_episode_start(agent, {'episode': 2})
            with open(manager.latest_path, 'wb') as handle:
                handle.write(b'corrupt')
            path, payload = manager.latest_valid()
            self.assertEqual(path, manager.previous_path)
            self.assertEqual(payload['identity']['episode'], 1)

    def test_half_episode_rollback_rerun_matches_no_fault_canonical_state(self):
        identity = {'stage_index': 2, 'local_episode': 1, 'global_episode': 401}
        with tempfile.TemporaryDirectory() as directory:
            baseline_agent = StatefulAgent()
            baseline = EpisodeTransaction(
                os.path.join(directory, 'baseline'), baseline_agent, identity,
            ).begin()
            for decision in range(1, 4):
                baseline.append(make_transition(baseline_agent, decision))
            baseline_marker = baseline.commit()
            baseline_state = canonical_digest(baseline_agent.full_state_dict())

            random.seed(31)
            np.random.seed(31)
            torch.manual_seed(31)
            recovered_agent = StatefulAgent()
            recovered = EpisodeTransaction(
                os.path.join(directory, 'recovered'), recovered_agent, identity,
            ).begin()
            recovered.append(make_transition(recovered_agent, 1))
            recovered.rollback()
            recovered.begin()
            for decision in range(1, 4):
                recovered.append(make_transition(recovered_agent, decision))
            recovered_marker = recovered.commit()
            recovered_state = canonical_digest(recovered_agent.full_state_dict())

            self.assertEqual(
                baseline_marker['canonical_transition_digests'],
                recovered_marker['canonical_transition_digests'],
            )
            self.assertEqual(baseline_state, recovered_state)
            self.assertEqual(recovered_agent.value, 3)


if __name__ == '__main__':
    unittest.main()
