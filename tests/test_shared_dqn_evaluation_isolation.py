import copy
import sys
import types
from collections import deque

import numpy as np
import torch

sys.modules.setdefault('libsumo', types.ModuleType('libsumo'))

from arterial.replay import split_local_transitions
from trainer.tsc_trainer import EvaluationIsolationGuard


class Agent:
    def __init__(self):
        self.model = torch.nn.Linear(2, 2)
        self.target_model = torch.nn.Linear(2, 2)
        self.optimizer = torch.optim.RMSprop(self.model.parameters())
        self.epsilon = 0.1
        self.replay_buffer = deque(split_local_transitions(
            ['i1'], np.zeros((1, 2)), np.zeros(1), np.zeros(1),
            np.zeros(1), np.ones((1, 2)), np.zeros(1), False, False,
            'scene', 1, 1), maxlen=8)

    def remember(self, *args):
        raise AssertionError


class Trainer:
    def __init__(self):
        self.agents = [Agent()]
        self.dataset = []
        self.trajectory_writer = None
        self.gradient_updates = 2
        self.global_decision_step = 3
        self.evaluation_isolation_checks = []


def test_shared_dataclass_replay_remains_unchanged_during_evaluation():
    trainer = Trainer()
    before = copy.deepcopy(list(trainer.agents[0].replay_buffer))
    with EvaluationIsolationGuard(trainer, 'TEST'):
        pass
    after = list(trainer.agents[0].replay_buffer)
    assert np.array_equal(before[0].state, after[0].state)
    assert trainer.evaluation_isolation_checks[-1]['replay_unchanged'] is True
