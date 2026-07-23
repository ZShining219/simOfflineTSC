import copy
import unittest

import numpy as np
import torch
from torch import nn

from agent.offline_dqn import (
    BatchDQNAgent, CQLDQNAgent, build_full_q_mse_target,
    compute_offline_loss_tensors,
)
from dataset.offline_trajectory_dataset import OfflineBatch
from trainer.offline_tsc_trainer import should_update_target


class TableQNet(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, observations, train=True):
        return self.values[: len(observations)]


def make_agent(cls, alpha, values, target_values, learning_rate=0.01):
    agent = cls.__new__(cls)
    agent.model = TableQNet(values)
    agent.target_model = TableQNet(target_values)
    agent.gamma = 0.5
    agent.grad_clip = 100000.0
    agent.cql_alpha = float(alpha)
    agent.criterion = nn.MSELoss(reduction='mean')
    agent.optimizer = torch.optim.RMSprop(
        agent.model.parameters(), lr=learning_rate, alpha=.9, centered=False, eps=1e-7
    )
    return agent


class FullQVectorMSEGradientTest(unittest.TestCase):
    def _check_batch(self, batch_size, action_dim):
        q_values = torch.arange(
            batch_size * action_dim, dtype=torch.float32, requires_grad=True
        ).reshape(batch_size, action_dim)
        q_values.retain_grad()
        actions = (torch.arange(batch_size) % action_dim).to(torch.int32)
        td_target = torch.arange(batch_size, dtype=torch.float32) + 100.0
        target, normalized = build_full_q_mse_target(q_values, actions, td_target)
        self.assertEqual(torch.long, normalized.dtype)
        self.assertEqual((batch_size,), tuple(normalized.shape))
        self.assertFalse(target.requires_grad)
        self.assertIsNone(target.grad_fn)
        for row in range(batch_size):
            for action in range(action_dim):
                expected = td_target[row] if action == normalized[row] else q_values.detach()[row, action]
                self.assertEqual(float(expected), float(target[row, action]))
        loss = nn.MSELoss()(q_values, target)
        loss.backward()
        for row in range(batch_size):
            mask = torch.ones(action_dim, dtype=torch.bool)
            mask[normalized[row]] = False
            self.assertTrue(torch.equal(q_values.grad[row, mask], torch.zeros(action_dim - 1)))
            self.assertNotEqual(0.0, float(q_values.grad[row, normalized[row]]))

    def test_batch_size_one_and_64_with_variable_action_dims(self):
        for batch_size, action_dim in ((1, 3), (64, 2), (64, 11)):
            with self.subTest(batch_size=batch_size, action_dim=action_dim):
                self._check_batch(batch_size, action_dim)

    def test_full_loss_exposes_q_and_parameter_gradients_only_at_data_action(self):
        model = TableQNet([[1., 2., 3., 4.]])
        target = TableQNet([[10., 20., 30., 40.]])
        tensors = compute_offline_loss_tensors(
            model, target, np.zeros((1, 1), np.float32),
            np.zeros((1, 1), np.float32), np.asarray([[2]], np.int32),
            np.asarray([5.], np.float32), .5, 0., nn.MSELoss(), retain_q_grad=True,
        )
        tensors.total_loss.backward()
        self.assertEqual(torch.long, tensors.actions.dtype)
        self.assertFalse(tensors.full_target.requires_grad)
        self.assertTrue(torch.equal(tensors.q_values.grad[0, [0, 1, 3]], torch.zeros(3)))
        self.assertNotEqual(0.0, float(tensors.q_values.grad[0, 2]))
        self.assertTrue(torch.equal(model.values.grad, tensors.q_values.grad))
        self.assertIsNone(target.values.grad)

    def test_invalid_action_shape_dtype_and_range_are_rejected(self):
        q = torch.zeros((2, 3), requires_grad=True)
        td = torch.ones(2)
        for actions, error in (([0.0, 1.0], TypeError), ([True, False], TypeError), ([0], ValueError), ([0, 3], ValueError)):
            with self.subTest(actions=actions), self.assertRaises(error):
                build_full_q_mse_target(q, actions, td)


class BatchCQLExactEquivalenceTest(unittest.TestCase):
    def test_alpha_zero_has_identical_loss_gradients_optimizer_and_target_update(self):
        values = [[1., 2., 3.], [4., 5., 6.]]
        targets = [[2., 3., 4.], [5., 6., 7.]]
        batch_agent = make_agent(BatchDQNAgent, 0., values, targets)
        cql_agent = make_agent(CQLDQNAgent, 0., values, targets)
        batch = OfflineBatch(
            observations=np.zeros((2, 1), np.float32),
            actions=np.asarray([0, 2], np.int64), rewards=np.asarray([1., -1.], np.float32),
            next_observations=np.zeros((2, 1), np.float32),
            terminated=np.asarray([False, False]), truncated=np.asarray([False, True]),
            source_network='n',
        )
        first = batch_agent.train_offline_batch(batch)
        first_grad = batch_agent.model.values.grad.detach().clone()
        second = cql_agent.train_offline_batch(batch)
        second_grad = cql_agent.model.values.grad.detach().clone()
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(first_grad, second_grad))
        self.assertTrue(torch.equal(batch_agent.model.values, cql_agent.model.values))
        batch_agent.update_target_network()
        cql_agent.update_target_network()
        self.assertTrue(torch.equal(batch_agent.target_model.values, cql_agent.target_model.values))
        self.assertTrue(torch.equal(batch_agent.target_model.values, batch_agent.model.values))


class CQLNumericalStabilityTest(unittest.TestCase):
    def test_extreme_q_values_have_finite_mean_loss_and_gradients(self):
        for values, action in (([[1000., 999., -1000.]], 1), ([[-1000., -1001., -999.]], 0)):
            with self.subTest(values=values):
                model = TableQNet(values)
                target = TableQNet([[0., 0., 0.]])
                tensors = compute_offline_loss_tensors(
                    model, target, np.zeros((1, 1), np.float32),
                    np.zeros((1, 1), np.float32), np.asarray([action]),
                    np.asarray([0.], np.float32), .95, 1., nn.MSELoss(),
                )
                tensors.total_loss.backward()
                self.assertTrue(torch.isfinite(tensors.cql_loss))
                self.assertTrue(torch.isfinite(tensors.total_loss))
                self.assertTrue(torch.isfinite(model.values.grad).all())
                self.assertIsNone(target.values.grad)

    def test_regularizer_is_per_row_then_batch_mean_and_gather_is_correct(self):
        values = [[3., 1., 0.], [-2., 4., 2.]]
        model = TableQNet(values)
        target = TableQNet([[0., 0., 0.], [0., 0., 0.]])
        tensors = compute_offline_loss_tensors(
            model, target, np.zeros((2, 1), np.float32),
            np.zeros((2, 1), np.float32), np.asarray([0, 2]), np.zeros(2, np.float32),
            0., 2., nn.MSELoss(),
        )
        q = torch.tensor(values)
        expected = ((torch.logsumexp(q[0], 0) - q[0, 0]) + (torch.logsumexp(q[1], 0) - q[1, 2])) / 2
        self.assertEqual(float(expected), float(tensors.cql_loss))
        self.assertEqual(float(tensors.td_loss + 2 * expected), float(tensors.total_loss))


class TargetUpdateBoundaryTest(unittest.TestCase):
    def test_completed_update_boundaries_are_explicit(self):
        expected = {0: False, 1: False, 9: False, 10: True, 11: False, 20: True}
        self.assertEqual(expected, {update: should_update_target(update, 10) for update in expected})

    def test_resume_at_9_or_10_preserves_next_sync_boundary(self):
        self.assertTrue(should_update_target(10, 10))  # resume at 9, next completed update
        self.assertFalse(should_update_target(11, 10))  # resume at 10, next completed update


if __name__ == '__main__':
    unittest.main()
