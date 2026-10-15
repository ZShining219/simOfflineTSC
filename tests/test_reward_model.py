import unittest

import numpy as np
import torch

from sequential.reward_model import (
    PreferenceRewardNet, evaluate_pairs, pair_loss, predict_segment_scores)


def _pairs(n, tie=0.0):
    rng = np.random.RandomState(0)
    pairs = []
    for i in range(n):
        y = 0.5 if rng.rand() < tie else float(rng.randint(0, 2))
        pairs.append({
            'left_segment': int(rng.randint(0, 20)),
            'right_segment': int(rng.randint(0, 20)),
            'training_y': y,
            'clean_y': y,
        })
    return pairs


class RewardModelTest(unittest.TestCase):
    def test_architecture_is_128_128_1(self):
        net = PreferenceRewardNet(56)
        shapes = [tuple(p.shape) for p in net.parameters()]
        self.assertIn((128, 56), shapes)
        self.assertIn((128, 128), shapes)
        self.assertIn((1, 128), shapes)

    def test_forward_shape(self):
        net = PreferenceRewardNet(56)
        out = net(torch.zeros(4, 56))
        self.assertEqual((4,), tuple(out.shape))

    def test_pair_loss_drives_preference(self):
        torch.manual_seed(0)
        net = PreferenceRewardNet(56)
        rng = np.random.RandomState(1)
        feats = rng.randn(40, 6, 56).astype(np.float32)
        # synthetic: label = left feature sum > right
        pairs = []
        for _ in range(200):
            l, r = rng.randint(0, 40, size=2)
            if l == r:
                continue
            y = 1.0 if feats[l].sum() > feats[r].sum() else 0.0
            pairs.append({'left_segment': l, 'right_segment': r,
                          'training_y': y, 'clean_y': y})
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        x = torch.as_tensor(feats)
        first = None
        for _ in range(50):
            xl = x[[p['left_segment'] for p in pairs]]
            xr = x[[p['right_segment'] for p in pairs]]
            y = torch.as_tensor(
                [p['training_y'] for p in pairs], dtype=torch.float32)
            opt.zero_grad()
            loss = pair_loss(net, xl, xr, y)
            loss.backward()
            opt.step()
            if first is None:
                first = float(loss)
        self.assertLess(float(loss), first)

    def test_evaluate_reports_tie_metrics(self):
        torch.manual_seed(0)
        net = PreferenceRewardNet(8)
        feats = np.random.RandomState(2).randn(30, 6, 8).astype(np.float32)
        pairs = _pairs(60, tie=0.3)
        pairs = [p for p in pairs
                 if p['left_segment'] != p['right_segment']]
        m = evaluate_pairs(net, feats, pairs)
        self.assertIn('bce', m)
        self.assertIn('strict_accuracy', m)
        self.assertIn('tie_prob_error', m)
        self.assertEqual(m['count'], len(pairs))
        self.assertGreater(m['tie_count'], 0)

    def test_predict_segment_scores(self):
        net = PreferenceRewardNet(8)
        feats = np.zeros((3, 4, 8), dtype=np.float32)
        s = predict_segment_scores(net, feats)
        self.assertEqual((3,), s.shape)


if __name__ == '__main__':
    unittest.main()
