"""Preference reward model for the HITL B channel (spec §9.1).

Network: x -> 128 -> 128 -> 1 (ReLU), trained with BCE on pairwise
preferences where the segment score S = mean_t f_phi(x_t) and
P(left better) = sigmoid((S_left - S_right) / temperature).

The frozen artifact is a plain state_dict + normalization stats of f over
the fit partition, so the policy-side reward shaping uses
r = r0 + lambda * clip((f_hat - mu) / max(sigma, 1e-6), -3, 3).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

TEMPERATURE = 0.25


class PreferenceRewardNet(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def segment_score(self, x):
        """x: (B, W, D) -> per-segment mean score (B,)."""
        return self.net(x).squeeze(-1).mean(dim=1)


def pair_loss(model, x_left, x_right, y):
    """BCE on P_left = sigmoid((S_l - S_r)/0.25); supports y=0.5 ties."""
    logits = (model.segment_score(x_left)
              - model.segment_score(x_right)) / TEMPERATURE
    return nn.functional.binary_cross_entropy_with_logits(
        logits, y.float())


def predict_segment_scores(model, features):
    """features: (N, W, D) numpy -> frozen mean scores (N,) numpy."""
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        return model.segment_score(x).cpu().numpy()


def evaluate_pairs(model, features, pairs):
    """Holdout metrics: BCE, strict non-tie accuracy, tie |P-0.5| error."""
    if not pairs:
        return {'bce': None, 'strict_accuracy': None,
                'tie_prob_error': None, 'count': 0}
    scores = predict_segment_scores(model, features)
    losses, correct, nontie, tie_err, ties = [], 0, 0, [], 0
    for p in pairs:
        logits = (scores[p['left_segment']]
                  - scores[p['right_segment']]) / TEMPERATURE
        prob = 1.0 / (1.0 + np.exp(-logits))
        y = float(p['clean_y'])
        losses.append(-(y * np.log(prob + 1e-9)
                        + (1 - y) * np.log(1 - prob + 1e-9)))
        if y == 0.5:
            ties += 1
            tie_err.append(abs(prob - 0.5))
        else:
            nontie += 1
            correct += int((prob >= 0.5) == (y == 1.0))
    return {
        'bce': float(np.mean(losses)),
        'strict_accuracy': (correct / nontie) if nontie else None,
        'tie_prob_error': float(np.mean(tie_err)) if tie_err else None,
        'count': len(pairs),
        'nontie_count': nontie,
        'tie_count': ties,
    }
