"""Plan 3/4 sequential DQN support, isolated from existing Online paths."""

from .config import load_sequential_config
from .core import (
    ReplayMetadata, ReplayRecord, SequentialReplay, TrainingPayload,
    TargetUpdateScheduler, canonical_digest,
)
from .hybrid import HybridReplayPool
from .evaluation_matrix import validate_lower_triangle, expected_cells

__all__ = [
    'ReplayMetadata', 'ReplayRecord', 'SequentialReplay', 'TrainingPayload',
    'TargetUpdateScheduler', 'canonical_digest', 'load_sequential_config',
    'HybridReplayPool', 'validate_lower_triangle', 'expected_cells',
]
