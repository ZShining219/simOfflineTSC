"""Common deterministic evaluator contracts and fixed-time reference gate."""
from copy import deepcopy
from .checkpoint import training_state_digest
from sequential.core import canonical_digest
from sequential.io import atomic_json

from .config import HISTORICAL_SCHEDULE, SCENES


def frozen_evaluate(agent, run_episode, *, snapshot=None):
    """Run an evaluator callback on a copy and return result plus isolation proof."""
    before=training_state_digest(agent)
    candidate=deepcopy(agent)
    result=run_episode(candidate, snapshot=snapshot, deterministic=True)
    after=training_state_digest(agent)
    if before != after: raise AssertionError("Frozen evaluation changed training state")
    return {"result":result,"training_state_digest_before":before,"training_state_digest_after":after,"isolated":True}


def build_fixedtime_reference(scenes, evaluate, output_path=None):
    """Execute each FixedTime scene twice under one callback protocol.

    ``evaluate(scene)`` must return ``{"travel_time": number,
    "trajectory_digest": str, ...}``; equality is intentionally exact.
    """
    references={}
    for scene in scenes:
        first=evaluate(scene); second=evaluate(scene)
        if first.get("travel_time") != second.get("travel_time") or first.get("trajectory_digest") != second.get("trajectory_digest"):
            raise ValueError(f"FixedTime repeatability gate failed for {scene}")
        references[scene]={"travel_time":first["travel_time"],"trajectory_digest":first["trajectory_digest"],"repeated":True}
    payload={"schema_version":1,"protocol":"Plan5 common FixedTime evaluator","scenes":references,"digest":canonical_digest(references)}
    if output_path: atomic_json(output_path,payload)
    return payload


def formal_evaluation_schedule(run_type, *, scene=None, source_scene=None,
                               target_scene=None):
    """Return the frozen episode/scene evaluation cells for one logical run."""
    cells = []
    if run_type == 'ANCHOR':
        if scene not in SCENES:
            raise ValueError('Anchor evaluation requires one frozen scene')
        for episode in range(101):
            cells.append({
                'episode': episode, 'scene': scene, 'role': 'current',
            })
        return cells
    if run_type != 'TRANSITION':
        raise ValueError('Unknown Plan5 run type')
    if source_scene not in SCENES or target_scene not in SCENES \
            or source_scene == target_scene:
        raise ValueError('Transition evaluation scenes are invalid')
    for episode in range(101):
        cells.append({
            'episode': episode, 'scene': target_scene, 'role': 'current',
        })
    for episode in HISTORICAL_SCHEDULE:
        cells.append({
            'episode': episode, 'scene': source_scene, 'role': 'historical',
        })
    for other in SCENES:
        if other not in {source_scene, target_scene}:
            cells.append({
                'episode': 100, 'scene': other,
                'role': 'secondary_generalization',
            })
    return sorted(cells, key=lambda item: (
        item['episode'], item['scene'], item['role'],
    ))


def full_checkpoint_schedule(run_type):
    if run_type == 'ANCHOR':
        return (0, 100)
    if run_type == 'TRANSITION':
        return HISTORICAL_SCHEDULE
    raise ValueError('Unknown Plan5 run type')
