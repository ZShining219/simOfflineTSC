"""Immutable Plan5 fixed-probe manifest and row validation."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from sequential.io import atomic_json, sha256_file
from sequential.core import canonical_digest
from .config import SCENES

PROBE_ID = "plan5_fixed_probe_v1"


def _jsonl_sha256(rows):
    digest=hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row,sort_keys=True,separators=(",",":"))+"\n").encode())
    return digest.hexdigest()


def validate_rows(rows, expected_count=7200):
    rows=list(rows)
    if len(rows) != expected_count: raise ValueError(f"Probe requires {expected_count} rows, got {len(rows)}")
    required={"probe_id","split","scene","network","source_training_seed","decision_index","simulation_time_s","raw16","ctx4","arrival_counts","source_action","source_checkpoint_path","source_checkpoint_sha256","source_trajectory_sha256","resolved_sumo_command_digest"}
    trajectories = {}
    for row in rows:
        if set(row) < required: raise ValueError(f"Probe row missing fields: {sorted(required-set(row))}")
        if row["probe_id"] != PROBE_ID: raise ValueError("Probe ID mismatch")
        if row["split"] not in {"main","heldout"}: raise ValueError("Invalid probe split")
        if row["scene"] not in SCENES: raise ValueError("Unknown probe scene")
        seed = int(row["source_training_seed"])
        if not 0 <= seed <= 4: raise ValueError("Invalid source seed")
        if row['network'] != SCENES[row['scene']]: raise ValueError('Probe scene/network mismatch')
        expected_split = 'main' if seed <= 3 else 'heldout'
        if row['split'] != expected_split: raise ValueError('Probe seed/split mismatch')
        if not 1 <= int(row["decision_index"]) <= 360: raise ValueError("Invalid decision index")
        if len(row["raw16"]) != 16 or len(row["ctx4"]) != 4 or len(row["arrival_counts"]) != 4: raise ValueError("Probe vector dimensions are frozen")
        counts = row['arrival_counts']
        if any(isinstance(x, bool) or int(x) != x or int(x) < 0 for x in counts): raise ValueError("Arrival counts must be non-negative integers")
        if any(not math.isfinite(float(x)) for x in row['raw16'] + row['ctx4']): raise ValueError('Probe vectors must be finite')
        if [float(x) for x in row['ctx4']] != [int(x) / 60.0 for x in counts]: raise ValueError('Probe ctx4/count vectors disagree')
        if not 0 <= int(row['source_action']) < 8: raise ValueError('Probe action is outside the frozen action space')
        expected_time = (int(row['decision_index']) - 1) * 10
        if float(row['simulation_time_s']) != float(expected_time): raise ValueError('Probe decision time is not frozen')
        key = (row['scene'], seed)
        trajectory = trajectories.setdefault(key, {
            'decisions': set(),
            'checkpoint': (row['source_checkpoint_path'], row['source_checkpoint_sha256']),
            'trajectory_sha256': row['source_trajectory_sha256'],
            'command_digest': row['resolved_sumo_command_digest'],
        })
        if int(row['decision_index']) in trajectory['decisions']: raise ValueError('Duplicate probe decision row')
        trajectory['decisions'].add(int(row['decision_index']))
        if trajectory['checkpoint'] != (row['source_checkpoint_path'], row['source_checkpoint_sha256']) or trajectory['trajectory_sha256'] != row['source_trajectory_sha256'] or trajectory['command_digest'] != row['resolved_sumo_command_digest']:
            raise ValueError('Probe trajectory identity changed within a controller')
    expected_main=4*4*360; expected_heldout=4*360
    if sum(r["split"]=="main" for r in rows)!=expected_main or sum(r["split"]=="heldout" for r in rows)!=expected_heldout: raise ValueError("Probe split counts are invalid")
    expected_trajectories = {(scene, seed) for scene in SCENES for seed in range(5)}
    if set(trajectories) != expected_trajectories or any(
            item['decisions'] != set(range(1, 361))
            for item in trajectories.values()):
        raise ValueError('Probe controller/decision matrix is incomplete')
    return {"valid":True,"rows":len(rows),"main_rows":expected_main,"heldout_rows":expected_heldout}


def write_probe(rows, directory, metadata=None):
    validation=validate_rows(rows)
    directory=Path(directory)
    if directory.exists():
        raise FileExistsError(f"Frozen probe directory already exists: {directory}")
    directory.parent.mkdir(parents=True,exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f'.{directory.name}.staging-', dir=directory.parent,
    ))
    try:
        paths={}
        for split in ("main","heldout"):
            subset=[r for r in rows if r["split"]==split]
            filename=f"{split}.jsonl"
            path=staging/filename
            with open(path,"w",encoding="utf-8") as handle:
                for row in subset:
                    handle.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
                handle.flush(); os.fsync(handle.fileno())
            paths[split]=filename
            paths[f"{split}_sha256"]=_jsonl_sha256(subset)
        manifest={"schema_version":1,"probe_id":PROBE_ID,"validation":validation,"paths":paths}
        if metadata:
            overlap = set(manifest) & set(metadata)
            if overlap:
                raise ValueError(f'Probe metadata overwrites frozen fields: {sorted(overlap)}')
            manifest.update(metadata)
        manifest["manifest_digest"]=canonical_digest(manifest)
        atomic_json(staging/"probe_manifest.json",manifest)
        os.replace(staging, directory)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def generate_probe(reference_manifest, output_dir, evaluation_root,
                   interface='traci'):
    """Generate the immutable 4 scenes x 5 controllers x 360-row probe."""
    from .sumo_runtime import (
        Plan1DQNPolicy, causal_gate_report, evaluate_policy,
    )
    reverse_scenes = {network: scene for scene, network in SCENES.items()}
    output_dir = Path(output_dir)
    evaluation_root = Path(evaluation_root)
    if output_dir.exists() or evaluation_root.exists():
        raise FileExistsError(
            'Frozen probe or evaluation directory already exists'
        )
    if output_dir.parent != evaluation_root.parent:
        raise ValueError('Probe and evaluation directories must share a parent')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    generation_staging = Path(tempfile.mkdtemp(
        prefix=f'.{PROBE_ID}.generation-', dir=output_dir.parent,
    ))
    staged_output = generation_staging / output_dir.name
    staged_evaluations = generation_staging / evaluation_root.name
    rows = []
    trajectories = []
    mapping_by_scene = {}
    evaluation_promoted = False
    try:
        for asset in sorted(
            reference_manifest['assets'],
            key=lambda item: (
                reverse_scenes[item['scene']], item['training_seed']
            ),
        ):
            if asset['availability'] != 'available':
                raise ValueError(
                    'Fixed probe requires all 20 canonical Plan1 assets available'
                )
            scene = reverse_scenes[asset['scene']]
            seed = int(asset['training_seed'])
            split = 'main' if seed <= 3 else 'heldout'
            run_dir = staged_evaluations / scene / f'seed_{seed}'
            identity = {
                'probe_id': PROBE_ID, 'split': split,
                'scene': scene, 'network': asset['scene'],
                'source_training_seed': seed,
                'source_checkpoint_path': asset['checkpoint_path'],
                'source_checkpoint_sha256': asset['checkpoint_sha256'],
            }
            evaluation = evaluate_policy(
                asset['scene'], run_dir,
                Plan1DQNPolicy(asset['checkpoint_path']),
                interface=interface, collect_probe=True,
                source_identity=identity,
            )
            command_digest = evaluation['summary'][
                'resolved_sumo_command_digest'
            ]
            for row in evaluation['decisions']:
                row['resolved_sumo_command_digest'] = command_digest
            causal = causal_gate_report(evaluation)
            if not causal['valid']:
                raise RuntimeError(
                    f'Plan5 causal context gates failed for {scene} seed {seed}'
                )
            mapping = evaluation['summary']['boundary_entry_mapping']
            previous = mapping_by_scene.setdefault(scene, mapping)
            if previous != mapping:
                raise RuntimeError(
                    f'Boundary geometry mapping changed in {scene}'
                )
            rows.extend(evaluation['decisions'])
            trajectories.append({
                'scene': scene, 'network': asset['scene'],
                'source_training_seed': seed, 'split': split,
                'source_checkpoint_path': asset['checkpoint_path'],
                'source_checkpoint_sha256': asset['checkpoint_sha256'],
                'source_trajectory_sha256': evaluation['summary'][
                    'trajectory_digest'
                ],
                'resolved_sumo_command': evaluation['summary'][
                    'resolved_sumo_command'
                ],
                'resolved_sumo_command_digest': command_digest,
                'causal_gates': causal,
            })
        manifest = write_probe(rows, staged_output, metadata={
            'trajectories': trajectories,
            'boundary_entry_mapping': mapping_by_scene,
            'all_causal_gates_valid': all(
                item['causal_gates']['valid'] for item in trajectories
            ),
        })
        os.replace(staged_evaluations, evaluation_root)
        evaluation_promoted = True
        try:
            os.replace(staged_output, output_dir)
        except Exception:
            os.replace(evaluation_root, staged_evaluations)
            evaluation_promoted = False
            raise
        return manifest
    finally:
        if evaluation_promoted and not output_dir.exists():
            os.replace(evaluation_root, staged_evaluations)
        shutil.rmtree(generation_staging, ignore_errors=True)


def validate_probe_manifest(path):
    path = Path(path)
    with open(path,encoding="utf-8") as handle: manifest=json.load(handle)
    if manifest.get("probe_id") != PROBE_ID: raise ValueError("Probe manifest ID mismatch")
    expected_digest = manifest.pop('manifest_digest', None)
    if expected_digest != canonical_digest(manifest): raise ValueError('Probe manifest digest mismatch')
    manifest['manifest_digest'] = expected_digest
    rows = []
    for split in ("main","heldout"):
        file_path=Path(manifest["paths"][split])
        if not file_path.is_absolute(): file_path = path.parent / file_path
        if sha256_file(file_path) != manifest["paths"][f"{split}_sha256"]: raise ValueError(f"Probe {split} hash mismatch")
        with open(file_path, encoding='utf-8') as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    validate_rows(rows)
    return manifest


def validate_probe_against_references(probe_manifest_path,
                                      reference_manifest_path):
    """Bind all 20 frozen trajectories back to the canonical whitelist."""
    manifest = validate_probe_manifest(probe_manifest_path)
    with open(reference_manifest_path, encoding='utf-8') as handle:
        reference = json.load(handle)
    reverse = {network: scene for scene, network in SCENES.items()}
    expected = {
        (reverse[item['scene']], int(item['training_seed'])): (
            item['scene'], item['checkpoint_path'], item['checkpoint_sha256']
        )
        for item in reference.get('assets', [])
        if item.get('availability') == 'available'
        and item.get('scene') in reverse
    }
    trajectories = manifest.get('trajectories', [])
    actual_keys = {
        (item.get('scene'), int(item.get('source_training_seed', -1)))
        for item in trajectories
    }
    if len(trajectories) != 20 or set(expected) != actual_keys:
        raise ValueError('Plan5 probe/canonical trajectory matrix mismatch')
    from .validation import validate_sumo_command
    for item in trajectories:
        key = (item['scene'], int(item['source_training_seed']))
        network, checkpoint, digest = expected[key]
        if (item.get('network'), item.get('source_checkpoint_path'),
                item.get('source_checkpoint_sha256')) != (
                network, checkpoint, digest):
            raise ValueError('Plan5 probe source is not canonical')
        if not item.get('source_trajectory_sha256') \
                or not item.get('resolved_sumo_command_digest'):
            raise ValueError('Plan5 probe trajectory digest is incomplete')
        validate_sumo_command(item.get('resolved_sumo_command', ()))
        causal = item.get('causal_gates', {})
        if not causal.get('valid') or not all(
                causal.get(field) for field in (
                    'past_only', 'future_mutation_invariant',
                    'reset_zero_padding', 'offline_recompute_exact')):
            raise ValueError('Plan5 probe trajectory causal gate failed')
    mapping = manifest.get('boundary_entry_mapping', {})
    if set(mapping) != set(SCENES) or any(
            not scene_mapping for scene_mapping in mapping.values()):
        raise ValueError('Plan5 probe boundary geometry mapping is incomplete')
    if not manifest.get('all_causal_gates_valid'):
        raise ValueError('Plan5 probe global causal gate is false')
    return {
        'valid': True, 'trajectories': 20,
        'manifest_digest': manifest['manifest_digest'],
        'main_sha256': manifest['paths']['main_sha256'],
        'heldout_sha256': manifest['paths']['heldout_sha256'],
        'all_causal_gates_valid': True,
    }


def load_probe_rows(manifest_path, split):
    if split not in {'main', 'heldout'}:
        raise ValueError('Probe split must be main or heldout')
    manifest_path = Path(manifest_path)
    manifest = validate_probe_manifest(manifest_path)
    path = Path(manifest['paths'][split])
    if not path.is_absolute():
        path = manifest_path.parent / path
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fixed_probe_inference(agent, rows):
    """Run immutable probe rows without touching exploration or training RNG."""
    rows = list(rows)
    algorithm_id = agent.algorithm_id
    if algorithm_id not in {'DDQN', 'CTXDDQN', 'PPO'}:
        raise ValueError('Unsupported Plan5 probe algorithm')
    before = canonical_digest(agent.full_state_dict())
    actions = []
    policy_vectors = []
    target_policy_vectors = []
    critic_values = []
    with torch.no_grad():
        for row in rows:
            raw = np.asarray(row['raw16'], dtype=np.float32)
            if raw.shape != (16,):
                raise ValueError('Probe inference requires raw16')
            if algorithm_id == 'CTXDDQN':
                context = np.asarray(row['ctx4'], dtype=np.float32)
                feature = np.concatenate([raw, context])
            else:
                feature = raw
            tensor = torch.as_tensor(feature[None, :], dtype=torch.float32)
            if algorithm_id == 'PPO':
                distribution, value = agent.model(tensor)
                action = int(torch.argmax(distribution.probs, dim=1).item())
                policy_vectors.append(
                    distribution.probs[0].detach().cpu().tolist()
                )
                critic_values.append(float(value.reshape(-1)[0].item()))
            else:
                values = agent.model(tensor)
                target_values = agent.target_model(tensor)
                action = int(torch.argmax(values, dim=1).item())
                policy_vectors.append(values[0].detach().cpu().tolist())
                target_policy_vectors.append(
                    target_values[0].detach().cpu().tolist()
                )
            actions.append(action)
    after = canonical_digest(agent.full_state_dict())
    if before != after:
        raise RuntimeError('Fixed-probe inference changed training state')
    result = {
        'algorithm_id': algorithm_id, 'row_count': len(actions),
        'actions': actions, 'actions_digest': canonical_digest(actions),
        'policy_vector_kind': (
            'categorical_probability' if algorithm_id == 'PPO' else 'q_value'
        ),
        'policy_vectors': policy_vectors,
        'policy_vectors_digest': canonical_digest(policy_vectors),
        'target_policy_vectors': (
            target_policy_vectors
            if algorithm_id in {'DDQN', 'CTXDDQN'} else None
        ),
        'target_policy_vectors_digest': (
            canonical_digest(target_policy_vectors)
            if algorithm_id in {'DDQN', 'CTXDDQN'} else None
        ),
        'critic_values': critic_values if algorithm_id == 'PPO' else None,
        'critic_values_digest': (
            canonical_digest(critic_values) if algorithm_id == 'PPO' else None
        ),
        'training_state_digest_before': before,
        'training_state_digest_after': after,
        'isolated': True,
    }
    if algorithm_id in {'DDQN', 'CTXDDQN'}:
        residuals = []
        for index, row in enumerate(rows[:-1]):
            following = rows[index + 1]
            same_trajectory = all(
                field in row and field in following
                for field in ('scene', 'source_training_seed', 'decision_index')
            ) and (
                row.get('scene') == following.get('scene')
                and row.get('source_training_seed')
                == following.get('source_training_seed')
                and int(following.get('decision_index', -1))
                == int(row.get('decision_index', -2)) + 1
            )
            if not same_trajectory:
                continue
            action = int(row['source_action'])
            reward = float(row['reward'])
            next_action = int(np.argmax(policy_vectors[index + 1]))
            target = reward + 0.95 * target_policy_vectors[
                index + 1
            ][next_action]
            residuals.append(abs(policy_vectors[index][action] - target))
        result['bellman_residual_mean'] = (
            float(np.mean(residuals)) if residuals else None
        )
        result['bellman_residual_count'] = len(residuals)
    else:
        result['bellman_residual_mean'] = None
        result['bellman_residual_count'] = 0
    return result
