"""Fail-closed Plan5 run identities and canonical reference manifests."""
import csv
import os
from pathlib import Path
import tempfile

from sequential.core import canonical_digest
from sequential.io import atomic_json, sha256_file
from .config import SCENES, TRANSITIONS, ALGORITHMS, SEEDS
from .schema import RUN_MANIFEST_COLUMNS, TABLE_SCHEMAS

PPO_CALIBRATION_SEEDS = (100, 101, 102)
PPO_CALIBRATION_CANDIDATES = tuple(
    (learning_rate, entropy_coefficient)
    for learning_rate in (1e-4, 2.5e-4, 5e-4)
    for entropy_coefficient in (0.001, 0.01)
)


def calibration_run_id(learning_rate, entropy_coefficient, seed):
    key = (float(learning_rate), float(entropy_coefficient))
    if key not in PPO_CALIBRATION_CANDIDATES:
        raise ValueError('Unknown Plan5 PPO calibration candidate')
    if int(seed) not in PPO_CALIBRATION_SEEDS:
        raise ValueError('Unknown Plan5 PPO calibration seed')
    lr_tag = {1e-4: 'LR1E4', 2.5e-4: 'LR25E5', 5e-4: 'LR5E4'}[key[0]]
    entropy_tag = {0.001: 'EC001', 0.01: 'EC01'}[key[1]]
    return f'P5-CAL-PPO-S2-{lr_tag}-{entropy_tag}-SD{int(seed)}'


def build_ppo_calibration_manifest(config, source_commit):
    rows = []
    for learning_rate, entropy_coefficient in PPO_CALIBRATION_CANDIDATES:
        for seed in PPO_CALIBRATION_SEEDS:
            rows.append({
                'logical_run_id': calibration_run_id(
                    learning_rate, entropy_coefficient, seed,
                ),
                'algorithm_id': 'PPO', 'run_type': 'CALIBRATION',
                'scene': 'S2', 'training_seed': seed,
                'learning_rate': learning_rate,
                'entropy_coefficient': entropy_coefficient,
                'config_sha256': config.sha256(),
                'source_commit': source_commit, 'status': 'planned',
            })
    validate_ppo_calibration_manifest(rows)
    return rows


def validate_ppo_calibration_row(row):
    required = {
        'logical_run_id', 'algorithm_id', 'run_type', 'scene',
        'training_seed', 'learning_rate', 'entropy_coefficient',
        'config_sha256', 'source_commit',
    }
    missing = required - set(row)
    if missing:
        raise ValueError(
            f'PPO calibration row missing fields: {sorted(missing)}'
        )
    if row['algorithm_id'] != 'PPO' or row['run_type'] != 'CALIBRATION' \
            or row['scene'] != 'S2':
        raise ValueError('PPO calibration identity is not S2/PPO')
    expected = calibration_run_id(
        row['learning_rate'], row['entropy_coefficient'],
        row['training_seed'],
    )
    if row['logical_run_id'] != expected:
        raise ValueError('PPO calibration logical run ID mismatch')
    if not row['config_sha256'] or not row['source_commit']:
        raise ValueError('PPO calibration config/source identity is empty')
    return True


def validate_ppo_calibration_manifest(rows):
    rows = list(rows)
    if len(rows) != 18:
        raise ValueError('PPO calibration manifest requires 18 runs')
    for row in rows:
        validate_ppo_calibration_row(row)
    identities = {
        (float(row['learning_rate']), float(row['entropy_coefficient']),
         int(row['training_seed']))
        for row in rows
    }
    expected = {
        (learning_rate, entropy_coefficient, seed)
        for learning_rate, entropy_coefficient in PPO_CALIBRATION_CANDIDATES
        for seed in PPO_CALIBRATION_SEEDS
    }
    if identities != expected or len({row['logical_run_id'] for row in rows}) != 18:
        raise ValueError('PPO calibration matrix is incomplete or duplicated')
    return True


def write_ppo_calibration_manifest(rows, output_path):
    validate_ppo_calibration_manifest(rows)
    columns = TABLE_SCHEMAS['ppo_calibration_run_manifest.csv']
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix='.tmp-plan5-ppo-calibration-manifest-',
        dir=output_path.parent,
    )
    try:
        with os.fdopen(descriptor, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(
                handle, fieldnames=columns, extrasaction='raise',
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        if output_path.exists():
            if sha256_file(temporary) != sha256_file(output_path):
                raise FileExistsError(
                    'Refusing to replace PPO calibration manifest: '
                    f'{output_path}'
                )
            os.unlink(temporary)
            temporary = None
        else:
            os.replace(temporary, output_path)
            temporary = None
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return {
        'path': str(output_path.resolve()),
        'sha256': sha256_file(output_path), 'rows': len(rows),
    }


def run_id(algorithm_id, run_type, source_scene, target_scene=None, seed=0):
    if algorithm_id not in ALGORITHMS: raise ValueError("Unauthorized algorithm")
    if run_type == "ANCHOR":
        if source_scene not in SCENES: raise ValueError("Unknown anchor scene")
        return f"P5-ANCHOR-{algorithm_id}-S{source_scene[1:]}-SD{int(seed)}"
    if run_type == "TRANSITION":
        transition = next((name for name, pair in TRANSITIONS.items() if pair == (source_scene, target_scene)), None)
        if transition is None: raise ValueError("Unknown Plan5 transition")
        return f"P5-{algorithm_id}-{transition}-SD{int(seed)}"
    raise ValueError("run_type must be ANCHOR or TRANSITION")


def build_run_manifest(config, source_commit, config_sha256=None):
    rows = []
    for algorithm in config.algorithms:
        for scene in SCENES:
            for seed in SEEDS:
                rows.append({"logical_run_id":run_id(algorithm,"ANCHOR",scene,seed=seed),"algorithm_id":algorithm,"run_type":"ANCHOR","scene":scene,"training_seed":seed,"config_sha256":config_sha256 or config.sha256(),"source_commit":source_commit,"status":"planned"})
        for transition, (source, target) in TRANSITIONS.items():
            for seed in SEEDS:
                rows.append({"logical_run_id":run_id(algorithm,"TRANSITION",source,target,seed),"algorithm_id":algorithm,"run_type":"TRANSITION","transition_id":transition,"source_scene":source,"target_scene":target,"training_seed":seed,"config_sha256":config_sha256 or config.sha256(),"source_commit":source_commit,"status":"planned"})
    if len({row["logical_run_id"] for row in rows}) != len(rows): raise ValueError("Duplicate logical run identity")
    return rows


def write_run_manifest(rows, output_path):
    """Validate and atomically write the frozen Plan5 CSV run manifest."""
    validate_run_manifest(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix='.tmp-plan5-run-manifest-', dir=output_path.parent,
    )
    try:
        with os.fdopen(descriptor, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(
                handle, fieldnames=RUN_MANIFEST_COLUMNS, extrasaction='raise',
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        if output_path.exists():
            if sha256_file(temporary) != sha256_file(output_path):
                raise FileExistsError(
                    f'Refusing to replace Plan5 run manifest: {output_path}'
                )
            os.unlink(temporary)
            temporary = None
        else:
            os.replace(temporary, output_path)
            temporary = None
    except Exception:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return {
        'path': str(output_path.resolve()),
        'sha256': sha256_file(output_path),
        'rows': len(rows),
    }


def validate_run_manifest(rows, existing_ids=()):
    ids = [row.get("logical_run_id") for row in rows]
    if any(not item for item in ids) or len(ids) != len(set(ids)): raise ValueError("Manifest contains duplicate/missing logical run IDs")
    collision = set(ids) & set(existing_ids)
    if collision: raise ValueError(f"Refusing to overwrite existing runs: {sorted(collision)}")
    required = {"logical_run_id","algorithm_id","run_type","training_seed","config_sha256","source_commit"}
    for row in rows:
        missing = required - set(row)
        if missing: raise ValueError(f"Manifest row missing fields: {sorted(missing)}")
        if row["algorithm_id"] not in ALGORITHMS: raise ValueError("Unauthorized algorithm in manifest")
        if int(row["training_seed"]) not in SEEDS: raise ValueError("Unauthorized training seed")
        if row['run_type'] == 'ANCHOR':
            scene = row.get('scene')
            expected_id = run_id(
                row['algorithm_id'], 'ANCHOR', scene,
                seed=row['training_seed'],
            )
            if any(row.get(field) not in (None, '') for field in (
                    'transition_id', 'source_scene', 'target_scene')):
                raise ValueError('Anchor manifest row contains transition fields')
        elif row['run_type'] == 'TRANSITION':
            transition = row.get('transition_id')
            if transition not in TRANSITIONS:
                raise ValueError('Unknown Plan5 transition in manifest')
            source, target = TRANSITIONS[transition]
            if (row.get('source_scene'), row.get('target_scene')) != (
                    source, target):
                raise ValueError('Plan5 transition source/target mismatch')
            if row.get('scene') not in (None, ''):
                raise ValueError('Transition manifest row contains anchor scene')
            expected_id = run_id(
                row['algorithm_id'], 'TRANSITION', source, target,
                row['training_seed'],
            )
        else:
            raise ValueError('Manifest run_type must be ANCHOR or TRANSITION')
        if row['logical_run_id'] != expected_id:
            raise ValueError('Manifest logical run ID does not match row identity')
        if not row['config_sha256'] or not row['source_commit']:
            raise ValueError('Manifest config/source identity must be non-empty')
    return True


def canonical_reference_manifest(whitelist_path, output_path=None):
    from sequential.parents import ParentSource, validate_parent_source
    assets=[]; seen=set()
    with open(whitelist_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scene=row.get("network"); seed=int(row["behavior_training_seed"])
            if scene not in SCENES.values() or seed not in SEEDS: raise ValueError("Whitelist row outside frozen Plan1 matrix")
            key=(scene,seed)
            if key in seen: raise ValueError(f"Duplicate canonical whitelist row: {key}")
            seen.add(key)
            run_path=row["run_path"]
            checkpoint = Path(run_path) / "checkpoints/resumable/episode_0100.pt"
            validation = None
            error = None
            if checkpoint.is_file():
                try:
                    validation = validate_parent_source(
                        ParentSource(run_path, scene, seed),
                        checkpoint_episode=100,
                    )
                    status = "available"
                    digest = validation["checkpoint_file_sha256"]
                    path = validation["checkpoint_path"]
                except Exception as exception:
                    status = "missing"
                    digest = None
                    path = str(checkpoint.resolve())
                    error = f"{type(exception).__name__}: {exception}"
            elif run_path.startswith("/projects/"):
                status="external"; digest=None; path=str(checkpoint)
            else:
                status="missing"; digest=None; path=str(checkpoint)
            assets.append({
                "scene": scene, "training_seed": seed,
                "run_path": run_path, "episode": 100,
                "checkpoint_path": path, "checkpoint_sha256": digest,
                "availability": status, "validation_error": error,
                "source_run_id": (
                    None if validation is None else validation["source_run_id"]
                ),
                "dimensions": (
                    None if validation is None else validation["dimensions"]
                ),
                "state_digests": (
                    None if validation is None else validation["digests"]
                ),
            })
    expected={(network,seed) for network in SCENES.values() for seed in SEEDS}
    if seen != expected: raise ValueError(f"Canonical whitelist must contain 20 rows; got {len(seen)}")
    payload={"schema_version":1,"plan_id":"Plan5","source":"Plan1 episode-100 canonical whitelist","assets":assets,"available_counts":{status:sum(a["availability"]==status for a in assets) for status in ("available","external","missing")}}
    payload["manifest_sha256"]=canonical_digest(payload)
    if output_path: atomic_json(output_path,payload)
    return payload


def validate_same_start(h34, l32):
    keys=("algorithm_id","training_seed","source_scene","source_checkpoint_path","source_checkpoint_sha256","model_state_digest","optimizer_state_digest","algorithm_state_digest","rng_state_digest")
    if h34.get("transition_id")!="H34" or l32.get("transition_id")!="L32": raise ValueError("same-start requires H34 and L32")
    if any(h34.get(key)!=l32.get(key) for key in keys): raise ValueError("H34/L32 source checkpoint identity is not exact")
    if h34.get("source_scene")!="S3" or l32.get("source_scene")!="S3": raise ValueError("same-start source must be S3")
    return True


def checkpoint_source_identity(path):
    """Return immutable component identities for a Plan5 full checkpoint."""
    from .checkpoint import load_checkpoint
    payload = load_checkpoint(path)
    state = payload['agent_state']
    model = state.get('online_model_state_dict', state.get('model_state_dict'))
    optimizer = state.get('optimizer_state_dict')
    if model is None or optimizer is None:
        raise ValueError('Plan5 checkpoint lacks model/optimizer state')
    algorithm_state = {
        key: value for key, value in state.items()
        if key not in {
            'online_model_state_dict', 'target_model_state_dict',
            'model_state_dict', 'optimizer_state_dict', 'rng_state',
        }
    }
    return {
        'source_checkpoint_path': str(Path(path).resolve()),
        'source_checkpoint_sha256': sha256_file(path),
        'model_state_digest': canonical_digest(model),
        'optimizer_state_digest': canonical_digest(optimizer),
        'algorithm_state_digest': canonical_digest(algorithm_state),
        'rng_state_digest': canonical_digest(payload['rng_state']),
        'checkpoint_identity': copy_identity(payload['identity']),
    }


def copy_identity(identity):
    return {key: identity[key] for key in sorted(identity)}
