"""Fail-closed phase planning and launch authorization."""
from dataclasses import dataclass
from pathlib import Path
import json

from sequential.io import atomic_json, read_json, sha256_file
from sequential.launcher import AttemptLineage, LogicalRunLock
from .config import ALGORITHMS, SCENES, TRANSITIONS, SEEDS
from .manifest import (
    build_ppo_calibration_manifest, checkpoint_source_identity, run_id,
    validate_run_manifest, write_ppo_calibration_manifest, write_run_manifest,
)
from .validation import (
    validate_checkpoint_binding, validate_launch_row,
    validate_same_start_matrix,
)
from .schema import validate_json_fields


@dataclass
class PhaseGate:
    environment_ready: bool = False
    probe_ready: bool = False
    source_ready: bool = False
    tests_passed: bool = False

    @property
    def ready(self): return all((self.environment_ready,self.probe_ready,self.source_ready,self.tests_passed))


class Plan5Launcher:
    def __init__(self, output_root):
        self.output_root=Path(output_root)

    def build_phase1(self, algorithms=ALGORITHMS, config_sha256="", source_commit=""):
        algorithms=tuple(algorithms)
        if any(a not in ALGORITHMS for a in algorithms): raise ValueError("Phase1 algorithm is not authorized")
        rows=[]
        for algorithm in algorithms:
            for scene in SCENES:
                for seed in SEEDS:
                    rows.append({"logical_run_id":run_id(algorithm,"ANCHOR",scene,seed=seed),"algorithm_id":algorithm,"run_type":"ANCHOR","scene":scene,"training_seed":seed,"config_sha256":config_sha256,"source_commit":source_commit,"status":"planned"})
        return rows

    def build_ppo_calibration(self, config, source_commit):
        return build_ppo_calibration_manifest(config, source_commit)

    def bind_ppo_config(self, rows, config_path, *, require_ready=True):
        validate_json_fields(config_path, 'ppo_formal_config.json')
        payload = read_json(config_path)
        if require_ready and payload['formal_status'] != 'READY':
            raise ValueError('PPO formal config is not READY')
        digest = sha256_file(config_path)
        bound = []
        for original in rows:
            row = dict(original)
            if row.get('algorithm_id') == 'PPO':
                row['ppo_config_path'] = str(Path(config_path).resolve())
                row['ppo_config_sha256'] = digest
            bound.append(row)
        validate_run_manifest(bound)
        return bound

    def write_launch_batch(self, rows, path):
        """Freeze one fully-bound batch without occupying the final manifest."""
        rows = [dict(row) for row in rows]
        calibration = [
            row for row in rows if row.get('run_type') == 'CALIBRATION'
        ]
        if calibration and len(calibration) != len(rows):
            raise ValueError('Plan5 launch batch cannot mix calibration/formal rows')
        for row in rows:
            if row.get('run_type') == 'CALIBRATION':
                from .manifest import validate_ppo_calibration_row
                validate_ppo_calibration_row(row)
            else:
                validate_launch_row(row)
        if calibration:
            return write_ppo_calibration_manifest(rows, path)
        transitions = [
            row for row in rows if row.get('run_type') == 'TRANSITION'
        ]
        if transitions:
            validate_same_start_matrix(transitions)
        return write_run_manifest(rows, path)

    def build_phase2(self, algorithms, anchor_status, config_sha256="", source_commit=""):
        rows=[]
        for algorithm in algorithms:
            if not anchor_status.get(algorithm,False): continue
            for transition,(source,target) in TRANSITIONS.items():
                for seed in SEEDS:
                    rows.append({"logical_run_id":run_id(algorithm,"TRANSITION",source,target,seed),"algorithm_id":algorithm,"run_type":"TRANSITION","transition_id":transition,"source_scene":source,"target_scene":target,"training_seed":seed,"config_sha256":config_sha256,"source_commit":source_commit,"status":"planned"})
        return rows

    def bind_phase2_sources(self, rows, anchor_checkpoints):
        """Bind every continuation to its exact same-algorithm anchor."""
        bound = []
        for original in rows:
            row = dict(original)
            if row.get('run_type') != 'TRANSITION':
                raise ValueError('Phase2 source binding accepts transitions only')
            key = (
                row['algorithm_id'], row['source_scene'],
                int(row['training_seed']),
            )
            path = anchor_checkpoints.get(key)
            if path is None:
                raise ValueError(f'Missing Plan5 anchor checkpoint: {key}')
            source = checkpoint_source_identity(path)
            checkpoint_identity = source.pop('checkpoint_identity')
            if checkpoint_identity.get('algorithm_id') != row['algorithm_id'] \
                    or int(checkpoint_identity.get('training_seed', -1)) != int(
                        row['training_seed']
                    ) \
                    or checkpoint_identity.get('scene') != row['source_scene']:
                raise ValueError('Anchor checkpoint identity does not bind transition')
            row.update(source)
            validate_checkpoint_binding(row)
            bound.append(row)
        validate_run_manifest(bound)
        validate_same_start_matrix(bound)
        return bound

    def authorize(self, gate, *, phase, user_confirmed=False):
        if phase not in ("Phase1","Phase2"): raise ValueError("Only Phase1/Phase2 are launchable")
        if not gate.ready: raise RuntimeError("Plan5 Phase0 gates are not all passed")
        if not user_confirmed: raise PermissionError(f"{phase} requires explicit user confirmation")
        return {"authorized":True,"phase":phase}

    def write_plan(self, rows, path):
        validate_run_manifest(rows)
        payload={"schema_version":1,"phase":"Plan5","runs":rows,"launch_authorized":False}
        atomic_json(path,payload); return payload

    def reserve_attempt(self, row, *, resume_from=None):
        if row.get('run_type') == 'CALIBRATION':
            from .manifest import validate_ppo_calibration_row
            validate_ppo_calibration_row(row)
        else:
            validate_launch_row(row)
        logical_id = row['logical_run_id']
        runs_root = self.output_root / 'runs'
        lock_path = runs_root / logical_id / 'logical_run.lock'
        with LogicalRunLock(lock_path):
            lineage = AttemptLineage(runs_root, logical_id)
            if lineage.manifest.get('status') == 'completed':
                raise FileExistsError(
                    f'Plan5 logical run is already complete: {logical_id}'
                )
            attempt = lineage.create_attempt(resume_from=resume_from)
            row_path = Path(attempt['attempt_dir']) / 'run_row.json'
            atomic_json(row_path, row)
            lineage.update_attempt(
                attempt['attempt_id'], 'planned',
                run_row_path=str(row_path.resolve()),
            )
            return attempt

    def update_attempt(self, logical_run_id, attempt_id, status, **fields):
        if status not in {'running', 'completed', 'failed', 'invalid'}:
            raise ValueError('Unknown Plan5 attempt status')
        runs_root = self.output_root / 'runs'
        lock_path = runs_root / logical_run_id / 'logical_run.lock'
        with LogicalRunLock(lock_path):
            lineage = AttemptLineage(runs_root, logical_run_id)
            return lineage.update_attempt(attempt_id, status, **fields)
