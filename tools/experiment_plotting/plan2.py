"""Strict validation and aggregation for Plan 2 offline experiment runs."""

import csv
import hashlib
import json
import math
import shutil
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from utils.logger import verify_config_archive


PLAN2_EVALUATION_UPDATES = (0, 14400, 36000, 72000, 108000, 144000)
PLAN2_OFFLINE_SEEDS = (1000, 1001, 1002, 1003, 1004)
PLAN2_BEHAVIOR_SEEDS = (0, 1, 2, 3, 4)
PLAN2_NETWORKS = (
    'sumohz1x1_config2', 'sumohz1x1',
    'sumohz1x1_config4', 'sumohz1x1_config3',
)
RUN_LIST_FIELDS = (
    'algorithm', 'dataset_id', 'dataset_kind', 'dataset_stage',
    'evaluation_network', 'offline_training_seed', 'run_dir', 'include',
)
TRUE_VALUES = {'1', 'true', 'yes', 'y'}
FALSE_VALUES = {'0', 'false', 'no', 'n'}


@dataclass(frozen=True)
class Plan2RunSpec:
    algorithm: str
    dataset_id: str
    dataset_kind: str
    dataset_stage: str
    evaluation_network: str
    offline_training_seed: int
    run_dir: Path
    include: bool


def _load_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parse_include(value, line_number):
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f'Invalid include value at line {line_number}: {value!r}')


def load_plan2_run_list(path):
    source = Path(path).expanduser().resolve()
    specs = []
    seen = set()
    with source.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        missing = [] if reader.fieldnames is None else [
            field for field in RUN_LIST_FIELDS if field not in reader.fieldnames
        ]
        if reader.fieldnames is None or missing:
            raise ValueError(f'Plan 2 run-list missing fields: {missing or RUN_LIST_FIELDS}')
        for line_number, row in enumerate(reader, start=2):
            if not any((value or '').strip() for value in row.values()):
                continue
            run_dir = Path(row['run_dir']).expanduser().resolve()
            if run_dir in seen:
                raise ValueError(f'Duplicate Plan 2 run_dir: {run_dir}')
            seen.add(run_dir)
            specs.append(Plan2RunSpec(
                algorithm=row['algorithm'].strip(),
                dataset_id=row['dataset_id'].strip(),
                dataset_kind=row['dataset_kind'].strip(),
                dataset_stage=row['dataset_stage'].strip(),
                evaluation_network=row['evaluation_network'].strip(),
                offline_training_seed=int(row['offline_training_seed']),
                run_dir=run_dir,
                include=_parse_include(row['include'], line_number),
            ))
    included = [spec for spec in specs if spec.include]
    if not included:
        raise ValueError('Plan 2 run-list has no included runs')
    return source, specs, included


def _load_records(path):
    records = []
    with Path(path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f'Blank offline metric line at {path}:{line_number}')
            record = json.loads(line)
            if record.get('schema_version') != 1:
                raise ValueError(f'Unsupported offline metric schema at {path}:{line_number}')
            records.append(record)
    if not records:
        raise ValueError(f'No offline metric records: {path}')
    return records


def validate_plan2_run(spec, formal=True):
    if spec.algorithm not in {'batch_dqn', 'cql_dqn'}:
        raise ValueError(f'Unsupported Plan 2 algorithm: {spec.algorithm}')
    if spec.dataset_kind not in {'single_scene', 'leave_one_out'}:
        raise ValueError(f'Unsupported Plan 2 dataset kind: {spec.dataset_kind}')
    if not spec.run_dir.is_dir():
        raise FileNotFoundError(f'Plan 2 run does not exist: {spec.run_dir}')

    manifest = _load_json(spec.run_dir / 'run_manifest.json')
    status = _load_json(spec.run_dir / 'run_status.json')
    metadata = _load_json(spec.run_dir / 'offline_run_metadata.json')
    summary = _load_json(spec.run_dir / 'evaluation' / 'summary.json')
    verify_config_archive(str(spec.run_dir / 'config'))
    if status.get('status') != '已完成' or status.get('exit_code') != 0:
        raise ValueError(f'Plan 2 run did not complete successfully: {spec.run_dir}')
    expected = {
        'agent': spec.algorithm,
        'dataset_id': spec.dataset_id,
        'dataset_kind': spec.dataset_kind,
        'dataset_stage': spec.dataset_stage,
        'evaluation_network': spec.evaluation_network,
        'offline_training_seed': spec.offline_training_seed,
    }
    mismatches = [
        f'{field}: run={manifest.get(field)!r}, list={value!r}'
        for field, value in expected.items() if manifest.get(field) != value
    ]
    if mismatches:
        raise ValueError('Plan 2 run-list identity mismatch: ' + '; '.join(mismatches))
    if manifest.get('training_mode') != 'pure_offline':
        raise ValueError(f'Run is not marked pure_offline: {spec.run_dir}')
    resolved_config = spec.run_dir / 'config' / 'resolved_config.yaml'
    if manifest.get('config_hash') != _sha256(resolved_config):
        raise ValueError(f'Offline run config_hash mismatch: {spec.run_dir}')
    if metadata.get('training_environment_interactions') != 0:
        raise ValueError(f'Offline run reports training environment interactions: {spec.run_dir}')
    dataset = metadata.get('dataset', {})
    dataset_manifest_path = Path(metadata['dataset_manifest'])
    if (
        not dataset_manifest_path.is_file()
        or _sha256(dataset_manifest_path) != metadata.get('dataset_manifest_sha256')
    ):
        raise ValueError(f'Offline source dataset manifest changed or is missing: {spec.run_dir}')
    if dataset.get('evaluation_network') != spec.evaluation_network:
        raise ValueError(f'Offline metadata target mismatch: {spec.run_dir}')
    sources = set(dataset.get('source_networks', []))
    if spec.dataset_kind == 'leave_one_out' and spec.evaluation_network in sources:
        raise ValueError(f'Leave-one-out target leakage: {spec.run_dir}')
    if spec.dataset_kind == 'single_scene' and sources != {spec.evaluation_network}:
        raise ValueError(f'Single-scene source mismatch: {spec.run_dir}')

    total_updates = int(metadata['total_updates'])
    evaluation_updates = tuple(metadata['evaluation_updates'])
    if formal:
        if total_updates != 144000 or evaluation_updates != PLAN2_EVALUATION_UPDATES:
            raise ValueError(f'Non-formal update/evaluation schedule: {spec.run_dir}')
        if tuple(metadata.get('behavior_training_seeds', [])) != PLAN2_BEHAVIOR_SEEDS:
            raise ValueError(f'Non-formal behavior seed set: {spec.run_dir}')
        if spec.offline_training_seed not in PLAN2_OFFLINE_SEEDS:
            raise ValueError(f'Non-formal offline training seed: {spec.run_dir}')
        expected_training_config = {
            'network_hidden_layers': [20, 20],
            'batch_size': 64,
            'learning_rate': 0.001,
            'gamma': 0.95,
            'grad_clip': 5.0,
            'optimizer': 'RMSprop',
            'optimizer_alpha': 0.9,
            'optimizer_centered': False,
            'optimizer_eps': 1e-7,
            'target_update_interval': 10,
            'cql_alpha': 0.0 if spec.algorithm == 'batch_dqn' else 1.0,
            'td_target_terminal_mask': False,
            'td_target_double_dqn': False,
            'loss': 'full_q_vector_mse',
        }
        if metadata.get('training_config') != expected_training_config:
            raise ValueError(f'Offline training configuration drift: {spec.run_dir}')

    records = _load_records(spec.run_dir / 'metrics' / 'offline_records.jsonl')
    for record in records:
        identity = (
            record.get('algorithm'), record.get('network'),
            record.get('offline_training_seed'), record.get('dataset_id'),
            record.get('dataset_kind'), record.get('dataset_stage'),
        )
        expected_identity = (
            spec.algorithm, spec.evaluation_network, spec.offline_training_seed,
            spec.dataset_id, spec.dataset_kind, spec.dataset_stage,
        )
        if identity != expected_identity:
            raise ValueError(f'Offline metric identity mismatch: {spec.run_dir}')
        for field in ('loss', 'travel_time', 'delay', 'real_delay', 'queue', 'throughput'):
            value = record.get(field)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f'Invalid offline metric {field}: {spec.run_dir}')
    evaluation_records = [
        record for record in records
        if record['record_type'] in {'EVALUATION', 'FINAL_EVALUATION'}
    ]
    if tuple(record['training_update'] for record in evaluation_records) != evaluation_updates:
        raise ValueError(f'Incomplete offline evaluation records: {spec.run_dir}')
    if evaluation_records[-1]['record_type'] != 'FINAL_EVALUATION':
        raise ValueError(f'Missing final offline evaluation: {spec.run_dir}')
    train_updates = [
        record['training_update'] for record in records if record['record_type'] == 'TRAIN'
    ]
    if not train_updates or train_updates[-1] != total_updates:
        raise ValueError(f'Offline training records do not reach total updates: {spec.run_dir}')
    if tuple(summary.get('evaluation_updates', [])) != evaluation_updates:
        raise ValueError(f'Offline evaluation summary schedule mismatch: {spec.run_dir}')
    if summary.get('final_update') != total_updates:
        raise ValueError(f'Offline final checkpoint update mismatch: {spec.run_dir}')
    for update in evaluation_updates:
        for checkpoint_type in ('evaluation', 'resumable'):
            checkpoint = (
                spec.run_dir / 'checkpoints' / checkpoint_type /
                f'update_{update:06d}.pt'
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(f'Missing offline checkpoint: {checkpoint}')
            import torch
            try:
                payload = torch.load(checkpoint, map_location='cpu')
            except Exception as error:
                raise IOError(f'Cannot load offline checkpoint: {checkpoint}') from error
            if (
                payload.get('schema_version') != 1
                or payload.get('training_mode') != 'pure_offline'
                or payload.get('checkpoint_type') != checkpoint_type
                or payload.get('training_update') != update
                or payload.get('algorithm') != spec.algorithm
                or payload.get('dataset_manifest_sha256')
                != metadata['dataset_manifest_sha256']
                or payload.get('resume_fingerprint') != metadata['resume_fingerprint']
            ):
                raise ValueError(f'Offline checkpoint identity mismatch: {checkpoint}')
            online_state = payload.get('online_model_state_dict')
            if not isinstance(online_state, dict) or not online_state:
                raise ValueError(f'Offline checkpoint has no policy state: {checkpoint}')
            if any(
                not torch.isfinite(tensor).all().item()
                for tensor in online_state.values()
            ):
                raise ValueError(f'Offline checkpoint contains non-finite policy weights: {checkpoint}')
            if checkpoint_type == 'resumable':
                required_resume = {
                    'target_model_state_dict', 'optimizer_state_dict',
                    'dataset_rng_state', 'python_random_state', 'numpy_random_state',
                    'torch_cpu_rng_state', 'torch_cuda_rng_states',
                }
                if not required_resume <= set(payload):
                    raise ValueError(f'Incomplete resumable checkpoint: {checkpoint}')
    return {
        'spec': spec,
        'manifest': manifest,
        'metadata': metadata,
        'summary': summary,
        'records': records,
        'config_hash': manifest['config_hash'],
        'dataset_manifest_sha256': metadata['dataset_manifest_sha256'],
    }


def _write_csv(path, rows, fields=None):
    path = Path(path)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values):
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def _aggregate_final(validated):
    groups = {}
    for run in validated:
        spec = run['spec']
        key = (
            spec.algorithm, spec.dataset_id, spec.dataset_kind,
            spec.dataset_stage, spec.evaluation_network,
        )
        groups.setdefault(key, []).append(run)
    rows = []
    for key, runs in sorted(groups.items()):
        final_values = [run['summary']['final_travel_time'] for run in runs]
        best_values = [run['summary']['best_travel_time'] for run in runs]
        final_mean, final_std = _mean_std(final_values)
        best_mean, best_std = _mean_std(best_values)
        rows.append({
            'algorithm': key[0], 'dataset_id': key[1], 'dataset_kind': key[2],
            'dataset_stage': key[3], 'evaluation_network': key[4],
            'seed_count': len(runs),
            'offline_training_seeds': json.dumps(sorted(
                run['spec'].offline_training_seed for run in runs
            )),
            'final_travel_time_mean': final_mean,
            'final_travel_time_std': final_std,
            'best_travel_time_mean': best_mean,
            'best_travel_time_std': best_std,
        })
    return rows


def _validate_matrix(validated):
    groups = {}
    for run in validated:
        spec = run['spec']
        key = (
            spec.algorithm, spec.dataset_id, spec.dataset_kind,
            spec.dataset_stage, spec.evaluation_network,
        )
        groups.setdefault(key, set()).add(spec.offline_training_seed)
    incomplete = {
        key: sorted(set(PLAN2_OFFLINE_SEEDS) - seeds)
        for key, seeds in groups.items() if seeds != set(PLAN2_OFFLINE_SEEDS)
    }
    if incomplete:
        raise ValueError(f'Incomplete Plan 2 five-seed matrix: {incomplete}')
    dataset_ids = {run['spec'].dataset_id for run in validated}
    if len(dataset_ids) != 1:
        raise ValueError(f'Formal Plan 2 analysis requires one dataset_id: {dataset_ids}')
    present = {
        (
            run['spec'].algorithm, run['spec'].dataset_kind,
            run['spec'].dataset_stage, run['spec'].evaluation_network,
        )
        for run in validated
    }
    required = set()
    for algorithm in ('batch_dqn', 'cql_dqn'):
        for network in PLAN2_NETWORKS:
            for stage in ('Q1', 'Q4', 'full'):
                required.add((algorithm, 'single_scene', stage, network))
            required.add((algorithm, 'leave_one_out', 'full', network))
    missing = sorted(required - present)
    if missing:
        raise ValueError(f'Incomplete formal Plan 2 experiment cells: {missing}')


def run_plan2_analysis(args):
    run_list, all_specs, included = load_plan2_run_list(args.run_list)
    output = Path(args.output_root).expanduser().resolve() / args.analysis_id
    if output.exists():
        raise FileExistsError(f'Plan 2 analysis output already exists: {output}')
    inputs = output / 'inputs'
    tables = output / 'tables'
    inputs.mkdir(parents=True)
    tables.mkdir()
    try:
        shutil.copy2(run_list, inputs / 'runs.csv')
        validated = [
            validate_plan2_run(spec, formal=not args.allow_incomplete)
            for spec in included
        ]
        if not args.allow_incomplete:
            _validate_matrix(validated)

        curves = []
        summaries = []
        dataset_statistics = []
        seen_dataset_statistics = set()
        for run in validated:
            spec = run['spec']
            for record in run['records']:
                curves.append({
                    'algorithm': spec.algorithm,
                    'dataset_id': spec.dataset_id,
                    'dataset_kind': spec.dataset_kind,
                    'dataset_stage': spec.dataset_stage,
                    'evaluation_network': spec.evaluation_network,
                    'offline_training_seed': spec.offline_training_seed,
                    'run_dir': str(spec.run_dir),
                    **record,
                    'action_distribution': json.dumps(
                        record.get('action_distribution', {}), sort_keys=True
                    ),
                    'source_network_counts': json.dumps(
                        record.get('source_network_counts', {}), sort_keys=True
                    ),
                })
            summaries.append({
                'algorithm': spec.algorithm, 'dataset_id': spec.dataset_id,
                'dataset_kind': spec.dataset_kind, 'dataset_stage': spec.dataset_stage,
                'evaluation_network': spec.evaluation_network,
                'offline_training_seed': spec.offline_training_seed,
                'run_dir': str(spec.run_dir),
                'best_update': run['summary']['best_update'],
                'best_travel_time': run['summary']['best_travel_time'],
                'final_update': run['summary']['final_update'],
                'final_travel_time': run['summary']['final_travel_time'],
                'backend': run['metadata']['backend']['name'],
                'dataset_manifest_sha256': run['dataset_manifest_sha256'],
                'config_hash': run['config_hash'],
            })
            for source, stats in run['metadata']['dataset_statistics'].items():
                stats_key = (run['dataset_manifest_sha256'], source)
                if stats_key in seen_dataset_statistics:
                    continue
                seen_dataset_statistics.add(stats_key)
                dataset_statistics.append({
                    'dataset_id': spec.dataset_id,
                    'dataset_kind': spec.dataset_kind,
                    'dataset_stage': spec.dataset_stage,
                    'evaluation_network': spec.evaluation_network,
                    'source_network': source,
                    **stats,
                    'action_counts': json.dumps(stats['action_counts'], sort_keys=True),
                })
        aggregates = _aggregate_final(validated)
        _write_csv(tables / 'offline_training_curves.csv', curves)
        _write_csv(tables / 'run_summary.csv', summaries)
        _write_csv(tables / 'dataset_statistics.csv', dataset_statistics)
        _write_csv(
            tables / 'same_scene_results.csv',
            [row for row in aggregates if row['dataset_kind'] == 'single_scene' and row['dataset_stage'] == 'full'],
        )
        _write_csv(
            tables / 'data_stage_results.csv',
            [row for row in aggregates if row['dataset_kind'] == 'single_scene'],
        )
        _write_csv(
            tables / 'leave_one_scene_out_results.csv',
            [row for row in aggregates if row['dataset_kind'] == 'leave_one_out'],
        )
        report = [
            '# Plan 2 Offline 汇总报告', '',
            f'- 纳入运行：{len(validated)}',
            f'- 完整五 seed 验证：{"否（允许不完整）" if args.allow_incomplete else "是"}',
            f'- 生成时间：{datetime.now(timezone.utc).isoformat()}', '',
            '本报告只汇总已完成、配置归档有效、离线数据身份一致且固定评估/checkpoint 完整的运行。',
        ]
        (output / 'plan2_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
        manifest = {
            'schema_version': 1,
            'analysis_type': 'plan2',
            'analysis_id': args.analysis_id,
            'run_list_source': str(run_list),
            'listed_run_count': len(all_specs),
            'included_run_count': len(validated),
            'allow_incomplete': args.allow_incomplete,
            'tables': sorted(path.name for path in tables.iterdir()),
            'report': 'plan2_report.md',
        }
        (output / 'plotting_manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return output
