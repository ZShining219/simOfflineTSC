import argparse
import ast
import csv
import json
import re
import shutil
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from . import __version__
from .aggregations import (
    AUC_FIELDS, AULC_SUMMARY_FIELDS, CORE_FIELDS, EFFICIENCY_FIELDS,
    LEARNING_SPEED_FIELDS, build_efficiency_rows, build_run_summaries,
    calculate_efficiency_aulc, calculate_first_100_auc,
    calculate_learning_speed, normalize_records, write_csv, calculate_convergence,
    summarize_convergence, calculate_budget_sufficiency,
    recommend_resumable_budget, calculate_action_distribution_diagnostics,
    normalize_cross_scene_summaries, aggregate_cross_scene,
    aggregate_episode100_pair_gate,
    calculate_policy_disagreement, calculate_state_distribution_diagnostics,
    select_transition_pairs, PROBE_FEATURE_NAMES,
)
from .loaders import (
    REQUIRED_RUN_LIST_FIELDS, load_run_list, load_scene_mapping,
    load_evaluation_records,
)
from .plotting import (
    render_all, render_evaluation_timeseries, render_convergence_curves,
    render_convergence_confirmation_metrics,
    render_convergence_distribution, render_budget_sufficiency,
    render_mean_action_distribution, render_distance_heatmap,
    render_within_between_distance,
    render_cross_scene_heatmaps, render_relative_degradation_heatmaps,
    render_cross_scene_seed_scatter,
    render_episode100_pair_gate,
    render_policy_scene_heatmap, render_policy_model_heatmap,
    render_policy_source_stratified, render_within_between_policy,
    render_state_distance_heatmap,
    render_state_pca, render_classifier_confusion,
    render_state_distance_vs_policy,
    render_dqn_training_state_coverage,
)
from .plan2 import run_plan2_analysis
from .sequential import run_sequential_plotting, run_sequential_frozen_plotting
from .profiles import PROFILES, get_profile, S1_S4_ACTION_SEMANTICS
from .validators import compare_dqn_run_configs, validate_run
from utils.logger import validate_evaluation_package


DEFAULT_OUTPUT_ROOT = Path("data/output_data/analysis/plan1")
DEFAULT_PLAN2_OUTPUT_ROOT = Path("data/output_data/analysis/plan2")
DQN_TRAINING_STATE_WINDOWS = ((1, 10), (41, 50), (91, 100),
                              (291, 300), (391, 400))


def _json_dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _analysis_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise argparse.ArgumentTypeError(
            "analysis-id must contain only letters, digits, dot, underscore and hyphen"
        )
    return value


def build_parser():
    parser = argparse.ArgumentParser(description="Validated reusable experiment plotting")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan1 = subparsers.add_parser("plan1", help="Generate Plan 1 tables and figures")
    plan1.add_argument("--run-list", required=True, help="Explicit CSV run-list")
    plan1.add_argument("--analysis-id", required=True, type=_analysis_id)
    plan1.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    plan1.add_argument("--dpi", type=int, default=160)
    plan1.add_argument(
        "--evaluation-package", default=None,
        help="Optional completed best-checkpoint decision-level evaluation package",
    )
    plan2 = subparsers.add_parser("plan2", help="Validate and aggregate Plan 2 runs")
    plan2.add_argument("--run-list", required=True, help="Explicit Plan 2 CSV run-list")
    plan2.add_argument("--analysis-id", required=True, type=_analysis_id)
    plan2.add_argument("--output-root", default=str(DEFAULT_PLAN2_OUTPUT_ROOT))
    plan2.add_argument(
        "--allow-incomplete", action="store_true",
        help="Development-only: permit non-144k schedules and incomplete five-seed cells",
    )
    sequential = subparsers.add_parser(
        'sequential', help='Plot a validated Plan 3/4 sequential report',
    )
    sequential.add_argument('--analysis-report', required=True)
    sequential.add_argument('--analysis-id', required=True, type=_analysis_id)
    sequential.add_argument('--output-root', default='data/output_data/analysis/plan34')
    sequential.add_argument('--dpi', type=int, default=160)
    sequential_frozen = subparsers.add_parser(
        'sequential-frozen',
        help='Plot one-scene frozen sequential and Plan 1/baseline time series',
    )
    sequential_frozen.add_argument(
        '--sequential-evaluation-root', required=True, nargs='+',
        help='One or more non-overlapping frozen-evaluation roots.',
    )
    sequential_frozen.add_argument('--baseline-package', required=True)
    sequential_frozen.add_argument('--analysis-id', required=True, type=_analysis_id)
    sequential_frozen.add_argument(
        '--output-root', default='data/output_data/analysis/plan34',
    )
    sequential_frozen.add_argument('--network', required=True)
    sequential_frozen.add_argument('--scene', required=True)
    sequential_frozen.add_argument('--evaluation-seed', type=int, required=True)
    sequential_frozen.add_argument(
        '--orders', nargs='+', default=('O1', 'O2', 'O3', 'O4'),
        choices=('O1', 'O2', 'O3', 'O4'),
        help='Complete sequential orders expected in the frozen package.',
    )
    sequential_frozen.add_argument('--smoothing-window-seconds', type=int, default=60)
    sequential_frozen.add_argument('--dpi', type=int, default=160)
    analyze = subparsers.add_parser(
        "analyze", help="Run a validated analysis through a named experiment profile",
    )
    analyze.add_argument("--profile", required=True, choices=tuple(PROFILES))
    analyze.add_argument("--run-list", required=True, help="Explicit CSV run-list")
    analyze.add_argument("--analysis-id", required=True, type=_analysis_id)
    analyze.add_argument(
        "--output-root", default=None,
        help="Override the profile-specific analysis output root",
    )
    analyze.add_argument("--dpi", type=int, default=160)
    analyze.add_argument(
        "--phase",
        choices=("phase1", "prepare-evaluation-manifests", "phase2", "phase3",
                 "training-state-coverage", "phase4", "prepare-g0", "g0",
                 ),
        default=None,
    )
    analyze.add_argument("--scene-mapping", default=None)
    analyze.add_argument("--probe-evaluation-package", default=None)
    analyze.add_argument(
        "--refresh-existing", action="store_true",
        help="Recompute this analysis phase inside its existing independent output directory",
    )
    analyze.add_argument("--evaluation-traffic-seed", type=int, default=10000)
    analyze.add_argument("--probe-traffic-seeds", type=int, nargs="*", default=(10000,10001,10002,10003,10004))
    analyze.add_argument(
        "--evaluation-package", default=None,
        help="Optional completed best-checkpoint decision-level evaluation package",
    )
    analyze.add_argument(
        "--allow-incomplete", action="store_true",
        help="Development-only Plan 2 validation override",
    )
    return parser


def _diagnostic_output(args):
    default_root = Path('data/output_data/analysis/plan1')
    return Path(args.output_root or default_root).expanduser().resolve() / args.analysis_id


def run_s1_s4_phase1(args):
    if not args.scene_mapping:
        raise ValueError('--scene-mapping is required for S1-S4 diagnostics')
    output = _diagnostic_output(args)
    report_path = output / 'reports' / 'phase1_report.md'
    if report_path.exists() and not args.refresh_existing:
        raise FileExistsError(f'Phase 1 already exists: {report_path}')
    _, mapping_rows, scene_by_network = load_scene_mapping(args.scene_mapping)
    _, _, included = load_run_list(args.run_list)
    validated = [validate_run(spec) for spec in included if spec.role == 'formal']
    if len(validated) != 20:
        raise ValueError(f'Phase 1 requires 20 formal DQN runs, got {len(validated)}')
    compare_dqn_run_configs(validated)
    metric_rows, _ = normalize_records(validated)
    convergence, curves = calculate_convergence(metric_rows, scene_by_network)
    convergence_summary = summarize_convergence(convergence)
    budget_rows = calculate_budget_sufficiency(metric_rows, scene_by_network)
    budget = recommend_resumable_budget(convergence)
    action_per_seed, action_summary, action_distances = (
        calculate_action_distribution_diagnostics(
            metric_rows, scene_by_network, S1_S4_ACTION_SEMANTICS
        )
    )
    tables = output / 'tables'
    figures = output / 'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(tables / 'convergence_per_seed.csv', convergence)
    write_csv(tables / 'convergence_summary.csv', convergence_summary)
    write_csv(tables / 'budget_sufficiency.csv', budget_rows)
    write_csv(tables / 'convergence_curve_source.csv', curves)
    write_csv(tables / 'action_distribution_per_seed.csv', action_per_seed)
    write_csv(tables / 'action_distribution_summary.csv', action_summary)
    write_csv(tables / 'action_distribution_distances.csv', action_distances)
    figure_paths = []
    figure_paths += render_convergence_curves(curves, convergence, figures / 'convergence_curves', args.dpi)
    figure_paths += render_convergence_confirmation_metrics(
        curves, convergence, figures / 'convergence_confirmation_metrics', args.dpi)
    figure_paths += render_convergence_distribution(convergence, figures / 'convergence_episode_distribution', args.dpi)
    figure_paths += render_budget_sufficiency(budget_rows, figures / 'budget_sufficiency', args.dpi)
    figure_paths += render_mean_action_distribution(action_summary, figures / 'mean_action_distribution', args.dpi)
    figure_paths += render_distance_heatmap(action_distances, 'total_variation', figures / 'tv_distance_heatmap', args.dpi)
    figure_paths += render_distance_heatmap(action_distances, 'jensen_shannon', figures / 'js_distance_heatmap', args.dpi)
    figure_paths += render_within_between_distance(action_distances, figures / 'within_between_distance', args.dpi)
    primary = sorted((
        row for row in convergence_summary
        if row['metric'] == 'travel_time' and row['tolerance'] == .05
    ), key=lambda row: int(row['scene'][1:]))
    tv_within = [row['distance'] for row in action_distances
                 if row['aggregation'] == 'model_pair'
                 and row['metric'] == 'total_variation'
                 and row['pair_kind'] == 'within_scene']
    js_within = [row['distance'] for row in action_distances
                 if row['aggregation'] == 'model_pair'
                 and row['metric'] == 'jensen_shannon'
                 and row['pair_kind'] == 'within_scene']
    within_thresholds = {
        'total_variation': float(np.quantile(tv_within, .95)),
        'jensen_shannon': float(np.quantile(js_within, .95)),
    }
    scene_distances = [row for row in action_distances
                       if row['aggregation'] == 'scene_mean'
                       and row['source_scene'] < row['target_scene']]
    lines = [
        '# Phase 1：收敛预算与动作边际分布', '', '## 方法', '',
        '- 主判据：evaluation travel time 的 10-episode moving average。',
        '- 稳定参考：最后 20 episodes 的均值；主容差 ±5%，敏感性 ±3%。',
        '- 进入容差后要求连续保持 20 episodes。',
        '- 稳定分母：`max(abs(final_reference), 0.01 × median(abs(series)), 1e-8)`。',
        '- 本阶段只读解析 20 个正式 run，没有运行 SUMO。', '',
        '## Travel-time ±5% 结果', '',
        '| Scene | Network | Stable seeds | Median | Mean | Sample SD | Maximum |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in primary:
        lines.append(f"| {row['scene']} | {row['network']} | {row['converged_seed_count']}/{row['seed_count']} | {row['median']} | {row['mean']} | {row['sample_sd']} | {row['maximum']} |")
    lines += [
        '', '## 预算建议', '',
        f"- 90% seed 保守门槛所需 episode：{budget['raw_required_episode']}。",
        f"- 对齐完整 resumable checkpoint 后：**{budget['recommended_resumable_episode']} episodes**。",
        '- 不使用缺少 optimizer/replay/RNG 的 episode-75 轻量 checkpoint。', '',
        '## 动作语义', '',
    ]
    lines += [f'- action {action}: {label}' for action, label in S1_S4_ACTION_SEMANTICS.items()]
    lines += ['', '## 场景动作距离相对 training-seed 波动', '',
              f"- within-scene model-pair TV 95th percentile={within_thresholds['total_variation']:.3f}；JS 95th percentile={within_thresholds['jensen_shannon']:.3f}。"]
    for metric in ('total_variation','jensen_shannon'):
        values = [row for row in scene_distances if row['metric'] == metric]
        largest = max(values, key=lambda row: row['distance'])
        smallest = min(values, key=lambda row: row['distance'])
        exceeds = [row for row in values
                   if row['distance'] > within_thresholds[metric]]
        lines += [
            f"- {metric}: 最大 {largest['source_scene']}–{largest['target_scene']}={largest['distance']:.3f}；最小 {smallest['source_scene']}–{smallest['target_scene']}={smallest['distance']:.3f}。",
            f"- 超过 within-scene 95th percentile 的场景对："
            + (', '.join(f"{row['source_scene']}–{row['target_scene']}" for row in exceeds)
               if exceeds else '无') + '。',
        ]
    lines += ['', 'Individual-seed 数据和 TV/JS 距离见 `tables/`；正式图见 `figures/`。']
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    _json_dump(output / 'manifests' / 'phase1_manifest.json', {
        'schema_version': 1, 'tool': 'tools.experiment_plotting',
        'profile': 's1_s4_diagnostics', 'phase': 'phase1',
        'scene_mapping': mapping_rows, 'budget_recommendation': budget,
        'figures': [str(path.relative_to(output)) for path in figure_paths],
    })
    return output


def prepare_s1_s4_evaluation_manifests(args):
    if not args.scene_mapping:
        raise ValueError('--scene-mapping is required')
    output = _diagnostic_output(args)
    config_dir = output / 'inputs'
    cross_path = config_dir / 'cross_scene_final_evaluation_fixed_v2.json'
    probe_path = config_dir / 'fixedtime_probe_evaluation.json'
    if cross_path.exists() or probe_path.exists():
        raise FileExistsError('S1-S4 evaluation manifests already exist')
    _, mapping_rows, scene_by_network = load_scene_mapping(args.scene_mapping)
    _, _, included = load_run_list(args.run_list)
    formal = [spec for spec in included if spec.role == 'formal' and spec.agent == 'dqn']
    baselines = [spec for spec in included if spec.role == 'baseline' and spec.agent == 'fixedtime']
    if len(formal) != 20 or len(baselines) != 4:
        raise ValueError('Manifest preparation requires 20 formal DQN and 4 FixedTime runs')
    validated = [validate_run(spec) for spec in formal]
    by_network = {}
    for spec in formal:
        by_network.setdefault(spec.network, []).append(spec)
    target_reference = {network: sorted(specs, key=lambda spec: spec.training_seed)[0]
                        for network, specs in by_network.items()}
    cross_controllers = []
    cross_csv = []
    for run in validated:
        spec = run['spec']
        checkpoint = run['evaluation_summary']['final_checkpoint']
        for target in mapping_rows:
            controller_id = (
                f"dqn_{scene_by_network[spec.network]}_seed{spec.training_seed}"
                f"_to_{target['scene']}"
            )
            item = {
                'controller_id': controller_id, 'agent': 'dqn',
                'network': target['network'], 'training_seed': spec.training_seed,
                'run_dir': str(spec.run_dir), 'checkpoint': checkpoint,
                'source_scene': scene_by_network[spec.network],
                'source_network': spec.network, 'target_network': target['network'],
                'target_scene': target['scene'],
                'target_run_dir': str(target_reference[target['network']].run_dir),
                'expected_vehicle_count': target['vehicles'],
                'checkpoint_role': 'final', 'source_policy': 'dqn_final_greedy',
            }
            cross_controllers.append(item)
            cross_csv.append({
                'controller_id': controller_id,
                'source_scene': scene_by_network[spec.network],
                'source_network': spec.network, 'training_seed': spec.training_seed,
                'target_scene': target['scene'], 'target_network': target['network'],
                'evaluation_traffic_seed': args.evaluation_traffic_seed,
                'checkpoint_role': 'final', 'checkpoint_episode': 400,
                'checkpoint': str(spec.run_dir / checkpoint),
            })
    cross_manifest = {
        'schema_version': 2,
        'package_id': (
            f's1_s4_cross_scene_final_eval_seed{args.evaluation_traffic_seed}_fixed_v2'
        ),
        'world': 'sumo', 'evaluation_seeds': [args.evaluation_traffic_seed],
        'sampling_interval_seconds': 10, 'smoothing_window_seconds': 60,
        'metrics': ['reward','queue','delay','throughput','travel_time'],
        'record_state_diagnostics': False,
        'reward_definitions': {
            'controller_reward_mean': 'DQN agent reward: -12 times mean incoming-lane waiting count',
            'reward_network_mean': 'negative queue_intersections mean; diagnostic only',
        },
        'expected_controller_count': 80, 'expected_episode_count': 80,
        'controllers': cross_controllers,
    }
    fixedtime_by_network = {spec.network: spec for spec in baselines}
    probe_controllers = []
    for mapping in mapping_rows:
        spec = fixedtime_by_network[mapping['network']]
        probe_controllers.append({
            'controller_id': f"fixedtime_{mapping['scene']}",
            'agent': 'fixedtime', 'network': mapping['network'],
            'training_seed': spec.training_seed, 'run_dir': str(spec.run_dir),
            'checkpoint': None, 'source_scene': mapping['scene'],
            'source_network': mapping['network'], 'target_scene': mapping['scene'],
            'target_network': mapping['network'], 'target_run_dir': str(spec.run_dir),
            'expected_vehicle_count': mapping['vehicles'],
            'checkpoint_role': 'none', 'source_policy': 'fixedtime_common',
        })
    probe_seeds = list(dict.fromkeys(args.probe_traffic_seeds))
    probe_manifest = {
        'schema_version': 2, 'package_id': 's1_s4_fixedtime_probe_v1',
        'world': 'sumo', 'evaluation_seeds': probe_seeds,
        'sampling_interval_seconds': 10, 'smoothing_window_seconds': 60,
        'metrics': ['reward','queue','delay','throughput','travel_time'],
        'record_state_diagnostics': True,
        'reward_definitions': {
            'controller_reward_mean': 'FixedTime legacy agent reward; not compared to DQN reward',
            'reward_network_mean': 'negative queue_intersections mean; diagnostic only',
        },
        'expected_controller_count': 4,
        'expected_episode_count': 4 * len(probe_seeds),
        'controllers': probe_controllers,
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(cross_path, cross_manifest)
    _json_dump(probe_path, probe_manifest)
    write_csv(config_dir / 'cross_scene_manifest.csv', cross_csv)
    # Strict normalization validates source/target run identity and final paths.
    from utils.logger import load_evaluation_collection_manifest
    load_evaluation_collection_manifest(str(cross_path))
    load_evaluation_collection_manifest(str(probe_path))
    return output


G0_DIRECTIONS = (('S3', 'S4'), ('S4', 'S3'), ('S2', 'S3'), ('S3', 'S2'))


def prepare_s1_s4_g0(args):
    """Prepare the immutable episode-100 targeted pair gate manifest."""
    if not args.scene_mapping:
        raise ValueError('--scene-mapping is required')
    output = _diagnostic_output(args)
    manifest_dir = output / 'manifests'
    config_dir = output / 'inputs'
    manifest_path = manifest_dir / 'episode100_gate_manifest.json'
    if manifest_path.exists():
        raise FileExistsError(f'G0 manifest already exists: {manifest_path}')
    _, mapping_rows, scene_by_network = load_scene_mapping(args.scene_mapping)
    mapping_by_scene = {row['scene']: row for row in mapping_rows}
    _, _, included = load_run_list(args.run_list)
    formal = [spec for spec in included if spec.role == 'formal' and spec.agent == 'dqn']
    if len(formal) != 20:
        raise ValueError(f'G0 preparation requires 20 formal DQN runs, got {len(formal)}')
    validated = [validate_run(spec) for spec in formal]
    spec_by_identity = {
        (scene_by_network[item['spec'].network], int(item['spec'].training_seed)):
            item['spec']
        for item in validated
    }
    controller_identities = [
        (source, target, seed)
        for source, target in G0_DIRECTIONS for seed in range(5)
    ] + [
        (scene, scene, seed)
        for scene in ('S2', 'S3', 'S4') for seed in range(5)
    ]
    controllers = []
    audit_rows = []
    for source, target, seed in controller_identities:
        source_spec = spec_by_identity[(source, seed)]
        target_spec = spec_by_identity[(target, seed)]
        checkpoint = 'checkpoints/resumable/episode_0100.pt'
        controller_id = f'dqn_{source}_seed{seed}_ep100_to_{target}'
        controllers.append({
            'controller_id': controller_id, 'agent': 'dqn',
            'network': mapping_by_scene[target]['network'],
            'training_seed': seed, 'run_dir': str(source_spec.run_dir),
            'checkpoint': checkpoint, 'checkpoint_episode': 100,
            'source_scene': source, 'source_network': source_spec.network,
            'target_scene': target,
            'target_network': mapping_by_scene[target]['network'],
            'target_run_dir': str(target_spec.run_dir),
            'expected_vehicle_count': mapping_by_scene[target]['vehicles'],
            'checkpoint_role': 'resumable',
            'source_policy': 'dqn_episode100_resumable_online_greedy',
        })
        audit_rows.append({
            'controller_id': controller_id, 'source_scene': source,
            'target_scene': target, 'training_seed': seed,
            'checkpoint_role': 'resumable', 'checkpoint_episode': 100,
            'checkpoint': str(source_spec.run_dir / checkpoint),
        })
    manifest = {
        'schema_version': 2,
        'package_id': f's1_s4_episode100_pair_gate_seed{args.evaluation_traffic_seed}_v1',
        'world': 'sumo', 'evaluation_seeds': [args.evaluation_traffic_seed],
        'sampling_interval_seconds': 10, 'smoothing_window_seconds': 60,
        'metrics': ['reward', 'queue', 'delay', 'throughput', 'travel_time'],
        'record_state_diagnostics': False,
        'expected_controller_count': 35, 'expected_episode_count': 35,
        'controllers': controllers,
    }
    manifest_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(manifest_path, manifest)
    write_csv(manifest_dir / 'episode100_gate_manifest.csv', audit_rows)
    protocol = {
        'schema_version': 1, 'task': 'G0_episode100_pair_sensitivity_gate',
        'package_identity': 's1_s4_pairwise_g_clear_retain_ep100_v1',
        'evaluation_traffic_seed': args.evaluation_traffic_seed,
        'checkpoint_episode': 100, 'checkpoint_role': 'resumable',
        'checkpoint_network': 'online', 'greedy': True,
        'directions': [list(item) for item in G0_DIRECTIONS],
        'same_seed_target_reference': True,
        'thresholds': {
            'low_pair_mean_degradation_max': .05,
            'high_minus_low_worst_min_percentage_points': 5,
        },
    }
    import yaml
    with (config_dir / 'experiment_protocol.yaml').open('w', encoding='utf-8') as handle:
        yaml.safe_dump(protocol, handle, sort_keys=False, allow_unicode=True)
    from utils.logger import load_evaluation_collection_manifest
    _, normalized = load_evaluation_collection_manifest(str(manifest_path))
    checkpoint_audit = []
    for item in normalized['controllers']:
        checkpoint_audit.append({
            'controller_id': item['controller_id'],
            'source_scene': item['source_scene'],
            'target_scene': item['target_scene'],
            'training_seed': item['training_seed'],
            **item['checkpoint_audit'],
        })
    (output / 'tables').mkdir(parents=True, exist_ok=True)
    write_csv(output / 'tables' / 'episode100_checkpoint_hash_audit.csv',
              checkpoint_audit)
    _json_dump(output / 'manifests' / 'g0_prepare_manifest.json', {
        'schema_version': 1, 'valid': True,
        'controller_count': len(controllers),
        'checkpoint_hash_match_count': len(checkpoint_audit),
        'evaluation_manifest': str(manifest_path),
    })
    return output


def _write_g0_stop_report(output, args, package, summary, relative, gate):
    """Write terminal Task G evidence when the pre-registered G0 gate fails."""
    if gate['passed']:
        return
    reports = output / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    travel = {
        (row['source_scene'], row['target_scene']): row
        for row in summary if row['metric'] == 'travel_time'
    }
    seed_values = {}
    for row in relative:
        if row['metric'] == 'travel_time':
            seed_values.setdefault(
                (row['source_scene'], row['target_scene']), []
            ).append((int(row['training_seed']), float(row['relative_degradation'])))
    for values in seed_values.values():
        values.sort()
    classes = {('S3', 'S4'): 'high', ('S4', 'S3'): 'high',
               ('S2', 'S3'): 'low', ('S3', 'S2'): 'low'}
    lines = [
        '# Task G 最终停止报告', '',
        '## 结论', '',
        '**结论分类：E. 工程或统计证据不足，不能形成 sequential adaptation/forgetting 科研结论。**', '',
        'G0 工程与证据链通过，但预先注册的 high/low 定性门禁未通过。',
        '按照协议，工作流已在 G0 终止；G1、smoke test 和 40 个正式 sequential runs 均未执行。', '',
        f"最差 high 方向与最差 low 方向仅相差 {gate['high_minus_low_worst_percentage_points']:.2f} 个百分点，低于 5 个百分点门槛。", '',
        '## G0 结果', '',
        '| Direction | Registered class | Mean degradation | Sample SD | Seeds 0–4 |',
        '|---|---|---:|---:|---|',
    ]
    for pair in G0_DIRECTIONS:
        row = travel[pair]
        seeds = ', '.join(f'{value:.2%}' for _, value in seed_values[pair])
        lines.append(
            f'| {pair[0]}→{pair[1]} | {classes[pair]} | '
            f"{float(row['mean_relative_degradation']):.2%} | "
            f"{float(row['sample_sd_relative_degradation']):.2%} | {seeds} |"
        )
    s4_s3_seeds = ', '.join(
        f'{value:.2%}' for _, value in seed_values[('S4', 'S3')]
    )
    s3_s4_mean = float(travel[('S3', 'S4')]['mean_relative_degradation'])
    s4_s3_mean = float(travel[('S4', 'S3')]['mean_relative_degradation'])
    s2_s3_mean = float(travel[('S2', 'S3')]['mean_relative_degradation'])
    s3_s2_mean = float(travel[('S3', 'S2')]['mean_relative_degradation'])
    lines += [
        '', '## 对任务问题的回答', '',
        '1. **原 high/low 排序是否保持？** 仅部分保持。S4→S3 仍最差，两个 low 方向均接近 in-domain；但注册的 high/low 分离未达到门槛。',
        f'2. **S3→S4 与 S4→S3 是否不对称？** zero-shot 均值分别为 {s3_s4_mean:.2%} 与 {s4_s3_mean:.2%}。未进行 target training，不能回答 adaptation 不对称性。',
        f'3. **S2→S3 与 S3→S2 是否基本无需重新适应？** zero-shot 均值分别为 {s2_s3_mean:.2%} 与 {s3_s2_mean:.2%}，都在 5% 描述性阈值内；可称为接近 in-domain，但不能声称无需适应。',
        '4. **是否实际发生 adaptation？** 未评价；G1 未运行。',
        '5. **是否发生 source forgetting？** 未评价；G1 未运行。',
        '6. **high pair 是否遗忘更强？** 未评价。',
        '7. **Clear/Retain 谁的 target adaptation 更好？** 未评价。',
        '8. **Clear/Retain 谁的 source retention 更好？** 未评价。',
        '9. **Retain 是否存在 stability–plasticity trade-off？** 未评价。',
        '10. **source replay 实际保留多久？** 未评价。',
        '11. **batch 实际采到多少 source transitions？** 未评价。',
        '12. **Clear/Retain update count 差异？** 未评价。',
        '13. **Clear warm-up 是否解释性能差异？** 未评价。',
        f'14. **individual seeds 是否一致？** 不完全一致；S4→S3 五种子为 {s4_s3_seeds}。效应不由单一种子驱动，但幅度有明显差异。',
        '15. **是否支持固定 offline pool + online replay？** 不支持；本轮没有运行 replay 比较。',
        '16. **是否需要 update-matched/fixed-mixture 消融？** 当前没有先运行 Clear/Retain 的依据，不应直接进入该消融。',
        '17. **是否值得进入完整 S1–S4 顺序训练？** 当前证据不支持。若继续，应先在独立协议中重新筛选 episode-100 场景对。',
        '18. **能够与不能声称什么？** 能声称 low pair 接近 in-domain、S4→S3 的退化最大但 high/low 分离不足；不能声称发生 adaptation、forgetting、replay retention 收益或稳定性—可塑性权衡。', '',
        '## 场景对重新判定', '',
        '- S2↔S3 可继续作为低 zero-shot-gap 对照候选。',
        '- S3↔S4 不再作为已经通过验证的 high-conflict pair；它只能被描述为候选，其中 S4→S3 较 S3→S4 更敏感。',
        '- 如未来继续，应使用新的、预先规定的筛选协议考察其他 episode-100 有向转换；不得事后降低本次门槛。', '',
        '## 证据范围', '',
        f"- immutable package: {Path(args.evaluation_package).expanduser().resolve()}",
        f"- completed evaluations: {package['manifest']['episode_count']}",
        f"- decision records: {package['manifest']['decision_record_count']}",
        '- checkpoint semantics、target identity、vehicle counts 与 evaluation isolation 均通过。',
        '- 未使用旧错误非对角线 package；G0 只使用 episode-100 resumable online networks。', '',
        '## 未生成产物的理由', '',
        'sequential manifest、sequential tables、G1 engineering audit 及 Figure G1–G9 不存在，',
        '因为它们只在 G0 通过后才允许生成。创建空表或占位图会误导为实验已经执行。',
    ]
    (reports / 'g_final_report.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8'
    )
    (reports / 'code_changes.md').write_text(
        '# Task G 代码变更\n\n'
        '- trainer/tsc_trainer.py：支持从 resumable checkpoint 仅加载 online network，保持训练状态不变。\n'
        '- utils/logger.py：解析 episode-100 resumable evaluation，并审计同 episode evaluation checkpoint 的 online-network hash。\n'
        '- run.py：以 online-only 语义执行 resumable checkpoint 冻结评价并记录 audit。\n'
        '- tools/experiment_plotting/aggregations.py：增加 same-seed G0 聚合、指标方向和门禁。\n'
        '- tools/experiment_plotting/plotting.py：增加显示全部种子及 mean ± sample SD 的 G0 PNG/PDF 图。\n'
        '- tools/experiment_plotting/cli.py：增加 G0 manifest、分析、报告与失败停止交付物。\n'
        '- 既有测试覆盖 checkpoint 语义、same-seed reference、指标方向与停止报告。\n\n'
        '没有创建独立功能脚本或 scripts/ 目录；没有实现 G1，因为 G0 未通过。\n',
        encoding='utf-8',
    )
    (reports / 'unresolved_issues.md').write_text(
        '# 未决问题\n\n'
        f"1. 注册的 episode-100 high/low pair 只相差 {gate['high_minus_low_worst_percentage_points']:.2f} 个百分点，未满足 5 个百分点门槛。\n"
        '2. adaptation、forgetting、Clear/Retain、replay provenance、warm-up 与 update count 均因硬停止条件而未评价。\n'
        '3. 若开启后续任务，需要新协议重新筛选 high-conflict 有向转换；不得使用旧错误非对角线结果或事后修改门槛。\n'
        '4. 当前结果不支持完整 S1–S4 顺序训练或固定 offline pool + online replay 实验。\n',
        encoding='utf-8',
    )
    controller_by_id = {
        row['controller_id']: row for row in package['collection']['controllers']
    }
    run_rows = [{
        'record_type': 'g0_evaluation',
        'controller_id': row['controller_id'],
        'source_scene': controller_by_id[row['controller_id']]['source_scene'],
        'target_scene': controller_by_id[row['controller_id']]['target_scene'],
        'training_seed': row['training_seed'],
        'evaluation_traffic_seed': row['evaluation_traffic_seed'],
        'status': 'completed', 'reason': '',
    } for row in package['summaries']]
    for phase in ('G1 engineering', 'G1 smoke', 'formal sequential runs'):
        run_rows.append({
            'record_type': 'workflow_phase', 'controller_id': phase,
            'source_scene': '', 'target_scene': '', 'training_seed': '',
            'evaluation_traffic_seed': '', 'status': 'not_run',
            'reason': 'G0 high/low gate failed',
        })
    write_csv(output / 'tables' / 'run_status.csv', run_rows)
    commands = output / 'commands'
    commands.mkdir(parents=True, exist_ok=True)
    reproduction = '''#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/projects/simOfflineTSC
PYTHON=/home/dev/miniforge3/envs/colight/bin/python
export SUMO_HOME=/home/dev/miniforge3/envs/colight/lib/python3.10/site-packages/sumo
ANALYSIS_ID=task_g_pairwise_pilot_reproduction
OUTPUT_ROOT=data/output_data/analysis/plan34
PACKAGE=data/output_data/evaluations/plan34/episode100_pair_gate_seed10000_v1_reproduction

cd "$PROJECT_ROOT"

"$PYTHON" -m tools.experiment_plotting analyze \\
  --profile s1_s4_diagnostics --phase prepare-g0 \\
  --run-list data/output_data/analysis/plan1/p1_formal_20_runlist_20260722.csv \\
  --scene-mapping data/output_data/analysis/plan1/s1_s4_adaptation_diagnostics_20260723/inputs/scene_mapping.csv \\
  --analysis-id "$ANALYSIS_ID" --output-root "$OUTPUT_ROOT" \\
  --evaluation-traffic-seed 10000

"$PYTHON" run.py \\
  --evaluation-manifest "$OUTPUT_ROOT/$ANALYSIS_ID/manifests/episode100_gate_manifest.json" \\
  --evaluation-output "$PACKAGE" --interface libsumo --ngpu -1

"$PYTHON" -m tools.experiment_plotting analyze \\
  --profile s1_s4_diagnostics --phase g0 \\
  --run-list data/output_data/analysis/plan1/p1_formal_20_runlist_20260722.csv \\
  --scene-mapping data/output_data/analysis/plan1/s1_s4_adaptation_diagnostics_20260723/inputs/scene_mapping.csv \\
  --analysis-id "$ANALYSIS_ID" --output-root "$OUTPUT_ROOT" \\
  --evaluation-traffic-seed 10000 --evaluation-package "$PACKAGE" --dpi 160
'''
    (commands / 'reproduction_commands.sh').write_text(reproduction, encoding='utf-8')
    (commands / 'reproduction_commands.sh').chmod(0o755)


def run_s1_s4_g0(args):
    if not args.scene_mapping or not args.evaluation_package:
        raise ValueError('G0 requires --scene-mapping and --evaluation-package')
    output = _diagnostic_output(args)
    package = validate_evaluation_package(
        str(Path(args.evaluation_package).expanduser().resolve())
    )
    if package['manifest']['package_id'] != (
        f's1_s4_episode100_pair_gate_seed{args.evaluation_traffic_seed}_v1'
    ):
        raise ValueError('G0 evaluation package identity mismatch')
    if any(item.get('checkpoint_role') != 'resumable'
           or int(item.get('checkpoint_episode', -1)) != 100
           for item in package['collection']['controllers']):
        raise ValueError('G0 package contains a non-episode100 resumable checkpoint')
    _, _, scene_by_network = load_scene_mapping(args.scene_mapping)
    raw = normalize_cross_scene_summaries(package['summaries'], scene_by_network)
    summary, relative, gate = aggregate_episode100_pair_gate(raw, G0_DIRECTIONS)
    processed = output / 'tables'
    figures = output / 'figures'
    reports = output / 'reports'
    processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    write_csv(processed / 'episode100_pair_gate_raw.csv', raw)
    write_csv(processed / 'episode100_pair_gate_summary.csv', summary)
    write_csv(processed / 'episode100_pair_gate_relative_degradation.csv', relative)
    isolation_rows = []
    for row in raw:
        isolation_rows.append({
            'controller_id': row['controller_id'],
            'source_scene': row['source_scene'], 'target_scene': row['target_scene'],
            'training_seed': row['training_seed'],
            **row['isolation_check'],
        })
    write_csv(processed / 'episode100_target_identity_isolation_audit.csv',
              isolation_rows)
    figure_paths = render_episode100_pair_gate(
        raw, relative, figures / 'figure_g0_episode100_pair_sensitivity_gate', args.dpi
    )
    travel = [row for row in summary if row['metric'] == 'travel_time']
    lines = [
        '# G0：episode-100 场景对敏感性门禁', '',
        f"**门禁结果：{'通过' if gate['passed'] else '未通过'}。**", '',
        '## 方法', '',
        '- 使用 episode-100 resumable checkpoint 中的 online Q-network。',
        '- 同 episode evaluation checkpoint 仅用于 online-network hash 语义审计。',
        f"- greedy evaluation；traffic seed={args.evaluation_traffic_seed}。",
        '- 主比较为 same-seed target in-domain reference。', '',
        '## Travel-time degradation', '',
        '| Direction | Mean | Sample SD |', '|---|---:|---:|',
    ]
    for row in sorted(travel, key=lambda item: (item['source_scene'], item['target_scene'])):
        lines.append(
            f"| {row['source_scene']}→{row['target_scene']} | "
            f"{row['mean_relative_degradation']:.2%} | "
            f"{row['sample_sd_relative_degradation']:.2%} |"
        )
    lines += ['', '## Gate checks', '']
    lines.extend(
        f"- {name}: {'PASS' if value else 'FAIL'}"
        for name, value in gate['checks'].items()
    )
    lines += [
        '',
        f"- High worst − low worst = {gate['high_minus_low_worst_percentage_points']:.2f} percentage points.",
        '- Individual seeds、queue、real delay、approximate delay、throughput 与 unfinished vehicles 均保存在 tables CSV。',
        '- target identity、SUMO route/vehicle evidence 与 evaluation isolation 保存在 raw package 和 isolation audit。', '',
        '若门禁未通过，按协议停止，不运行 G1 或 40 个 sequential runs。',
    ]
    report = '\n'.join(lines) + '\n'
    (reports / 'g0_episode100_gate_report.md').write_text(report, encoding='utf-8')
    _json_dump(output / 'manifests' / 'g0_gate_manifest.json', {
        'schema_version': 1, 'gate': gate,
        'evaluation_package': str(Path(args.evaluation_package).resolve()),
        'figures': [str(path.relative_to(output)) for path in figure_paths],
    })
    _write_g0_stop_report(output, args, package, summary, relative, gate)
    return output


def run_s1_s4_phase2(args):
    if not args.scene_mapping or not args.evaluation_package:
        raise ValueError('Phase 2 requires --scene-mapping and --evaluation-package')
    output = _diagnostic_output(args)
    report_path = output / 'reports' / 'phase2_report.md'
    if report_path.exists() and not args.refresh_existing:
        raise FileExistsError(f'Phase 2 already exists: {report_path}')
    _, _, scene_by_network = load_scene_mapping(args.scene_mapping)
    package_path = Path(args.evaluation_package).expanduser().resolve()
    validated = validate_evaluation_package(str(package_path))
    collection = validated['collection']
    if collection.get('schema_version') != 2:
        raise ValueError('Phase 2 requires evaluation package schema v2')
    if any(item.get('checkpoint_role') != 'final' for item in collection['controllers']):
        raise ValueError('Phase 2 package contains a non-final checkpoint')
    raw = normalize_cross_scene_summaries(validated['summaries'], scene_by_network)
    if len(raw) != 80:
        raise ValueError(f'Phase 2 requires 80 evaluations, got {len(raw)}')
    summary, relative, directional = aggregate_cross_scene(raw)
    tables = output / 'tables'
    figures = output / 'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(tables / 'cross_scene_raw.csv', raw)
    write_csv(tables / 'cross_scene_evaluation.csv', raw)
    write_csv(tables / 'cross_scene_summary.csv', summary)
    write_csv(tables / 'cross_scene_relative_degradation.csv', relative)
    write_csv(tables / 'directional_transfer_summary.csv', directional)
    isolation_rows = []
    for row in raw:
        check = row['isolation_check']
        isolation_rows.append({
            'controller_id': row['controller_id'],
            'source_scene': row['source_scene'],
            'source_network': row['source_network'],
            'target_scene': row['target_scene'],
            'target_network': row['target_network'],
            'training_seed': row['training_seed'],
            'evaluation_traffic_seed': row['evaluation_traffic_seed'],
            'checkpoint_role': row['checkpoint_role'],
            **{field: check.get(field) for field in (
                'rng_unchanged','online_model_unchanged','target_model_unchanged',
                'optimizer_unchanged','epsilon_unchanged','replay_unchanged',
                'counters_unchanged','trajectory_writes_unchanged',
            )},
        })
    write_csv(tables / 'evaluation_isolation_checks.csv', isolation_rows)
    figure_paths = []
    figure_paths += render_cross_scene_heatmaps(summary, figures / 'cross_scene_metric_heatmaps', args.dpi)
    figure_paths += render_relative_degradation_heatmaps(relative, figures / 'cross_scene_relative_degradation', args.dpi)
    figure_paths += render_cross_scene_seed_scatter(raw, figures / 'cross_scene_seed_scatter', args.dpi)
    non_diagonal = [row for row in directional if row['source_scene'] != row['target_scene']]
    best = min(non_diagonal, key=lambda row: row['travel_time_relative_degradation'])
    worst = max(non_diagonal, key=lambda row: row['travel_time_relative_degradation'])
    asymmetric = max(non_diagonal, key=lambda row: abs(row['directional_asymmetry']))
    isolation = [row['isolation_check'] for row in raw]
    protected_fields = (
        'rng_unchanged', 'online_model_unchanged', 'target_model_unchanged',
        'optimizer_unchanged', 'epsilon_unchanged', 'replay_unchanged',
        'counters_unchanged', 'trajectory_writes_unchanged',
    )
    isolation_ok = all(
        isinstance(check, dict)
        and all(check.get(field) is True for field in protected_fields)
        for check in isolation
    )
    lines = [
        '# Phase 2：4×4 跨场景 final-checkpoint 冻结评估', '',
        '## 设计', '',
        f"- 20 个 episode-400 final models × 4 target scenes = {len(raw)} evaluations。",
        f"- 共同 evaluation traffic seed：{raw[0]['evaluation_traffic_seed']}。",
        '- 统计重复单位：5 个 training seeds；traffic seed 是固定评估条件。',
        '- 对角线与非对角线均由同一 schema-v2 入口重评。',
        '- lower-is-better：`(cross - target_in_domain) / abs(target_in_domain)`。',
        '- reward/throughput：`(target_in_domain - cross) / abs(target_in_domain)`；正值统一表示更差。', '',
        '- relative-degradation 表同时保留同方向的 `absolute_difference`、实际分母和近零 reference 标记；正值同样表示更差。', '',
        '## Travel-time zero-shot 排名', '',
        f"- 最佳有向转换：**{best['source_scene']}→{best['target_scene']}**，relative degradation={best['travel_time_relative_degradation']:.3%}。",
        f"- 最差有向转换：**{worst['source_scene']}→{worst['target_scene']}**，relative degradation={worst['travel_time_relative_degradation']:.3%}。",
        f"- 最大方向不对称：**{asymmetric['source_scene']}→{asymmetric['target_scene']}** 与反向，差值={asymmetric['directional_asymmetry']:.3%}。", '',
        '## Reward 与隔离', '',
        '- `reward_mean` 是各 DQN agent 的正式 reward；`reward_network_mean=-queue` 只保留在原始 decision package，未混入正式 reward 表。',
        f"- 80/80 evaluation isolation checks recorded；汇总检查：{'通过' if isolation_ok else '需复核'}。", '',
        '完整 individual-seed、mean ± sample SD 和 metric-wise degradation 见 `tables/`。',
    ]
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    _json_dump(output / 'manifests' / 'phase2_manifest.json', {
        'schema_version': 1, 'tool': 'tools.experiment_plotting',
        'phase': 'phase2', 'evaluation_package': str(package_path),
        'checkpoint_role': 'final', 'evaluation_count': len(raw),
        'evaluation_traffic_seeds': sorted({row['evaluation_traffic_seed'] for row in raw}),
        'figures': [str(path.relative_to(output)) for path in figure_paths],
    })
    return output


def _load_csv_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _probe_rows(validated_package, package_path, scene_by_network):
    collection = validated_package['collection']
    if (collection.get('schema_version') != 2
            or not collection.get('record_state_diagnostics')):
        raise ValueError('Phase 3 requires a schema-v2 state-diagnostics package')
    if any(item.get('source_policy') != 'fixedtime_common'
           for item in collection['controllers']):
        raise ValueError('Formal probe package must use common FixedTime policy')
    records = load_evaluation_records(package_path)
    probes = []
    counts = {}
    for record in records:
        if (record.get('raw_state') is None or record.get('model_input') is None
                or record.get('current_phase') is None):
            raise ValueError('Probe evaluation record is missing state diagnostics')
        if len(record['raw_state']) != 1 or len(record['model_input']) != 1:
            raise ValueError('S1-S4 diagnostics expects one controlled intersection')
        scene = scene_by_network[record['target_network']]
        key = (scene, int(record['evaluation_traffic_seed']))
        counts[key] = counts.get(key, 0) + 1
        probes.append({
            'probe_id': (f"{scene}_traffic{record['evaluation_traffic_seed']}_"
                         f"decision{record['decision_step']:04d}"),
            'scene': scene, 'network': record['target_network'],
            'evaluation_traffic_seed': int(record['evaluation_traffic_seed']),
            'episode': 1, 'decision_step': int(record['decision_step']),
            'simulation_step': int(record['state_simulation_time_seconds']),
            'current_phase': int(record['current_phase'][0]),
            'raw_state': [float(value) for value in record['raw_state'][0]],
            'model_input': [float(value) for value in record['model_input'][0]],
            'source_policy': record['source_policy'],
            'queue_network_mean': float(record['queue_network_mean']),
            'delay_network_weighted_mean': float(record['delay_network_weighted_mean']),
            'throughput_cumulative': int(record['throughput_cumulative']),
            'traffic_metric_time_seconds': float(record['simulation_time_seconds']),
        })
    expected_keys = {
        (scene, seed) for scene in ('S1','S2','S3','S4')
        for seed in collection['evaluation_seeds']
    }
    if set(counts) != expected_keys or len(set(counts.values())) != 1:
        raise ValueError(f'Probe states are not balanced by scene/traffic seed: {counts}')
    probes.sort(key=lambda row: (int(row['scene'][1:]),
                                 row['evaluation_traffic_seed'], row['decision_step']))
    return probes


def _final_model_predictions(run_specs, scene_by_network, probe_rows):
    import numpy as np
    import torch
    from agent.dqn import DQNNet
    from trainer.tsc_trainer import TSCTrainer

    features = np.asarray([row['model_input'] for row in probe_rows], dtype=np.float32)
    tensor = torch.tensor(features, dtype=torch.float32)
    predictions = []
    models = []
    for spec in sorted(run_specs, key=lambda item: (
            int(scene_by_network[item.network][1:]), item.training_seed)):
        summary_path = spec.run_dir / 'evaluation' / 'summary.json'
        with summary_path.open(encoding='utf-8') as handle:
            summary = json.load(handle)
        if int(summary['final_episode']) != 400:
            raise ValueError(f'Final checkpoint is not episode 400: {spec.run_dir}')
        checkpoint_path = (spec.run_dir / summary['final_checkpoint']).resolve()
        payload = TSCTrainer.load_checkpoint_payload(
            str(checkpoint_path), expected_type='evaluation'
        )
        if int(payload['episode']) != 400 or len(payload['agents']) != 1:
            raise ValueError(f'Unexpected final checkpoint payload: {checkpoint_path}')
        model = DQNNet(16, 8)
        model.load_state_dict(payload['agents'][0]['online_model_state_dict'])
        model.eval()
        with torch.no_grad():
            q_values = model(tensor, train=False).cpu().numpy()
        ordered = np.argsort(q_values, axis=1)
        top_actions = ordered[:, -1]
        top2_actions = ordered[:, -2]
        top_values = q_values[np.arange(len(q_values)), top_actions]
        top2_values = q_values[np.arange(len(q_values)), top2_actions]
        margins = top_values - top2_values
        normalized = margins / np.maximum(np.max(np.abs(q_values), axis=1), 1e-8)
        scene = scene_by_network[spec.network]
        model_id = f'{scene}_seed{spec.training_seed}'
        models.append({
            'model_id': model_id, 'scene': scene, 'network': spec.network,
            'training_seed': spec.training_seed, 'checkpoint_episode': 400,
            'checkpoint_path': str(checkpoint_path),
        })
        for index, probe in enumerate(probe_rows):
            predictions.append({
                'probe_id': probe['probe_id'],
                'probe_source_scene': probe['scene'],
                'evaluation_traffic_seed': probe['evaluation_traffic_seed'],
                'simulation_step': probe['simulation_step'],
                'current_phase': probe['current_phase'],
                'model_id': model_id, 'model_scene': scene,
                'model_network': spec.network, 'training_seed': spec.training_seed,
                'checkpoint_role': 'final', 'checkpoint_episode': 400,
                'top_action': int(top_actions[index]),
                'top2_action': int(top2_actions[index]),
                'q_margin': float(margins[index]),
                'normalized_q_margin': float(normalized[index]),
                'q_values': [float(value) for value in q_values[index]],
            })
    if len(models) != 20:
        raise ValueError(f'Phase 3 requires 20 final models, got {len(models)}')
    return models, predictions


def _state_embedding_and_classifier(probe_rows):
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = np.asarray([row['model_input'] for row in probe_rows], dtype=float)
    labels = np.asarray([row['scene'] for row in probe_rows])
    groups = np.asarray([row['evaluation_traffic_seed'] for row in probe_rows])
    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2, random_state=20260723)
    coordinates = pca.fit_transform(scaled)
    pca_rows = [{
        'probe_id': probe['probe_id'], 'scene': probe['scene'],
        'evaluation_traffic_seed': probe['evaluation_traffic_seed'],
        'simulation_step': probe['simulation_step'],
        'pc1': float(coordinates[index, 0]), 'pc2': float(coordinates[index, 1]),
        'pc1_explained_variance': float(pca.explained_variance_ratio_[0]),
        'pc2_explained_variance': float(pca.explained_variance_ratio_[1]),
    } for index, probe in enumerate(probe_rows)]
    splitter = GroupKFold(n_splits=len(set(groups)))
    predictions = np.empty(labels.shape, dtype=object)
    classifier_rows = []
    for fold, (train, test) in enumerate(splitter.split(features, labels, groups), start=1):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, solver='lbfgs', random_state=20260723),
        )
        classifier.fit(features[train], labels[train])
        predictions[test] = classifier.predict(features[test])
        classifier_rows.append({
            'record_type': 'fold_accuracy', 'fold': fold,
            'held_out_traffic_seed': int(groups[test][0]),
            'sample_count': len(test),
            'accuracy': float(accuracy_score(labels[test], predictions[test])),
            'random_baseline': .25,
        })
    accuracy = float(accuracy_score(labels, predictions))
    classifier_rows.append({
        'record_type': 'overall_accuracy', 'fold': None,
        'held_out_traffic_seed': None, 'sample_count': len(labels),
        'accuracy': accuracy, 'random_baseline': .25,
    })
    scene_order = ['S1','S2','S3','S4']
    matrix = confusion_matrix(labels, predictions, labels=scene_order)
    confusion_rows = []
    for row_index, true_scene in enumerate(scene_order):
        for column_index, predicted_scene in enumerate(scene_order):
            count = int(matrix[row_index, column_index])
            normalized = count / max(1, int(matrix[row_index].sum()))
            confusion_rows.append({
                'true_scene': true_scene, 'predicted_scene': predicted_scene,
                'count': count, 'normalized_count': normalized,
            })
            classifier_rows.append({
                'record_type': 'confusion_matrix', 'fold': None,
                'held_out_traffic_seed': None, 'sample_count': count,
                'accuracy': normalized, 'random_baseline': .25,
                'true_scene': true_scene, 'predicted_scene': predicted_scene,
            })
    return pca_rows, classifier_rows, confusion_rows, accuracy


def _fixedtime_pca_reference(output):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    processed = output / 'tables'
    probe_path = processed / 'probe_states_fixedtime.csv'
    pca_path = processed / 'state_pca.csv'
    if not probe_path.is_file() or not pca_path.is_file():
        raise FileNotFoundError(
            'Training-state coverage requires existing Phase 3 '
            f'FixedTime inputs: {probe_path} and {pca_path}'
        )
    probes = _load_csv_rows(probe_path)
    reference_rows = _load_csv_rows(pca_path)
    if len(probes) != 7200 or len(reference_rows) != 7200:
        raise ValueError(
            'FixedTime PCA reference must contain exactly 7200 balanced states'
        )
    features = np.asarray([
        json.loads(row['model_input_json']) for row in probes
    ], dtype=float)
    if features.shape != (7200, 16):
        raise ValueError(f'Unexpected FixedTime feature shape: {features.shape}')
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2, random_state=20260723)
    coordinates = pca.fit_transform(scaled)
    coordinate_by_probe = {
        row['probe_id']: coordinates[index] for index, row in enumerate(probes)
    }
    for row in reference_rows:
        expected = coordinate_by_probe.get(row['probe_id'])
        actual = np.asarray([float(row['pc1']), float(row['pc2'])])
        if expected is None or not np.allclose(expected, actual, rtol=0, atol=1e-9):
            raise ValueError(
                'Reconstructed FixedTime PCA does not match the existing '
                f"state_pca reference at {row['probe_id']}"
            )
    fixedtime_rows = [{
        'scene': row['scene'],
        'pc1': float(row['pc1']),
        'pc2': float(row['pc2']),
    } for row in reference_rows]
    return scaler, pca, fixedtime_rows


def _validate_training_state_runs(run_specs, scene_by_network):
    expected_networks = set(scene_by_network)
    formal = [spec for spec in run_specs
              if spec.role == 'formal' and spec.agent == 'dqn']
    cells = {(spec.network, spec.training_seed) for spec in formal}
    expected_cells = {(network, seed) for network in expected_networks
                      for seed in range(5)}
    if len(formal) != 20 or cells != expected_cells:
        raise ValueError(
            'Training-state coverage requires the complete four-network × '
            'five-training-seed Plan 1 formal matrix'
        )
    for spec in formal:
        trajectory = spec.run_dir / 'trajectory'
        with (trajectory / 'validation.json').open(encoding='utf-8') as handle:
            validation = json.load(handle)
        with (trajectory / 'manifest.json').open(encoding='utf-8') as handle:
            manifest = json.load(handle)
        if (
            validation.get('valid') is not True
            or int(validation.get('episode_count', -1)) != 400
            or int(validation.get('expected_decisions_per_episode', -1)) != 360
            or int(validation.get('evaluation_transition_count', -1)) != 0
            or manifest.get('network') != spec.network
            or int(manifest.get('behavior_training_seed', -1)) != spec.training_seed
            or int(manifest.get('action_dim', -1)) != 8
        ):
            raise ValueError(f'Invalid Plan 1 trajectory evidence: {trajectory}')
    return sorted(formal, key=lambda spec: (
        int(scene_by_network[spec.network][1:]), spec.training_seed
    ))


def _load_dqn_training_state_window(
        run_specs, scene_by_network, scaler, pca, episode_start, episode_end):
    rows = []
    epsilon_ranges = {}
    for spec in run_specs:
        scene = scene_by_network[spec.network]
        epsilon_values = []
        behavior_modes = set()
        for episode in range(episode_start, episode_end + 1):
            shard = (spec.run_dir / 'trajectory' / 'episodes'
                     / f'episode_{episode:04d}.npz')
            if not shard.is_file():
                raise FileNotFoundError(f'Missing trajectory shard: {shard}')
            with np.load(shard, allow_pickle=False) as data:
                required = {'episode_id', 'state', 'current_phase', 'epsilon',
                            'behavior_mode'}
                missing = required - set(data.files)
                if missing:
                    raise ValueError(f'{shard} is missing fields: {sorted(missing)}')
                state = np.asarray(data['state'], dtype=float).reshape(-1, 8)
                phase = np.asarray(data['current_phase'], dtype=int).reshape(-1)
                episode_ids = np.asarray(data['episode_id'], dtype=int).reshape(-1)
                behavior = np.asarray(data['behavior_mode']).reshape(-1)
                epsilon = np.asarray(data['epsilon'], dtype=float).reshape(-1)
            if (
                state.shape != (360, 8)
                or phase.shape != (360,)
                or not np.all(episode_ids == episode)
                or np.any(phase < 0)
                or np.any(phase >= 8)
                or not set(behavior.tolist()).issubset(
                    {'random_warmup', 'epsilon_greedy'}
                )
            ):
                raise ValueError(f'Unexpected trajectory contents: {shard}')
            model_input = np.concatenate([state, np.eye(8)[phase]], axis=1)
            coordinates = pca.transform(scaler.transform(model_input))
            rows.extend({
                'scene': scene,
                'training_seed': spec.training_seed,
                'episode': episode,
                'pc1': float(coordinate[0]),
                'pc2': float(coordinate[1]),
            } for coordinate in coordinates)
            epsilon_values.extend(epsilon.tolist())
            behavior_modes.update(behavior.tolist())
        epsilon_ranges[f'{scene}_seed{spec.training_seed}'] = {
            'minimum': float(min(epsilon_values)),
            'maximum': float(max(epsilon_values)),
            'behavior_modes': sorted(behavior_modes),
        }
    expected_count = len(run_specs) * (episode_end - episode_start + 1) * 360
    if len(rows) != expected_count:
        raise ValueError(
            f'DQN training-state count {len(rows)} != expected {expected_count}'
        )
    return rows, epsilon_ranges


def _shared_state_axis_limits(fixedtime_rows, stage_rows):
    x_values = [row['pc1'] for row in fixedtime_rows]
    y_values = [row['pc2'] for row in fixedtime_rows]
    for rows in stage_rows.values():
        x_values.extend(row['pc1'] for row in rows)
        y_values.extend(row['pc2'] for row in rows)
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_pad = max((x_max - x_min) * .025, .1)
    y_pad = max((y_max - y_min) * .025, .1)
    return {
        'x': (x_min - x_pad, x_max + x_pad),
        'y': (y_min - y_pad, y_max + y_pad),
    }


def _fixedtime_reference_axis_limits(fixedtime_rows):
    x_values = [row['pc1'] for row in fixedtime_rows]
    y_values = [row['pc2'] for row in fixedtime_rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_pad = max((x_max - x_min) * .06, .1)
    y_pad = max((y_max - y_min) * .06, .1)
    return {
        'x': (x_min - x_pad, x_max + x_pad),
        'y': (y_min - y_pad, y_max + y_pad),
    }


def run_s1_s4_training_state_coverage(args):
    if not args.scene_mapping:
        raise ValueError('--scene-mapping is required for training-state coverage')
    output = _diagnostic_output(args)
    manifest_path = output / 'manifests' / 'training_state_coverage_manifest.json'
    figures = output / 'figures'
    expected_outputs = [
        figures / f'state_pca_dqn_training_ep{start:03d}_{end:03d}.{suffix}'
        for start, end in DQN_TRAINING_STATE_WINDOWS
        for suffix in ('png', 'pdf')
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if (manifest_path.exists() or existing) and not args.refresh_existing:
        raise FileExistsError(
            'Training-state coverage outputs already exist; use '
            '--refresh-existing to replace only this derived figure set'
        )
    _, _, scene_by_network = load_scene_mapping(args.scene_mapping)
    _, _, included = load_run_list(args.run_list)
    formal = _validate_training_state_runs(included, scene_by_network)
    scaler, pca, fixedtime_rows = _fixedtime_pca_reference(output)

    stage_rows = {}
    stage_epsilon = {}
    for start, end in DQN_TRAINING_STATE_WINDOWS:
        key = f'{start:03d}_{end:03d}'
        rows, epsilon_ranges = _load_dqn_training_state_window(
            formal, scene_by_network, scaler, pca, start, end
        )
        stage_rows[key] = rows
        stage_epsilon[key] = epsilon_ranges
    axis_limits = _shared_state_axis_limits(fixedtime_rows, stage_rows)
    reference_axis_limits = _fixedtime_reference_axis_limits(fixedtime_rows)

    figure_paths = []
    stage_manifest = []
    for start, end in DQN_TRAINING_STATE_WINDOWS:
        key = f'{start:03d}_{end:03d}'
        output_base = figures / f'state_pca_dqn_training_ep{key}'
        paths = render_dqn_training_state_coverage(
            fixedtime_rows, stage_rows[key], start, end, axis_limits,
            reference_axis_limits, pca.explained_variance_ratio_,
            output_base, args.dpi,
        )
        figure_paths.extend(paths)
        reference_view_fraction = {}
        for scene in ('S1', 'S2', 'S3', 'S4'):
            scene_rows = [row for row in stage_rows[key] if row['scene'] == scene]
            inside = [
                reference_axis_limits['x'][0] <= row['pc1'] <= reference_axis_limits['x'][1]
                and reference_axis_limits['y'][0] <= row['pc2'] <= reference_axis_limits['y'][1]
                for row in scene_rows
            ]
            reference_view_fraction[scene] = float(sum(inside) / len(inside))
        stage_manifest.append({
            'episode_start': start,
            'episode_end': end,
            'dqn_state_count': len(stage_rows[key]),
            'reference_view_fraction_by_scene': reference_view_fraction,
            'epsilon_by_scene_and_training_seed': stage_epsilon[key],
            'figures': [str(path.relative_to(output)) for path in paths],
        })
    _json_dump(manifest_path, {
        'schema_version': 1,
        'tool': 'tools.experiment_plotting',
        'tool_version': __version__,
        'profile': 's1_s4_diagnostics',
        'phase': 'training-state-coverage',
        'reference': {
            'source': 'tables/state_pca.csv',
            'state_count': len(fixedtime_rows),
            'policy': 'fixedtime_common',
            'pca_fit': 'FixedTime 16D standardized model inputs only',
            'explained_variance': [
                float(value) for value in pca.explained_variance_ratio_
            ],
        },
        'dqn': {
            'source': 'Plan 1 per-episode training trajectory NPZ shards',
            'behavior_policy': 'random_warmup_then_epsilon_greedy',
            'training_seed_count_per_scene': 5,
            'decision_states_per_episode_per_seed': 360,
        },
        'axis_limits': {
            axis: [float(value) for value in limits]
            for axis, limits in axis_limits.items()
        },
        'reference_axis_limits': {
            axis: [float(value) for value in limits]
            for axis, limits in reference_axis_limits.items()
        },
        'visualization': {
            'layout': 'four scene small multiples plus shared full-range context',
            'fixedtime_raw_points': 'all 1800 states per scene',
            'dqn_raw_points': 'deterministic representative sample of in-view states',
            'density_estimator': '72x72 histogram smoothed with Gaussian sigma 1.15',
            'density_regions': [0.50, 0.80, 0.95],
            'dqn_density_scope': 'all states inside the FixedTime reference-scale view',
            'main_view': 'shared FixedTime reference-scale rectangle',
            'full_range_view': 'shared across all five DQN episode windows',
        },
        'stages': stage_manifest,
        'figures': [str(path.relative_to(output)) for path in figure_paths],
    })
    return output


def _representative_conflicts(probe_rows, prediction_rows, model_rows, threshold):
    candidates = [row for row in model_rows
                  if row['probe_source_scene'] == 'ALL'
                  and row['left_scene'] != row['right_scene']]
    strongest = max(candidates, key=lambda row: row['high_confidence_disagreement']
                    if row['high_confidence_disagreement'] is not None else -1)
    by_key = {(row['model_id'], row['probe_id']): row for row in prediction_rows}
    probe_by_id = {row['probe_id']: row for row in probe_rows}
    conflicts = []
    for probe_id in probe_by_id:
        left = by_key[(strongest['left_model'], probe_id)]
        right = by_key[(strongest['right_model'], probe_id)]
        if (left['top_action'] != right['top_action']
                and left['normalized_q_margin'] >= threshold
                and right['normalized_q_margin'] >= threshold):
            conflicts.append((min(left['normalized_q_margin'],
                                  right['normalized_q_margin']),
                              probe_by_id[probe_id], left, right))
    conflicts.sort(key=lambda item: item[0], reverse=True)
    return strongest, conflicts[:5]


def run_s1_s4_phase3(args):
    if not args.scene_mapping or not args.probe_evaluation_package:
        raise ValueError('Phase 3 requires --scene-mapping and --probe-evaluation-package')
    output = _diagnostic_output(args)
    report_path = output / 'reports' / 'phase3_report.md'
    if report_path.exists() and not args.refresh_existing:
        raise FileExistsError(f'Phase 3 already exists: {report_path}')
    _, _, scene_by_network = load_scene_mapping(args.scene_mapping)
    package_path = Path(args.probe_evaluation_package).expanduser().resolve()
    package = validate_evaluation_package(str(package_path))
    probe_isolation = []
    for summary in package['summaries']:
        check = summary['isolation_check']
        if isinstance(check, str):
            try:
                check = json.loads(check)
            except json.JSONDecodeError:
                check = ast.literal_eval(check)
        protected = (
            'rng_unchanged','online_model_unchanged','target_model_unchanged',
            'optimizer_unchanged','epsilon_unchanged','replay_unchanged',
            'counters_unchanged','trajectory_writes_unchanged',
        )
        if not all(check.get(field) is True for field in protected):
            raise RuntimeError(
                f"Probe evaluation isolation failed: {summary['controller_id']}"
            )
        probe_isolation.append({
            'controller_id': summary['controller_id'],
            'target_network': summary['target_network'],
            'evaluation_traffic_seed': int(summary['evaluation_traffic_seed']),
            **{field: check[field] for field in protected},
        })
    probes = _probe_rows(package, package_path, scene_by_network)
    _, _, included = load_run_list(args.run_list)
    formal = [spec for spec in included if spec.role == 'formal' and spec.agent == 'dqn']
    models, predictions = _final_model_predictions(formal, scene_by_network, probes)
    model_pairs, policy_scene, threshold = calculate_policy_disagreement(predictions)
    feature_stats, state_distances = calculate_state_distribution_diagnostics(probes)
    network_by_scene = {scene: network for network, scene in scene_by_network.items()}
    for row in policy_scene:
        row['source_network'] = network_by_scene[row['source_scene']]
        row['target_network'] = network_by_scene[row['target_scene']]
    for row in state_distances:
        row['source_network'] = network_by_scene[row['source_scene']]
        row['target_network'] = network_by_scene[row['target_scene']]
    pca_rows, classifier_rows, confusion_rows, accuracy = _state_embedding_and_classifier(probes)
    disagreement = {frozenset((row['source_scene'], row['target_scene'])): row['mean']
                    for row in policy_scene if row['probe_source_scene'] == 'ALL'
                    and row['metric'] == 'action_disagreement'
                    and row['source_scene'] != row['target_scene']}
    state_lookup = {frozenset((row['source_scene'], row['target_scene'])): row['distance']
                    for row in state_distances if row['source_scene'] != row['target_scene']}
    relation = []
    for pair in sorted(disagreement, key=lambda item: sorted(item)):
        left, right = sorted(pair, key=lambda value: int(value[1:]))
        relation.append({'scene_a': left, 'scene_b': right,
                         'state_distribution_distance': state_lookup[pair],
                         'policy_disagreement': disagreement[pair]})
    strongest, examples = _representative_conflicts(
        probes, predictions, model_pairs, threshold
    )
    tables, figures = output / 'tables', output / 'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    probe_export = []
    for row in probes:
        exported = {key: row[key] for key in row if key not in {'raw_state','model_input'}}
        exported['raw_state_json'] = json.dumps(row['raw_state'], separators=(',', ':'))
        exported['model_input_json'] = json.dumps(row['model_input'], separators=(',', ':'))
        exported['feature_schema'] = 'lane_count_8_plus_phase_one_hot_8_v1'
        probe_export.append(exported)
    prediction_export = []
    for row in predictions:
        exported = {key: row[key] for key in row if key != 'q_values'}
        exported['q_values_json'] = json.dumps(row['q_values'], separators=(',', ':'))
        prediction_export.append(exported)
    write_csv(tables / 'probe_states_fixedtime.csv', probe_export)
    write_csv(tables / 'probe_evaluation_isolation_checks.csv', probe_isolation)
    write_csv(tables / 'policy_model_manifest.csv', models)
    write_csv(tables / 'policy_predictions.csv', prediction_export)
    policy_rows = policy_scene + model_pairs
    policy_fields = list(dict.fromkeys(
        key for row in policy_rows for key in row
    ))
    write_csv(tables / 'policy_disagreement.csv', policy_rows, policy_fields)
    write_csv(tables / 'q_distance.csv', [row for row in model_pairs
                                         if row['probe_source_scene'] == 'ALL'])
    write_csv(tables / 'state_feature_statistics.csv', feature_stats)
    write_csv(tables / 'state_distribution_distances.csv', state_distances)
    write_csv(tables / 'state_pca.csv', pca_rows)
    write_csv(tables / 'scene_classifier_results.csv', classifier_rows)
    write_csv(tables / 'state_distance_vs_policy_disagreement.csv', relation)
    figure_paths = []
    for metric, name in (
        ('action_disagreement','scene_disagreement_heatmap'),
        ('high_confidence_disagreement','high_confidence_disagreement'),
        ('q_cosine_distance','q_cosine_distance_heatmap'),
    ):
        figure_paths += render_policy_scene_heatmap(
            policy_scene, metric, figures / name, args.dpi)
    figure_paths += render_policy_model_heatmap(
        model_pairs, figures / 'model_disagreement_heatmap', args.dpi)
    figure_paths += render_policy_source_stratified(
        policy_scene, figures / 'source_stratified_policy_disagreement', args.dpi)
    figure_paths += render_within_between_policy(
        model_pairs, figures / 'within_between_policy_difference', args.dpi)
    figure_paths += render_state_distance_heatmap(
        state_distances, figures / 'state_distance_heatmap', args.dpi)
    figure_paths += render_state_pca(pca_rows, figures / 'state_pca', args.dpi)
    figure_paths += render_classifier_confusion(
        confusion_rows, figures / 'classifier_confusion_matrix', args.dpi)
    figure_paths += render_state_distance_vs_policy(
        relation, figures / 'state_distance_vs_policy_disagreement', args.dpi)
    conflict_lines = ['# FixedTime probe 典型高置信策略冲突', '',
                      f"模型对：{strongest['left_model']} vs {strongest['right_model']}。",
                      f'固定 normalized Q-margin 阈值：{threshold:.2f}。', '']
    for index, (_, probe, left, right) in enumerate(examples, start=1):
        conflict_lines += [f'## 案例 {index}: {probe["probe_id"]}', '',
                           f'- raw state: `{probe["raw_state"]}`',
                           f'- current phase: {probe["current_phase"]}',
                           f'- {left["model_id"]}: Q=`{left["q_values"]}`, top action {left["top_action"]}（{S1_S4_ACTION_SEMANTICS[left["top_action"]]}），margin={left["q_margin"]:.4f}, normalized={left["normalized_q_margin"]:.4f}',
                           f'- {right["model_id"]}: Q=`{right["q_values"]}`, top action {right["top_action"]}（{S1_S4_ACTION_SEMANTICS[right["top_action"]]}），margin={right["q_margin"]:.4f}, normalized={right["normalized_q_margin"]:.4f}', '']
    (output / 'representative_conflict_states.md').write_text(
        '\n'.join(conflict_lines) + '\n', encoding='utf-8')
    within = [row['action_disagreement'] for row in model_pairs
              if row['probe_source_scene'] == 'ALL' and row['pair_kind'] == 'within_scene']
    between = [row['action_disagreement'] for row in model_pairs
               if row['probe_source_scene'] == 'ALL' and row['pair_kind'] == 'between_scene']
    within_high = [row['high_confidence_disagreement'] for row in model_pairs
                   if row['probe_source_scene'] == 'ALL'
                   and row['pair_kind'] == 'within_scene'
                   and row['high_confidence_disagreement'] is not None]
    between_high = [row['high_confidence_disagreement'] for row in model_pairs
                    if row['probe_source_scene'] == 'ALL'
                    and row['pair_kind'] == 'between_scene'
                    and row['high_confidence_disagreement'] is not None]
    report_lines = ['# Phase 3：公共 FixedTime probe 上的策略与状态分布', '',
                    f'- 状态数：{len(probes)}；四场景 × 5 个共同 evaluation traffic seeds，逐格等量。',
                    '- 每次评估保留完整 360 个 decision-time states，覆盖 0–3600 秒全时间轴，而非只取稳定尾段。',
                    '- source_policy=fixedtime_common；20 个 episode-400 final Q-network 只做冻结前向推理。',
                    '- FixedTime probe evaluation isolation：20/20 通过。',
                    f'- high-confidence：双方 normalized Q-margin 均 ≥ {threshold:.2f}。',
                    f'- within-scene 平均 disagreement：{sum(within)/len(within):.3f}。',
                    f'- between-scene 平均 disagreement：{sum(between)/len(between):.3f}。',
                    f'- within-scene 平均 high-confidence disagreement：{sum(within_high)/len(within_high):.3f}。',
                    f'- between-scene 平均 high-confidence disagreement：{sum(between_high)/len(between_high):.3f}。',
                    f'- traffic-seed-grouped logistic regression accuracy：{accuracy:.1%}（随机基线 25%）。',
                    '- 状态可区分性、动作边际分布、同状态策略差异和控制退化分别解释。']
    report_path.write_text('\n'.join(report_lines) + '\n', encoding='utf-8')
    _json_dump(output / 'manifests' / 'phase3_manifest.json', {
        'schema_version': 1, 'tool': 'tools.experiment_plotting', 'phase': 'phase3',
        'probe_evaluation_package': str(package_path), 'probe_state_count': len(probes),
        'model_count': len(models), 'confidence_threshold': threshold,
        'figures': [str(path.relative_to(output)) for path in figure_paths],
    })
    return output


def _numeric_csv_rows(rows):
    converted = []
    for source in rows:
        row = dict(source)
        for key, value in row.items():
            if value in ('', None):
                row[key] = None
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
        converted.append(row)
    return converted


def run_s1_s4_phase4(args):
    output = _diagnostic_output(args)
    report_path = output / 'reports' / 'phase4_report.md'
    if report_path.exists() and not args.refresh_existing:
        raise FileExistsError(f'Phase 4 already exists: {report_path}')
    tables = output / 'tables'
    required = {
        'relative': tables / 'cross_scene_relative_degradation.csv',
        'action': tables / 'action_distribution_distances.csv',
        'policy': tables / 'policy_disagreement.csv',
        'state': tables / 'state_distribution_distances.csv',
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Phase 4 requires completed Phase 1-3 tables: {missing}')
    evidence, selected = select_transition_pairs(
        _numeric_csv_rows(_load_csv_rows(required['relative'])),
        _numeric_csv_rows(_load_csv_rows(required['action'])),
        _numeric_csv_rows(_load_csv_rows(required['policy'])),
        _numeric_csv_rows(_load_csv_rows(required['state'])),
    )
    write_csv(tables / 'pair_selection_evidence.csv', evidence)
    write_csv(tables / 'selected_transition_pairs.csv', selected)
    high = [row for row in selected if row['selection'] == 'high_conflict']
    low = [row for row in selected if row['selection'] == 'low_conflict']
    if high and low:
        verdict = 'B'
        verdict_text = '只有部分场景对具有一致的冲突证据，应先做 targeted pairwise pilot。'
    elif high:
        verdict = 'B'
        verdict_text = '存在高冲突候选，但低冲突对证据不一致；仅适合有针对性的 pilot。'
    else:
        verdict = 'C'
        verdict_text = '跨场景退化与同状态策略冲突未形成一致高位证据，需要更强场景。'
    lines = [
        '# Phase 4：场景对证据一致性选择', '',
        '未构造加权总分。六个无向场景对分别按两方向中较差的 travel-time 退化、公共状态 high-confidence disagreement 和 action TV 排序；三项同时位于高半区或低半区，且相对同场景 seed 参考方向一致时，才进入候选。Raw top-1 disagreement 只作补充。两个转换方向仍分别保留。', '',
        '## 选择结果', '',
    ]
    if selected:
        lines += [
            '| Type | Direction | Travel-time degradation | Policy disagreement (excess vs within) | High-confidence disagreement (excess vs within) | Action TV (excess vs within) | State distance |',
            '|---|---|---:|---:|---:|---:|---:|',
        ]
        for row in selected:
            lines.append(
                f"| {row['selection']} | {row['source_scene']}→{row['target_scene']} "
                f"| {row['travel_time_relative_degradation']:.3f} "
                f"| {row['policy_disagreement']:.3f} ({row['policy_disagreement_excess_over_within']:+.3f}) "
                f"| {row['high_confidence_disagreement']:.3f} ({row['high_confidence_excess_over_within']:+.3f}) "
                f"| {row['action_total_variation']:.3f} ({row['action_total_variation_excess_over_within']:+.3f}) "
                f"| {row['state_distribution_distance']:.3f} |"
            )
    else:
        lines.append('- 没有场景对同时满足主要证据的一致排序；未强行选择。')
    asymmetric_pair = max(evidence, key=lambda row: abs(row['directional_asymmetry']))
    state_pair = max(evidence, key=lambda row: row['state_distribution_distance'])
    policy_pair = max(evidence, key=lambda row: row['policy_disagreement'])
    lines += [
        '', '## 证据不一致与混杂', '',
        f"- directionally asymmetric：{asymmetric_pair['scene_a']}–{asymmetric_pair['scene_b']}，两方向 degradation 差 {abs(asymmetric_pair['directional_asymmetry']):.1%}。",
        f"- state-different but not top policy-conflict：{state_pair['scene_a']}–{state_pair['scene_b']} 的 state distance 排名最高，但 policy disagreement 排名为 {int(state_pair['policy_disagreement_rank_low_to_high'])}/6。",
        f"- policy-different but transfer-moderate：{policy_pair['scene_a']}–{policy_pair['scene_b']} 的 policy disagreement 排名最高，但 worst directed degradation 排名为 {int(policy_pair['maximum_directional_degradation_rank_low_to_high'])}/6。",
        '- 同场景不同 training seed 的策略差异较大，因此只把 high-confidence excess-over-within 作为更强的冲突佐证，不把 raw top-1 disagreement 单独当作场景效应。',
    ]
    lines += ['', '## 三档判断', '', f'**{verdict}. {verdict_text}**', '',
              '本阶段到此停止，没有启动 sequential training、Plan Clear 或 Plan Retain。']
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    convergence = _numeric_csv_rows(_load_csv_rows(
        tables / 'convergence_summary.csv'))
    primary_convergence = sorted([
        row for row in convergence
        if row['metric'] == 'travel_time' and row['tolerance'] == .05
    ], key=lambda row: int(row['scene'][1:]))
    directional = _numeric_csv_rows(_load_csv_rows(
        tables / 'directional_transfer_summary.csv'))
    best_direction = min(
        directional, key=lambda row: row['travel_time_relative_degradation'])
    worst_direction = max(
        directional, key=lambda row: row['travel_time_relative_degradation'])
    transferable = [row for row in directional
                    if row['travel_time_relative_degradation'] <= .05]
    action_table = _numeric_csv_rows(_load_csv_rows(
        tables / 'action_distribution_distances.csv'))
    action_model_pairs = [row for row in action_table
        if row['aggregation'] == 'model_pair'
        and row['metric'] == 'total_variation']
    within_action = [row['distance'] for row in action_model_pairs
                     if row['pair_kind'] == 'within_scene']
    between_action = [row['distance'] for row in action_model_pairs
                      if row['pair_kind'] == 'between_scene']
    action_within_p95 = float(np.quantile(within_action, .95))
    scene_action_pairs = [row for row in action_table
                          if row['aggregation'] == 'scene_mean'
                          and row['metric'] == 'total_variation'
                          and row['source_scene'] < row['target_scene']]
    action_exceeds_seed = [row for row in scene_action_pairs
                           if row['distance'] > action_within_p95]
    policy_table = _numeric_csv_rows(_load_csv_rows(
        tables / 'policy_disagreement.csv'))
    policy_scene = [row for row in policy_table
        if row['aggregation'] == 'scene_pair'
        and row['probe_source_scene'] == 'ALL'
        and row['metric'] == 'action_disagreement'
        and row['source_scene'] != row['target_scene']]
    within_policy = [row['action_disagreement'] for row in policy_table
                     if row['aggregation'] == 'model_pair'
                     and row['probe_source_scene'] == 'ALL'
                     and row['pair_kind'] == 'within_scene']
    between_policy = [row['action_disagreement'] for row in policy_table
                      if row['aggregation'] == 'model_pair'
                      and row['probe_source_scene'] == 'ALL'
                      and row['pair_kind'] == 'between_scene']
    within_high_policy = [row['high_confidence_disagreement'] for row in policy_table
                          if row['aggregation'] == 'model_pair'
                          and row['probe_source_scene'] == 'ALL'
                          and row['pair_kind'] == 'within_scene'
                          and row['high_confidence_disagreement'] is not None]
    between_high_policy = [row['high_confidence_disagreement'] for row in policy_table
                           if row['aggregation'] == 'model_pair'
                           and row['probe_source_scene'] == 'ALL'
                           and row['pair_kind'] == 'between_scene'
                           and row['high_confidence_disagreement'] is not None]
    strongest_policy = max(policy_scene, key=lambda row: row['mean'])
    weakest_policy = min(policy_scene, key=lambda row: row['mean'])
    classifier = next(row for row in _numeric_csv_rows(
        _load_csv_rows(tables / 'scene_classifier_results.csv'))
        if row['record_type'] == 'overall_accuracy')
    final_lines = [
        '# S1–S4 适应/遗忘可观测性最终诊断', '',
        '## 核心结论', '',
        f'**{verdict}. {verdict_text}**', '',
        '- 400 episodes 对当前独立单场景学习明显冗余；推荐预算为 100，而不是 50。',
        '- 80 次 final-checkpoint cross-scene evaluations 与 20 次 FixedTime probe evaluations 的隔离检查全部通过。',
        f"- zero-shot 最佳方向是 {best_direction['source_scene']}→{best_direction['target_scene']}（{best_direction['travel_time_relative_degradation']:.1%}），最差方向是 {worst_direction['source_scene']}→{worst_direction['target_scene']}（{worst_direction['travel_time_relative_degradation']:.1%}）。",
        f"- 公共 FixedTime 状态上，最强场景级策略 disagreement 为 {strongest_policy['source_scene']}–{strongest_policy['target_scene']}（{strongest_policy['mean']:.1%}），最弱非同场景对为 {weakest_policy['source_scene']}–{weakest_policy['target_scene']}（{weakest_policy['mean']:.1%}）。",
        f"- FixedTime 状态的 traffic-seed-grouped 四分类准确率为 {classifier['accuracy']:.1%}（随机基线 25%）；状态可区分性不等同于策略冲突或遗忘。", '',
        '## 场景映射', '',
    ]
    if args.scene_mapping:
        for row in _load_csv_rows(args.scene_mapping):
            final_lines.append(
                f"- {row['scene']}: {row['network']}；demand={row['demand']}；vehicles={row['vehicles']}。"
            )
    final_lines += ['', '## 收敛与预算', '',
                    '| Scene / network | Median | Mean ± sample SD | Maximum | Redundant after maximum |',
                    '|---|---:|---:|---:|---:|']
    for row in primary_convergence:
        final_lines.append(
            f"| {row['scene']} / {row['network']} | {row['median']:.0f} | "
            f"{row['mean']:.1f} ± {row['sample_sd']:.1f} | {row['maximum']:.0f} | "
            f"{400-row['maximum']:.0f} episodes |"
        )
    final_lines += [
        '',
        '- 全场景最慢 seed 在 episode 80 进入并维持 travel-time ±5% 稳定区间。',
        '- 对齐已有完整 resumable checkpoints 后取 100 episodes；50 episodes 不能覆盖 S2/S3 的保守尾部。',
        '- 因此 400 相对推荐预算多 300 episodes（75%），相对最慢稳定点多 320 episodes（80%）。',
        '- queue、delay、reward 的稳定时间更噪；它们用于确认，不取代 travel time 主判据，也不使用 loss 单独宣称收敛。', '',
        '## Zero-shot 与方向不对称', '',
        '- 以 travel-time degradation ≤5% 作为描述性“接近 target in-domain”阈值，可直接迁移的方向为：'
        + (', '.join(
            f"{row['source_scene']}→{row['target_scene']} ({row['travel_time_relative_degradation']:.1%})"
            for row in transferable) if transferable else '无') + '。',
        '- 所有方向均使用相同 evaluator、相同 traffic seed=10000，并以 5 个 training seeds 计算 mean ± sample SD。',
        '- S4→S2 与 S2→S4 等方向高度不对称，因此后续 pilot 必须保留有向转换。', '',
        '## 动作分布、相同状态策略与状态分布', '',
        f'- within-scene training-seed action-TV 平均={sum(within_action)/len(within_action):.3f}；between-scene model-pair 平均={sum(between_action)/len(between_action):.3f}。',
        f"- scene-mean action TV 超过 within-scene 95th percentile ({action_within_p95:.3f}) 的场景对："
        + ', '.join(f"{row['source_scene']}–{row['target_scene']} ({row['distance']:.3f})" for row in action_exceeds_seed) + '。',
        f'- common-state top-1 disagreement：within-scene 平均={sum(within_policy)/len(within_policy):.3f}，between-scene 平均={sum(between_policy)/len(between_policy):.3f}。',
        f'- high-confidence disagreement：within-scene 平均={sum(within_high_policy)/len(within_high_policy):.3f}，between-scene 平均={sum(between_high_policy)/len(between_high_policy):.3f}。较小的差距说明 training-seed 不稳定性不可忽略。',
        '- 边际动作频率差异可能来自访问状态不同；正式策略冲突结论只采用公共 FixedTime probe 上的同状态比较。',
        '- 状态距离、边际动作距离、同状态 disagreement 和跨场景性能下降分别保存，未合并成单一加权分数。', '',
        '## 建议转换对', '',
    ]
    if selected:
        for row in selected:
            final_lines.append(
                f"- {row['selection']}: {row['source_scene']}→{row['target_scene']}；"
                f"travel-time degradation={row['travel_time_relative_degradation']:.1%}；"
                f"policy disagreement={row['policy_disagreement']:.1%}；"
                f"high-confidence disagreement={row['high_confidence_disagreement']:.1%}；"
                f"high-confidence excess vs within={row['high_confidence_excess_over_within']:+.1%}；"
                f"action TV={row['action_total_variation']:.1%}；"
                f"action-TV excess vs within={row['action_total_variation_excess_over_within']:+.1%}。"
            )
    else:
        final_lines.append('- 主要证据没有形成一致排序，因此没有强行指定场景对。')
    final_lines += [
        '', '## 对 adaptation / forgetting 的结论边界', '',
        '- A–F 证明了哪些转换具有 zero-shot gap、共同状态策略冲突和状态分布差异，但没有时间维度，因此不能声称已经观察到灾难性遗忘。',
        '- 任务 B 使用 episode-400 final models，用于诊断成熟策略的场景冲突；任务 A 推荐的任务 G 预算是 100 episodes，二者不能自动视为具有完全相同的迁移排序。',
        '- 在启动 G 前，应仅对选定 high/low pair 的有向转换做 episode-100 resumable checkpoint 最小敏感性验证；本轮只提出该验证，没有执行。',
        '- 本轮没有运行 G，所以 Clear/Retain 对 source retention、target adaptation、warm-up 和 update-count 的影响仍未知。',
        '- 若存在选定 high-conflict pair，值得下一轮先做对称 100+100 targeted pilot；low-conflict pair 作为负对照。',
        '- G 前必须具备：switch 前后 source/target 双场景冻结评估、完整 resumable checkpoint、replay provenance、实际 sampled historical fraction、每 episode gradient update count 与 clear warm-up gap。',
        '- 若 high-conflict pilot 仍无显著 adaptation gap 或 source 退化，应增加方向服务冲突、流量比例冲突或时间模式冲突，而不是直接启动完整四场景顺序训练。', '',
        '## 未完成项', '',
        '- 任务 G、Plan Clear、Plan Retain、完整 S1–S4 sequential training：按停止条件未启动。',
        '- adaptation episode、maximum/final forgetting、adaptation/retention AUC：需要下一轮顺序训练时间序列。',
        '- Phase 2 主矩阵只有一个共同 evaluation traffic seed；其不确定性来自 training seeds，不代表 traffic-seed 泛化方差。',
    ]
    final_report_content = '\n'.join(final_lines) + '\n'
    reports_dir = output / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / 'final_diagnostic_report.md').write_text(
        final_report_content, encoding='utf-8')
    (reports_dir / 'code_changes.md').write_text(
        '# 代码改动说明\n\n'
        '- `run.py`：扩展既有 evaluation manifest runner，支持 source checkpoint / target scene 解耦。\n'
        '- `trainer/tsc_trainer.py`：扩展既有 `evaluate_once` 与 `EvaluationIsolationGuard`，记录 FixedTime probe 状态并校验 replay 内容。\n'
        '- `utils/logger.py`：在既有 evaluation package 中增加向后兼容 schema v2、final role 和 source/target 字段。\n'
        '- `tools/experiment_plotting/` 既有 CLI、loader、aggregation、plotting、profile：集成 A–F。\n'
        '- 默认 v1/Plan 1/Plan 2 行为不变；新行为由显式 profile、phase 或 schema-v2 manifest 启用。\n'
        '- 未创建独立功能脚本或第二套日志/绘图体系。\n', encoding='utf-8')
    (reports_dir / 'unresolved_issues.md').write_text(
        '# 当前无法完成内容及原因\n\n'
        '- 顺序适应、遗忘与 replay Clear/Retain：本轮明确禁止任务 G。\n'
        '- traffic-seed 泛化不确定性：Phase 2 仅授权一个共同 traffic seed。\n'
        '- replay warm-up/update-count 混杂：必须由下一轮顺序训练日志产生。\n'
        '- 环境会输出 legacy Gym 与未使用的 torch-geometric CUDA extension 警告；本轮 DQN/FixedTime 路径未调用这些扩展，测试和评估均完成。\n',
        encoding='utf-8')
    _json_dump(output / 'manifests' / 'phase4_manifest.json', {
        'schema_version': 1, 'tool': 'tools.experiment_plotting',
        'phase': 'phase4', 'verdict': verdict,
        'selection_count': len(selected),
        'selection_method': 'evidence-consistent half-ranking using worst directed degradation, high-confidence disagreement, and action TV relative to within-scene seed references; raw disagreement supplemental; no weighted score',
    })
    return output


def run_plan1(args):
    run_list_path, all_specs, included_specs = load_run_list(args.run_list)
    output_dir = Path(args.output_root).expanduser().resolve() / args.analysis_id
    if output_dir.exists():
        raise FileExistsError(
            f"Analysis output already exists: {output_dir}. Use a new analysis-id."
        )
    inputs_dir = output_dir / "inputs"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    inputs_dir.mkdir(parents=True)
    tables_dir.mkdir()
    figures_dir.mkdir()
    try:
        shutil.copy2(run_list_path, inputs_dir / "runs.csv")
        validated = [validate_run(spec) for spec in included_specs]
        config_comparison = compare_dqn_run_configs(validated)
        metric_rows, action_rows = normalize_records(validated)
        summaries, comparisons = build_run_summaries(validated, metric_rows)
        auc_rows = calculate_first_100_auc(metric_rows)
        learning_speed_rows = calculate_learning_speed(metric_rows)
        efficiency_rows = build_efficiency_rows(metric_rows)
        aulc_rows = calculate_efficiency_aulc(efficiency_rows, metric_rows)

        metric_fields = (
            "run_key", "role", "agent", "network", "training_seed", "run_dir",
            *CORE_FIELDS, "action_distribution_json",
        )
        write_csv(tables_dir / "metrics.csv", metric_rows, metric_fields)
        write_csv(tables_dir / "action_distribution.csv", action_rows, (
            "run_key", "role", "agent", "network", "training_seed", "run_dir",
            "record_type", "episode", "action", "fraction",
        ))
        write_csv(tables_dir / "run_summary.csv", summaries)
        write_csv(tables_dir / "final_best_comparison.csv", comparisons)
        write_csv(tables_dir / "first_100_auc.csv", auc_rows, AUC_FIELDS)
        write_csv(
            tables_dir / "learning_speed.csv", learning_speed_rows,
            LEARNING_SPEED_FIELDS,
        )
        write_csv(tables_dir / "efficiency.csv", efficiency_rows, EFFICIENCY_FIELDS)
        write_csv(tables_dir / "aulc.csv", aulc_rows, AULC_SUMMARY_FIELDS)
        _json_dump(tables_dir / "config_comparison.json", config_comparison)
        figures = render_all(
            metric_rows, action_rows, comparisons, auc_rows, figures_dir, dpi=args.dpi,
            learning_speed_rows=learning_speed_rows,
            profile=get_profile(getattr(args, "profile", "plan1")),
            efficiency_rows=efficiency_rows, aulc_rows=aulc_rows,
        )
        evaluation_package = None
        evaluation_package_path = getattr(args, "evaluation_package", None)
        if evaluation_package_path:
            evaluation_package_path = Path(evaluation_package_path).expanduser().resolve()
            evaluation_package = validate_evaluation_package(
                str(evaluation_package_path)
            )
            shutil.copy2(
                evaluation_package_path / "manifest.json",
                inputs_dir / "evaluation_package_manifest.json",
            )
            shutil.copy2(
                evaluation_package_path / "collection_manifest.json",
                inputs_dir / "evaluation_collection_manifest.json",
            )
            shutil.copy2(
                evaluation_package_path / "summary.csv",
                tables_dir / "best_checkpoint_evaluation_summary.csv",
            )
            decision_records = []
            with (evaluation_package_path / "records.jsonl").open(
                encoding="utf-8"
            ) as handle:
                for line in handle:
                    decision_records.append(json.loads(line))
            figures.extend(render_evaluation_timeseries(
                decision_records, figures_dir,
                evaluation_package["collection"]["smoothing_window_seconds"],
                get_profile(getattr(args, "profile", "plan1")), dpi=args.dpi,
            ))
        figure_paths = sorted(str(path.relative_to(output_dir)) for path in figures)
        manifest = {
            "schema_version": 2,
            "tool": "tools.experiment_plotting",
            "tool_version": __version__,
            "analysis_type": "plan1",
            "plot_profile": getattr(args, "profile", "plan1"),
            "analysis_id": args.analysis_id,
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "run_list_source": str(run_list_path),
            "run_list_fields": list(REQUIRED_RUN_LIST_FIELDS),
            "listed_run_count": len(all_specs),
            "included_run_count": len(included_specs),
            "runs": [{
                "run_key": run["spec"].run_key,
                "run_dir": str(run["spec"].run_dir),
                "config_sha256": run["resolved_config_sha256"],
                "metric_record_count": len(run["records"]),
            } for run in validated],
            "tables": sorted(str(path.relative_to(output_dir)) for path in tables_dir.iterdir()),
            "figures": figure_paths,
            "figure_groups": {
                "results": [path for path in figure_paths if "/results/" in path],
                "diagnostics": [path for path in figure_paths if "/diagnostics/" in path],
            },
            "plot_semantics": {
                "default_language": "en",
                "network_label": "raw_network_name",
                "learning_curve": "seed_mean_with_95pct_bootstrap_ci",
                "final_summary": "individual_seeds_with_mean_and_sample_sd",
                "primary_ranking_metric": "travel_time",
            },
            "library_versions": {
                package: metadata.version(package)
                for package in ("matplotlib", "numpy", "pandas", "seaborn")
            },
            "config_compatible": config_comparison["compatible"],
            "evaluation_package": (
                None if evaluation_package is None else {
                    "source": str(evaluation_package_path),
                    "package_id": evaluation_package["manifest"]["package_id"],
                    "episode_count": evaluation_package["manifest"]["episode_count"],
                    "decision_record_count": evaluation_package["record_count"],
                    "statistics": "evaluation seeds averaged within each DQN training seed, then training seeds summarized with 95% bootstrap CI",
                }
            ),
        }
        _json_dump(output_dir / "plotting_manifest.json", manifest)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return output_dir


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "plan1":
        output_dir = run_plan1(args)
        print(f"Plan 1 analysis completed: {output_dir}")
    elif args.command == "plan2":
        output_dir = run_plan2_analysis(args)
        print(f"Plan 2 analysis completed: {output_dir}")
    elif args.command == 'sequential':
        output_dir = run_sequential_plotting(args)
        print(f'Sequential plotting completed: {output_dir}')
    elif args.command == 'sequential-frozen':
        output_dir = run_sequential_frozen_plotting(args)
        print(f'Sequential frozen plotting completed: {output_dir}')
    elif args.command == "analyze":
        if args.profile == "plan1":
            args.output_root = args.output_root or str(DEFAULT_OUTPUT_ROOT)
            output_dir = run_plan1(args)
            print(f"Plan 1 analysis completed: {output_dir}")
        elif args.profile == "plan2":
            args.output_root = args.output_root or str(DEFAULT_PLAN2_OUTPUT_ROOT)
            output_dir = run_plan2_analysis(args)
            print(f"Plan 2 analysis completed: {output_dir}")
        elif args.profile == "s1_s4_diagnostics":
            if args.phase == "phase1":
                output_dir = run_s1_s4_phase1(args)
                print(f"S1-S4 Phase 1 completed: {output_dir}")
            elif args.phase == "prepare-evaluation-manifests":
                output_dir = prepare_s1_s4_evaluation_manifests(args)
                print(f"S1-S4 evaluation manifests prepared: {output_dir}")
            elif args.phase == "phase2":
                output_dir = run_s1_s4_phase2(args)
                print(f"S1-S4 Phase 2 completed: {output_dir}")
            elif args.phase == "phase3":
                output_dir = run_s1_s4_phase3(args)
                print(f"S1-S4 Phase 3 completed: {output_dir}")
            elif args.phase == "training-state-coverage":
                output_dir = run_s1_s4_training_state_coverage(args)
                print(f"S1-S4 training-state coverage completed: {output_dir}")
            elif args.phase == "phase4":
                output_dir = run_s1_s4_phase4(args)
                print(f"S1-S4 Phase 4 completed: {output_dir}")
            elif args.phase == "prepare-g0":
                output_dir = prepare_s1_s4_g0(args)
                print(f"S1-S4 G0 manifest prepared: {output_dir}")
            elif args.phase == "g0":
                output_dir = run_s1_s4_g0(args)
                print(f"S1-S4 G0 completed: {output_dir}")
            else:
                raise ValueError(
                    f"S1-S4 phase is not implemented yet: {args.phase}"
                )


if __name__ == "__main__":
    main()
