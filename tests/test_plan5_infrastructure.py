import copy
import json

import numpy as np
import pytest
import torch

from sequential.io import sha256_file

from plan5.cli import run_manifest_destination
from plan5.algorithms import ddqn_target, dqn_full_vector_target
from plan5.analysis import (
    context_distinction, current_adaptation, historical_metrics,
    freeze_ppo_calibration, paired_cluster_bootstrap, policy_disagreement,
    probe_internal_metrics,
)
from plan5.analysis_package import build_inference_tables
from plan5.context import ContextHistory, context_from_events, direction_from_geometry
from plan5.config import load_config
from plan5.evaluator import (
    build_fixedtime_reference, formal_evaluation_schedule,
    full_checkpoint_schedule,
)
from plan5.frozen_evaluator import SnapshotDQNPolicy
from plan5.manifest import (
    build_ppo_calibration_manifest, build_run_manifest, calibration_run_id,
    run_id, validate_ppo_calibration_manifest, validate_run_manifest,
    validate_same_start, write_ppo_calibration_manifest, write_run_manifest,
)
from plan5.launcher import Plan5Launcher
from plan5.probe import (
    PROBE_ID, fixed_probe_inference, generate_probe,
    validate_probe_manifest, write_probe,
)
from plan5.provenance import collect_source_provenance, is_source_scope_path
from plan5.schema import (
    JSON_REQUIRED_FIELDS, RUN_MANIFEST_COLUMNS, TABLE_SCHEMAS,
    validate_json_fields, validate_table_header, write_schema_index,
    write_table,
)
from plan5.runtime import Plan5FormalRunner
from plan5.sumo_runtime import causal_gate_report
from plan5.validation import (
    validate_formal_records, validate_launch_row, validate_same_start_matrix,
)


def test_frozen_config_and_identity():
    config = load_config()
    assert config.decisions_per_episode == 360
    assert run_id("DDQN", "TRANSITION", "S3", "S4", 3) == "P5-DDQN-H34-SD3"


def test_ddqn_target_uses_online_argmax_and_bootstraps_truncation():
    online = torch.tensor([[1.0, 3.0]])
    target = torch.tensor([[9.0, 8.0]])
    assert torch.equal(ddqn_target([1.0], online, target), torch.tensor([8.6]))
    assert torch.equal(ddqn_target([1.0], online, target, terminated=[True]), torch.tensor([1.0]))
    vanilla = torch.as_tensor([1.0]) + .95 * target.max(dim=1).values
    assert torch.equal(vanilla, torch.tensor([9.55]))
    assert not torch.equal(ddqn_target([1.0], online, target), vanilla)
    predicted = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    target_full = dqn_full_vector_target(predicted, [2], [7.0])
    assert torch.equal(target_full, torch.tensor([[1.0, 2.0, 7.0, 4.0]]))
    assert torch.nn.functional.mse_loss(predicted, target_full) == 4.0


def test_context_is_causal_and_zero_padded():
    history = ContextHistory()
    assert history.value(0.0) == (0.0, 0.0, 0.0, 0.0)
    history.observe_entry(1, "North"); history.observe_entry(60, "East")
    assert history.value(60) == context_from_events([(1, "North"), (60, "East")], 60)
    assert history.value(61)[0] == 0.0
    assert direction_from_geometry((0, 0), (0, 4)) == "North"


def test_analysis_formulas_and_bootstrap():
    historical = historical_metrics({e: 100 + e for e in (0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100)})
    assert historical["episode100_degradation"] == 1.0
    current = current_adaptation({e: 100 for e in range(101)}, 100)
    assert current["current_adaptation_aulc"] == 1.0 and current["time_to_reference"] == 0
    assert current['time_to_reference_censored'] is False
    censored = current_adaptation(
        {episode: 200 for episode in range(101)}, 100,
    )
    assert censored['time_to_reference'] == '>100'
    assert censored['time_to_reference_censored'] is True
    result = paired_cluster_bootstrap([1, 2, 3, 4, 5], [0, 0, 0, 0, 0], resamples=100)
    assert len(result["effects"]) == 5 and result["bootstrap_resamples"] == 100
    assert policy_disagreement([0, 1, 1], [0, 0, 1])['disagreement'] == pytest.approx(1 / 3)
    distinction = context_distinction([
        {'scene': 'S1', 'raw16': [0] * 16,
         'arrival_counts': [1, 0, 0, 0]},
        {'scene': 'S2', 'raw16': [0] * 16,
         'arrival_counts': [0, 1, 0, 0]},
        # Same-scene duplicates are not inferential cross-scene pairs.
        {'scene': 'S2', 'raw16': [0] * 16,
         'arrival_counts': [0, 1, 0, 0]},
    ])
    assert distinction['pair_weighted_distinguished_fraction'] == 1.0
    assert distinction['matched_pairs'] == 2
    assert distinction['matched_raw_state_groups'] == 1


def test_probe_internal_metrics_cover_dqn_and_ppo_definitions():
    dqn0 = {
        'algorithm_id': 'DDQN', 'actions': [0, 1],
        'policy_vectors': [[2, 1, 0, 0, 0, 0, 0, 0],
                           [0, 2, 1, 0, 0, 0, 0, 0]],
        'bellman_residual_mean': 1.0,
    }
    dqn1 = copy.deepcopy(dqn0)
    dqn1['policy_vectors'][0][0] = 3
    metrics = probe_internal_metrics(dqn0, dqn1)
    assert metrics['q_drift_l1'] == 0.5
    assert metrics['q_drift_l2'] == 0.5
    assert metrics['bellman_residual'] == 1.0
    ppo0 = {
        'algorithm_id': 'PPO', 'actions': [0, 1],
        'policy_vectors': [[.5, .5, 0, 0, 0, 0, 0, 0],
                           [.4, .6, 0, 0, 0, 0, 0, 0]],
        'critic_values': [1.0, 2.0],
    }
    ppo1 = copy.deepcopy(ppo0)
    ppo1['policy_vectors'] = [
        [.6, .4, 0, 0, 0, 0, 0, 0],
        [.3, .7, 0, 0, 0, 0, 0, 0],
    ]
    ppo1['critic_values'] = [2.0, 4.0]
    metrics = probe_internal_metrics(ppo0, ppo1)
    assert metrics['policy_tv_from_episode0'] == pytest.approx(.1)
    assert metrics['critic_value_drift'] == 1.5


def test_same_start_rejects_different_source():
    row = {"transition_id": "H34", "algorithm_id": "DDQN", "training_seed": 0, "source_scene": "S3", "source_checkpoint_path": "x", "source_checkpoint_sha256": "y", "model_state_digest": "m", "optimizer_state_digest": "o", "algorithm_state_digest": "a", "rng_state_digest": "r"}
    other = dict(row, transition_id="L32")
    assert validate_same_start(row, other)
    other["source_checkpoint_sha256"] = "different"
    try: validate_same_start(row, other)
    except ValueError: pass
    else: raise AssertionError("same-start must reject mismatched checkpoint SHA")


def test_same_start_matrix_accepts_hold_subset_and_rejects_missing():
    rows = []
    transitions = {
        'H34': ('S3', 'S4'), 'H43': ('S4', 'S3'),
        'L23': ('S2', 'S3'), 'L32': ('S3', 'S2'),
    }
    for transition, (source, target) in transitions.items():
        for seed in range(5):
            rows.append({
                'run_type': 'TRANSITION', 'algorithm_id': 'DDQN',
                'transition_id': transition, 'training_seed': seed,
                'source_scene': source, 'target_scene': target,
                'source_checkpoint_path': f'{source}-{seed}.pt',
                'source_checkpoint_sha256': f'{source}-{seed}-sha',
                'model_state_digest': f'{source}-{seed}-model',
                'optimizer_state_digest': f'{source}-{seed}-optimizer',
                'algorithm_state_digest': f'{source}-{seed}-algorithm',
                'rng_state_digest': f'{source}-{seed}-rng',
            })
    assert validate_same_start_matrix(rows)
    with pytest.raises(ValueError, match='incomplete'):
        validate_same_start_matrix(rows[:-1])


def test_fixedtime_repeatability_gate():
    calls = {"n": 0}
    def evaluate(scene):
        calls["n"] += 1
        return {"travel_time": 10, "trajectory_digest": f"{scene}-digest"}
    result = build_fixedtime_reference(["S1", "S2"], evaluate)
    assert calls["n"] == 4 and result["scenes"]["S1"]["repeated"]


def test_fixedtime_repeatability_rejects_trajectory_change():
    calls = {'n': 0}
    def evaluate(scene):
        calls['n'] += 1
        return {'travel_time': 10, 'trajectory_digest': str(calls['n'])}
    with pytest.raises(ValueError, match='repeatability'):
        build_fixedtime_reference(['S1'], evaluate)


def test_formal_evaluation_and_checkpoint_schedules_are_frozen():
    anchor = formal_evaluation_schedule('ANCHOR', scene='S3')
    assert len(anchor) == 101
    assert anchor[0] == {'episode': 0, 'scene': 'S3', 'role': 'current'}
    transition = formal_evaluation_schedule(
        'TRANSITION', source_scene='S3', target_scene='S4',
    )
    assert len(transition) == 118
    episode100 = [row for row in transition if row['episode'] == 100]
    assert {row['scene'] for row in episode100} == {'S1', 'S2', 'S3', 'S4'}
    assert full_checkpoint_schedule('ANCHOR') == (0, 100)
    assert full_checkpoint_schedule('TRANSITION')[-1] == 100


def _formal_record_fixture(row):
    width = 20 if row['algorithm_id'] == 'CTXDDQN' else 16
    summaries = []
    for episode in range(1, 101):
        summary = {
            'episode': episode, 'decision_steps': 360,
            'mean_reward': -1.0, 'reward_std': 0.25,
            'action_counts': [45] * 8,
            'action_frequencies': [0.125] * 8,
            'model_input_summary': {
                'feature_dim': width,
                'mean': [0.0] * width, 'std': [1.0] * width,
                'min': [-1.0] * width, 'max': [1.0] * width,
            },
            'counters_end': {'global_decision_step': episode * 360},
            'trajectory_digest': f'trajectory-{episode}',
        }
        if row['algorithm_id'] in {'DDQN', 'CTXDDQN'}:
            summary.update({
                'gradient_updates': 1, 'loss_mean': 0.5,
                'epsilon_end': 0.1, 'replay_size_end': 5000,
            })
        else:
            summary['minibatch_updates'] = 16
        summaries.append(summary)
    schedule = formal_evaluation_schedule(
        'ANCHOR', scene=row['scene'],
    )
    evaluations = [{
        **cell,
        'summary': {
            'network': load_config().scenes[cell['scene']],
            'decision_steps': 360, 'simulation_duration_s': 3600,
            'travel_time': 100.0, 'throughput': 10,
            'mean_reward': -1.0, 'mean_queue': 1.0,
            'mean_delay': 0.5,
            'resolved_sumo_command': ['sumo', '-c', 'scene.sumocfg'],
        },
    } for cell in schedule]
    probes = [{
        'episode': episode, 'scene': row['scene'], 'split': split,
        'row_count': 1440 if split == 'main' else 360,
        'actions_digest': 'actions', 'policy_vectors_digest': 'vectors',
    } for episode in (0, 100) for split in ('main', 'heldout')]
    return summaries, evaluations, probes


def test_formal_completion_records_fail_closed_on_counts_and_finiteness():
    row = {
        'logical_run_id': 'P5-ANCHOR-DDQN-S2-SD0',
        'algorithm_id': 'DDQN', 'run_type': 'ANCHOR',
        'scene': 'S2', 'training_seed': 0,
    }
    summaries, evaluations, probes = _formal_record_fixture(row)
    report = validate_formal_records(row, summaries, evaluations, probes)
    assert report == {
        'episode_count': 100, 'evaluation_count': 101,
        'probe_record_count': 4, 'finite_metrics': True,
        'late_metrics_present': True,
    }
    broken = copy.deepcopy(summaries)
    broken[-1]['mean_reward'] = float('nan')
    with pytest.raises(ValueError, match='non-finite'):
        validate_formal_records(row, broken, evaluations, probes)
    with pytest.raises(ValueError, match='evaluation matrix'):
        validate_formal_records(row, summaries, evaluations[:-1], probes)
    with pytest.raises(ValueError, match='probe record matrix'):
        validate_formal_records(row, summaries, evaluations, probes[:-1])


class _ProbeAgent:
    def __init__(self, algorithm_id, input_dim):
        self.algorithm_id = algorithm_id
        self.model = torch.nn.Linear(input_dim, 8)
        self.target_model = self.model
    def full_state_dict(self):
        return {'model': copy.deepcopy(self.model.state_dict())}


def test_fixed_probe_inference_is_deterministic_and_isolated():
    rows = [{
        'raw16': [0.0] * 16, 'ctx4': [0.0] * 4,
    }] * 3
    for algorithm, width in (('DDQN', 16), ('CTXDDQN', 20)):
        agent = _ProbeAgent(algorithm, width)
        result = fixed_probe_inference(agent, rows)
        assert result['row_count'] == 3 and result['isolated']
        assert result['training_state_digest_before'] == result[
            'training_state_digest_after'
        ]


def test_frozen_dqn_policy_contract_is_argmax_not_sampling():
    assert 'argmax' in SnapshotDQNPolicy.action.__code__.co_names


def test_manifest_csv_and_fail_closed_identity(tmp_path):
    config = load_config()
    rows = build_run_manifest(config, 'a' * 40)
    result = write_run_manifest(rows, tmp_path / 'plan5_run_manifest.csv')
    assert result['rows'] == 120
    assert write_run_manifest(
        rows, tmp_path / 'plan5_run_manifest.csv'
    )['sha256'] == result['sha256']
    assert tuple((tmp_path / 'plan5_run_manifest.csv').read_text().splitlines()[0].split(',')) == RUN_MANIFEST_COLUMNS
    changed = copy.deepcopy(rows)
    changed[0]['status'] = 'complete'
    with pytest.raises(FileExistsError, match='Refusing to replace'):
        write_run_manifest(changed, tmp_path / 'plan5_run_manifest.csv')

    duplicate = rows + [dict(rows[0])]
    with pytest.raises(ValueError, match='duplicate'):
        validate_run_manifest(duplicate)
    bad_transition = copy.deepcopy(rows)
    transition = next(row for row in bad_transition if row['run_type'] == 'TRANSITION')
    transition['source_scene'] = 'S1'
    with pytest.raises(ValueError, match='source/target'):
        validate_run_manifest(bad_transition)
    bad_id = copy.deepcopy(rows)
    bad_id[0]['logical_run_id'] = 'P5-FAKE'
    with pytest.raises(ValueError, match='logical run ID'):
        validate_run_manifest(bad_id)
    missing = copy.deepcopy(rows)
    del missing[0]['source_commit']
    with pytest.raises(ValueError, match='missing fields'):
        validate_run_manifest(missing)


def test_precommit_manifest_cannot_claim_formal_identity(tmp_path):
    source = {
        'head': 'a' * 40, 'formal_source_ready': False,
        'source_scope_digest': 'c' * 64,
    }
    draft = run_manifest_destination(tmp_path, source, 'b' * 64)
    assert draft.name.startswith(
        'plan5_run_manifest_draft_aaaaaaaaaaaa_bbbbbbbbbbbb_'
    )
    assert draft.suffix == '.csv'
    source['formal_source_ready'] = True
    assert run_manifest_destination(
        tmp_path, source, 'b' * 64
    ).name.startswith('plan5_run_manifest_draft_')
    assert run_manifest_destination(
        tmp_path, source, 'b' * 64, formal=True,
    ).name == 'plan5_run_manifest.csv'
    source['formal_source_ready'] = False
    with pytest.raises(ValueError, match='clean source'):
        run_manifest_destination(
            tmp_path, source, 'b' * 64, formal=True,
        )


def test_source_scope_ignores_docs_and_outputs_but_blocks_code_config():
    assert is_source_scope_path('plan5/runtime.py')
    assert is_source_scope_path('configs/sequential/plan5.yml')
    assert is_source_scope_path('tests/test_plan5.py')
    assert not is_source_scope_path('docs/plan.md')
    assert not is_source_scope_path('core')
    assert not is_source_scope_path('data/output_data/result.json')


def test_source_provenance_hashes_dirty_source_scope():
    source = collect_source_provenance()
    assert len(source['head']) == 40
    assert len(source['source_scope_digest']) == 64
    assert source['formal_source_ready'] is source['source_scope_clean']
    assert {item['path'] for item in source['source_files']} == set(
        source['tracked_changes'] + source['untracked_source_changes']
    )


def test_ppo_calibration_identity_manifest_and_runner_contract(tmp_path):
    config = load_config()
    rows = build_ppo_calibration_manifest(config, 'a' * 40)
    assert len(rows) == 18
    assert validate_ppo_calibration_manifest(rows)
    assert calibration_run_id(2.5e-4, 0.01, 102) == (
        'P5-CAL-PPO-S2-LR25E5-EC01-SD102'
    )
    manifest_path = tmp_path / 'ppo_calibration_run_manifest.csv'
    written = write_ppo_calibration_manifest(rows, manifest_path)
    assert written['rows'] == 18
    assert validate_table_header(manifest_path)['valid']
    runner = Plan5FormalRunner(
        rows[0], tmp_path / 'calibration_attempt', probe_manifest=None,
        verify_source=False,
    )
    assert len(runner._schedule()) == 101
    assert runner._identity()['learning_rate'] == 1e-4
    assert runner._probe_cache == {}


def test_all_final_csv_and_json_schemas_are_frozen(tmp_path):
    assert 'ppo_calibration_summary.csv' in TABLE_SCHEMAS
    assert 'episode100_degradation' in TABLE_SCHEMAS[
        'plan5_transition_summary.csv'
    ]
    assert 'time_to_reference_censored' in TABLE_SCHEMAS[
        'plan5_transition_summary.csv'
    ]
    assert 'ppo_formal_config.json' in JSON_REQUIRED_FIELDS
    index = tmp_path / 'result_schema.json'
    write_schema_index(index)
    payload = json.loads(index.read_text())
    assert set(payload['tables']) == set(TABLE_SCHEMAS)
    assert set(payload['json_required_fields']) == set(JSON_REQUIRED_FIELDS)

    table = tmp_path / 'ppo_calibration_summary.csv'
    table.write_text(','.join(TABLE_SCHEMAS[table.name]) + '\n')
    assert validate_table_header(table)['valid']
    formal = tmp_path / 'ppo_formal_config.json'
    formal.write_text(json.dumps({
        field: None for field in JSON_REQUIRED_FIELDS[formal.name]
    }))
    assert validate_json_fields(formal)['valid']
    context = [{
        'probe_id': 'plan5_fixed_probe_v1',
        'pair_weighted_distinguished_fraction': 0.5,
        'unique_raw_state_weighted_distinguished_fraction': 0.25,
        'matched_raw_state_groups': 2, 'matched_pairs': 4, 'valid': True,
    }]
    written = write_table(
        tmp_path / 'plan5_context_distinction.csv',
        'plan5_context_distinction.csv', context,
    )
    assert written['rows'] == 1
    assert write_table(
        tmp_path / 'plan5_context_distinction.csv',
        'plan5_context_distinction.csv', context,
    )['sha256'] == written['sha256']
    changed = copy.deepcopy(context); changed[0]['matched_pairs'] = 5
    with pytest.raises(FileExistsError, match='immutable'):
        write_table(
            tmp_path / 'plan5_context_distinction.csv',
            'plan5_context_distinction.csv', changed,
        )


def test_inference_tables_preserve_pairs_clusters_and_cross_algorithm_scope():
    runs = []
    for algorithm_index, algorithm in enumerate(('DDQN', 'CTXDDQN', 'PPO')):
        for transition_index, transition in enumerate(
                ('H34', 'H43', 'L23', 'L32')):
            for seed in range(5):
                base = algorithm_index + transition_index + seed / 10
                summary = {
                    'algorithm_id': algorithm, 'transition_id': transition,
                    'training_seed': seed, 'target_scene': (
                        'S4' if transition == 'H34' else
                        'S3' if transition in {'H43', 'L23'} else 'S2'
                    ),
                    **{field: base for field in (
                        'worst_historical_degradation',
                        'episode100_degradation',
                        'max_policy_disagreement_main',
                        'current_adaptation_aulc',
                    )},
                }
                runs.append({
                    'summary': summary,
                    'source_checkpoint_sha256': (
                        f'{algorithm}-S3-{seed}'
                        if transition in {'H34', 'L32'}
                        else f'{algorithm}-{transition}-{seed}'
                    ),
                })
    tables = build_inference_tables(runs)
    assert len(tables['same_start']) == 72
    assert len(tables['secondary']) == 60
    assert len(tables['cross_algorithm']) == 24
    aggregates = [
        row for row in tables['same_start']
        if row['row_type'] == 'aggregate'
    ]
    assert len(aggregates) == 12
    assert all(row['bootstrap_resamples'] == 10000 for row in aggregates)
    assert all(
        row['algorithm_left'] == 'DDQN'
        and row['algorithm_right'] == 'CTXDDQN'
        for row in tables['cross_algorithm']
    )


def test_ppo_calibration_freeze_is_complete_deterministic_and_hashed(tmp_path):
    rows = []
    for learning_rate in (1e-4, 2.5e-4, 5e-4):
        for entropy in (0.001, 0.01):
            for seed in (100, 101, 102):
                run_summary = tmp_path / (
                    f'run-{learning_rate}-{entropy}-{seed}.json'
                )
                run_summary.write_text(json.dumps({'valid': True}))
                # The smallest LR/entropy pair wins every ordered criterion.
                penalty = learning_rate * 10000 + entropy
                rows.append({
                    'learning_rate': learning_rate,
                    'entropy_coefficient': entropy,
                    'training_seed': seed,
                    'logical_run_id': calibration_run_id(
                        learning_rate, entropy, seed,
                    ),
                    'status': 'completed', 'tt_episode0': 200.0,
                    'tt_late_median': 100.0 + penalty,
                    'tt_fixedtime': 150.0,
                    'current_adaptation_aulc': 1.0 + penalty,
                    'run_summary_path': str(run_summary),
                    'run_summary_sha256': sha256_file(run_summary),
                })
    result = freeze_ppo_calibration(
        rows, tmp_path / 'ppo_calibration_summary.csv',
        tmp_path / 'ppo_formal_config.json', 'c' * 64,
    )
    assert result['formal_status'] == 'READY'
    config = json.loads((tmp_path / 'ppo_formal_config.json').read_text())
    assert config['learning_rate'] == 1e-4
    assert config['entropy_coefficient'] == 0.001
    assert config['eligible_seed_count'] == 3
    assert validate_table_header(
        tmp_path / 'ppo_calibration_summary.csv'
    )['valid']


def test_launcher_preserves_attempt_lineage_and_completed_outputs(tmp_path):
    config = load_config()
    row = build_run_manifest(config, 'a' * 40)[0]
    launcher = Plan5Launcher(tmp_path)
    first = launcher.reserve_attempt(row)
    assert first['attempt_id'] == 'attempt_1'
    launcher.update_attempt(
        row['logical_run_id'], first['attempt_id'], 'failed',
        failure_class='synthetic',
    )
    second = launcher.reserve_attempt(row, resume_from='checkpoint.pt')
    assert second['attempt_id'] == 'attempt_2'
    launcher.update_attempt(
        row['logical_run_id'], second['attempt_id'], 'completed',
    )
    with pytest.raises(FileExistsError, match='already complete'):
        launcher.reserve_attempt(row)


def test_launch_batch_freezes_only_fully_bound_rows(tmp_path):
    config = load_config()
    launcher = Plan5Launcher(tmp_path)
    anchors = [
        row for row in build_run_manifest(config, 'a' * 40)
        if row['algorithm_id'] == 'DDQN' and row['run_type'] == 'ANCHOR'
    ]
    result = launcher.write_launch_batch(
        anchors, tmp_path / 'launch_manifests' / 'ddqn_anchors.csv',
    )
    assert result['rows'] == 20
    ppo = next(
        row for row in build_run_manifest(config, 'a' * 40)
        if row['algorithm_id'] == 'PPO' and row['run_type'] == 'ANCHOR'
    )
    with pytest.raises(ValueError, match='PPO launch row'):
        launcher.write_launch_batch(
            [ppo], tmp_path / 'launch_manifests' / 'bad_ppo.csv',
        )


def test_launch_validator_rejects_unbound_ppo_and_transition():
    rows = build_run_manifest(load_config(), 'a' * 40)
    ppo = next(
        row for row in rows
        if row['algorithm_id'] == 'PPO' and row['run_type'] == 'ANCHOR'
    )
    with pytest.raises(ValueError, match='PPO launch row'):
        validate_launch_row(ppo)
    transition = next(
        row for row in rows
        if row['algorithm_id'] == 'DDQN'
        and row['run_type'] == 'TRANSITION'
    )
    with pytest.raises(ValueError, match='checkpoint-bound'):
        validate_launch_row(transition)


def _probe_rows():
    rows = []
    for scene, network in load_config().scenes.items():
        for seed in range(5):
            split = 'main' if seed <= 3 else 'heldout'
            for decision in range(1, 361):
                counts = [decision % 3, 0, 0, 0]
                rows.append({
                    'probe_id': PROBE_ID, 'split': split,
                    'scene': scene, 'network': network,
                    'source_training_seed': seed,
                    'decision_index': decision,
                    'simulation_time_s': (decision - 1) * 10,
                    'raw16': [0.0] * 16,
                    'ctx4': [counts[0] / 60.0, 0.0, 0.0, 0.0],
                    'arrival_counts': counts, 'source_action': 0,
                    'source_checkpoint_path': f'/checkpoint/{scene}/{seed}',
                    'source_checkpoint_sha256': f'checkpoint-{scene}-{seed}',
                    'source_trajectory_sha256': f'trajectory-{scene}-{seed}',
                    'resolved_sumo_command_digest': f'command-{scene}',
                })
    return rows


def test_probe_atomic_promotion_and_full_validation(tmp_path):
    destination = tmp_path / PROBE_ID
    manifest = write_probe(_probe_rows(), destination)
    assert destination.is_dir()
    assert manifest['validation']['rows'] == 7200
    assert validate_probe_manifest(destination / 'probe_manifest.json')[
        'manifest_digest'
    ] == manifest['manifest_digest']
    with pytest.raises(FileExistsError):
        write_probe(_probe_rows(), destination)


def test_probe_failure_leaves_no_partial_directory(tmp_path):
    destination = tmp_path / PROBE_ID
    with pytest.raises(ValueError, match='overwrites frozen fields'):
        write_probe(_probe_rows(), destination, metadata={'probe_id': 'bad'})
    assert not destination.exists()
    assert not list(tmp_path.glob(f'.{PROBE_ID}.staging-*'))


def test_probe_rejects_split_and_hash_tampering(tmp_path):
    rows = _probe_rows()
    rows[0]['split'] = 'heldout'
    with pytest.raises(ValueError, match='seed/split'):
        write_probe(rows, tmp_path / 'bad-split')

    destination = tmp_path / PROBE_ID
    write_probe(_probe_rows(), destination)
    with open(destination / 'main.jsonl', 'a', encoding='utf-8') as handle:
        handle.write(json.dumps({'tampered': True}) + '\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_probe_manifest(destination / 'probe_manifest.json')


def test_causal_gate_report_covers_all_four_gates():
    decisions = [
        {'simulation_time_s': 0.0, 'arrival_counts': [0, 0, 0, 0]},
        {'simulation_time_s': 10.0, 'arrival_counts': [1, 0, 0, 0]},
    ]
    evaluation = {
        'summary': {'entry_events': [
            {'time_s': 1.0, 'direction': 'North', 'vehicle_id': 'v1'},
        ]},
        'decisions': decisions,
    }
    report = causal_gate_report(evaluation)
    assert report['valid'] and all(
        report[key] for key in (
            'past_only', 'future_mutation_invariant',
            'reset_zero_padding', 'offline_recompute_exact',
        )
    )
    corrupted = copy.deepcopy(evaluation)
    corrupted['decisions'][1]['arrival_counts'][0] = 2
    assert not causal_gate_report(corrupted)['valid']


def test_probe_generation_failure_does_not_promote_trajectories(
        tmp_path, monkeypatch):
    assets = []
    for scene, network in load_config().scenes.items():
        for seed in range(5):
            assets.append({
                'scene': network, 'training_seed': seed,
                'availability': 'available',
                'checkpoint_path': f'/checkpoint/{scene}/{seed}',
                'checkpoint_sha256': f'checkpoint-{scene}-{seed}',
            })
    calls = {'count': 0}
    def failing_evaluation(network, run_dir, policy, **kwargs):
        calls['count'] += 1
        if calls['count'] == 7:
            raise RuntimeError('synthetic SUMO failure')
        identity = kwargs['source_identity']
        decisions = []
        for decision in range(1, 361):
            decisions.append({
                **identity, 'decision_index': decision,
                'simulation_time_s': float((decision - 1) * 10),
                'raw16': [0.0] * 16, 'ctx4': [0.0] * 4,
                'arrival_counts': [0] * 4, 'source_action': 0,
                'source_trajectory_sha256': f'trajectory-{calls["count"]}',
            })
        return {
            'decisions': decisions,
            'summary': {
                'resolved_sumo_command_digest': 'command',
                'trajectory_digest': f'trajectory-{calls["count"]}',
                'resolved_sumo_command': ['sumo', '-c', network],
                'boundary_entry_mapping': {'edge': 'North'},
                'entry_events': [],
            },
        }
    monkeypatch.setattr(
        'plan5.sumo_runtime.evaluate_policy', failing_evaluation,
    )
    output = tmp_path / PROBE_ID
    evaluations = tmp_path / 'evaluation_runs'
    with pytest.raises(RuntimeError, match='synthetic SUMO failure'):
        generate_probe(
            {'assets': assets}, output, evaluations,
        )
    assert not output.exists() and not evaluations.exists()
    assert not list(tmp_path.glob(f'.{PROBE_ID}.generation-*'))
