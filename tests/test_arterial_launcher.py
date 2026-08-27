import yaml

from arterial_run import command_for_stage
from arterial.experiment import (build_cross_scene_evaluation_manifest,
                                 stage_overlay, validate_experiment_config)


def test_all_three_orders_can_generate_each_stage_command():
    base = yaml.safe_load(open('configs/arterial/sequential_online.yml'))
    for order in ('order_1', 'order_2', 'order_3'):
        config = validate_experiment_config({**base, 'scene_order': order})
        for stage in range(4):
            overlay = stage_overlay(config, stage)
            command = command_for_stage(config, stage, 'overlay.yml', 'run')
            assert overlay['model']['scene_id'] == config['scene_order'][stage]
            assert '--experiment-config' in command
            assert command[command.index('-n') + 1].startswith('sumoarterial1x6_')


def test_full_history_uses_frozen_noncausal_upper_bound_label():
    raw = yaml.safe_load(open('configs/arterial/semi_offline_full_history.yml'))
    config = validate_experiment_config(raw)
    overlay = stage_overlay(config, 0)
    assert overlay['model']['history_access_mode'] == 'full'
    assert overlay['run_metadata']['history_mode'] == (
        'non_causal_full_history_upper_bound')
    assert overlay['run_metadata']['non_causal_full_history_upper_bound'] is True
    assert overlay['run_metadata']['full_history_noncausal_upper_bound'] is True


def test_cli_style_seed_and_budget_overrides_validate():
    raw = yaml.safe_load(open('configs/arterial/independent_online.yml'))
    raw.update(training_seed=1000, episode_budget=225, scene_order='order_2')
    config = validate_experiment_config(raw)
    overlay = stage_overlay(config, 0)
    assert overlay['run_metadata']['training_seed'] == 1000
    assert overlay['trainer']['episodes'] == 225
    assert overlay['model']['scene_id'] == '700_0.6'


def test_cross_scene_manifest_contains_four_explicit_targets(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'run_manifest.json').write_text(
        '{"agent":"shared_dqn","network":"sumoarterial1x6_300_06"}')
    targets = {scene: tmp_path / scene for scene in (
        '300_0.3', '300_0.6', '700_0.3', '700_0.6')}
    payload = build_cross_scene_evaluation_manifest(
        tmp_path / 'manifest.json', source, targets,
        'checkpoints/resumable/episode_0001.pt', 1, '300_0.6', 0)
    assert len(payload['controllers']) == 4
    assert {x['evaluation_scene'] for x in payload['controllers']} == set(targets)
    assert all(x['checkpoint_role'] == 'resumable'
               for x in payload['controllers'])
