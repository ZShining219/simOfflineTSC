import json
import subprocess
import sys
from pathlib import Path

import yaml

from arterial.experiment import ARTERIAL_SCENES


def test_stage1_manifest_has_frozen_twenty_run_matrix(tmp_path):
    output = tmp_path / 'manifest.json'
    overlays = tmp_path / 'overlays'
    subprocess.run([
        sys.executable, 'tools/build_arterial_stage1_manifest.py',
        '--output', str(output), '--overlay-dir', str(overlays),
    ], check=True)
    payload = json.loads(output.read_text(encoding='utf-8'))
    tasks = payload['tasks']
    assert len(tasks) == 20
    assert {(task['scene'], task['training_seed']) for task in tasks} == {
        (scene, seed) for scene in ('300_0.3', '300_0.6', '700_0.3', '700_0.6')
        for seed in range(5)}
    assert all(task['episode_budget'] == 400 for task in tasks)
    assert all(task['offline_ratio'] == 0.0 for task in tasks)
    assert len({task['output_path'] for task in tasks}) == 20
    assert payload['collector_training_seeds'] == [0, 1, 2, 3, 4]
    assert payload['sequential_training_seeds'] == [1000, 1001, 1002, 1003, 1004]
    for task in tasks:
        command = task['command']
        assert command[command.index('--scene') + 1] == task['scene']
        assert command[command.index('--system-profile-hash') + 1] == (
            payload['system_profile_hash'])
        assert command[command.index('--system-profile-path') + 1] == (
            payload['system_profile_path'])
        subprocess.run([
            *command[:-1],
        ], check=True, capture_output=True, text=True)
        overlay_path = Path(command[command.index('--output-overlay') + 1])
        overlay = yaml.safe_load(overlay_path.read_text(encoding='utf-8'))
        assert overlay['model']['scene_id'] == task['scene']
        assert overlay['model']['scene_order'] == [task['scene']]
        assert overlay['run_metadata']['scene'] == task['scene']
        assert overlay['run_metadata']['simulator_network'] == (
            ARTERIAL_SCENES[task['scene']])
