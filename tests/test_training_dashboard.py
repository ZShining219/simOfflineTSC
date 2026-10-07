"""Read-only experiment projection and TensorBoard continuity regressions."""
import hashlib
import json
import socket
import subprocess
import sys
import time
from urllib.parse import urlencode
from urllib.request import urlopen

import pytest

from tools.training_dashboard import ExperimentProjection, ScalarMirror


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_manifest(tmp_path):
    run = tmp_path / 'train/normal_s7'
    tasks = []
    for episode in (5, 10):
        folder = tmp_path / 'validation/normal_s7' / f'episode_{episode:04d}'
        tasks.append(dict(kind='evaluate', role='validation', run='normal_s7',
                          episode=episode, output_path=str(folder)))
        cases = [dict(kind=kind, system_time_per_vehicle=j, mean_queue=3,
                      arrived=2700, forced_removed=0)
                 for kind, j in [('normal', 340 + episode), ('road_closure', 350 + episode),
                                 ('road_closure', 370 + episode)]]
        write(folder / 'results.json', dict(protocol_hash='frozen', checkpoint_episode=episode, cases=cases))
    path = run / 'monitor/episodes.jsonl'
    path.parent.mkdir(parents=True)
    row = dict(record_type='TRAIN', episode=1, reward_per_node_mean=-.2,
               loss_mean=None, epsilon=.8)
    path.write_text(json.dumps(row) + '\n' + '{"episode": 2')
    write(run / 'environment/train/episode_0001.json',
          dict(system_time_per_vehicle=400, mean_queue=30, arrived=2600, forced_removed=1))
    # Uncommitted/future traffic records must not appear as completed episodes.
    write(run / 'environment/train/episode_0002.json', dict(system_time_per_vehicle=1))
    manifest = tmp_path / 'manifest.json'
    write(manifest, dict(root=str(tmp_path), protocol_hash='frozen', config={'episodes': 150},
                        runs=[dict(run_id='normal_s7', regime='normal', seed=7, output_path=str(run))],
                        tasks=tasks))
    return manifest


def test_projection_groups_checkpoints_and_scenarios_without_touching_sources(tmp_path):
    manifest = fixture_manifest(tmp_path)
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob('*') if p.is_file()}
    projection = ExperimentProjection(manifest)
    points = projection.points()
    assert {run for run, _, _ in points} == {'N150_seed7'}
    assert points[('N150_seed7', 'Validation/road_closure/J_seconds', 5)] == 365
    assert points[('N150_seed7', 'Validation/road_closure/J_seconds', 10)] == 370
    assert points[('N150_seed7', 'Validation/normal/J_seconds', 5)] == 345
    assert ('N150_seed7', 'Train/td_loss', 1) not in points
    assert ('N150_seed7', 'Train/J_seconds', 2) not in points
    assert points[('N150_seed7', 'Train/removed_vehicles', 1)] == 1
    assert projection.points() == points
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())


def test_projection_refreshes_changed_and_removed_records(tmp_path):
    projection = ExperimentProjection(fixture_manifest(tmp_path))
    projection.points()
    path = tmp_path / 'train/normal_s7/environment/train/episode_0001.json'
    write(path, dict(system_time_per_vehicle=390))
    assert projection.points()[('N150_seed7', 'Train/J_seconds', 1)] == 390
    path.unlink()
    assert ('N150_seed7', 'Train/J_seconds', 1) not in projection.points()


def test_projection_rejects_foreign_results(tmp_path):
    projection = ExperimentProjection(fixture_manifest(tmp_path))
    path = tmp_path / 'validation/normal_s7/episode_0005/results.json'
    row = json.loads(path.read_text())
    row['protocol_hash'] = 'different'
    write(path, row)
    with pytest.raises(ValueError, match='another protocol'):
        projection.points()


@pytest.mark.parametrize('backend', ['legacy', 'plugin'])
def test_mirror_refresh_restart_and_out_of_order_checkpoints(tmp_path, backend):
    if backend == 'legacy':
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    else:
        from tensorboard.backend.event_processing.plugin_event_accumulator import EventAccumulator
    from tensorboard.util import tensor_util
    mirror = ScalarMirror(tmp_path / 'events')
    points = {('N150_seed7', 'Train/td_loss', 20): .1,
              ('N150_seed7', 'Validation/normal/J_seconds', 10): 350}
    try:
        assert mirror.sync(points) == 2
        accumulator = EventAccumulator(str(tmp_path / 'events/N150_seed7'), size_guidance={
            'scalars' if backend == 'legacy' else 'tensors': 0})
        def values(tag):
            if backend == 'legacy':
                return [(x.step, x.value) for x in accumulator.Scalars(tag)]
            return [(x.step, float(tensor_util.make_ndarray(x.tensor_proto))) for x in accumulator.Tensors(tag)]
        accumulator.Reload()
        assert mirror.sync(points) == 0
        points[('N150_seed7', 'Validation/normal/J_seconds', 5)] = 400
        mirror.sync(points)
        accumulator.Reload()
        assert values('Validation/normal/J_seconds') == [(5, 400), (10, 350)]
        assert len(values('Train/td_loss')) == 1
        # Model rollback/revision must replace, not duplicate or retain old points.
        points[('N150_seed7', 'Validation/normal/J_seconds', 5)] = 380
        del points[('N150_seed7', 'Validation/normal/J_seconds', 10)]
        mirror.sync(points)
        accumulator.Reload()
        assert values('Validation/normal/J_seconds') == [(5, 380)]
        assert len(values('Train/td_loss')) == 1
        assert mirror.sync(points) == 0
    finally:
        mirror.close()


def test_live_tensorboard_first_late_checkpoint_does_not_duplicate_training(tmp_path):
    # Exercise the real HTTP backend, not only the older scalar accumulator.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    mirror = ScalarMirror(tmp_path / 'events')
    points = {('E150_seed27', 'Train/td_loss', i): .1 / i for i in range(1, 7)}
    points[('E150_seed27', 'Validation/normal/J_seconds', 10)] = 350
    mirror.sync(points)
    with (tmp_path / 'server.log').open('w') as log:
        server = subprocess.Popen([sys.executable, '-m', 'tensorboard.main',
            '--logdir', str(tmp_path / 'events'), '--host', '127.0.0.1', '--port', str(port),
            '--reload_interval', '1'], stdout=log, stderr=subprocess.STDOUT)
        def expect(tag, expected_steps):
            deadline = time.monotonic() + 15
            last = None
            while time.monotonic() < deadline:
                assert server.poll() is None, (tmp_path / 'server.log').read_text()
                try:
                    url = f'http://127.0.0.1:{port}/data/plugin/scalars/scalars?' + urlencode(
                        dict(run='E150_seed27', tag=tag, format='json'))
                    with urlopen(url, timeout=2) as response:
                        last = json.load(response)
                    if [row[1] for row in last] == expected_steps:
                        return last
                except OSError:
                    pass
                time.sleep(.1)
            pytest.fail(f'Unexpected live points: {last}')
        try:
            expect('Train/td_loss', list(range(1, 7)))
            expect('Validation/normal/J_seconds', [10])
            points[('E150_seed27', 'Validation/normal/J_seconds', 0)] = 1400
            mirror.sync(points)
            assert [r[2] for r in expect('Validation/normal/J_seconds', [0, 10])] == [1400, 350]
            expect('Train/td_loss', list(range(1, 7)))
            # A second repair and a new training point still stay unique.
            points[('E150_seed27', 'Validation/normal/J_seconds', 5)] = 700
            points[('E150_seed27', 'Train/td_loss', 7)] = .01
            mirror.sync(points)
            expect('Validation/normal/J_seconds', [0, 5, 10])
            expect('Train/td_loss', list(range(1, 8)))
            assert mirror.sync(points) == 0
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            mirror.close()
