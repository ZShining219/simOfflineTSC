"""P0 evidence for physical events under corrected signals and immutable inputs.

No P1 training, strength calibration or formal evaluation is performed here.
All generated evidence is written under pytest's temporary directory.
"""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from world.sumo_signal_control import install_signal_control
from world.sumo_events import load_schedule
from world.sumo_events import __main__ as validation


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('seed,seconds', [(7, 3600), (17, 300)])
def test_physical_events_with_corrected_fixedtime(seed, seconds, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    original = validation.World

    def controlled_world(*args, **kwargs):
        world = original(*args, **kwargs)
        install_signal_control(world, {'version': 'sumo-green-yellow-v1', 'yellow_seconds': 5})
        return world

    monkeypatch.setattr(validation, 'World', controlled_world)
    args = SimpleNamespace(schedule='configs/events/hz4x4.yml', seconds=seconds,
                           sim_config='configs/sim/hz4x4.cfg', interface='libsumo',
                           seeds=[seed], repeats=2, phase_seconds=20,
                           output=str(tmp_path / 'physical'))
    summary = validation.validate(args)
    assert summary['status'] == 'passed'
    assert len({r['trace_sha256'] for r in summary['runs']}) == 1
    summary['signal_control'] = {'version': 'sumo-green-yellow-v1', 'yellow_seconds': 5}
    summary['invocation'] = [sys.executable, *sys.argv]
    for path in ['world/sumo_signal_control.py', 'agent/fixedtime.py',
                 'tests/test_paper_p0.py', 'configs/tsc/paper_p0.yml']:
        summary['files_sha256'][path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    (tmp_path / 'p0_validation.json').write_text(json.dumps(summary, indent=2) + '\n')


def test_frozen_grounding_dependencies_preserved():
    manifest = json.loads((ROOT / 'world/sumo_events/grounding_v1.json').read_text())
    changed = [name for name, digest in manifest['files_sha256'].items()
               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    assert changed == []


def test_fixed_schedule_snapshot_does_not_follow_file_changes(tmp_path):
    source = ROOT / 'configs/events/hz4x4.yml'
    path = tmp_path / 'schedule.yml'
    path.write_bytes(source.read_bytes())
    schedule = load_schedule(path)
    digest = hashlib.sha256(json.dumps(schedule.to_dict(), sort_keys=True).encode()).hexdigest()
    path.write_text('schema_version: sumo-events-v1\nevents: []\n')
    assert len(schedule.events) == 3
    assert hashlib.sha256(json.dumps(schedule.to_dict(), sort_keys=True).encode()).hexdigest() == digest
    assert load_schedule(path).events == ()
