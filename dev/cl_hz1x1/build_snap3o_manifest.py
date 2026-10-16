"""Build a same-scene evaluation collection manifest over snap3o reruns.

Each snap3o stage-3 rerun (o3 st3, c3->s1) wrote resumable checkpoints at
episodes {0,5,10,15,20,50,100} — these hold the ONLINE model weights
(model/{e}_0.pt only stores the target net for dqn-family).  The manifest
lists every (backbone, condition, seed, episode) checkpoint as a controller,
evaluated on sumohz1x1 with identical evaluation seeds -> identical vehicle
realizations across controllers.

Output: artifacts/cl_hz1x1/snap3o_eval/collection_manifest.json
Run it via: python run.py --evaluation-manifest <m> --evaluation-output <pkg>

Run: python dev/cl_hz1x1/build_snap3o_manifest.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'output_data' / 'tsc'
ART = ROOT / 'artifacts' / 'cl_hz1x1' / 'snap3o_eval'

EPS = (0, 5, 10, 15, 20, 50, 100)
EVAL_SEEDS = [101]
CELLS = {
    'ddqn': ('sumo_dqn', {'clear': 'snap3o_ddqn_clear_o3',
                         'semi': 'snap3o_ddqn_semi_o3',
                         'p': 'snap3o_ddqn_p_o3'}),
    'duel': ('sumo_dqn', {'clear': 'snap3o_duel_clear_o3',
                          'semi': 'snap3o_duel_semi_o3',
                          'p': 'snap3o_duel_p_o3'}),
}
# snap3o run id: snap3o_{bb}_{cond}_o3_sd{seed}_st3 — prefixes above end at
# _o3 so the run dir is prefix + _sd{seed}_st3.


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ART.mkdir(parents=True, exist_ok=True)
    controllers, missing = [], []
    for bb, (adir, prefixes) in CELLS.items():
        for cond, prefix in prefixes.items():
            for sd in (0, 1, 2):
                run_dir = OUT / adir / 'sumohz1x1' / f'{prefix}_sd{sd}_st3'
                if not (run_dir / 'run_status.json').exists():
                    missing.append(str(run_dir))
                    continue
                for ep in EPS:
                    ckpt = (run_dir / 'checkpoints' / 'resumable'
                            / f'episode_{ep:04d}.pt')
                    if not ckpt.exists():
                        missing.append(str(ckpt))
                        continue
                    controllers.append({
                        'controller_id': f'{bb}_{cond}_sd{sd}_e{ep}',
                        'agent': 'dqn',
                        'network': 'sumohz1x1',
                        'training_seed': sd,
                        'run_dir': str(run_dir),
                        'checkpoint': str(ckpt),
                        'checkpoint_path': str(ckpt),
                        'checkpoint_episode': ep,
                        'checkpoint_role': 'resumable',
                        'checkpoint_sha256': sha256(ckpt),
                        'evaluation_steps': 3600,
                    })
    manifest = {
        'schema_version': 1,
        'package_id': 'cl_hz1x1_snap3o_timeseries',
        'world': 'sumo',
        'evaluation_seeds': EVAL_SEEDS,
        'sampling_interval_seconds': 10,
        'smoothing_window_seconds': 60,
        'metrics': ['reward', 'queue', 'delay', 'throughput', 'travel_time'],
        'record_state_diagnostics': False,
        'controllers': controllers,
    }
    path = ART / 'collection_manifest.json'
    path.write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print(f'controllers={len(controllers)} missing={len(missing)}')
    for m in missing:
        print('  missing:', m)
    print('manifest:', path)


if __name__ == '__main__':
    main()
