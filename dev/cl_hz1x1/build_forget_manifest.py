"""Forgetting-side microscopic manifest: old-scene s1 evals along a chain.

For the worst-forgetting cells (o2 order s1->c4->c2->c3), evaluate every
stage-end checkpoint back on s1 under one eval seed — the same online-net
resumable payloads as snap3o — to see WHAT the ~5-10s old-scene ΔATT looks
like microscopically (queue(t), throughput(t), per-lane starvation).

Extra mid-stage checkpoints (st4 ep0/ep50 for frap, st3 ep0/ep50 for duel)
show WHERE inside a stage the old-scene skill erodes.

Schema v2 (source_network != target_network allowed): each stage run dir
keeps its own scene config; target_run_dir points at the same-backbone s1
baseline run so the eval world is sumohz1x1 for every controller.

Run: python dev/cl_hz1x1/build_forget_manifest.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'output_data' / 'tsc'
ART = ROOT / 'artifacts' / 'cl_hz1x1' / 'forget_eval'

EVAL_SEEDS = [101]
# o2 stage -> trained scene
O2_SCENE = {1: 'sumohz1x1', 2: 'sumohz1x1_config4',
            3: 'sumohz1x1_config2', 4: 'sumohz1x1_config3'}
# (bb, agent_dir, seed): {cond: stage run prefix}
CELLS = {
    'frap': {
        'adir': 'sumo_frap', 'seeds': [0, 1],
        'replay': 'so1_sofrap_o2', 'p': 'soa1_sofrapp_o2',
        'midstage': {4: (0, 50, 100)},  # st4 intra-stage erosion
    },
    'duel': {
        'adir': 'sumo_dqn', 'seeds': [2],
        'replay': 'so1_soduel_o2', 'p': 'soa1_soduelp_o2',
        'midstage': {3: (0, 50, 100)},  # duel sd2 st3 transient forgetting
    },
}
BASE_RUN = {
    'frap': 'sumo_frap/sumohz1x1/cl1_frap_base_s1_sd{sd}',
    'duel': 'sumo_dqn/sumohz1x1/cl1_duel_base_s1_sd{sd}',
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def controller(cid, agent, run_dir, ckpt, ep, sd, source_net, target_dir):
    return {
        'controller_id': cid,
        'agent': agent,
        'network': 'sumohz1x1',
        'source_network': source_net,
        'target_network': 'sumohz1x1',
        'target_run_dir': str(target_dir),
        'training_seed': sd,
        'run_dir': str(run_dir),
        'checkpoint': str(ckpt),
        'checkpoint_path': str(ckpt),
        'checkpoint_episode': ep,
        'checkpoint_role': 'resumable',
        'checkpoint_sha256': sha256(ckpt),
        'source_policy': agent,
        'evaluation_steps': 3600,
    }


def main():
    ART.mkdir(parents=True, exist_ok=True)
    controllers, missing = [], []
    for bb, spec in CELLS.items():
        adir, agent = spec['adir'], ('frap' if bb == 'frap' else 'dqn')
        for sd in spec['seeds']:
            target_dir = (OUT / BASE_RUN[bb].format(sd=sd))
            for cond, pref in (('r', spec['replay']), ('p', spec['p'])):
                for st in (1, 2, 3, 4):
                    scene = O2_SCENE[st]
                    run_dir = OUT / adir / scene / f'{pref}_sd{sd}_st{st}'
                    eps = set(spec['midstage'].get(st, ())) | {100}
                    for ep in sorted(eps):
                        ckpt = (run_dir / 'checkpoints' / 'resumable'
                                / f'episode_{ep:04d}.pt')
                        cid = f'{bb}{cond}_sd{sd}_st{st}_e{ep}'
                        if not ckpt.exists():
                            missing.append(str(ckpt))
                            continue
                        controllers.append(controller(
                            cid, agent, run_dir, ckpt, ep, sd, scene,
                            target_dir))
            # same-scene baseline reference
            bdir = OUT / BASE_RUN[bb].format(sd=sd)
            bck = bdir / 'checkpoints' / 'resumable' / 'episode_0100.pt'
            if bck.exists():
                controllers.append(controller(
                    f'{bb}_base_sd{sd}', agent, bdir, bck, 100, sd,
                    'sumohz1x1', target_dir))
            else:
                missing.append(str(bck))
    manifest = {
        'schema_version': 2,
        'package_id': 'cl_hz1x1_forget_micro',
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
