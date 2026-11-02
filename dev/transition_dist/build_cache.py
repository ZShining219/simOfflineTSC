#!/usr/bin/env python3
"""Build transition-distribution cache over ALL usable trajectory assets.

Trajectory schemas:
  A) tsc/sumo_dqn/<scene>/<run>/trajectory/episodes/episode_XXXX.npz
     has instantaneous 'queue' (per decision point) + reward.
  B) <root>/<...>/trajectory/episodes/stage_XX_episode_*.npz
     no queue field -> uniform queue proxy q_hat = -reward * n_lanes / 12
     (reward = -12 * mean incoming-lane waiting count; verified numerically:
      -2/3*r never exceeds sum(state), while -8*r violates ~90-99%).

Uniform cross-schema queue: q_hat = -2/3 * reward  (8 in-lanes, hz1x1).
Schema A additionally stores instantaneous 'queue' -> qi arrays.

Scene resolution:
  A) parent dir name.
  B) attempt/run child_run_manifest.json 'networks'[stage-1]; fallbacks:
     logical_run_manifest -> campaign manifest children[logical_run_id];
     HITL run-id tag map (-QC/-KN/-SB -> config2/3/4); else 'unknown'.

Output: dev/transition_dist/cache/<group>__<label>.npz with arrays
  q, qpre, dq, a  (+ qi for schema A)
and datasets_index.json describing each unit.
"""
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data/output_data')
CACHE = os.path.join(ROOT, 'dev/transition_dist/cache')
INDEX = os.path.join(CACHE, 'datasets_index.json')

CAP_A = 100  # max episodes per schema-A run
CAP_B = 12   # max episodes per schema-B attempt/run

STAGE_RE = re.compile(r'stage_(\d+)_episode_(\d+)')
HITL_TAG = {'QC': 'sumohz1x1_config2', 'KN': 'sumohz1x1_config3',
            'SB': 'sumohz1x1_config4'}


def short(net):
    return 'base' if net == 'sumohz1x1' else net.replace('sumohz1x1_', '')


def take_stride(files, cap):
    if len(files) <= cap:
        return files
    step = len(files) / cap
    return [files[int(i * step)] for i in range(cap)]


def load_npz(path):
    with np.load(path, allow_pickle=False) as d:
        r = d['reward'].reshape(-1)
        q_hat = -2.0 * r / 3.0
        a = d['action'].reshape(-1).astype(np.int64)
        qi = (d['queue'].reshape(-1).astype(np.float64)
              if 'queue' in d.files else np.empty(0, dtype=np.float64))
    return q_hat, qi, a


def resolve_networks(cand_dirs):
    """first dir containing a manifest with 'networks' wins."""
    for d in cand_dirs:
        for name in ('child_run_manifest.json', 'logical_run_manifest.json'):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                try:
                    nets = json.load(open(p)).get('networks')
                    if nets:
                        return nets
                except Exception:
                    pass
        for p in sorted(glob.glob(os.path.join(d, '*manifest*.json'))):
            try:
                nets = json.load(open(p)).get('networks')
                if nets:
                    return nets
            except Exception:
                continue
    return []


def emit(label, uid, group, scene, src, run, files_used, buffers):
    if not buffers['q']:
        return None
    q = np.concatenate(buffers['q'])
    rec = {
        'label': label, 'group': group, 'scene': scene, 'source': src,
        'run': run, 'n_files': int(files_used), 'n_transitions': int(len(q)),
        'q_mean': float(q.mean()), 'q_p50': float(np.median(q)),
        'q_p90': float(np.quantile(q, 0.9)), 'q_max': float(q.max()),
    }
    if buffers['qi']:
        qi = np.concatenate(buffers['qi'])
        rec['qi_mean'] = float(qi.mean())
        rec['qi_vs_qhat_corr'] = float(np.corrcoef(qi, q)[0, 1])
    arrs = {'q': q, 'qpre': np.concatenate(buffers['qpre']),
            'dq': np.concatenate(buffers['dq']),
            'a': np.concatenate(buffers['a'])}
    if buffers['qi']:
        arrs['qi'] = np.concatenate(buffers['qi'])
    np.savez_compressed(
        os.path.join(CACHE, f'{uid}.npz'), **arrs)
    rec['uid'] = uid
    return rec


def fill(buf, f):
    q, qi, a = load_npz(f)
    buf['q'].append(q)
    buf['qpre'].append(q[:-1])
    buf['dq'].append(q[1:] - q[:-1])
    buf['a'].append(a[1:])
    if qi.size:
        buf['qi'].append(qi)


def collect_schema_a(index):
    for scene_dir in sorted(glob.glob(os.path.join(OUT, 'tsc/sumo_dqn/*/'))):
        scene = os.path.basename(scene_dir.rstrip('/'))
        for run in sorted(os.listdir(scene_dir)):
            epdir = os.path.join(scene_dir, run, 'trajectory/episodes')
            if not os.path.isdir(epdir):
                continue
            files = take_stride(
                sorted(glob.glob(os.path.join(epdir, '*.npz'))), CAP_A)
            if not files:
                continue
            buf = {'q': [], 'qi': [], 'qpre': [], 'dq': [], 'a': []}
            for f in files:
                try:
                    fill(buf, f)
                except Exception as e:
                    print('  skip', f, e)
            label = f'{short(scene)}::{run[:40]}'
            uid = f'A_train__{scene}__{run[:60]}'
            rec = emit(label, uid, 'A_train', scene,
                       'train_npz', run, len(files), buf)
            if rec:
                index.append(rec)


def collect_schema_b(index):
    specs = [
        ('ha_sodqn', 'ha_sodqn/*/*/attempts/*/trajectory/episodes'),
        ('seq8',     'sequential/seq8_*/*/attempts/*/trajectory/episodes'),
        ('plan34',   'sequential/plan34_*/*/attempts/*/trajectory/episodes'),
        ('hitl',     'hitl_probe/*/runs/*/attempt_*/trajectory/episodes'),
        ('hitl',     'hitl_probe/*/runs/*/trajectory/episodes'),
        ('p1freeze', 'plan1_freeze_reproduction_67f5840_20260723/**/'
                     'trajectory/episodes'),
    ]
    seen = set()
    for group, pat in specs:
        for epdir in sorted(glob.glob(os.path.join(OUT, pat), recursive=True)):
            if epdir in seen:
                continue
            seen.add(epdir)
            rel = os.path.relpath(epdir, OUT).split(os.sep)
            batch = rel[1]  # <batch|probe-group> right under the root
            holder = os.path.dirname(os.path.dirname(epdir))  # attempt or run dir
            if os.path.basename(holder).startswith('attempt'):
                attempt = holder
                run_dir = os.path.dirname(attempt)
                if os.path.basename(run_dir) == 'attempts':
                    run_dir = os.path.dirname(run_dir)
                run_id = f'{os.path.basename(run_dir)}__{os.path.basename(attempt)}'
            else:  # trajectory sits directly under the run dir (e.g. multiscene)
                attempt, run_dir = holder, os.path.dirname(holder)
                run_id = os.path.basename(attempt)
            networks = resolve_networks([attempt, run_dir])
            if not networks and group == 'hitl':
                for tag, net in HITL_TAG.items():
                    if f'-{tag}-' in run_id:
                        networks = [net]
                        break
                if not networks:
                    networks = ['sumohz1x1']
            files = take_stride(
                sorted(glob.glob(os.path.join(epdir, '*.npz'))), CAP_B)
            by_scene = {}
            for f in files:
                m = STAGE_RE.search(os.path.basename(f))
                st = int(m.group(1)) - 1 if m else 0
                scene = (networks[st] if st < len(networks)
                         else ('unknown' if group != 'p1freeze' else
                               scene_from_p1freeze(f)))
                if scene == 'unknown' and group == 'hitl':
                    scene = 'sumohz1x1'
                by_scene.setdefault(scene, []).append(f)
            for scene, fl in by_scene.items():
                buf = {'q': [], 'qi': [], 'qpre': [], 'dq': [], 'a': []}
                for f in fl:
                    try:
                        fill(buf, f)
                    except Exception as e:
                        print('  skip', f, e)
                label = f'{short(scene)}::{batch}::{run_id[:36]}'
                uid = f'{group}__{batch}__{scene}__{run_id[:48]}'
                rec = emit(label, uid, group, scene,
                           'eval_npz', run_id, len(fl), buf)
                if rec:
                    index.append(rec)


def scene_from_p1freeze(f):
    m = re.search(r'/sumo_dqn/([^/]+)/', f)
    return m.group(1) if m else 'unknown'


def main():
    os.makedirs(CACHE, exist_ok=True)
    index = []
    collect_schema_a(index)
    collect_schema_b(index)
    with open(INDEX, 'w') as fh:
        json.dump(index, fh, indent=1, ensure_ascii=False)
    print(f'wrote {len(index)} dataset units -> {INDEX}')


if __name__ == '__main__':
    main()
