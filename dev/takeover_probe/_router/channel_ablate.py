"""Channel attribution on the frozen routers: state channel vs report text.

Under the legacy 'flag' layout the lane token's feature 3 (event_flag)
and the text vector's lane indicator always agree — both come from
``_report_fields`` — so attention-binding evidence cannot tell whether
the mechanism reads the text or just tracks the flag.  Under the
'identity' layout the redundancy is removed: lane identity dims carry
edge membership, the text edge field shares that index space, and the
ablation separates *where the report points* from *what lanes claim*.

This script recovers snapshots from ``probe_repr/z_dump.npz``
(raw = lanes + text + global), re-encodes them under each frozen router
with the channels perturbed independently, and measures

  binding   attention mass / argmax on true vs alternative event edge
            (attn: cross-attn row; selfattn: received column mass)
  z-shift   cosine distance from the full-input z
  q-flip    router argmax controller change rate

Conditions, flag layout (legacy):
  full            as recorded
  no_flag         lanes[:,3] = 0, text intact
  no_text         report vector replaced by the kind='none' encoding
  no_text_edge    lane-indicator block zeroed, kind/scalars kept
  text_wrong      text points at a DIFFERENT edge, flag stays true
  flag_wrong      flag moved to a different edge, text stays true

Conditions, identity layout:
  full / no_text / no_text_edge / text_wrong   as above (edge block)
  no_lane_id      lane identity dims zeroed — binding impossible
  laneid_wrong    true/wrong edges swap lane identities — if attention
                  follows the REPORT name it lands on the wrong lanes
  (+ no_flag / flag_wrong when the optional flag dim is present)

No SUMO, no retraining — encoder forward passes only.
Usage:
    python -m dev.takeover_probe.channel_ablate \
        --dump output_data/dev_takeover_probe/probe_repr/z_dump.npz \
        --records-dir output_data/dev_takeover_probe/eval_p2_obj3 \
        --out output_data/dev_takeover_probe/probe_repr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dev.takeover_probe.dispatch_model import (  # noqa: E402
    DispatchRouter, collate,
)
from tools.sumo_live_broadcast.injects import EVENT_KINDS  # noqa: E402

ARMS3 = ['attn', 'selfattn', 'concat']
N_KINDS = len(EVENT_KINDS)
SCALARS = 6
PHYS = 3
# TEST_EVENTS order == episode prefix e0..e3 used by probe_repr
CELL_EDGES = ['road_0_1_0', 'road_1_0_1', 'road_2_1_2', '']
CONDS_FLAG = ['full', 'no_flag', 'no_text', 'no_text_edge',
              'text_wrong', 'flag_wrong']
CONDS_ID = ['full', 'no_text', 'no_text_edge', 'text_wrong',
            'no_lane_id', 'laneid_wrong']


def lane_order(records_dir):
    """fb.lanes order = sorted lane_queue keys of any record row."""
    f = sorted(Path(records_dir).glob('records_*.jsonl'))[0]
    row = json.loads(open(f).readline())
    return sorted(row['lane_queue'].keys())


def edge_lanes(lanes, edge):
    return [i for i, l in enumerate(lanes)
            if l.rsplit('_', 1)[0] == edge]


def dump_spec(d, lanes=None):
    """Feature-layout spec for a z_dump npz.

    Dumps written with the identity layout carry a 'meta' json holding
    the exact dims; older dumps (no meta) are the legacy 'flag' layout,
    sized from the lane list.
    """
    if 'meta' in d.files:
        m = json.loads(d['meta'].item())
        return dict(layout=m['feature_layout'], n_lanes=m['n_lanes'],
                    n_edges=m['n_edges'], event_flag=m['event_flag'],
                    lane_dim=m['lane_dim'], text_dim=m['text_dim'],
                    global_dim=m['global_dim'])
    if lanes is None:
        raise ValueError('legacy dump without meta needs lane names')
    n = len(lanes)
    return dict(layout='flag', n_lanes=n, n_edges=None, event_flag=True,
                lane_dim=PHYS + 1,
                text_dim=N_KINDS + n + SCALARS, global_dim=11)


def conds_for(spec):
    conds = (CONDS_ID if spec['layout'] == 'identity' else CONDS_FLAG)
    if spec['layout'] == 'identity' and spec['event_flag']:
        conds = conds + ['no_flag', 'flag_wrong']
    return conds


def unpack(raw, spec):
    n_l = spec['n_lanes']
    lanes = raw[: n_l * spec['lane_dim']].reshape(
        n_l, spec['lane_dim']).copy()
    o = n_l * spec['lane_dim']
    text = raw[o: o + spec['text_dim']].copy()
    glob = raw[o + spec['text_dim']:].copy()
    return lanes, text, glob


def none_text(text):
    v = np.zeros_like(text)
    v[EVENT_KINDS.index('none')] = 1.0
    return v


def perturb(raw, cond, spec, true_idx, wrong_idx,
            true_eidx=None, wrong_eidx=None):
    """Return snap dict for one condition."""
    lanes, text, glob = unpack(raw, spec)
    if spec['layout'] == 'identity':
        ne = spec['n_edges']
        tid = slice(N_KINDS, N_KINDS + ne)       # text edge onehot
        lid = slice(PHYS, PHYS + ne)             # lane identity block
        if cond == 'no_text':
            text = none_text(text)
        elif cond == 'no_text_edge':
            text[tid] = 0.0
        elif cond == 'text_wrong':
            text[tid] = 0.0
            text[N_KINDS + wrong_eidx] = 1.0
        elif cond == 'no_lane_id':
            lanes[:, lid] = 0.0
        elif cond == 'laneid_wrong':
            # swap identity claims between the true and wrong edges:
            # if binding follows the report name it lands on wrong lanes
            if true_idx and true_eidx is not None:
                lanes[true_idx, lid] = 0.0
                lanes[wrong_idx, lid] = 0.0
                lanes[true_idx, PHYS + wrong_eidx] = 1.0
                lanes[wrong_idx, PHYS + true_eidx] = 1.0
            else:
                lanes[wrong_idx, lid] = 0.0
        elif cond == 'no_flag' and spec['event_flag']:
            lanes[:, -1] = 0.0
        elif cond == 'flag_wrong' and spec['event_flag']:
            lanes[:, -1] = 0.0
            lanes[wrong_idx, -1] = 1.0
    else:
        n_l = spec['n_lanes']
        if cond == 'no_flag':
            lanes[:, PHYS] = 0.0
        elif cond == 'no_text':
            text = none_text(text)
        elif cond == 'no_text_edge':
            text[N_KINDS: N_KINDS + n_l] = 0.0
        elif cond == 'text_wrong':
            text[N_KINDS: N_KINDS + n_l] = 0.0
            for i in wrong_idx:
                text[N_KINDS + i] = 1.0
        elif cond == 'flag_wrong':
            lanes[:, PHYS] = 0.0
            for i in wrong_idx:
                lanes[i, PHYS] = 1.0
    return {'lanes': lanes.astype(np.float32),
            'text': text.astype(np.float32),
            'global': glob.astype(np.float32),
            'event_lane_idx': []}


def encode(router, snap):
    import torch
    batch = collate([snap])
    with torch.no_grad():
        z, w = router.encode(batch)
        q, _ = router.q_values(batch)
    z = z.cpu().numpy()[0]
    q = q.cpu().numpy()[0]
    if w is None:
        return z, q, None
    w = w.detach().cpu().numpy()
    if w.ndim == 2:                          # selfattn: (L,L) -> received
        w = w.mean(axis=0)
    return z, q, w.ravel()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dump',
                   default='output_data/dev_takeover_probe/'
                           'probe_repr/z_dump.npz')
    p.add_argument('--records-dir',
                   default='output_data/dev_takeover_probe/eval_p2_obj3')
    p.add_argument('--ckpt', action='append', default=[],
                   help='NAME=PATH override/add router arms '
                        '(default: the three obj3 checkpoints)')
    p.add_argument('--out',
                   default='output_data/dev_takeover_probe/probe_repr')
    args = p.parse_args()

    lanes = lane_order(args.records_dir)
    n_lanes = len(lanes)
    d = np.load(args.dump, allow_pickle=True)
    raw = d['raw']
    episodes = [str(e) for e in d['episodes']]
    spec = dump_spec(d, lanes)
    conds = conds_for(spec)
    if spec['n_lanes'] != n_lanes:
        raise ValueError(
            f'dump lanes={spec["n_lanes"]} vs records lanes={n_lanes}')
    edges = sorted({l.rsplit('_', 1)[0] for l in lanes})
    expected = (n_lanes * spec['lane_dim'] + spec['text_dim']
                + spec['global_dim'])
    if raw.shape[1] != expected:
        raise ValueError(
            f'raw width {raw.shape[1]} != layout {spec["layout"]} '
            f'expected {expected}')

    routers = {}
    arm_order = ARMS3
    if args.ckpt:
        arm_order = []
        for kv in args.ckpt:
            nm, ck = kv.split('=', 1)
            routers[nm], _ = DispatchRouter.load(ck)
            arm_order.append(nm)
    else:
        for name in ARMS3:
            ck = (f'output_data/dev_takeover_probe/router_{name}_obj3/'
                  f'router_{name}.pt')
            routers[name], _ = DispatchRouter.load(ck)

    # only windows with a reported event edge carry the disentangling
    # signal: flag layout -> lane flag; identity -> text edge block
    if spec['layout'] == 'identity':
        eoff = n_lanes * spec['lane_dim'] + N_KINDS
        active = raw[:, eoff: eoff + spec['n_edges']].sum(axis=1) > 0
    else:
        lanes_x = raw[:, : n_lanes * spec['lane_dim']].reshape(
            -1, n_lanes, spec['lane_dim'])
        active = lanes_x[:, :, PHYS].sum(axis=1) > 0
    idx_active = np.where(active)[0]

    results = {}
    for name, router in routers.items():
        res = {c: {'z_cos': [], 'q_flip': [], 'bind_true': [],
                   'mass_true': [], 'mass_wrong': []}
               for c in conds}
        for i in range(len(raw)):
            cell = int(episodes[i].split('_')[0][1])
            true_edge = CELL_EDGES[cell]
            true_idx = edge_lanes(lanes, true_edge) if true_edge else []
            alt = [e for e in CELL_EDGES if e and e != true_edge]
            wrong_edge = alt[cell % len(alt)] or alt[0]
            wrong_idx = edge_lanes(lanes, wrong_edge)
            true_eidx = (edges.index(true_edge)
                         if true_edge in edges else None)
            wrong_eidx = edges.index(wrong_edge)
            snap_full = perturb(raw[i], 'full', spec,
                                true_idx, wrong_idx,
                                true_eidx, wrong_eidx)
            z0, q0, w0 = encode(router, snap_full)
            act0 = int(np.argmax(q0))
            for cond in conds:
                snap = snap_full if cond == 'full' else perturb(
                    raw[i], cond, spec, true_idx, wrong_idx,
                    true_eidx, wrong_eidx)
                z, q, w = encode(router, snap)
                r = res[cond]
                denom = (np.linalg.norm(z) * np.linalg.norm(z0))
                r['z_cos'].append(
                    float(np.dot(z, z0) / denom) if denom else 1.0)
                r['q_flip'].append(float(int(np.argmax(q)) != act0))
                if w is not None and i in idx_active and true_idx:
                    am = int(np.argmax(w))
                    r['bind_true'].append(float(am in true_idx))
                    r['mass_true'].append(
                        float(sum(w[j] for j in true_idx)))
                    r['mass_wrong'].append(
                        float(sum(w[j] for j in wrong_idx)))
        results[name] = {
            c: {k: (round(float(np.mean(v)), 4) if v else None)
                for k, v in rr.items()}
            for c, rr in res.items()}

    payload = {'n_samples': int(len(raw)),
               'n_flag_active': int(len(idx_active)),
               'feature_layout': spec['layout'],
               'lanes': lanes, 'results': results}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fname = ('channel_ablate.json' if not args.ckpt
             else 'channel_ablate_' + '_'.join(arm_order) + '.json')
    (out / fname).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n')

    for name in arm_order:
        print(f'\n=== {name} ===')
        print(f'{"cond":<13}{"cos(z,full)":>12}{"q_flip":>8}'
              f'{"bind_true":>11}{"mass_true":>11}{"mass_wrong":>11}')
        for c in conds:
            r = results[name][c]
            print(f'{c:<13}'
                  f'{r["z_cos"] if r["z_cos"] is not None else "":>12}'
                  f'{r["q_flip"] if r["q_flip"] is not None else "":>8}'
                  f'{r["bind_true"] if r["bind_true"] is not None else "":>11}'
                  f'{r["mass_true"] if r["mass_true"] is not None else "":>11}'
                  f'{r["mass_wrong"] if r["mass_wrong"] is not None else "":>11}')


if __name__ == '__main__':
    main()
