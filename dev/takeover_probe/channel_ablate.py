"""Channel attribution on the frozen routers: event_flag vs report text.

The lane token's feature 3 (event_flag) and the text vector's edge
indicator always agree in training/eval featurization — both come from
``_report_fields`` — so the attention-binding evidence (fig4) cannot
tell whether the mechanism reads the text or just tracks the flag.
This script breaks that tie offline: recover snapshots from
``probe_repr/z_dump.npz`` (raw = lanes + text + global), re-encode them
under each frozen router with the two channels perturbed
independently, and measure

  binding   attention mass / argmax on true vs alternative event edge
            (attn: cross-attn row; selfattn: received column mass)
  z-shift   cosine distance from the full-input z
  q-flip    router argmax controller change rate

Conditions (applied on top of the recorded snapshot):
  full            as recorded
  no_flag         lanes[:,3] = 0, text intact
  no_text         report vector replaced by the kind='none' encoding
  no_text_edge    lane-indicator block zeroed, kind/scalars kept
  text_wrong      text points at a DIFFERENT edge, flag stays true
  flag_wrong      flag moved to a different edge, text stays true

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
# TEST_EVENTS order == episode prefix e0..e3 used by probe_repr
CELL_EDGES = ['road_0_1_0', 'road_1_0_1', 'road_2_1_2', '']
CONDS = ['full', 'no_flag', 'no_text', 'no_text_edge',
         'text_wrong', 'flag_wrong']


def lane_order(records_dir):
    """fb.lanes order = sorted lane_queue keys of any record row."""
    f = sorted(Path(records_dir).glob('records_*.jsonl'))[0]
    row = json.loads(open(f).readline())
    return sorted(row['lane_queue'].keys())


def edge_lanes(lanes, edge):
    return [i for i, l in enumerate(lanes)
            if l.rsplit('_', 1)[0] == edge]


def unpack(raw, n_lanes):
    lanes = raw[: n_lanes * 4].reshape(n_lanes, 4).copy()
    text = raw[n_lanes * 4: n_lanes * 4 + N_KINDS + n_lanes + SCALARS].copy()
    glob = raw[n_lanes * 4 + N_KINDS + n_lanes + SCALARS:].copy()
    return lanes, text, glob


def none_text(text):
    v = np.zeros_like(text)
    v[EVENT_KINDS.index('none')] = 1.0
    return v


def perturb(raw, cond, n_lanes, true_idx, wrong_idx):
    """Return snap dict for one condition."""
    lanes, text, glob = unpack(raw, n_lanes)
    text = text.copy()
    lanes = lanes.copy()
    if cond == 'no_flag':
        lanes[:, 3] = 0.0
    elif cond == 'no_text':
        text = none_text(text)
    elif cond == 'no_text_edge':
        text[N_KINDS: N_KINDS + n_lanes] = 0.0
    elif cond == 'text_wrong':
        text[N_KINDS: N_KINDS + n_lanes] = 0.0
        for i in wrong_idx:
            text[N_KINDS + i] = 1.0
    elif cond == 'flag_wrong':
        lanes[:, 3] = 0.0
        for i in wrong_idx:
            lanes[i, 3] = 1.0
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
    p.add_argument('--out',
                   default='output_data/dev_takeover_probe/probe_repr')
    args = p.parse_args()

    lanes = lane_order(args.records_dir)
    n_lanes = len(lanes)
    d = np.load(args.dump, allow_pickle=True)
    raw = d['raw']
    episodes = [str(e) for e in d['episodes']]

    routers = {}
    for name in ARMS3:
        ck = (f'output_data/dev_takeover_probe/router_{name}_obj3/'
              f'router_{name}.pt')
        routers[name], _ = DispatchRouter.load(ck)

    # only windows with a reported (flagged) event edge carry the
    # disentangling signal
    lanes_x = raw[:, : n_lanes * 4].reshape(-1, n_lanes, 4)
    flag_on = lanes_x[:, :, 3].sum(axis=1) > 0
    idx_active = np.where(flag_on)[0]

    results = {}
    for name, router in routers.items():
        res = {c: {'z_cos': [], 'q_flip': [], 'bind_true': [],
                   'mass_true': [], 'mass_wrong': []}
               for c in CONDS}
        for i in range(len(raw)):
            cell = int(episodes[i].split('_')[0][1])
            true_edge = CELL_EDGES[cell]
            true_idx = edge_lanes(lanes, true_edge) if true_edge else []
            alt = [e for e in CELL_EDGES if e and e != true_edge]
            wrong_idx = edge_lanes(lanes, alt[cell % len(alt)] or alt[0])
            snap_full = perturb(raw[i], 'full', n_lanes,
                                true_idx, wrong_idx)
            z0, q0, w0 = encode(router, snap_full)
            act0 = int(np.argmax(q0))
            for cond in CONDS:
                snap = snap_full if cond == 'full' else perturb(
                    raw[i], cond, n_lanes, true_idx, wrong_idx)
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
               'lanes': lanes, 'results': results}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'channel_ablate.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n')

    for name in ARMS3:
        print(f'\n=== {name} ===')
        print(f'{"cond":<13}{"cos(z,full)":>12}{"q_flip":>8}'
              f'{"bind_true":>11}{"mass_true":>11}{"mass_wrong":>11}')
        for c in CONDS:
            r = results[name][c]
            print(f'{c:<13}'
                  f'{r["z_cos"] if r["z_cos"] is not None else "":>12}'
                  f'{r["q_flip"] if r["q_flip"] is not None else "":>8}'
                  f'{r["bind_true"] if r["bind_true"] is not None else "":>11}'
                  f'{r["mass_true"] if r["mass_true"] is not None else "":>11}'
                  f'{r["mass_wrong"] if r["mass_wrong"] is not None else "":>11}')


if __name__ == '__main__':
    main()
