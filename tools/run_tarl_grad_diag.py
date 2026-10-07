"""Offline gradient decomposition diagnostic for TARL text arms (Test 2).

Reconstructs TD-loss gradients from a resumable checkpoint + its replay
buffer. This is NOT a replay of the historical training trajectory: batches
are re-sampled from the frozen replay state with a fixed diagnostic RNG, so
metrics are "replay-conditioned gradient statistics at checkpoint epN".

Per sampled batch (batch_size drawn like training):
  - replicates train(): target = r + g(1-d) max target-Q ; prediction online-Q;
    MSE loss; backward (NO optimizer step, NO clip on reported norms)
  - reports parameter-group grad norms:
      theta_traffic = traffic_mlp + state_projection
      theta_text    = text_projection (+ attn_* counted in fusion)
      theta_fusion  = attn_* | gate + fusion_fc*
      theta_graph   = gat_layers
      theta_head    = q_head
  - activation-space alignment: cos( dL/d e_p , dL/d s_p )  (flattened)
  - shared-parameter conflict: cos( grad fusion_fc1.W[:, :256] ,
                                    grad fusion_fc1.W[:, 256:] )
    (for gating: same split on gate.W)

Usage:
  python tools/run_tarl_grad_diag.py --run-dir <run_dir> \
      --episodes 50 100 150 200 --batches 16 --out grad_diag.csv
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ACTIONS = 8
TEXT_DIM = 768
GAMMA = 0.99

GROUPS = {
    'theta_traffic': ('policy.traffic_mlp.', 'policy.state_projection.'),
    'theta_text': ('policy.text_projection.',),
    'theta_fusion': ('policy.attn_', 'policy.gate.', 'policy.fusion_fc'),
    'theta_graph': ('policy.gat_layers.',),
    'theta_head': ('policy.q_head.',),
}


def infer_method(state_dict):
    keys = set(state_dict)
    if any(k.startswith('policy.attn_') for k in keys):
        return 'attention'
    if any(k.startswith('policy.gate') for k in keys):
        return 'gating'
    if any(k.startswith('policy.gat_layers') for k in keys):
        return 'gat'
    return 'sensor'


def build_wrapper(method, traffic_dim, lane_dim):
    from arterial.control import stable_intersection_order
    from reproduction.tarl_tsc.parent_adapter import _QWrapper
    graph = json.loads(
        (ROOT / 'reproduction/tarl_tsc/data/hangzhou_4x4_gudang_18041610_1h'
         '/adjacency.json').read_text())
    nodes = list(stable_intersection_order(graph.keys()))
    adjacency = torch.tensor(
        [[a == b or b in graph[a] for b in nodes] for a in nodes])
    return _QWrapper(traffic_dim, ACTIONS, method, adjacency, lane_dim,
                     impl='paper')


def features(pay_obs, pay_phase):
    phase = np.eye(ACTIONS, dtype=np.float32)[np.asarray(pay_phase, dtype=int)]
    return np.concatenate([pay_obs, phase], axis=-1)


def diagnose_checkpoint(ckpt_path, n_batches, seed):
    ck = torch.load(ckpt_path, map_location='cpu')
    ag = ck['agents'][0]
    sd = ag['online_model_state_dict']
    method = infer_method(sd)
    traffic_dim = sd['policy.traffic_mlp.0.weight'].shape[1]
    items = ag['replay_state']['items']
    lane_dim = None  # deduced from first payload
    pay = items[0][1]
    x_dim = features(pay[0], pay[1]).shape[-1]
    lane_dim = x_dim - TEXT_DIM - ACTIONS

    online = build_wrapper(method, traffic_dim, lane_dim)
    online.load_state_dict(sd)
    online.train()
    target = build_wrapper(method, traffic_dim, lane_dim)
    target.load_state_dict(ag['target_model_state_dict'])
    for p in target.parameters():
        p.requires_grad_(False)

    act_grads = {}
    if method in ('attention', 'gating'):
        def retain(name):
            def _h(m, i, o):
                o.retain_grad()
                act_grads[name] = o
            return _h
        online.policy.state_projection.register_forward_hook(retain('s_p'))
        online.policy.text_projection.register_forward_hook(retain('e_p'))

    rng = random.Random(seed)
    rows = []
    for b in range(n_batches):
        samples = rng.sample(items, 128)
        pays = [s[1] for s in samples]
        before = torch.as_tensor(
            np.stack([features(v[0], v[1]) for v in pays]), dtype=torch.float32)
        after = torch.as_tensor(
            np.stack([features(v[4], v[5]) for v in pays]), dtype=torch.float32)
        rewards = torch.as_tensor(
            np.stack([v[3] for v in pays]), dtype=torch.float32)
        actions = torch.as_tensor(
            np.stack([v[2] for v in pays]), dtype=torch.long)
        terminal = torch.as_tensor(
            [v[6] for v in pays], dtype=torch.float32).unsqueeze(-1)
        before = torch.as_tensor(before, dtype=torch.float32)
        after = torch.as_tensor(after, dtype=torch.float32)
        with torch.no_grad():
            tgt = rewards + GAMMA * (1 - terminal) * target(after).max(-1).values
        pred = online(before, train=True).gather(
            -1, actions.unsqueeze(-1)).squeeze(-1)
        loss = torch.nn.functional.mse_loss(pred, tgt)
        online.zero_grad(set_to_none=True)
        loss.backward()

        rec = {'batch': b, 'loss': float(loss.item()), 'method': method}
        for gname, prefixes in GROUPS.items():
            tot = 0.0
            for pn, p in online.named_parameters():
                if p.grad is not None and pn.startswith(prefixes):
                    tot += float(p.grad.pow(2).sum())
            rec[f'norm_{gname}'] = float(np.sqrt(tot))
        # activation-space cosine
        if 'e_p' in act_grads and act_grads['e_p'].grad is not None:
            ge = act_grads['e_p'].grad.reshape(-1)
            gs = act_grads['s_p'].grad.reshape(-1)
            rec['cos_act_text_traffic'] = float(
                torch.nn.functional.cosine_similarity(ge, gs, dim=0, eps=1e-8))
            rec['norm_dL_de'] = float(ge.norm())
            rec['norm_dL_ds'] = float(gs.norm())
        # shared fusion-parameter conflict (text half vs traffic half)
        if method == 'attention':
            w = online.policy.fusion_fc1.weight.grad
        elif method == 'gating':
            w = online.policy.gate.weight.grad
        else:
            w = None
        if w is not None:
            gt, gs = w[:, :256].reshape(-1), w[:, 256:].reshape(-1)
            rec['cos_fusion_param_halves'] = float(
                torch.nn.functional.cosine_similarity(gt, gs, dim=0, eps=1e-8))
            rec['fusion_text_half_norm'] = float(gt.norm())
            rec['fusion_traffic_half_norm'] = float(gs.norm())
        rows.append(rec)
        act_grads.clear()
    return rows, method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True,
                    help='run output dir containing checkpoints/resumable/')
    ap.add_argument('--episodes', nargs='+', type=int,
                    default=[50, 100, 150, 200])
    ap.add_argument('--batches', type=int, default=16)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rdir = Path(args.run_dir)
    all_rows = []
    for ep in args.episodes:
        ck = rdir / 'checkpoints' / 'resumable' / f'episode_{ep:04d}.pt'
        if not ck.exists():
            ck = rdir / 'checkpoints' / 'resumable' / f'episode_{ep}.pt'
        if not ck.exists():
            print(f'skip missing {ck}')
            continue
        rows, method = diagnose_checkpoint(ck, args.batches, args.seed + ep)
        for r in rows:
            r['episode'] = ep
        all_rows.extend(rows)
        print(f'ep{ep}: {len(rows)} batches ({method})')
    import csv
    keys = sorted({k for r in all_rows for k in r})
    with open(args.out, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=['episode'] + keys)
        wr.writeheader()
        wr.writerows(all_rows)
    print(f'wrote {args.out} ({len(all_rows)} rows)')


if __name__ == '__main__':
    main()
