#!/usr/bin/env python3
"""Alignment retrieval probe for the stage-3 CAREL-style arm.

Runs one eval episode with a frozen checkpoint while recording every
decision step's (lane obs, scene) pair, then replays the episode through
``EpisodeAlignAccumulator`` to rebuild (event, traffic-window) pairs offline
and scores the *frozen* encoders:

  * event -> window retrieval: Top-1 accuracy and
    positive-vs-best-negative cosine margin;
  * per-neg-type margins (wrong_node / no_event / cross-event);
  * a same-kind wrong-location sanity row when a second local event shares
    the event kind (rare; usually reported as absent).

No training and no checkpoint mutation.  Usage mirrors
tools/run_sga_counterfactual.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AGENTS_REF = []          # patched init stores the live agent here


def run(args):
    import agent.colight as colight
    from agent.scene_alignment import EpisodeAlignAccumulator

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    saved_state = checkpoint['agents'][0]['online_model_state_dict']
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / 'retrieval_records.jsonl'

    agent_class = {'sga_flx_colight': colight.SGAFlxColightAgent}[args.agent_class]
    original_init = agent_class.__init__
    captured = {'rows': []}

    def patched_init(self, world, rank):
        original_init(self, world, rank)
        missing = [k for k in saved_state
                   if k not in self.model.state_dict()]
        self.model.load_state_dict(saved_state, strict=not missing)
        self.target_model.load_state_dict(
            {k: v for k, v in saved_state.items()
             if k in self.target_model.state_dict()}, strict=not missing)
        self.model.eval()
        self.target_model.eval()
        # Retrieval must score the frozen checkpoint: the trainer still calls
        # train() during rollout, so make it an instance-level no-op.
        self.train = lambda: 0.0
        AGENTS_REF.append(self)

    def patched_get_action(self, ob, phase, test=False):
        observed_at = float(self.world.get_current_time())
        snapshot = self.scene_context_at(observed_at)
        captured['rows'].append({
            't': observed_at,
            'obs': np.stack([np.asarray(r, dtype=np.float32) for r in ob]),
            'snapshot': snapshot,
        })
        return original_get_action(self, ob, phase, test=test)

    original_get_action = agent_class.get_action
    agent_class.__init__ = patched_init
    agent_class.get_action = patched_get_action
    config = ROOT / args.config
    sys.argv = [str(ROOT / 'run.py'), '-w', 'sumo', '-a', args.agent_class,
                '-n', args.network, '--seed', str(args.seed), '--ngpu', '-1',
                '--sumo_seed', str(args.sumo_seed if args.sumo_seed is not None else args.seed),
                '--interface', 'libsumo', '--prefix', args.prefix,
                '--experiment-config', str(config)]
    runpy.run_path(str(ROOT / 'run.py'), run_name='__main__')

    # ---- offline scoring -------------------------------------------------
    agent = AGENTS_REF[0] if AGENTS_REF else None
    if agent is None or not hasattr(agent.model, 'align_window_encoder'):
        raise RuntimeError('checkpoint/agent lacks align_window_encoder; '
                           'retrieval probe requires an alignment-arm run')
    acc = EpisodeAlignAccumulator(
        window_cap_steps=args.window_cap, min_active_steps=2)
    for row in captured['rows']:
        acc.step(row['obs'], row['snapshot'])
    rng = np.random.default_rng(args.seed)
    pairs = acc.finish(rng)
    if not pairs:
        records_path.write_text(json.dumps(
            {'episode': 'eval', 'pairs': 0, 'note': 'no active events'}) + '\n')
        print('no events in eval episode; nothing to score')
        return
    transition = hasattr(agent.model.align_window_encoder, 'net') and \
        agent.model.align_window_encoder.net[0].in_features == \
        2 * agent.model.align_window_encoder.feat_dim
    # score only events with a valid (baseline, observed) pair in transition
    # mode; in traffic mode every pair qualifies.
    scored = [p for p in pairs if not transition or p.pos_before is not None]
    if not scored:
        records_path.write_text(json.dumps(
            {'episode': 'eval', 'pairs': len(pairs),
             'note': 'no pairs with baseline window'}) + '\n')
        print('no scorable pairs'); return
    z_e = agent._encode_events_for_align(
        [p.report for p in scored]).detach()
    windows, meta, befores = [], [], []
    for i, pair in enumerate(scored):
        meta.append({'event': i, 'kind': 'pos'})
        windows.append(pair.pos)
        if transition:
            befores.append(pair.pos_before)
        for name, w in pair.negs.items():
            if transition and name not in pair.negs_before:
                continue
            meta.append({'event': i, 'kind': 'neg_' + name})
            windows.append(w)
            if transition:
                befores.append(pair.negs_before[name])
    if transition:
        z_w = agent.model.align_window_encoder.encode_batch(
            befores, windows).detach()
    else:
        z_w = agent.model.align_window_encoder.encode_batch(windows).detach()
    sims = (z_e @ z_w.T).cpu().numpy()          # [E, W]
    pos_of = {i: [w for w, m in enumerate(meta)
                  if m['event'] == i] for i in range(len(scored))}
    results = []
    for i, pair in enumerate(scored):
        pos_idx = pos_of[i][0]
        neg_idx = [w for w, m in enumerate(meta) if not (
            m['event'] == i and m['kind'] == 'pos')]
        own_negs = [w for w, m in enumerate(meta)
                    if m['event'] == i and m['kind'].startswith('neg_')]
        other = [w for w, m in enumerate(meta)
                 if m['event'] != i]
        row = {
            'event_id': getattr(pair.report, 'event_id', None),
            'event_kind': getattr(pair.report, 'event_kind', None)
                          or getattr(pair.report, 'event_type', None),
            'sim_pos': float(sims[i, pos_idx]),
            'margin_all': float(sims[i, pos_idx] - sims[i, neg_idx].max()),
            'margin_own_negs': (float(sims[i, pos_idx] - sims[i, own_negs].max())
                                if own_negs else None),
            'margin_cross_events': (float(sims[i, pos_idx] - sims[i, other].max())
                                    if other else None),
            'rank': int((sims[i] > sims[i, pos_idx]).sum() + 1),
            'n_windows': len(windows),
        }
        results.append(row)
    top1 = float(np.mean([r['rank'] == 1 for r in results]))
    payload = {
        'checkpoint': str(args.checkpoint),
        'episodes_eval_rows': len(captured['rows']),
        'pairs': len(scored),
        'windows': len(windows),
        'top1': top1,
        'margin_all_mean': float(np.mean([r['margin_all'] for r in results])),
        'events': results,
    }
    with records_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
    print(json.dumps({k: payload[k] for k in ('pairs', 'windows', 'top1',
                                              'margin_all_mean')}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--network', default='hz4x4')
    parser.add_argument('--agent-class', choices=('sga_flx_colight',),
                        default='sga_flx_colight')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--sumo-seed', type=int, default=None)
    parser.add_argument('--window-cap', type=int, default=60)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
