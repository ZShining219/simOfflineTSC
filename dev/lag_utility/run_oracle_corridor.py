#!/usr/bin/env python3
"""Oracle corridor-throttle probe — closure upstream headroom test (att004).

Follow-up to run_oracle_deadlane.py: masking dead-lane observations produced
almost no action change and ~0 paired utility.  This probe tests the other
realizable lever directly — during a road_closure window, OVERRIDE the action
at the feeder junction j1 (edge `from`) and its controlled upstream neighbours
with the phase that serves the fewest corridor-bound links (protected 'G'
counts 1.0, permissive 's' counts 0.4).  If this best-available throttle also
yields ~0, the current event regime carries no realizable anti-corridor
headroom and the text channel cannot pay for itself there regardless of how
well attention is trained.

Window: [begin - lead, end) per event.  lead=0 reproduces the pure active
window; lead>0 additionally exercises the announce (pre-positioning) window
produced by TARL_ANNOUNCE_LEAD_S.

Mirrors tools/run_tarl_struct_cf.py (same eval_v2 contract + npz dump schema)
and additionally writes per-step corridor/oracle records.

Usage:
    python3 dev/lag_utility/run_oracle_corridor.py \
        --checkpoint <ckpt> --config <generated ev2 yml> \
        --agent tarl_attention --network hz4x4 --prefix <p> \
        --output <probe_dir> --dump_dir <dump_dir> --seed 7 --sumo_seed 9000 \
        --eval_protocol eval_v2 --phase-map dev/lag_utility/hz4x4_phase_map.json \
        [--lead 0] [--no-upstream]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

STATE_W = {'G': 1.0, 'g': 1.0, 's': 0.4}


def _flush_dump(dump_dir, state):
    buf = state['buffer']
    if not buf:
        return
    state['episode'] += 1
    arrays = {}
    for key in buf[0]:
        vals = [r[key] for r in buf]
        if vals[0] is None:
            continue
        arrays[key] = np.stack(vals) if key != 't' else np.asarray(vals)
    np.savez_compressed(dump_dir / f'ep{state["episode"]:03d}.npz', **arrays)
    state['buffer'] = []
    state['last_t'] = -1.0


def _check_eval_protocol(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    per_episode = bool((cfg.get('trainer') or {}).get('sumo_seed_per_episode'))
    if args.eval_protocol == 'eval_v2':
        if not per_episode:
            raise SystemExit('eval_v2 requires trainer.sumo_seed_per_episode: true')
        if not args.dump_dir:
            raise SystemExit('eval_v2 requires --dump_dir')
    elif per_episode:
        raise SystemExit('config sets sumo_seed_per_episode; pass eval_v2')


def run(args):
    import agent.tarl  # noqa: F401  (registers tarl_* classes)
    from common.registry import Registry
    from reproduction.tarl_tsc.models.paper_tarl import ZTASK_FIELDS

    _check_eval_protocol(args)
    agent_class = Registry.mapping['model_mapping'][args.agent]
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    saved_state = checkpoint['agents'][0]['online_model_state_dict']
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / 'oracle_corridor_records.jsonl'
    phase_map = json.loads(Path(args.phase_map).read_text())
    edges = phase_map['edges']
    node_idx_of = {nid: v['intersection_index']
                   for nid, v in phase_map['nodes'].items()}
    # lane -> edge map: lane ids carry the edge id as prefix minus _<i>
    edge_of_lane = {lane: eid for eid, e in edges.items() for lane in e['lanes']}

    original_init = agent_class.__init__

    def patched_init(self, world, rank):
        original_init(self, world, rank)
        if getattr(self, 'event_input', 'bert') != 'structured':
            raise RuntimeError('corridor probe requires event_input=structured')
        self.model.load_state_dict(saved_state)
        self.model.eval()
        self._phase_score_cache = {}  # (node_id, corridor_key) -> scores list

    def _controlled_links(world, node_id):
        """Ordered controlled links [from_lane, to_lane] aligned with
        phase_states chars."""
        node = world.id2intersection[node_id]
        out = []
        for link in node.lanelinks:
            if not link:
                out.append(None)
                continue
            l0 = link[0]
            out.append((l0[0], l0[1]))
        return out

    def _phase_scores(self, node_id, corridor_to_lanes):
        """Per-phase service weight onto lanes whose to-lane is in
        corridor_to_lanes."""
        key = (node_id, tuple(sorted(corridor_to_lanes)))
        if key in self._phase_score_cache:
            return self._phase_score_cache[key]
        states = phase_map['nodes'][node_id]['phase_states']
        links = _controlled_links(self.world, node_id)
        if len(links) != len(states[0]):
            raise RuntimeError(
                f'{node_id}: link order mismatch {len(links)} vs {len(states[0])}')
        scores = []
        for st in states:
            s = 0.0
            for i, ch in enumerate(st):
                lk = links[i]
                if lk is None:
                    continue
                if lk[1] in corridor_to_lanes:
                    s += STATE_W.get(ch, 0.0)
            scores.append(s)
        self._phase_score_cache[key] = scores
        return scores

    def _corridor_for_closure(self, ev):
        """Return [(node_idx, node_id, role, corridor_to_lane_set)] to throttle."""
        e = edges.get(ev.edge_id)
        if e is None:
            return []
        feeder = e['from']
        targets = {}
        # feeder j1: links discharging into the closed edge's lanes
        if feeder in node_idx_of:
            targets[feeder] = ('feeder', set(e['lanes']))
        # upstream: incoming lanes of j1 that feed the closed edge — their
        # edges' from-junctions throttle the whole corridor approach
        if not args.no_upstream and feeder in self.world.id2intersection:
            feeding_lanes = set()
            for links in self.world.id2intersection[feeder].lanelinks:
                if not links:
                    continue
                for l0 in links:
                    if l0[1] in set(e['lanes']):
                        feeding_lanes.add(l0[0])
            for fl in feeding_lanes:
                e0 = edge_of_lane.get(fl)
                if e0 is None:
                    continue
                j0 = edges[e0]['from']
                if j0 in node_idx_of and j0 != feeder:
                    targets.setdefault(j0, ('upstream', set(edges[e0]['lanes'])))
        return [(node_idx_of[nid], nid, role, lanes)
                for nid, (role, lanes) in targets.items()]

    def patched_get_action(self, ob, phase, test=False):
        with torch.no_grad():
            feats = torch.as_tensor(self._features(ob, phase), dtype=torch.float32)
            lane_dim = self.model.lane_dim
            q = self.model(feats).detach()
            base = q.argmax(-1)
            now = float(self.world.get_current_time())
            runtime = getattr(self.world, '_sumo_event_runtime', None)
            corridor = {}
            closures = []
            if runtime is not None:
                for ev in runtime.schedule.events:
                    if ev.kind != 'road_closure':
                        continue
                    if not (ev.begin - args.lead <= now < ev.end):
                        continue
                    e = edges.get(ev.edge_id)
                    if e is None:
                        continue
                    touched = []
                    for ni, nid, role, lanes in _corridor_for_closure(self, ev):
                        scores = _phase_scores(self, nid, lanes)
                        best = min(scores)
                        if int(base[ni]) < len(scores) and scores[int(base[ni])] == best:
                            oracle_a = int(base[ni])
                        else:
                            oracle_a = scores.index(best)
                        corridor[ni] = oracle_a
                        touched.append({'node': nid, 'role': role,
                                        'oracle': oracle_a,
                                        'argmax': int(base[ni]),
                                        'score_oracle': best,
                                        'score_argmax': scores[int(base[ni])]
                                        if int(base[ni]) < len(scores) else None})
                    closures.append({'event_id': ev.event_id, 'edge': ev.edge_id,
                                     'touched': touched})
            if getattr(self, '_guard_on', False):
                guarded, _ = self._starvation_guard(base.numpy())
            else:
                guarded = base.numpy()
            actions = guarded.copy()
            for ni, a in corridor.items():
                actions[ni] = a
            z = feats[:, lane_dim:lane_dim + ZTASK_FIELDS].numpy()
            rec = {
                'simulation_time': now,
                'n_present_nodes': int((z[:, 0] > 0.5).sum()),
                'queue_now': float(np.sum(
                    [g.generate() for g in self.queue_generators])),
                'lead_s': args.lead,
                'closures': closures,
                'oracle_action': {str(k): v for k, v in corridor.items()},
                'argmax_action': base.tolist(),
                'guarded_action': guarded.tolist(),
                'applied_action': actions.tolist(),
            }
            with records_path.open('a', encoding='utf-8') as h:
                h.write(json.dumps(rec, ensure_ascii=False) + '\n')
            return actions

    agent_class.__init__ = patched_init
    agent_class.get_action = patched_get_action

    if args.dump_dir:
        dump_dir = Path(args.dump_dir).resolve()
        dump_dir.mkdir(parents=True, exist_ok=True)
        state = {'buffer': [], 'episode': -1, 'last_t': -1.0}
        inner = agent_class.get_action

        def dumping_get_action(self, ob, phase, test=False):
            actions = inner(self, ob, phase, test=test)
            t = float(self.world.get_current_time())
            x = torch.as_tensor(self._features(ob, phase), dtype=torch.float32)
            lane_dim = self.model.lane_dim
            rec = {
                't': t,
                'ob_traffic': x[:, :lane_dim].numpy().astype(np.float32),
                'z_task': x[:, lane_dim:lane_dim + ZTASK_FIELDS].numpy()
                          .astype(np.float32),
                'phase_tail': x[:, lane_dim + ZTASK_FIELDS:].numpy()
                              .astype(np.float32),
                'action': np.asarray(actions).astype(np.int64),
                'queue_now': float(np.sum(
                    [g.generate() for g in self.queue_generators])),
                'queue_per_node': np.asarray(
                    [float(np.sum(g.generate())) for g in self.queue_generators],
                    dtype=np.float32),
            }
            if t <= state['last_t'] and state['buffer']:
                _flush_dump(dump_dir, state)
            state['last_t'] = t
            state['buffer'].append(rec)
            return actions

        import atexit
        atexit.register(lambda: _flush_dump(dump_dir, state))
        agent_class.get_action = dumping_get_action
        agent_class.sample = lambda self: self.get_action(
            self.get_ob(), self.get_phase(), test=True)

    sys.argv = [str(ROOT / 'run.py'), '-w', 'sumo', '-a', args.agent,
                '-n', args.network, '--seed', str(args.seed), '--ngpu', '-1',
                '--sumo_seed',
                str(args.sumo_seed if args.sumo_seed is not None else args.seed),
                '--interface', 'libsumo', '--prefix', args.prefix,
                '--experiment-config', str(ROOT / args.config)]
    runpy.run_path(str(ROOT / 'run.py'), run_name='__main__')

    if args.dump_dir:
        dump_dir = Path(args.dump_dir).resolve()
        plan_log = (ROOT / 'data/output_data/tsc' / f'sumo_{args.agent}'
                    / args.network / args.prefix / 'event_plan_log.jsonl')
        if plan_log.exists():
            (dump_dir / 'event_plan_log.jsonl').write_text(plan_log.read_text())
        meta = {'agent': args.agent, 'checkpoint': str(args.checkpoint),
                'config': str(args.config), 'seed': args.seed,
                'sumo_seed': args.sumo_seed, 'prefix': args.prefix,
                'eval_protocol': args.eval_protocol, 'probe': 'oracle_corridor',
                'lead_s': args.lead, 'upstream': not args.no_upstream}
        (dump_dir / 'manifest.json').write_text(json.dumps(meta, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--agent', required=True)
    parser.add_argument('--network', default='hz4x4')
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase-map',
                        default='dev/lag_utility/hz4x4_phase_map.json')
    parser.add_argument('--lead', type=float, default=0.0,
                        help='extend the throttle window to [begin-lead, end)')
    parser.add_argument('--no-upstream', action='store_true',
                        help='throttle only the feeder junction')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--sumo_seed', type=int, default=None)
    parser.add_argument('--eval_protocol', default='eval_v2',
                        choices=('eval_v2',))
    parser.add_argument('--dump_dir', default=None)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
