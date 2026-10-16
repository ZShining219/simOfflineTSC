#!/usr/bin/env python3
"""Oracle dead-lane mask probe — closure feeder-side headroom test (att004).

During an ACTIVE road_closure on edge E (from_junction j1 controlled), the
vehicles queueing on j1's approach lanes that turn into E can never discharge
(no rerouting is enabled; they wait for reopening).  Those approach lanes are
observed features of j1, so their queue looks like ordinary demand to a
text-blind policy and inflates its phase pressure.  A policy that *knew* the
closure would discount this "dead" demand.

This probe evaluates the realizable headroom of that knowledge with a
test-time oracle: on every decision, while a closure is active at j1 we zero
the E-feeding approach lanes in j1's observation vector and take the SAME
model's argmax there.  Other nodes keep the canonical (guarded) argmax.
Text condition stays canonical, so the contrast vs the paired canonical
eval_v2 run isolates the feeder-side benefit that perfectly-used text could
in principle deliver.

Mirrors tools/run_tarl_struct_cf.py (same eval_v2 contract + npz dump schema)
and additionally writes per-step dead-lane/oracle records.

Usage:
    python3 dev/lag_utility/run_oracle_deadlane.py \
        --checkpoint <ckpt> --config <generated ev2 yml> \
        --agent tarl_attention --network hz4x4 --prefix <p> \
        --output <probe_dir> --dump_dir <dump_dir> --seed 7 --sumo_seed 9000 \
        --eval_protocol eval_v2 --phase-map dev/lag_utility/hz4x4_phase_map.json
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
    records_path = output / 'oracle_deadlane_records.jsonl'
    phase_map = json.loads(Path(args.phase_map).read_text())
    edges = phase_map['edges']
    node_idx_of = {nid: v['intersection_index']
                   for nid, v in phase_map['nodes'].items()}

    original_init = agent_class.__init__

    def patched_init(self, world, rank):
        original_init(self, world, rank)
        if getattr(self, 'event_input', 'bert') != 'structured':
            raise RuntimeError('oracle probe requires event_input=structured')
        self.model.load_state_dict(saved_state)
        self.model.eval()
        self._lane_pos = None  # built lazily; generators are made by reset()

    def patched_get_action(self, ob, phase, test=False):
        with torch.no_grad():
            if self._lane_pos is None:
                self._lane_pos = []
                for gen in self.obs_generators:
                    lanes = [l for road in gen.lanes for l in road]
                    self._lane_pos.append({l: i for i, l in enumerate(lanes)})
            feats = torch.as_tensor(self._features(ob, phase), dtype=torch.float32)
            lane_dim = self.model.lane_dim
            q = self.model(feats).detach()
            base = q.argmax(-1)
            now = float(self.world.get_current_time())
            runtime = getattr(self.world, '_sumo_event_runtime', None)
            dead = {}
            closures = []
            if runtime is not None:
                for ev in runtime.schedule.events:
                    if ev.kind != 'road_closure' or not (ev.begin <= now < ev.end):
                        continue
                    e = edges.get(ev.edge_id)
                    if e is None:
                        continue
                    ni = node_idx_of.get(e['from'])
                    if ni is None:
                        continue  # uncontrolled feeder
                    lanes_e = set(e['lanes'])
                    hits = set()
                    node = self.world.id2intersection[e['from']]
                    for links in node.phase_available_lanelinks:
                        for fr, to in links:
                            if to in lanes_e:
                                hits.add(fr)
                    idxs = sorted({self._lane_pos[ni][l] for l in hits
                                   if l in self._lane_pos[ni]})
                    if idxs:
                        dead[ni] = idxs
                    closures.append({'event_id': ev.event_id, 'edge': ev.edge_id,
                                     'feeder_node': ni, 'dead_feats': idxs})
            oracle_actions = {}
            for ni, idxs in dead.items():
                x2 = feats.clone()
                x2[ni, idxs] = 0.0  # dead demand masked in j1's observation
                q2 = self.model(x2).detach()
                oracle_actions[ni] = int(q2[ni].argmax(-1))
            if getattr(self, '_guard_on', False):
                guarded, _ = self._starvation_guard(base.numpy())
            else:
                guarded = base.numpy()
            actions = guarded.copy()
            for ni, a in oracle_actions.items():
                actions[ni] = a
            z = feats[:, lane_dim:lane_dim + ZTASK_FIELDS].numpy()
            rec = {
                'simulation_time': now,
                'n_present_nodes': int((z[:, 0] > 0.5).sum()),
                'queue_now': float(np.sum(
                    [g.generate() for g in self.queue_generators])),
                'closures': closures,
                'oracle_action': {str(k): v for k, v in oracle_actions.items()},
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
                'eval_protocol': args.eval_protocol, 'probe': 'oracle_deadlane'}
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
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--sumo_seed', type=int, default=None)
    parser.add_argument('--eval_protocol', default='eval_v2',
                        choices=('eval_v2',))
    parser.add_argument('--dump_dir', default=None)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
