#!/usr/bin/env python3
"""Frozen-checkpoint paired utility probe for the structured-event TARL line.

At every decision of a single SUMO episode, evaluates the policy under several
STRUCTURED event-feature variants on the SAME live traffic state, logs
per-variant Q/actions (and semantic head outputs when semantic_aux is on),
and sends the canonical action to SUMO.

Variants (per-node z_task matrix, applied to the canonical feature rows):
    canonical      grounded features as built by ReportGrounder+TextEntityBinder
    empty          all-zero event slot
    wrong-location event rows rolled to different nodes (content kept)
    wrong-type     event_type cycled 1->2->3->1 on present rows (location kept)
    both-wrong     wrong-type composed with wrong-location
    foreign        a fabricated road_closure row placed on the last node (OOD)

Usage mirrors tools/run_tarl_text_cf.py but requires an event_input=structured
config and checkpoint.
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

VARIANTS = ('canonical', 'empty', 'wrong-location', 'wrong-type', 'both-wrong',
            'foreign')


def _wrong_type(z):
    out = z.copy()
    hit = out[:, 0] > 0.5
    out[hit, 1] = (out[hit, 1] % 3) + 1
    return out


def _foreign(z):
    out = np.zeros_like(z)
    out[-1] = np.array([1.0, 2.0, 2.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return out


def variant_z(name, z):
    if name == 'canonical':
        return z.copy()
    if name == 'empty':
        return np.zeros_like(z)
    if name == 'wrong-location':
        return np.roll(z, 4, axis=0)
    if name == 'wrong-type':
        return _wrong_type(z)
    if name == 'both-wrong':
        return np.roll(_wrong_type(z), 4, axis=0)
    if name == 'foreign':
        return _foreign(z)
    raise ValueError(name)


def _flush_dump(dump_dir, state):
    """Stack the buffered per-step records of one episode into an npz."""
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


def run(args):
    import agent.tarl  # noqa: F401  (registers tarl_* classes)
    from common.registry import Registry
    from reproduction.tarl_tsc.models.paper_tarl import ZTASK_FIELDS

    agent_class = Registry.mapping['model_mapping'][args.agent]
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    saved_state = checkpoint['agents'][0]['online_model_state_dict']
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / 'tarl_struct_cf_records.jsonl'

    original_init = agent_class.__init__

    def patched_init(self, world, rank):
        original_init(self, world, rank)
        if getattr(self, 'event_input', 'bert') != 'structured':
            raise RuntimeError('run_tarl_struct_cf requires event_input=structured')
        self.model.load_state_dict(saved_state)
        self.model.eval()

    def patched_get_action(self, ob, phase, test=False):
        with torch.no_grad():
            x = torch.as_tensor(self._features(ob, phase), dtype=torch.float32)
            lane_dim = self.model.lane_dim
            w = ZTASK_FIELDS
            head = x[..., :lane_dim]
            z_can = x[..., lane_dim:lane_dim + w].numpy()
            tail = x[..., lane_dim + w:]
            outputs = {}
            for name in VARIANTS:
                zv = torch.as_tensor(variant_z(name, z_can), dtype=torch.float32)
                xv = torch.cat([head, zv, tail], dim=-1)
                q = self.model(xv).detach()
                aux = getattr(self.model.policy, '_aux', None)
                outputs[name] = {
                    'q': q.cpu().tolist(),
                    'aux': (None if aux is None else {
                        k: v.cpu().tolist() for k, v in aux.items()}),
                }
                if name == 'canonical':
                    pol = self.model.policy
                    self._dump_tensors = (
                        pol._last_h_it, pol._last_z, pol.last_fusion_weights)
        base_q = np.asarray(outputs['canonical']['q'])
        base_actions = base_q.argmax(-1)
        # Repair arm: starvation guard applies to the control action so the
        # cf probe exercises the same decision path as rollout/eval.
        if getattr(self, '_guard_on', False):
            base_actions, _ = self._starvation_guard(base_actions)
        record = {
            'simulation_time': float(self.world.get_current_time()),
            'n_present_nodes': int((z_can[:, 0] > 0.5).sum()),
            'queue_now': float(np.sum([g.generate()
                                       for g in self.queue_generators])),
            'control_variant': 'canonical',
            'env_text_condition': getattr(self, 'text_condition', 'canonical'),
            'variants': {},
        }
        for name, value in outputs.items():
            q = np.asarray(value['q'])
            acts = q.argmax(-1)
            rec = {
                'action_changed_count': int(np.sum(acts != base_actions)),
                'q_l1_vs_canonical': float(np.abs(q - base_q).mean()),
                'action': acts.tolist(),
            }
            if value['aux'] is not None:
                aux = value['aux']
                rec['aux_present_prob'] = aux['present']
                rec['aux_type'] = aux['type']
                rec['aux_movement'] = aux['movement']
            record['variants'][name] = rec
        with records_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        return base_actions

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
            w = ZTASK_FIELDS
            policy = self.model.policy
            h_it, z_post, fusion_w = getattr(self, '_dump_tensors', (None,) * 3)
            rec = {
                't': t,
                'ob_traffic': x[..., :lane_dim].numpy().astype(np.float32),
                'z_task': x[..., lane_dim:lane_dim + w].numpy().astype(np.float32),
                'phase_tail': x[..., lane_dim + w:].numpy().astype(np.float32),
                'h_it': None if h_it is None else h_it.numpy().astype(np.float32),
                'z_post': None if z_post is None else z_post.numpy().astype(np.float32),
                'fusion_w': (None if fusion_w is None else
                             fusion_w.numpy().astype(np.float32)),
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
        # learning_start is huge in dump configs, so train-role episodes take
        # the sample() branch; redirect it to the deterministic canonical
        # policy so dumped trajectories are policy rollouts, not random.
        agent_class.sample = lambda self: self.get_action(
            self.get_ob(), self.get_phase(), test=True)

    sys.argv = [str(ROOT / 'run.py'), '-w', 'sumo', '-a', args.agent,
                '-n', args.network, '--seed', str(args.seed), '--ngpu', '-1',
                '--sumo_seed', str(args.sumo_seed if args.sumo_seed is not None else args.seed),
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
                'sumo_seed': args.sumo_seed, 'prefix': args.prefix}
        (dump_dir / 'manifest.json').write_text(json.dumps(meta, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--agent', required=True)
    parser.add_argument('--network', default='hz4x4')
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--sumo_seed', type=int, default=None)
    parser.add_argument('--dump_dir', default=None,
                        help='if set, dump per-decision arrays (obs/z_task/'
                             'h_it/z_post/fusion_w/action/queue) as per-episode '
                             'npz under this dir')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
