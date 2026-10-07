"""Representation-level probe for TARL text arms (ATT-ENTITY-003 diagnostic).

Runs ONE live evaluation episode with a trained ep200 checkpoint and records,
at every decision step, three representation levels plus event labels:

  z_text    = text_projection(e_t) output     (B,N,256)  text-path latent
  z_fuse    = _fuse() output h_it             (B,N,128)  post-fusion node state
  h_policy  = last GAT layer output z         (B,N,256)  pre-Q policy state

Labels per decision come from the project SUMO event runtime schedule
(privileged -- analysis only, never fed to the policy):
  event kinds active, per-node is_target mask, target junction id,
  movement class (lane_blockage), speed factor (global_rain).

Output: repr_records.npz (arrays) + repr_labels.jsonl (per-decision labels).

Usage:
  python tools/run_tarl_repr_probe.py --checkpoint <resumable ep200> \
      --config configs/tsc/tarl_paper_cfprobe.yml --agent tarl_gating \
      --prefix tarlrepr_X --output <dir> --seed 7
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def junction_of(event, catalog, edge_to_lanes):
    """Map an event to its controlled junction (signal owner)."""
    if event.kind == 'lane_blockage':
        return catalog[event.lane_id]['to_junction']
    if event.kind == 'road_closure':
        return catalog[edge_to_lanes[event.edge_id][0]]['to_junction']
    return 'global'


def run(args):
    import runpy
    import agent.tarl  # noqa: F401  (registers tarl_* classes)
    from common.registry import Registry

    agent_class = Registry.mapping['model_mapping'][args.agent]
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    saved_state = checkpoint['agents'][0]['online_model_state_dict']
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels_path = output / 'repr_labels.jsonl'

    z_text_buf, z_fuse_buf, h_pol_buf = [], [], []
    label_buf = []

    original_init = agent_class.__init__
    original_get_action = agent_class.get_action
    hooks = {}

    def patched_init(self, world, rank):
        original_init(self, world, rank)
        self.model.load_state_dict(saved_state)
        self.model.eval()
        policy = self.model.policy

        def cap(name):
            def _h(module, inp, out):
                hooks[name] = out.detach().cpu().numpy().copy()
            return _h

        # z_text: text_projection output; z_fuse: input to first GAT layer
        # (== h_it, post-fusion incl. residual); h_policy: last GAT output.
        policy.text_projection.register_forward_hook(cap('z_text'))
        policy.gat_layers[0].register_forward_pre_hook(
            lambda m, inp: hooks.__setitem__('z_fuse', inp[0].detach().cpu().numpy().copy()))
        policy.gat_layers[-1].register_forward_hook(cap('h_policy'))

    def patched_get_action(self, ob, phase, test=False):
        actions = original_get_action(self, ob, phase, test=test)
        runtime = getattr(self.world, '_sumo_event_runtime', None)
        now = float(self.world.get_current_time())
        active, target_nodes = [], np.zeros(len(self.node_ids), dtype=np.int64)
        junction = 'none'
        catalog, e2l = None, None
        if runtime is not None:
            catalog = runtime.network_catalog()
            e2l = {e: ls for e, ls in runtime._edges.items()}
            node_index = {n: i for i, n in enumerate(self.node_ids)}
            for ev in runtime.schedule.events:
                if ev.begin <= now < ev.end:
                    j = junction_of(ev, catalog, e2l)
                    active.append({'event_id': ev.event_id, 'kind': ev.kind,
                                   'junction': j,
                                   'lane_id': ev.lane_id, 'edge_id': ev.edge_id,
                                   'speed_factor': ev.speed_factor,
                                   'movements': (catalog[ev.lane_id]['movements']
                                                 if ev.kind == 'lane_blockage' else []),
                                   'position': ev.position})
                    if j in node_index:
                        target_nodes[node_index[j]] = 1
                    elif j == 'global':
                        target_nodes[:] = 1
                    if junction == 'none':
                        junction = j
        z_text_buf.append(hooks['z_text'].reshape(len(self.node_ids), -1))
        z_fuse_buf.append(hooks['z_fuse'].reshape(len(self.node_ids), -1))
        h_pol_buf.append(hooks['h_policy'].reshape(len(self.node_ids), -1))
        label_buf.append({
            'simulation_time': now,
            'n_active': len(active),
            'events': active,
            'junction': junction,
            'target_mask': target_nodes.tolist(),
            'actions': np.asarray(actions).tolist(),
        })
        return actions

    config_path = Path(args.config)
    if args.event_schedule or args.event_plan or args.episodes:
        import yaml
        cfg = yaml.safe_load((ROOT / config_path).read_text())
        if args.event_schedule:
            cfg['trainer']['event_schedule'] = str(ROOT / args.event_schedule)
            cfg['trainer'].pop('event_schedule_plan', None)
        if args.event_plan:
            cfg['trainer']['event_schedule_plan'] = str(ROOT / args.event_plan)
            cfg['trainer'].pop('event_schedule', None)
        if args.episodes:
            cfg['trainer']['episodes'] = int(args.episodes)
        config_path = output / 'resolved_probe_config.yml'
        config_path.write_text(yaml.safe_dump(cfg))

    agent_class.__init__ = patched_init
    agent_class.get_action = patched_get_action
    sys.argv = [str(ROOT / 'run.py'), '-w', 'sumo', '-a', args.agent,
                '-n', args.network, '--seed', str(args.seed), '--ngpu', '-1',
                '--sumo_seed', str(args.sumo_seed if args.sumo_seed is not None else args.seed),
                '--interface', 'libsumo', '--prefix', args.prefix,
                '--experiment-config', str(ROOT / config_path)]
    import runpy as _rp
    _rp.run_path(str(ROOT / 'run.py'), run_name='__main__')

    np.savez_compressed(
        output / 'repr_records.npz',
        z_text=np.stack(z_text_buf), z_fuse=np.stack(z_fuse_buf),
        h_policy=np.stack(h_pol_buf))
    with labels_path.open('w', encoding='utf-8') as fh:
        for row in label_buf:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'repr probe -> {output}: {len(label_buf)} decisions')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--agent', required=True)
    p.add_argument('--prefix', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--network', default='hz4x4')
    p.add_argument('--sumo_seed', type=int, default=None)
    p.add_argument('--event-schedule', default=None,
                   help='fixed schedule yaml (overrides config event_schedule)')
    p.add_argument('--event-plan', default=None,
                   help='planbook yaml (overrides config event_schedule_plan)')
    p.add_argument('--episodes', type=int, default=None)
    main_run(p.parse_args())


def main_run(args):
    run(args)


if __name__ == '__main__':
    main()
