#!/usr/bin/env python3
"""Frozen-checkpoint SGA mechanism counterfactual evaluation.

Runs one short SUMO episode while evaluating several Z/G variants on the
same live traffic state before one action is sent to SUMO. It does not train
or change the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
WRONG_LOCATION = {
    # Valid but deliberately distant policy nodes in hz4x4.  The event text
    # and Zscene stay unchanged; only its route-aligned Gscene column moves.
    'hz_lane_01': 'intersection_4_4',
    'hz_road_01': 'intersection_1_4',
}


def build_variant(rep, name):
    from agent.scene_attention import EventSceneRepresentation
    z = rep.z_events
    mask = rep.node_event_mask
    relation = rep.node_event_relation
    count = z.shape[0]
    if name == 'correct':
        return rep
    if name == 'zero_z':
        return EventSceneRepresentation(torch.zeros_like(z), rep.event_ids, mask, relation)
    if name == 'zero_g':
        return EventSceneRepresentation(z, rep.event_ids, torch.zeros_like(mask),
                                        torch.zeros_like(relation) if relation is not None else None)
    if name == 'shuffle_z':
        if count < 2:
            return None
        order = torch.arange(count - 1, -1, -1, device=z.device)
        return EventSceneRepresentation(z[order], rep.event_ids, mask, relation)
    if name == 'shuffle_g':
        if mask.shape[0] < 2:
            return None
        order = torch.roll(torch.arange(mask.shape[0], device=mask.device), shifts=1)
        shuffled_relation = None if relation is None else relation[order]
        return EventSceneRepresentation(z, rep.event_ids, mask[order], shuffled_relation)
    if name.startswith('remove_event_'):
        index = int(name.rsplit('_', 1)[1])
        if index >= count:
            return None
        keep = [i for i in range(count) if i != index]
        if not keep:
            return EventSceneRepresentation(z[:0], (), mask[:, :0],
                                            None if relation is None else relation[:, :0])
        keep = torch.tensor(keep, dtype=torch.long, device=z.device)
        return EventSceneRepresentation(
            z[keep], tuple(rep.event_ids[i] for i in keep.tolist()), mask[:, keep],
            None if relation is None else relation[:, keep])
    raise ValueError(f'Unknown counterfactual variant: {name}')


def build_wrong_location(rep, node_ids):
    """Move each local event's full relation column to a valid distant node."""
    mask = rep.node_event_mask.clone()
    relation = None if rep.node_event_relation is None else rep.node_event_relation.clone()
    moved = {}
    for event_index, event_id in enumerate(rep.event_ids):
        target_id = WRONG_LOCATION.get(event_id)
        if target_id is None or target_id not in node_ids:
            continue
        target = node_ids.index(target_id)
        source = mask[:, event_index].nonzero(as_tuple=False).flatten()
        if source.numel() == 0:
            continue
        moved[event_id] = {'from_nodes': [node_ids[i] for i in source.tolist()],
                           'to_node': target_id}
        mask[:, event_index] = False
        mask[target, event_index] = True
        if relation is not None:
            column = relation[:, event_index].clone()
            relation[:, event_index] = False
            relation[target, event_index] = column.any(dim=0)
            relation[target, event_index, 0] = True
    if not moved:
        return None, moved
    from agent.scene_attention import EventSceneRepresentation
    return EventSceneRepresentation(rep.z_events, rep.event_ids, mask, relation), moved


def run(args):
    import agent.colight as colight

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    saved_state = checkpoint['agents'][0]['online_model_state_dict']
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / 'counterfactual_records.jsonl'
    variants = ('correct', 'zero_z', 'zero_g', 'shuffle_z', 'shuffle_g',
                'wrong_location', 'remove_event_0', 'remove_event_1')

    agent_class = {'sga_colight': colight.SGAColightAgent,
                   'concat_colight': colight.ConcatColightAgent,
                   'sga_flx_colight': colight.SGAFlxColightAgent}[args.agent_class]
    original_init = agent_class.__init__

    def patched_init(self, world, rank):
        original_init(self, world, rank)
        self.model.load_state_dict(saved_state)
        self.target_model.load_state_dict(saved_state)
        self.model.eval()
        self.target_model.eval()

    def patched_get_action(self, ob, phase, test=False):
        from agent.scene_attention import EventSceneRepresentation
        import torch.nn.functional as F

        observed_at = float(self.world.get_current_time())
        snapshot = self.scene_context_at(observed_at)
        self._live_scene = snapshot
        observation = torch.tensor(self._network_input(ob, phase), dtype=torch.float32)
        data = Data(x=observation, edge_index=self.edge_idx)
        correct_rep = self.model.encode_scene(snapshot, train=False)
        node_ids = list(getattr(self, 'intersection_ids', self.world.intersection_ids))
        variants_here = []
        for name in variants:
            if name == 'wrong_location':
                rep, moved = build_wrong_location(correct_rep, node_ids)
            else:
                moved = {}
                rep = build_variant(correct_rep, name)
            if rep is not None:
                variants_here.append((name, rep))
        outputs = {}
        for name, rep in variants_here:
            q = self.model(x=data.x, edge_index=data.edge_index, train=False,
                           scene_embedding=rep).detach()
            outputs[name] = {
                'q': q.cpu().tolist(),
                'node_gate': getattr(self.model, 'last_scene_node_gate',
                                     torch.zeros((1, self.sub_agents))).detach().cpu().tolist(),
                'feature_gate_mean': getattr(
                    self.model, 'last_scene_feature_gate',
                    torch.ones((1, self.sub_agents, 128))).detach().mean(-1).cpu().tolist(),
                'attention': getattr(self.model, 'last_scene_attention',
                                     torch.zeros((1, self.sub_agents, 0))).detach().cpu().tolist(),
            }
            scene_attn_mod = getattr(self.model, 'scene_attention', None)
            for stat_name in ('semantic_score', 'direct_bias', 'relation_bias', 'relevance'):
                value = getattr(scene_attn_mod, 'last_' + stat_name, None)
                if value is not None:
                    flat = value.detach().float().reshape(-1)
                    outputs[name][stat_name + '_stats'] = {
                        'mean': float(flat.mean()) if flat.numel() else 0.0,
                        'std': float(flat.std(unbiased=False)) if flat.numel() else 0.0,
                        'min': float(flat.min()) if flat.numel() else 0.0,
                        'max': float(flat.max()) if flat.numel() else 0.0,
                    }
        base_q = np.asarray(outputs['correct']['q'], dtype=float)
        base_actions = np.asarray([
            int(np.argmax(row[:length]))
            for row, length in zip(base_q, self.phase_lengths)
        ])
        node_ids = list(getattr(self, 'intersection_ids', self.world.intersection_ids))
        record = {
            'episode': 1,
            'decision_step': int(getattr(self, '_scene_log_context', {}).get('decision_step') or 0),
            'simulation_time': observed_at,
            'event_ids': list(snapshot.event_ids),
            'event_count': len(snapshot.event_ids),
            'node_ids': node_ids,
            'direct_grounding_mask': snapshot.grounding.node_report_mask.astype(int).tolist(),
            'node_event_relation': snapshot.grounding.node_event_relation.astype(int).tolist(),
            'wrong_location_mapping': WRONG_LOCATION,
            'variants': {},
        }
        for name, value in outputs.items():
            q = np.asarray(value['q'], dtype=float)
            action = np.asarray([
                int(np.argmax(row[:length]))
                for row, length in zip(q, self.phase_lengths)
            ])
            value['q_l1_vs_correct'] = float(np.abs(q - base_q).mean())
            value['action_changed_count'] = int(np.sum(action != base_actions))
            value['action'] = action.tolist()
            record['variants'][name] = value
        selected = args.control_variant
        if selected not in outputs:
            selected = 'correct'
        with records_path.open('a', encoding='utf-8') as handle:
            record['control_variant'] = selected
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        selected_q = np.asarray(outputs[selected]['q'], dtype=float)
        return np.asarray([
            int(np.argmax(row[:length]))
            for row, length in zip(selected_q, self.phase_lengths)
        ])

    agent_class.__init__ = patched_init
    agent_class.get_action = patched_get_action
    config = ROOT / args.config
    sys.argv = [str(ROOT / 'run.py'), '-w', 'sumo', '-a', args.agent_class,
                '-n', args.network, '--seed', str(args.seed), '--ngpu', '-1',
                '--sumo_seed', str(args.sumo_seed if args.sumo_seed is not None else args.seed),
                '--interface', 'libsumo', '--prefix', args.prefix,
                '--experiment-config', str(config)]
    runpy.run_path(str(ROOT / 'run.py'), run_name='__main__')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--network', default='hz4x4')
    parser.add_argument('--agent-class', choices=('sga_colight', 'concat_colight',
                                                  'sga_flx_colight'),
                        default='sga_colight')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--sumo-seed', type=int, default=None)
    parser.add_argument('--control-variant', choices=('correct', 'wrong_location', 'zero_g'),
                        default='correct')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
