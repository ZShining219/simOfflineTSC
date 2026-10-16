#!/usr/bin/env python3
"""Frozen-checkpoint TARL text counterfactual evaluation.

At every decision of a single SUMO episode, evaluates the policy under several
text variants on the SAME live traffic state, logs per-variant Q/actions, and
sends the canonical action to SUMO.  Answers two questions with paired data:

  * empty vs canonical  — is the text pathway used at all?
  * foreign/relabeled   — does the policy discriminate text *content*?

Variants (broadcast text, identical for all nodes — paper semantics):
    canonical      runtime's current public reports, joined per policy_texts()
    empty          no event text at all
    foreign        a well-formed report for a road closure that is NOT in the
                   schedule (content discrimination; OOD probe only)
    wrong-location canonical text with every road_/intersection_ id rewritten
                   to a different valid node (alias of legacy 'relabeled')
    wrong-type     canonical locations kept; each report rewritten as a
                   different event kind (closure<->blockage; rain, which has
                   no location, becomes a closure on a fixed valid edge)
    both-wrong     wrong-type composed with wrong-location
    relabeled      legacy alias of wrong-location (not in VARIANTS)

Usage:
    python3 tools/run_tarl_text_cf.py \
        --checkpoint <run>/checkpoints/resumable/episode_0200.pt \
        --config configs/tsc/tarl_paper_attention_smoke.yml \
        --agent tarl_attention --prefix tarlcf_att --seed 7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import runpy
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FOREIGN_TEXT = (
    'Event cf_foreign_01. All lanes of directed road road_4_4_0, from junction '
    'intersection_4_4 to junction intersection_4_3 are closed to vehicle entry.'
)
VARIANTS = ('canonical', 'empty', 'wrong-location', 'wrong-type', 'both-wrong',
            'foreign')
NODE_RE = re.compile(r'(road|intersection)_(\d+)_(\d+)(_\d+)?')

# Sentence templates matching world/sumo_events/runtime.py:_render exactly.
_CLOSURE = re.compile(
    r'All lanes of directed road (\S+?), from junction (intersection_\d+_\d+) '
    r'to junction (intersection_\d+_\d+) are closed to vehicle entry\.')
_CLOSURE_CLEARED = re.compile(
    r'The entry restriction imposed by this event on directed road (\S+?), '
    r'from junction (intersection_\d+_\d+) to junction (intersection_\d+_\d+) '
    r'has been removed\.')
_BLOCKAGE = re.compile(
    r'A stationary obstacle locally blocks lane (\S+?) \(([^)]*)\), '
    r'on (\S+?) road (\S+?) approaching junction (intersection_\d+_\d+), '
    r'at [\d.]+ m from the lane start\.')
_BLOCKAGE_CLEARED = re.compile(
    r'The obstacle on lane (\S+?) \(([^)]*)\), on (\S+?) road (\S+?) '
    r'approaching junction (intersection_\d+_\d+) has been removed\.')
_RAIN = re.compile(
    r'Rain affects the whole network\. This event limits motor-vehicle '
    r'lane speeds to [\d.]+% of their normal limits\.')
_RAIN_CLEARED = re.compile(
    r'This network-wide rain event has ended and its speed restriction '
    r'has been removed\.')
def _edge_catalog(world):
    """edge_id -> {direction, from_junction, to_junction} from the live
    runtime's network catalog; empty when no project runtime is installed."""
    runtime = getattr(world, '_sumo_event_runtime', None)
    edges = {}
    if runtime is not None:
        for lane in runtime.network_catalog().values():
            edges.setdefault(lane['edge_id'], lane)
    return edges


def _rain_wrong_edge(edges):
    """Rain has no location; its wrong-type form necessarily invents one.
    Pick a fixed valid non-internal edge (documented confound)."""
    valid = sorted(e for e in edges if not e.startswith(':'))
    edge = 'road_1_0_1' if 'road_1_0_1' in edges else (valid[0] if valid
                                                     else 'road_1_0_1')
    lane = edges.get(edge) or {}
    return (edge, lane.get('from_junction', 'intersection_1_0'),
            lane.get('to_junction', 'intersection_1_1'))


def wrong_type(texts, edges):
    """Same locations, different event kind, per report sentence."""
    def c2b(match):
        edge, _frm, to = match.groups()
        lane = edges.get(edge) or {}
        direction = lane.get('direction', 'directed')
        return (f'A stationary obstacle locally blocks lane {edge}_0 '
                f'(through lane), on {direction} road {edge} '
                f'approaching junction {to}, at 150.00 m from the lane start.')

    def b2c(match):
        _lane, _label, _direction, edge, to = match.groups()
        lane = edges.get(edge) or {}
        frm = lane.get('from_junction', to)
        return (f'All lanes of directed road {edge}, from junction {frm} '
                f'to junction {to} are closed to vehicle entry.')

    rain_edge, rain_from, rain_to = _rain_wrong_edge(edges)
    rain_closure = (f'All lanes of directed road {rain_edge}, from junction '
                    f'{rain_from} to junction {rain_to} are closed to '
                    f'vehicle entry.')
    rain_cleared = (f'The entry restriction imposed by this event on directed '
                    f'road {rain_edge}, from junction {rain_from} to junction '
                    f'{rain_to} has been removed.')

    def convert(seg):
        # Dispatch on the ORIGINAL sentence so converted output is never
        # re-matched by a sibling template within the same pass.
        if _CLOSURE.search(seg):
            return _CLOSURE.sub(c2b, seg)
        if _BLOCKAGE.search(seg):
            return _BLOCKAGE.sub(b2c, seg)
        if _RAIN.search(seg):
            return _RAIN.sub(rain_closure, seg)
        if _CLOSURE_CLEARED.search(seg):
            return _CLOSURE_CLEARED.sub(
                lambda m: (f'The obstacle on lane {m.group(1)}_0 '
                           f'(through lane), on directed road {m.group(1)} '
                           f'approaching junction {m.group(3)} '
                           f'has been removed.'), seg)
        if _BLOCKAGE_CLEARED.search(seg):
            return _BLOCKAGE_CLEARED.sub(
                lambda m: (f'The entry restriction imposed by this event on '
                           f'directed road {m.group(4)}, from junction '
                           f'{(edges.get(m.group(4)) or {}).get("from_junction", m.group(5))} '
                           f'to junction {m.group(5)} has been removed.'), seg)
        if _RAIN_CLEARED.search(seg):
            return _RAIN_CLEARED.sub(rain_cleared, seg)
        return seg

    return [''.join(convert(seg) for seg in
                    re.split(r'(?=Event\s+\S+\.)', text)) for text in texts]


def relabel(texts, node_ids):
    """Rewrite every location id to a different in-range network node.

    Cyclic shift on (row,col) within the observed id range, so every
    rewritten id is still a syntactically valid, in-range reference and
    always differs from the original (unlike the legacy mirror map, which
    produced out-of-range ids for index-0 coordinates).
    """
    coords = []
    for nid in node_ids:
        m = NODE_RE.search(nid)
        coords.append((int(m.group(2)), int(m.group(3))) if m else (0, 0))
    max_r = max(r for r, _ in coords) or 4
    max_c = max(c for _, c in coords) or 4

    def swap(match):
        kind, r, c = match.group(1), int(match.group(2)), int(match.group(3))
        lane = match.group(4) or ''
        nr = r % max_r + 1 if r > 0 else 1
        nc = c % max_c + 1 if c > 0 else 1
        return f'{kind}_{nr}_{nc}{lane}'

    return [NODE_RE.sub(swap, t) for t in texts]


def run(args):
    import agent.tarl  # noqa: F401  (registers tarl_* classes)
    from common.registry import Registry

    agent_class = Registry.mapping['model_mapping'][args.agent]
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    saved_state = checkpoint['agents'][0]['online_model_state_dict']
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / 'tarl_text_cf_records.jsonl'

    original_init = agent_class.__init__
    original_texts = agent_class.policy_texts

    def patched_init(self, world, rank):
        original_init(self, world, rank)
        self.model.load_state_dict(saved_state)
        self.model.eval()

    def variant_texts(self, name, canonical, edges):
        n = len(self.node_ids)
        if name == 'canonical':
            return list(canonical)
        if name == 'empty':
            return [''] * n
        if name == 'foreign':
            return [FOREIGN_TEXT] * n
        if name in ('wrong-location', 'relabeled'):
            return relabel(canonical, self.node_ids)
        if name == 'wrong-type':
            return wrong_type(canonical, edges)
        if name == 'both-wrong':
            return relabel(wrong_type(canonical, edges), self.node_ids)
        raise ValueError(name)

    def patched_get_action(self, ob, phase, test=False):
        canonical = original_texts(self)
        with torch.no_grad():
            x = torch.as_tensor(self._features(ob, phase), dtype=torch.float32)
            lane_dim = self.model.lane_dim
            head, tail = x[..., :lane_dim], x[..., lane_dim + 768:]
            outputs = {}
            edges = _edge_catalog(self.world)
            for name in VARIANTS:
                texts = variant_texts(self, name, canonical, edges)
                emb = torch.as_tensor(
                    self.text_encoder(texts).numpy(), dtype=torch.float32)
                xv = torch.cat([head, emb, tail], dim=-1)
                q = self.model(xv).detach()
                outputs[name] = {
                    'q': q.cpu().tolist(),
                    'text': texts[0],
                    'fusion_weights': (
                        None if self.model.policy.last_fusion_weights is None
                        else self.model.policy.last_fusion_weights.cpu().tolist()),
                }
        base_q = np.asarray(outputs['canonical']['q'])
        base_actions = base_q.argmax(-1)
        record = {
            'simulation_time': float(self.world.get_current_time()),
            'canonical_text': canonical[0] if canonical else '',
            'control_variant': 'canonical',
            'variants': {},
        }
        for name, value in outputs.items():
            q = np.asarray(value['q'])
            acts = q.argmax(-1)
            record['variants'][name] = {
                'action_changed_count': int(np.sum(acts != base_actions)),
                'q_l1_vs_canonical': float(np.abs(q - base_q).mean()),
                'action': acts.tolist(),
                'fusion_weights': value['fusion_weights'],
                'text': value['text'],
            }
        with records_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        return base_actions

    agent_class.__init__ = patched_init
    agent_class.get_action = patched_get_action
    sys.argv = [str(ROOT / 'run.py'), '-w', 'sumo', '-a', args.agent,
                '-n', args.network, '--seed', str(args.seed), '--ngpu', '-1',
                '--sumo_seed', str(args.sumo_seed if args.sumo_seed is not None else args.seed),
                '--interface', 'libsumo', '--prefix', args.prefix,
                '--experiment-config', str(ROOT / args.config)]
    runpy.run_path(str(ROOT / 'run.py'), run_name='__main__')


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
    run(parser.parse_args())


if __name__ == '__main__':
    main()
