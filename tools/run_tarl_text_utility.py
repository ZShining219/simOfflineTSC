#!/usr/bin/env python3
"""Closed-loop text utility counterfactual for TARL text arms (Test 3).

Unlike run_tarl_text_cf.py (which only logs hypothetical variant Qs while the
canonical action drives SUMO), this tool EXECUTES the chosen variant's action
for the whole episode.  Physical event schedule, traffic demand and seeds are
identical across variant runs; ONLY the text fed to the policy differs.

That yields the utility comparison the sensitivity probe cannot give:

    Return(Correct)  vs  Return(Empty)  vs  Return(Foreign)  vs  Return(Relabeled)

Per-step records still contain every variant's Q/action (sensitivity), and the
run's normal metrics stream gives episode-level return/travel time/throughput/
unfinished (utility).

Usage (one run per variant, keep seeds/schedules identical):
    python3 tools/run_tarl_text_utility.py --variant canonical \
        --checkpoint <run>/checkpoints/resumable/episode_0200.pt \
        --config configs/tsc/tarl_paper_cfprobe.yml --agent tarl_gating \
        --prefix tarlutil_gatg_s7_canonical --output <dir> --seed 400007 \
        --event-plan configs/events/plans/hz4x4_random_v1.yml --episodes 4
"""
from __future__ import annotations

import argparse
import json
import re
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FOREIGN_TEXT = (
    'Event cf_foreign_01. All lanes of directed road road_4_4_0, from junction '
    'intersection_4_4 to junction intersection_4_3 are closed to vehicle entry.'
)
VARIANTS = ('canonical', 'empty', 'foreign', 'relabeled')
NODE_RE = re.compile(r'(road|intersection)_(\d+)_(\d+)(_\d+)?')


def relabel(texts):
    coords = []
    for t in texts:
        for m in NODE_RE.finditer(t):
            coords.append((int(m.group(2)), int(m.group(3))))
    if not coords:
        coords = [(2, 2)]
    max_r = max(r for r, _ in coords) or 4
    max_c = max(c for _, c in coords) or 4

    def swap(match):
        kind, r, c = match.group(1), int(match.group(2)), int(match.group(3))
        lane = match.group(4) or ''
        return f'{kind}_{max_r + 1 - r}_{max_c + 1 - c}{lane}'

    return [NODE_RE.sub(swap, t) for t in texts]


def run(args):
    import agent.tarl  # noqa: F401
    from common.registry import Registry

    agent_class = Registry.mapping['model_mapping'][args.agent]
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    saved_state = checkpoint['agents'][0]['online_model_state_dict']
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / 'text_utility_records.jsonl'

    original_init = agent_class.__init__
    original_texts = agent_class.policy_texts

    def patched_init(self, world, rank):
        original_init(self, world, rank)
        self.model.load_state_dict(saved_state)
        self.model.eval()

    def variant_texts(self, name, canonical):
        n = len(self.node_ids)
        if name == 'canonical':
            return list(canonical)
        if name == 'empty':
            return [''] * n
        if name == 'foreign':
            return [FOREIGN_TEXT] * n
        if name == 'relabeled':
            return relabel(canonical)
        raise ValueError(name)

    def patched_get_action(self, ob, phase, test=False):
        canonical = original_texts(self)
        with torch.no_grad():
            x = torch.as_tensor(self._features(ob, phase), dtype=torch.float32)
            lane_dim = self.model.lane_dim
            head, tail = x[..., :lane_dim], x[..., lane_dim + 768:]
            outputs = {}
            for name in VARIANTS:
                emb = torch.as_tensor(
                    self.text_encoder(variant_texts(self, name, canonical)).numpy(),
                    dtype=torch.float32)
                xv = torch.cat([head, emb, tail], dim=-1)
                q = self.model(xv).detach().cpu()
                outputs[name] = q
        base_q = outputs['canonical'].numpy()
        applied = outputs[args.variant].numpy().argmax(-1)
        record = {
            'simulation_time': float(self.world.get_current_time()),
            'executed_variant': args.variant,
            'canonical_text': canonical[0] if canonical else '',
            'action_changed_vs_canonical': int(
                np.sum(applied != base_q.argmax(-1))),
            'q_l1_executed_vs_canonical': float(
                np.abs(outputs[args.variant].numpy() - base_q).mean()),
            'applied_actions': applied.tolist(),
        }
        with records_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + '\n')
        return applied

    agent_class.__init__ = patched_init
    agent_class.get_action = patched_get_action

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
        config_path = output / 'resolved_utility_config.yml'
        config_path.write_text(yaml.safe_dump(cfg))

    sys.argv = [str(ROOT / 'run.py'), '-w', 'sumo', '-a', args.agent,
                '-n', args.network, '--seed', str(args.seed), '--ngpu', '-1',
                '--sumo_seed', str(args.sumo_seed if args.sumo_seed is not None else args.seed),
                '--interface', 'libsumo', '--prefix', args.prefix,
                '--experiment-config', str(ROOT / config_path)]
    runpy.run_path(str(ROOT / 'run.py'), run_name='__main__')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--agent', required=True)
    parser.add_argument('--variant', required=True, choices=VARIANTS)
    parser.add_argument('--network', default='hz4x4')
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--sumo_seed', type=int, default=None)
    parser.add_argument('--event-schedule', default=None)
    parser.add_argument('--event-plan', default=None)
    parser.add_argument('--episodes', type=int, default=None)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
