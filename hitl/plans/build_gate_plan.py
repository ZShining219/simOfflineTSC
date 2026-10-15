#!/usr/bin/env python
"""build_gate_plan.py — gate 触发策略消融 plan（训练层）。

探索件（hitl/plans/，登记 dev/INDEX.md）。克隆 plan_r2 里
HITL_EXPB（大池 r=0.25）模板，逐模式把 expert_pool_manifest
换成对应触发策略采集的纠偏池——唯一变量是"gate 以什么策略
问人产出的池"。臂名 HITL_GATE_{OD,RAND,MARGIN,PERIOD,WIN}
（五模式池均为 seed0-9 同规模 ~1210 转移，接管标注数随触发
策略自然分化）；基线行是既有 HITL_C0 与 HITL_EXPB（18 会话
大池，非同 seed 集，仅参照）。

用法：

    python hitl/plans/build_gate_plan.py \
        --out data/output_data/hitl_probe/pilot5/plan_gate.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

POOLS = {
    'OD': 'hitl_gate_on_demand',
    'RAND': 'hitl_gate_random',
    'MARGIN': 'hitl_gate_margin',
    'PERIOD': 'hitl_gate_periodic',
    'WIN': 'hitl_gate_window',
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--template', default=str(
            _REPO / 'data/output_data/hitl_probe/pilot5/plan_r2.json'))
    ap.add_argument('--pools-root', default=str(
        _REPO / 'data/output_data/hitl_probe/pools'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    plan = json.loads(Path(args.template).read_text())
    expb = [c for c in plan['children']
            if c.get('condition') == 'HITL_EXPB']
    assert len(expb) == 5, f'expect 5 EXPB children, got {len(expb)}'

    children = []
    for tag, pool_name in POOLS.items():
        pool = (Path(args.pools_root) / pool_name
                / 'expert_pool_manifest.json').resolve()
        assert pool.is_file(), f'missing pool: {pool}'
        for src in expb:
            c = copy.deepcopy(src)
            c['condition'] = f'HITL_GATE_{tag}'
            suffix = src['logical_run_id'].rsplit('-SD', 1)[1]
            c['logical_run_id'] = f'HITL_GATE_{tag}-SD{suffix}'
            c['expert_pool_manifest'] = str(pool)
            children.append(c)

    out = {
        'schema_version': plan.get('schema_version', 1),
        'protocol_id': plan.get('protocol_id'),
        'experiment_stage': 'hitl_gate_modes',
        'mode': plan.get('mode'),
        'model': plan.get('model'),
        'networks': plan.get('networks'),
        'trainer': plan.get('trainer'),
        'children': children,
    }
    Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'out': args.out,
                      'children': len(children)}, indent=1))


if __name__ == '__main__':
    main()
