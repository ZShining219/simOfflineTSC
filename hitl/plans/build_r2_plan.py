#!/usr/bin/env python
"""build_r2_plan.py — HITL r2 增补臂 plan 生成（pilot5 模板克隆法）。

探索件（hitl/plans/，登记 dev/INDEX.md）。不重新推导
initial_state/digest——直接把 pilot5 已验证的 child 当模板克隆，
只改实验变量字段，保证与既有臂"同初始权重/同 seed/同预算"可比。

臂：

  判官消融（模板=HITL_BP，三判官面板裁成单判官）：
    HITL_BPT   terms-only      —— 规则项独立承重？
    HITL_BPM   reward_model-only —— 学习奖励网独立承重？
    HITL_BPQ   qgap-only       —— Q-gap 判官独立承重？

  C 通道扩池（模板=HITL_EXPERT，专家池换大池/降配比）：
    HITL_EXPB  大纠偏池 r=0.25 —— hitl_ep50_big（18 会话）
    HITL_EXPL  大纠偏池 r=0.05 —— 低配比+大池（doc 建议 5-10%）

  A 通道扩示范（模板=HITL_APIQ）：
    HITL_APIQB 大示范池双预训  —— demo_expert_ep50_big.npz

用法：

    python hitl/plans/build_r2_plan.py \
        --pilot5-dir data/output_data/hitl_probe/pilot5 \
        --expert-pool data/output_data/hitl_probe/pools/hitl_ep50_big/expert_pool_manifest.json \
        --demo-data data/output_data/hitl_probe/demos/demo_expert_ep50_big.npz \
        --out data/output_data/hitl_probe/pilot5/plan_r2.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

JUDGE_ABLATION = {
    'HITL_BPT': 'terms',
    'HITL_BPM': 'reward_model',
    'HITL_BPQ': 'qgap',
}


def _clone(children, condition):
    return [c for c in children
            if c.get('condition') == condition]


def _renamed(child, new_condition):
    c = copy.deepcopy(child)
    c['condition'] = new_condition
    # 注意：HITL_EXPERT 的 logical_run_id 是历史名 HITL_EXP-SD*（非条件串），
    # 一律按 SD 后缀重建，不做子串替换。
    suffix = child['logical_run_id'].rsplit('-SD', 1)[1]
    c['logical_run_id'] = f'{new_condition}-SD{suffix}'
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pilot5-dir', required=True)
    ap.add_argument('--expert-pool', required=True,
                    help='大纠偏池 manifest 路径')
    ap.add_argument('--demo-data', required=True,
                    help='大示范集 npz 路径')
    ap.add_argument('--low-ratio', type=float, default=0.05)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    pdir = Path(args.pilot5_dir)
    plan_a = json.loads((pdir / 'plan_A.json').read_text())
    plan_bp = json.loads((pdir / 'plan_Bprime.json').read_text())
    plan_m = json.loads((pdir / 'plan_merged.json').read_text())

    children = []

    # --- B′ 逐判官消融：克隆 HITL_BP-SD*，judges 裁到单判官 ---
    bp = _clone(plan_bp['children'], 'HITL_BP')
    assert len(bp) == 5, f'expect 5 BP children, got {len(bp)}'
    for new_cond, keep_type in JUDGE_ABLATION.items():
        for src in bp:
            c = _renamed(src, new_cond)
            judges = c['trainer_overrides']['adjudication']['judges']
            c['trainer_overrides']['adjudication']['judges'] = [
                j for j in judges if j['type'] == keep_type]
            assert len(c['trainer_overrides']['adjudication']
                       ['judges']) == 1
            children.append(c)

    # --- C 扩池：克隆 HITL_EXPERT-SD* ---
    exp = _clone(plan_m['children'], 'HITL_EXPERT')
    assert len(exp) == 5, f'expect 5 EXPERT children, got {len(exp)}'
    for new_cond, ratio in (('HITL_EXPB', None), ('HITL_EXPL', args.low_ratio)):
        for src in exp:
            c = _renamed(src, new_cond)
            # 专家池字段在 child 顶层（与 plan_merged HITL_EXPERT 一致）
            c['expert_pool_manifest'] = str(Path(args.expert_pool).resolve())
            if ratio is not None:
                c['expert_ratio'] = ratio
            children.append(c)

    # --- A 扩示范：克隆 HITL_APIQ-SD* ---
    apiq = _clone(plan_a['children'], 'HITL_APIQ')
    assert len(apiq) == 5
    for src in apiq:
        c = _renamed(src, 'HITL_APIQB')
        c['demo_pretrain'] = copy.deepcopy(c.get('demo_pretrain') or {})
        c['demo_pretrain']['demo_data'] = str(Path(args.demo_data).resolve())
        # demo_pretrain 可能在 child 顶层而非 trainer_overrides（plan_A 实测如此）
        children.append(c)

    out = {
        'schema_version': plan_m.get('schema_version', 1),
        'protocol_id': plan_m.get('protocol_id'),
        'experiment_stage': 'hitl_pilot_r2',
        'mode': plan_m.get('mode'),
        'model': plan_m.get('model'),
        'networks': plan_m.get('networks'),
        'trainer': plan_m.get('trainer'),
        'children': children,
    }
    Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    conds = {}
    for c in children:
        conds.setdefault(c['condition'], 0)
        conds[c['condition']] += 1
    print(json.dumps({'out': args.out, 'arms': conds}, indent=1))


if __name__ == '__main__':
    main()
