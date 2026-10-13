"""Build the r3 multi-scene HITL plan: 4 arms × 5 seeds × 3 configs.

探索件（dev/hitl_live/，登记 dev/INDEX.md）。把 pilot5/r2 已验证的
代表性臂复制到另外三个杭州真实路口配置，检验 HITL 通道在不同流量
分布下的可迁移性（论文式"同拓扑换流量分布"多场景）：

- HITL_C0    无 HITL 基线（各配置自己的 ep0 初始态，seed 配对）
- HITL_APIQB A 通道：该配置专家会话打的 demo npz 双预训练
- HITL_B2    B 通道：零样本冻结奖励模型（RM 在 bc-tyc 上训练，
             直接部署到新路口——对应论文"基础场景预训练+直接部署"）
- HITL_EXPB  C 通道：该配置 hitl 会话打的纠偏池，ratio 0.25

模板取自 pilot5 已完成的 child（plan_A/plan_merged/plan_r2），按
配置重写 networks / initial_state（initial_state_catalog 已对
四个网络 × 5 seed 预登记 digest）/ 池与 demo 路径。

用法（repo 根，conda env: colight）：

    python dev/hitl_live/build_r3_plan.py

产物：``output_data/hitl_probe/pilot5/plan_r3.json``（60 children）
"""

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
PILOT = _REPO / 'output_data/hitl_probe/pilot5'
CATALOG = (_REPO / 'output_data/ha_sodqn/engineering_20260726/'
           'initial_state_catalog.json')
OUT = PILOT / 'plan_r3.json'

# tag → (network, ep50 训练配置名)
CONFIGS = {
    'QC': 'sumohz1x1_config2',
    'KN': 'sumohz1x1_config3',
    'SB': 'sumohz1x1_config4',
}
ARMS = ('HITL_C0', 'HITL_APIQB', 'HITL_B2', 'HITL_EXPB', 'HITL_FULL')
# r3 原件是前四臂；FULL 臂为 r4ms 增补（可单独生成）
SEEDS = range(5)

TEMPLATES = {
    'HITL_C0': 'plan_A.json',
    'HITL_APIQB': 'plan_r2.json',
    'HITL_B2': 'plan_merged.json',
    'HITL_EXPB': 'plan_r2.json',
    'HITL_FULL': 'plan_r4.json',   # 集成系统臂（A+B+C 全开）
}


def _ep0_path(network: str, seed: int, want_sha: str) -> Path:
    """seed0 有 _r2 重复 run 目录（ep0 内容不同）——按 catalog sha 选。"""
    pat = (str(_REPO / 'output_data/tsc/sumo_dqn' / network /
               f'p1_formal_dqn_{network}_seed{seed}_400ep_*' /
               'checkpoints/resumable/episode_0000.pt'))
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(f'no ep0 for {network} seed={seed}')
    for h in hits:
        with open(h, 'rb') as f:
            if hashlib.sha256(f.read()).hexdigest() == want_sha:
                return Path(h)
    raise FileNotFoundError(
        f'no ep0 matching catalog sha for {network} seed={seed}: {hits}')


def _catalog_index():
    cat = json.loads(CATALOG.read_text())
    return {(e['first_network'], e['training_seed']): e
            for e in cat['entries']}


def main() -> None:
    cat_idx = _catalog_index()
    templates = {}
    for arm, fname in TEMPLATES.items():
        plan = json.loads((PILOT / fname).read_text())
        templates[arm] = {x['training_seed']: x for x in plan['children']
                          if x['condition'] == arm}

    children = []
    for tag, network in CONFIGS.items():
        for arm in ARMS:
            for seed in SEEDS:
                child = dict(templates[arm][seed])
                entry = cat_idx[(network, seed)]
                ep0 = _ep0_path(network, seed,
                                entry['checkpoint_file_sha256'])
                child['condition'] = arm
                child['logical_run_id'] = f'{arm}-{tag}-SD{seed}'
                child['networks'] = [network]
                child['training_seed'] = seed
                child['initial_checkpoint'] = str(ep0)
                child['initial_checkpoint_file_sha256'] = (
                    entry['checkpoint_file_sha256'])
                ist = dict(entry)
                ist['checkpoint_path'] = str(ep0)
                ist.pop('source_run_path', None)
                child['initial_state'] = ist
                if 'demo_pretrain' in child:
                    dp = dict(child['demo_pretrain'])
                    dp['demo_data'] = str(
                        _REPO / 'output_data/hitl_probe/demos'
                        / f'demo_expert_ep50_{tag.lower()}.npz')
                    child['demo_pretrain'] = dp
                if 'expert_pool_manifest' in child:
                    child['expert_pool_manifest'] = str(
                        _REPO / 'output_data/hitl_probe/pools'
                        / f'hitl_ep50_{tag.lower()}'
                        / 'expert_pool_manifest.json')
                children.append(child)

    ids = [c['logical_run_id'] for c in children]
    assert len(set(ids)) == len(ids), 'duplicate logical_run_id'
    # FULL 臂单独出一份增补 plan（r4ms），不动已执行的 plan_r3
    full_children = [c for c in children if c['condition'] == 'HITL_FULL']
    children = [c for c in children if c['condition'] != 'HITL_FULL']
    # plan 级字段继承 r2 模板（model/trainer/protocol 等运行时必需）
    head = {k: v for k, v in
            json.loads((PILOT / 'plan_r2.json').read_text()).items()
            if k != 'children'}
    head['networks'] = sorted(set(CONFIGS.values()))
    head['experiment_stage'] = 'hitl_pilot_r3'
    out = dict(head)
    out['kind'] = 'hitl_r3_multiscene'
    out['description'] = (
        'r3 multi-scene replication: best arm per channel '
        'on 3 additional Hangzhou 1x1 flow distributions; '
        'B2 is zero-shot RM transfer from bc-tyc')
    out['children'] = children
    OUT.write_text(json.dumps(out, indent=1))
    print(f'wrote {OUT} with {len(children)} children')
    out_ms = dict(out)
    out_ms['kind'] = 'hitl_r4ms_full_multiscene'
    out_ms['description'] = ('FULL integrated-system arm on 3 scenes '
                             '(supplement to plan_r3)')
    out_ms['children'] = full_children
    (PILOT / 'plan_r4ms.json').write_text(json.dumps(out_ms, indent=1))
    print(f'wrote {PILOT}/plan_r4ms.json with {len(full_children)} children')


if __name__ == '__main__':
    main()
