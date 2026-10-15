#!/usr/bin/env python3
"""Generate EXPERIMENTS.md (human ledger) from runs.jsonl."""
import json, collections, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROWS = [json.loads(l) for l in open(ROOT / "ledger" / "runs.jsonl")]

META = {
"signal_control_v1": {
 "title": "SUMO 修正绿黄执行（sumo-green-yellow-v1）主线训练验证",
 "hypothesis": "在 run.py/TSCTrainer 主线启用 sumo-green-yellow-v1（5s 黄灯过渡、清空期锁相）后，DQN 在 hz4x4 上可正常训练收敛；与无过渡 legacy 臂的指标差异反映换相物理成本的口径变化（语义变更，GOVERNANCE §5.5，两臂指标不与历史无过渡 run 直接可比）。",
 "plan_id": "scv1_gy5_v1",
 "design": "artifacts/signal_control_v1/run_queue_dqn_gy5.json；configs/tsc/{dqn_gy5,dqn_gy5_smoke}.yml；适配器 world/sumo_signal_control.py；trainer/tsc_trainer.py signal_control 安装点",
 "conclusion": "进行中：2 臂 REGISTERED（dqn_gy5 + 同种子 legacy 对照），400ep train_frozen_eval seed=7。",
 "queues": "artifacts/signal_control_v1/run_queue_dqn_gy5.json",
},
"att_entity_004": {
 "title": "ATT-ENTITY-004 结构化事件输入 + 语义辅助（注意力线当前主线）",
 "hypothesis": "事件信息以结构化 z_task 槽位 + 语义辅助头进入 TARL 注意力，应同时提升场景区分与决策优化（H: 结构化+aux > 纯文本注入）。",
 "plan_id": "att004_wave1/fix/struct/formal/ablation/holdout/e200 波次队列",
 "design": "artifacts/att_entity_004/{run_queue_att004_*.json, run_state_*/}；"
           "configs/tsc/att_entity_004/generated/*.yml；"
           "artifacts/att_entity_004/{ABLATION_ARMS_DESIGN,DIVIDE_CONQUER_DESIGN,SGA_REPAIR_DESIGN}.md",
 "conclusion": "fixG/fixR/fixD/crossq/ablA/ablation/a2s/a3s/holdout 波次均完成并留档；"
               "\n- **fixG 收官测评收割（2026-10-07，eval 实为 10-05 已跑完、此前未汇总判读）**：fe_fixG_* 网格 5 seed×{ep50/100/150/200}×5 条件×3 eval seed 共 300 run 全存在。"
               "判读：①优化层——fixG vs a3s 同 seed/同 eval 配对 canonical TT ep50 −24.9s、ep100 −8.9s（收敛更快且更优）；"
               "②表征层——wrong_type−canonical 中位 +1.34s 非零，类型语义仍在影响行为；"
               "③**效用层未翻正**——canonical−empty 中位 +0.58s（均值 +2.27s 被 s? 单点 +28.7s 拉高），文本通道相对 empty 仍无净收益，独家信息设计需求不变。"
               "evidence：hz4x4/fe_fixG_*。stg2 波次登记未执行、2026-10-07 裁决转 SCOPE_DISCARD，stg1/ctl 部分臂 DROPPED（SCOPE_DISCARD）；"
               "SGA-GB 梯度线经 pilot 判定为负结果后停止（SGA_GB_PILOT_RESULT.md）；"
               "三指标+条件矩阵证据在 data/output_data/analysis/att_entity_004/ 与 baseline_cmp.md。",
 "queues": "run_queue_att004_{wave1,fix,struct,formal,ablation,ablA,holdout,stage2}.json + artifacts/att004_e200/run_queue_e200.json",
},
"att_entity_003": {
 "title": "ATT-ENTITY-003 TARL 对齐/传感矩阵（flx/tarlp/stage2-5）",
 "hypothesis": "文本-实体注意力对齐训练（align_trans/align_trans_meta）与随机事件分布随机化，应改善 TARL 在 hz4x4 事件场景下的鲁棒决策。",
 "plan_id": "att003 stage2/stage3(stage3_pilot/stage3_pilotB/main)/stage4/stage5_tarl",
 "design": "artifacts/att_entity_003/run_queue_stage{2,3*,4,5_tarl}.json；eval 包 artifacts/att_entity_003/{eval_v1,eval_stage4,eval_stage5_tarl,stage3_eval_v1}",
 "conclusion": "TARL 矩阵完成：20 训练 run + 20×14 评估包；文本反事实探针完成；终报与 wiki review v11 归档。随机事件臂 *_random_200（colight/sga_colight_text/sga_flx_colight_text/mplight/flx_align）属本 campaign。",
 "queues": "run_queue_stage{2,3_main,3_pilot,3_pilotB,4,5_tarl}.json + 各 eval 目录 run_queue.json",
},
"att_entity_002": {
 "title": "ATT-ENTITY-002 SGA/CoLight 文本事件 200ep 矩阵",
 "hypothesis": "SGA 变体（sga_colight/concat_colight/sga_mplight）注入事件文本应优于无文本基线（event vs normal 条件）。",
 "plan_id": "sga_concat_colight_200_v1 eval_v1/eval_v2",
 "design": "artifacts/sga_concat_colight_200_v1/run_queue.json + eval_v1/eval_v2 包目录",
 "conclusion": "wiki review v4 完成；30 个训练计划位中 24 个有效训练（部分种子中止/失败留档），48 评估包；H2 获支持、H3 未获支持；含闭环 grid、反事实、v2 回归等探针。",
 "queues": "run_queue.json + eval_v1/eval_v2/run_queue.json（partial_2352/quarantine/packages_INVALID 为历史残留桶）",
},
"arterial_1x6": {
 "title": "半离线 arterial 1×6 干线实验",
 "hypothesis": "shared_dqn 在 sumoarterial1x6 干线上按 stage0 采集→stage1 控制→stage2 流水线评估。",
 "plan_id": "artifacts/arterial_experiments/stage{0,1,2}",
 "design": "artifacts/arterial_experiments/**/run_queue.json + run_state",
 "conclusion": "stage0/stage1 采集与控制 run 501 完成、8 失败（含 stage1_invalid_scene_routing 隔离批）；stage2 部分留档。",
 "queues": "artifacts/arterial_experiments/**/run_queue.json",
},
"plan1_dqn": {
 "title": "Plan1 半离线 DQN 基线（hz1x1 系列）",
 "hypothesis": "sumohz1x1 系列网 DQN 400ep 正式批建立半离线基线。",
 "plan_id": "plan1_dqn 20260722",
 "design": "data/output_data/tsc/sumo_dqn/sumohz1x1*/p1_*",
 "conclusion": "p1_formal 初批 4 个 run 中止后由同身份 *_r2 重跑取代（SUPERSEDED）；pilot_v3 4 run ABORTED（状态残留）；正式批完成。",
 "queues": "无独立队列文件（直跑+run_status）",
},
"plan5_b100": {
 "title": "Plan5-B100 跨算法锚点/校准（Phase0）",
 "hypothesis": "DDQN/CTXDDQN 在 S1-S4 场景 100ep 锚点 + 校准，建立跨算法比较底座。",
 "plan_id": "plan5_b100 logical_run_id=P5-{ANCHOR,CAL}-<ALGO>-<场景>-SD<seed>",
 "design": "data/output_data/cross_algorithm/plan5_b100/{runs,manifest,audit}",
 "conclusion": "Phase0 基础设施（fixedtime 参照/probe/resume 等价/并发门禁/环境/provenance）与 DDQN/CTXDDQN 锚点、校准全部完成；Phase0 不构成正式批次结论。",
 "queues": "manifest/attempts JSON 审计链",
},
"tarl_reproduction": {
 "title": "TARL 复现线（v1/graph/fixed/mp/smoke）",
 "hypothesis": "复现 TARL 论文机制并做机制审计（v21 前置）。",
 "plan_id": "reproduction/tarl_tsc",
 "design": "reproduction/tarl_tsc/{scripts,tests,provenance.py}",
 "conclusion": "132 完成、11 失败（含审计/契约 smoke）；为 v21 正式批提供机制证据。",
 "queues": "reproduction/tarl_tsc 自有脚本族",
},
"tarl_v21_formal": {
 "title": "TARL v21 正式批",
 "hypothesis": "v21 修正后 TARL 正式对照批。",
 "plan_id": "reproduction/tarl_tsc v21 formal",
 "design": "reproduction/tarl_tsc/scripts/*v21*",
 "conclusion": "15 个正式训练 run 全部 DONE。",
 "queues": "同上",
},
"milestone0_infra": {
 "title": "Milestone0 实验基础设施烟测",
 "hypothesis": "m0f*/verify_sumo 系列验证 run.py+sumo+dqn 链路可用。",
 "plan_id": "milestone0",
 "design": "docs/plan.md",
 "conclusion": "12 个 smoke 全部 DONE。",
 "queues": "—",
},
"paper_infra_validation": {
 "title": "论文基础设施验证批次",
 "hypothesis": "paper_robustness/sumo_events/dashboard 示例等 pytest 级工程验证。",
 "plan_id": "—",
 "design": "tests/ + data/output_data/{paper_robustness,sumo_events,training_dashboard_examples}",
 "conclusion": "19 个批次根全部 DONE（内部 run 单元见各批次 notes）。",
 "queues": "—",
},
"misc_probes": {
 "title": "杂项一次性探针",
 "hypothesis": "readiness/inspect/frap_colight 适配探针等一次性验证。",
 "plan_id": "—",
 "design": "data/output_data/tsc 下散件",
 "conclusion": "1 DONE、5 FAILED、1 SCOPE_DISCARD（inspect_mplight，2026-10-07 裁决：基线已被 baseline_eval 覆盖）。",
 "queues": "—",
},
"cl_hz1x1": {
 "title": "cl_hz1x1 同质单路口持续学习（cl1 主线 + so1 半离线 + soa1 输出锚定）",
 "hypothesis": "值迭代型 TSC-RL 在顺序场景训练下发生机制性灾难性遗忘；25% 因果历史离线混合应以架构无关方式消除遗忘且不损害新场景学习；"
               "在离线样本上加 stage 末教师输出锚定（CLEAR 式 policy cloning 的 DQN 移植）应在同回放预算下进一步压缩残余退化。",
 "plan_id": "cl_hz1x1_v1 / cl_hz1x1_anchor_{screen,main,main_o3,rho,vdqn,vdqnb}",
 "design": "artifacts/cl_hz1x1/{run_queue_cl1.json, run_queue_so1.json, run_queue_soa_*.json}；"
           "dev/cl_hz1x1/{build_queue,build_queue_so,build_queue_soa,harvest_matrix,harvest_soa,plot_so_compare,plot_soa_compare}.py；"
           "矩阵 artifacts/cl_hz1x1/matrix_report{,_so,_soa_soa1}.json；图 artifacts/cl_hz1x1/fig_{so,soa}_{retention_trajectory,worst_degradation}.png；"
           "锚定实现 agent/{dqn,frap}.py::_anchor_loss，诊断经 metrics schema v5 train_diagnostics 落盘",
 "conclusion": "cl1 1476/1476 + so1 612/612 + soa screen 306/306 + soa1 主矩阵 612/612 + soa3 复现 306/306 + soar 1224/1224 + soav 204/204 + 基线 12/12 全 DONE。"
               "①ddqn/duel/frap 三骨干在 clear 下均崩（o2 最深退化 +597/+282/+140s），遗忘为 off-policy 机制性失败非实现特例；"
               "②结构调制易感性（duel>ddqn>frap）但不免疫；③25% 因果半离线三臂 18/18 链最深退化 ≤+9s，对角线新学习近似保持——"
               "结论=抗遗忘修复，非绝对性能提升。PPO 为 on-policy 未入 so 臂。"
               "④soa1（o2, n=3，口径=全部旧场景 cell 最深 ΔATT）：policy 锚 λ_π=1 把 worst 退化 ddqn 1.54→0.44s、duel 5.23→0.90s、frap 9.79→0.99s，"
               "对角线 ΔATT 均值变化 ≤0.4s 近似零塑性代价；λ_π=10 双指标恶化（筛除非单调剂量响应）；"
               "+value 锚 λ_q=0.1 一致略逊于纯 policy 锚（同预算下钉旧 Q 值与 TD 拉扯）。"
               "⑤o3 复现（仅 +P 臂）：worst 退化 ddqn 3.91→0.35s、duel 5.96→0.56s、frap 0.73→−0.03s（全 cell 优于基线），"
               "跨序一致；sd1 链 st3 对角线单点 +2.4s 为唯一塑性毛刺（臂均值仍 −0.10s）。"
               "⑥ρ 剂量-响应（soar, o2, n=3）：+P@0.25 全面优于 replay@0.5（0.17/0.39/0.65 vs 0.92/1.67/1.25）——"
               "锚定一半预算胜过加倍回放；+P@0.125 对 frap/duel 压过 replay@0.5（frap 1.05 vs 1.25、duel 1.45 vs 1.67）"
               "且远低于 replay@0.25 的 frap 7.03，ddqn 上则不及（1.48 vs 0.92）；+P 非单调——ρ=0.5 时 worst 回升（ddqn 0.17→1.00、duel 0.39→0.93），"
               "历史占比过高稀释当前学习信号，ρ=0.25 为甜点；对角线全程 ≤0.4s。"
               "⑦vanilla DQN 补格（soav, o2, n=3，double_dqn=false）：此前全部 dqn 臂实为 DDQN；"
               "补跑 replay-only + +P 双臂及 12 个单场景基线后 worst 退化 2.55/3.62 → 0.46/0.78（5.5×），"
               "vanilla 的 max-target 偏置使纯回放退化大于 ddqn（2.55 vs 1.07），+P 同样压回 <1s——"
               "锚定收益对 double/dueling/FRAP/原始 DQN 四种输出结构一致成立。"
               "⑧适应代价分析（plot_soa_adapt，复用既有 records 无新 run）：切换的入场冲击集中于 o3 st3(c3→s1)——"
               "ddqn/duel 纯在线 ep0 冻结评估相对基线最差 +157/+104s、前10ep 训练 ATT +8.2s、20ep 超额成本 3-4×；"
               "半离线将入场差距压回 ≤2s，+P 与其相当或略优；frap 同切换差异 <1s——"
               "结论=半离线/锚定收益含'新场景入场可靠性'一层，但具顺序与骨干特异性，非普适加速。"
               "⑨snap3 收敛探针（model_snapshot 新通道，324 eval 任务，o3 st3 ep5/10/15/20 快照评估）："
               "确认纯在线入场冲击真实且恢复缓慢但可恢复——ddqn clear 最差种子 ep5-20 仍高于基线+5%、"
               "至 ep100 才稳定；但同时发现回放/锚定臂存在中段回弹（ddqn semi sd1 ep10=89.2s、"
               "+P sd1 ep15=86.1s、duel +P sd2 ep15-20≈83-84s），'settled'时点在三条件间交错，"
               "锚定在新场景轴上并非单调更优——诚实的口径是'可靠入场+偶发中段回弹'而非'更快收敛'。"
               "注：dqn 系 model/{e}_0.pt 为 target 快照（保守口径），frap 为 online 快照。"
               "216 行 SCOPE_DISCARD 为超范围裁决留档。",
 "queues": "artifacts/cl_hz1x1/{run_queue_cl1.json, run_queue_so1.json, run_queue_soa_screen.json, run_queue_soa_main.json, run_queue_soa_main_o3.json, run_queue_soa_rho.json, run_queue_soa_vdqn.json, run_queue_soa_vdqnb.json, run_queue_soa_snap3.json}",
},
}

def fmt_metrics(m):
    if not m:
        return "—"
    keys = ("travel_time", "throughput", "unfinished_vehicles")
    vals = [m.get(k) for k in keys]
    if all(v is None for v in vals):
        return "—"
    def f(v):
        if v is None: return "—"
        if isinstance(v, float): return f"{v:.1f}"
        return str(v)
    return f"{f(vals[0])}/{f(vals[1])}/{f(vals[2])}"

SHORT = {
"att_entity_004": "当前主线：保真达标；canonical≈empty 修正为窗内小幅净负；closure 缺口待修",
"att_entity_003": "TARL 对齐/传感矩阵完成（flx）；文本反事实探针完成",
"att_entity_002": "SGA/concat 文本注入矩阵：H2 支持、H3 未支持",
"arterial_1x6": "半离线干线 stage0/1/2 流水线完成",
"plan1_dqn": "Plan1 在线 DQN 基线",
"plan5_b100": "Plan5 b100 半离线批",
"tarl_reproduction": "TARL 论文复现基线（完结，仅对照）",
"tarl_v21_formal": "TARL v2.1 正式评估（完结）",
"milestone0_infra": "基础设施验证",
"paper_infra_validation": "paper 基建验证",
"misc_probes": "零散探针",
"cl_hz1x1": "hz1x1 持续学习：off-policy 机制性遗忘三骨干证实；+policy 锚 λπ=1 把 worst 退化压至 ≤1.0s 近零塑性代价；ρ 剂量响应显示锚定可省回放预算且 ρ=0.25 为甜点",
}

camps = collections.Counter(r["campaign"] for r in ROWS)
order = ["att_entity_004", "att_entity_003", "att_entity_002", "arterial_1x6",
         "tarl_reproduction", "tarl_v21_formal", "plan5_b100", "plan1_dqn",
         "milestone0_infra", "paper_infra_validation", "misc_probes"]

# ---------- 明细文件：每 campaign 一篇 ----------
DETAIL = ROOT / "EXPERIMENTS"
DETAIL.mkdir(exist_ok=True)
idx_rows = []
for camp in order + sorted(set(camps) - set(order)):
    rs = [r for r in ROWS if r["campaign"] == camp]
    if not rs: continue
    meta = META.get(camp, {"title": camp, "hypothesis": "—", "plan_id": "—",
                           "design": "—", "conclusion": "—", "queues": "—"})
    d = []
    d.append(f"# {meta['title']}")
    d.append("")
    d.append("> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`")
    d.append("")
    d.append(f"- 假设：{meta['hypothesis']}")
    d.append(f"- plan_id/队列：{meta['plan_id']}")
    d.append(f"- 设计稿/证据位置：{meta['design']}")
    d.append(f"- 结论：{meta['conclusion']}")
    d.append(f"- 登记单元 {len(rs)}（train/eval/smoke/batch/calibration = "
             f"{collections.Counter(r['run_kind'] for r in rs).get('train',0)}/"
             f"{collections.Counter(r['run_kind'] for r in rs).get('eval',0)}/"
             f"{collections.Counter(r['run_kind'] for r in rs).get('smoke',0)}/"
             f"{collections.Counter(r['run_kind'] for r in rs).get('batch',0)}/"
             f"{collections.Counter(r['run_kind'] for r in rs).get('calibration',0)}）"
             f"（追溯登记 2026-10-07）")
    d.append("")
    core = [r for r in rs if r["run_kind"] != "eval"]
    core.sort(key=lambda r: (r["run_kind"], r["run_id"]))
    if core:
        d.append("| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |")
        d.append("|---|---|---|---|---|---|---|---|")
        for r in core:
            sid = f"{r.get('arm') or '—'}/{r.get('net') or '—'}/s{r.get('seed') if r.get('seed') is not None else '—'}"
            ep = r.get("episodes_total")
            note = (r.get("notes") or "").replace("|", "／")[:80]
            if r.get("alias_of"): note = f"alias→{r['alias_of'][:40]}; " + note
            d.append(f"| `{r['run_id'][:60]}` | {sid} | {ep if ep is not None else '—'} | "
                     f"{r['status']} | {(r.get('started') or '—')[:10]} | {fmt_metrics(r.get('metrics'))} | "
                     f"`{(r.get('run_dir') or '—')[:80]}` | {note} |")
        d.append("")
    evs = [r for r in rs if r["run_kind"] == "eval"]
    if evs:
        grp = collections.defaultdict(list)
        for r in evs:
            grp[r.get("behavior_source") or r.get("arm") or "?"].append(r)
        d.append(f"评估 attempt 共 {len(evs)} 行，按包/来源聚合：")
        d.append("")
        d.append("| 评估包/来源 | n DONE/FAILED/ABORTED | 条件集 | eval seeds |")
        d.append("|---|---|---|---|")
        for pkg in sorted(grp):
            g = grp[pkg]
            sc = collections.Counter(r["status"] for r in g)
            conds = sorted(set(r["condition"] for r in g if r["condition"]))
            ess = sorted(set(r["eval_seed"] for r in g if r["eval_seed"] is not None))
            d.append(f"| `{pkg[:60]}` | {sc.get('DONE',0)}/{sc.get('FAILED',0)}/{sc.get('ABORTED',0)} | "
                     f"{','.join(conds)[:60] or '—'} | {','.join(map(str,ess))[:40] or '—'} |")
        d.append("")
    # 明细文件尾部人工维护区：<!-- EXTRA-BEGIN/END --> 块原样回填（同索引层 PANEL 约定）。
    detail_path = DETAIL / f"{camp}.md"
    if detail_path.exists():
        xm = re.search(r"<!-- EXTRA-BEGIN -->.*?<!-- EXTRA-END -->",
                       detail_path.read_text(), re.S)
        if xm:
            d.append(xm.group(0))
    detail_path.write_text("\n".join(d))
    sc = collections.Counter(r["status"] for r in rs)
    idx_rows.append(f"| `{camp}` | {len(rs)} | "
                    f"{sc.get('DONE',0)}/{sc.get('FAILED',0)}/{sc.get('ABORTED',0)}/{sc.get('SCOPE_DISCARD',0)} | "
                    f"{SHORT.get(camp,'—')} | [明细](EXPERIMENTS/{camp}.md) |")

# ---------- 索引文件 ----------
out = []
out.append("# EXPERIMENTS — 实验台账（索引）")
out.append("")
out.append("> 追溯登记 2026-10-07（治理分支 gov/experiment-governance）。")
out.append("> 事实源 = `ledger/runs.jsonl`（每 run 一行机读）；本文件 = 索引层，明细按 campaign 分文件 `EXPERIMENTS/<campaign>.md`——**按需取读，勿通读**。")
out.append("> 口径：`run_kind=train|eval|smoke|batch|calibration`；eval 行为单次冻结评估 attempt，"
           "`behavior_source` 指回训练 run 包名；`alias_of` 非空者不作独立样本统计。")
out.append("> 机器一律代号 34/73；路径均为仓内相对路径。")
out.append("")
tot = collections.Counter(r["status"] for r in ROWS)
out.append("## 0. 总览")
out.append("")
out.append(f"登记 run 总数：**{len(ROWS)}**（train {sum(1 for r in ROWS if r['run_kind']=='train')}、"
           f"eval attempt {sum(1 for r in ROWS if r['run_kind']=='eval')}、"
           f"smoke {sum(1 for r in ROWS if r['run_kind']=='smoke')}、"
           f"calibration {sum(1 for r in ROWS if r['run_kind']=='calibration')}、"
           f"batch {sum(1 for r in ROWS if r['run_kind']=='batch')}）")
out.append("")
out.append("| campaign | DONE | FAILED | ABORTED | SUPERSEDED | REGISTERED | SCOPE_DISCARD | LEGACY_UNCLEAR | 合计 |")
out.append("|---|---|---|---|---|---|---|---|---|")
for c in order + sorted(set(camps) - set(order)):
    rs = [r for r in ROWS if r["campaign"] == c]
    if not rs: continue
    sc = collections.Counter(r["status"] for r in rs)
    out.append(f"| {c} | {sc.get('DONE',0)} | {sc.get('FAILED',0)} | {sc.get('ABORTED',0)} | "
               f"{sc.get('SUPERSEDED',0)} | {sc.get('REGISTERED',0)} | {sc.get('SCOPE_DISCARD',0)} | "
               f"{sc.get('LEGACY_UNCLEAR',0)} | {len(rs)} |")
sc = collections.Counter(r["status"] for r in ROWS)
out.append(f"| **合计** | {sc.get('DONE',0)} | {sc.get('FAILED',0)} | {sc.get('ABORTED',0)} | "
           f"{sc.get('SUPERSEDED',0)} | {sc.get('REGISTERED',0)} | {sc.get('SCOPE_DISCARD',0)} | "
           f"{sc.get('LEGACY_UNCLEAR',0)} | **{len(ROWS)}** |")
out.append("")
out.append("状态语义见 GOVERNANCE.md §2。LEGACY_UNCLEAR=0 表示全部可识别 run 均通过"
           "队列/状态文件/目录归属证据还原出 campaign；个别 run 的臂内身份仍以命名约定为据，"
           "已在 notes 标注处保持保守。")
out.append("")
out.append("## 1. campaign 索引")
out.append("")
out.append("| campaign | runs | D/F/A/SD | 一句话结论 | 明细 |")
out.append("|---|---|---|---|---|")
out.extend(idx_rows)
out.append("")
out.append("## 附：追溯口径说明")
out.append("")
out.append("- `eval` 行 = 冻结评估 attempt（评估包内 `attempts/` 或 tsc 输出树下独立 eval 目录），"
           "`behavior_source` 回指训练 run/包。")
out.append("- `partial_*`/`quarantine`/`packages_INVALID_no_ckpt_load` 桶中的 attempt 按隔离语义"
           "登记为 ABORTED/FAILED，不与 DONE 混计。")
out.append("- 残留 `运行中` 状态且核查无存活进程（2026-10-07）者登记 ABORTED，notes 记已写训练条数。")
out.append("- §4.4 seed 贯通自检：对全部 DONE 训练行按 (campaign,arm) 分组比对终值三指标，"
           "未发现同配置不同 seed 终值全同，无 flag 行。")
out.append("- 相同 run 的隔离/重复派发副本以 `alias_of` 指向 canonical run，统计时去重。")
out.append("")

TARGET = ROOT / "EXPERIMENTS.md"
# 当前面板为人工维护区：重生成时抽取 <!-- PANEL-BEGIN/END --> 块原样回填（置于「0. 总览」前）。
m = re.search(r"<!-- PANEL-BEGIN -->.*?<!-- PANEL-END -->", open(TARGET).read(), re.S)
if m:
    for i, line in enumerate(out):
        if line.startswith("## 0."):
            out[i:i] = [m.group(0), ""]
            break
open(TARGET, "w").write("\n".join(out))
print("written index", len(out), "lines; detail files:", len(list(DETAIL.glob('*.md'))))
