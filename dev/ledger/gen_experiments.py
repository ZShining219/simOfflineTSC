#!/usr/bin/env python3
"""Generate EXPERIMENTS.md (human ledger) from runs.jsonl."""
import json, collections, re

ROWS = [json.loads(l) for l in open(
    "/data/users/zfh/workspace/projects/simOfflineTSC-gov/ledger/runs.jsonl")]

META = {
"att_entity_004": {
 "title": "ATT-ENTITY-004 结构化事件输入 + 语义辅助（注意力线当前主线）",
 "hypothesis": "事件信息以结构化 z_task 槽位 + 语义辅助头进入 TARL 注意力，应同时提升场景区分与决策优化（H: 结构化+aux > 纯文本注入）。",
 "plan_id": "att004_wave1/fix/struct/formal/ablation/holdout/e200 波次队列",
 "design": "artifacts/att_entity_004/{run_queue_att004_*.json, run_state_*/}；"
           "configs/tsc/att_entity_004/generated/*.yml；"
           "artifacts/att_entity_004/{ABLATION_ARMS_DESIGN,DIVIDE_CONQUER_DESIGN,SGA_REPAIR_DESIGN}.md",
 "conclusion": "fixG/fixR/fixD/crossq/ablA/ablation/a2s/a3s/holdout 波次均完成并留档；"
               "stg2 波次登记未执行、2026-10-07 裁决转 SCOPE_DISCARD，stg1/ctl 部分臂 DROPPED（SCOPE_DISCARD）；"
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

out = []
out.append("# EXPERIMENTS — 实验台账（人读索引）")
out.append("")
out.append("> 追溯登记 2026-10-07（治理分支 gov/experiment-governance）。")
out.append("> 事实源 = `ledger/runs.jsonl`（每 run 一行机读，本文件为其人读投影）。")
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
camps = collections.Counter(r["campaign"] for r in ROWS)
order = ["att_entity_004", "att_entity_003", "att_entity_002", "arterial_1x6",
         "tarl_reproduction", "tarl_v21_formal", "plan5_b100", "plan1_dqn",
         "milestone0_infra", "paper_infra_validation", "misc_probes"]
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

for camp in order + sorted(set(camps) - set(order)):
    rs = [r for r in ROWS if r["campaign"] == camp]
    if not rs: continue
    meta = META.get(camp, {"title": camp, "hypothesis": "—", "plan_id": "—",
                           "design": "—", "conclusion": "—", "queues": "—"})
    out.append(f"## {meta['title']}")
    out.append("")
    out.append(f"- 假设：{meta['hypothesis']}")
    out.append(f"- plan_id/队列：{meta['plan_id']}")
    out.append(f"- 设计稿/证据位置：{meta['design']}")
    out.append(f"- 结论：{meta['conclusion']}")
    out.append(f"- 登记单元 {len(rs)}（train/eval/smoke/batch/calibration = "
               f"{collections.Counter(r['run_kind'] for r in rs).get('train',0)}/"
               f"{collections.Counter(r['run_kind'] for r in rs).get('eval',0)}/"
               f"{collections.Counter(r['run_kind'] for r in rs).get('smoke',0)}/"
               f"{collections.Counter(r['run_kind'] for r in rs).get('batch',0)}/"
               f"{collections.Counter(r['run_kind'] for r in rs).get('calibration',0)}）"
               f"（追溯登记 2026-10-07）")
    out.append("")
    # non-eval rows full table
    core = [r for r in rs if r["run_kind"] != "eval"]
    core.sort(key=lambda r: (r["run_kind"], r["run_id"]))
    if core:
        out.append("| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |")
        out.append("|---|---|---|---|---|---|---|---|")
        for r in core:
            sid = f"{r.get('arm') or '—'}/{r.get('net') or '—'}/s{r.get('seed') if r.get('seed') is not None else '—'}"
            ep = r.get("episodes_total")
            note = (r.get("notes") or "").replace("|", "／")[:80]
            if r.get("alias_of"): note = f"alias→{r['alias_of'][:40]}; " + note
            out.append(f"| `{r['run_id'][:60]}` | {sid} | {ep if ep is not None else '—'} | "
                       f"{r['status']} | {(r.get('started') or '—')[:10]} | {fmt_metrics(r.get('metrics'))} | "
                       f"`{(r.get('run_dir') or '—')[:80]}` | {note} |")
        out.append("")
    # eval attempts aggregated by package (behavior_source) or arm
    evs = [r for r in rs if r["run_kind"] == "eval"]
    if evs:
        grp = collections.defaultdict(list)
        for r in evs:
            grp[r.get("behavior_source") or r.get("arm") or "?"].append(r)
        out.append(f"评估 attempt 共 {len(evs)} 行，按包/来源聚合：")
        out.append("")
        out.append("| 评估包/来源 | n DONE/FAILED/ABORTED | 条件集 | eval seeds |")
        out.append("|---|---|---|---|")
        for pkg in sorted(grp):
            g = grp[pkg]
            sc = collections.Counter(r["status"] for r in g)
            conds = sorted(set(r["condition"] for r in g if r["condition"]))
            ess = sorted(set(r["eval_seed"] for r in g if r["eval_seed"] is not None))
            out.append(f"| `{pkg[:60]}` | {sc.get('DONE',0)}/{sc.get('FAILED',0)}/{sc.get('ABORTED',0)} | "
                       f"{','.join(conds)[:60] or '—'} | {','.join(map(str,ess))[:40] or '—'} |")
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

TARGET = "/data/users/zfh/workspace/projects/simOfflineTSC-gov/EXPERIMENTS.md"
# 当前面板为人工维护区：重生成时抽取 <!-- PANEL-BEGIN/END --> 块原样回填。
m = re.search(r"<!-- PANEL-BEGIN -->.*?<!-- PANEL-END -->", open(TARGET).read(), re.S)
if m:
    for i, line in enumerate(out):
        if line.startswith("## 0."):
            out[i:i] = [m.group(0), ""]
            break
open(TARGET, "w").write("\n".join(out))
print("written", len(out), "lines")
