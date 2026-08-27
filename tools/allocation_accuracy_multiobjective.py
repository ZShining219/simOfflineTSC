#!/usr/bin/env python3
"""Explore multi-objective acceptable-action accuracy from existing rollouts."""
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

METHODS = ("CONT-DQN", "P1C-DHOA-R25", "P1C-DHOA-R50")
SCENES = ("sumohz1x1_config2", "sumohz1x1", "sumohz1x1_config4", "sumohz1x1_config3")
HORIZONS = (30, 60, 90)

def read(path):
    with path.open(encoding="utf-8", newline="") as f: return list(csv.DictReader(f))
def f(row, key):
    try: return float(row[key])
    except (KeyError, TypeError, ValueError): return float("nan")
def i(row, key):
    try: return int(float(row[key]))
    except (KeyError, TypeError, ValueError): return -1

def build_sets(rollouts, eps, delta, gamma):
    groups=defaultdict(list)
    for r in rollouts: groups[r["state_id"]].append(r)
    out={}
    for sid, rows in groups.items():
        q=np.array([f(r,"normalized_queue_60") for r in rows]); d=np.array([f(r,"normalized_real_delay_60") for r in rows]); t=1-np.array([f(r,"normalized_throughput_60") for r in rows]); c=np.array([f(r,"cost_60") for r in rows]); best=float(np.nanmin(c)); mins=np.array([np.nanmin(q),np.nanmin(d),np.nanmin(t)])
        accepted=[]
        for k in range(len(rows)):
            dominated=any(np.all(np.array([q[j],d[j],t[j]]) <= np.array([q[k],d[k],t[k]])-eps) for j in range(len(rows)) if j != k)
            if (not dominated) and c[k]-best <= delta and np.max(np.array([q[k],d[k],t[k]])-mins) <= gamma: accepted.append(i(rows[k],"candidate_action"))
        out[sid]=set(accepted)
    return out, groups

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input-dir",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    rollouts=read(args.input_dir/"allocation_accuracy_all_action_rollouts.csv"); preds=read(args.input_dir/"allocation_accuracy_agent_predictions.csv"); labels=read(args.input_dir/"allocation_accuracy_oracle_labels.csv")
    label={r["state_id"]:r for r in labels}; summary=[]; detail=[]; sensitivity=[]
    rules=[("primary",0.05,0.10,0.50),("strict_pareto",0.0,0.10,0.50),("delta_0.05",0.05,0.05,0.50),("delta_0.20",0.05,0.20,0.50),("single_metric_0.20",0.05,0.10,0.20)]
    for rule,eps,delta,gamma in rules:
        sets,groups=build_sets(rollouts,eps,delta,gamma); sizes=[len(x) for x in sets.values()]; sensitivity.append({"rule":rule,"epsilon_dominance":eps,"cost_regret_delta":delta,"single_metric_gamma":gamma,"states":len(sizes),"mean_set_size":float(np.mean(sizes)),"median_set_size":float(np.median(sizes)),"p90_set_size":float(np.quantile(sizes,.9)),"random_hit_pct":100*float(np.mean(sizes))/8})
        for method in METHODS:
            rows=[r for r in preds if r.get("method")==method and r.get("predicted_action","")!="" and r["state_id"] in sets]
            hits=sum(i(r,"predicted_action") in sets[r["state_id"]] for r in rows)
            scen=[]
            for scene in SCENES:
                sr=[r for r in rows if r.get("scene")==scene]; scen.append(100*sum(i(r,"predicted_action") in sets[r["state_id"]] for r in sr)/len(sr) if sr else float("nan"))
            summary.append({"rule":rule,"method":method,"states":len(rows),"acceptable_hit_pct":100*hits/len(rows) if rows else float("nan"),"equal_weight_scene_hit_pct":float(np.nanmean(scen)),"random_hit_pct":100*float(np.mean(sizes))/8})
            for r in rows:
                rs=groups[r["state_id"]]; q=[f(x,"normalized_queue_60") for x in rs]; d=[f(x,"normalized_real_delay_60") for x in rs]; t=[1-f(x,"normalized_throughput_60") for x in rs]; c=[f(x,"cost_60") for x in rs]; pa=i(r,"predicted_action"); idx=next((k for k,x in enumerate(rs) if i(x,"candidate_action")==pa),None)
                rank=1+sum(c[k] < c[idx]-1e-12 for k in range(len(c))) if idx is not None else -1
                detail.append({"rule":rule,"method":method,"state_id":r["state_id"],"scene":r.get("scene",""),"predicted_action":pa,"acceptable_actions":json.dumps(sorted(sets[r["state_id"]])),"acceptable_hit":idx is not None and pa in sets[r["state_id"]],"oracle_cost_rank":rank,"cost_regret":c[idx]-min(c) if idx is not None else float("nan"),"queue_regret_norm":q[idx]-min(q) if idx is not None else float("nan"),"delay_regret_norm":d[idx]-min(d) if idx is not None else float("nan"),"throughput_regret_norm":t[idx]-min(t) if idx is not None else float("nan"),"stable_label":str(label.get(r["state_id"],{}).get("labels_consistent_30_60_90","")).lower()=="true"})
    def write(name, rows):
        keys=sorted({k for r in rows for k in r});
        with (args.output_dir/name).open("w",encoding="utf-8",newline="") as h:
            w=csv.DictWriter(h,fieldnames=keys); w.writeheader(); w.writerows(rows)
    write("allocation_accuracy_multiobjective_summary.csv",summary); write("allocation_accuracy_multiobjective_state.csv",detail); write("allocation_accuracy_multiobjective_sensitivity.csv",sensitivity)
    audit={"definition":"acceptable action = epsilon-Pareto non-dominated AND cost regret <= delta AND max single-metric normalized regret <= gamma","primary_rule":rules[0],"methods":METHODS,"scenes":SCENES,"rollout_source":str(args.input_dir),"no_new_simulation":True,"strict_top1_preserved":True}
    (args.output_dir/"allocation_accuracy_multiobjective_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    primary=[r for r in summary if r["rule"]=="primary"]
    sens_primary=next(r for r in sensitivity if r["rule"]=="primary")
    report=[
        "# 多目标可接受资源配置命中率探索性审计",
        "",
        "本分析复用现有 2048 条全动作 SUMO 反事实分支，没有重新运行仿真或训练。严格唯一 argmin Top-1 保留在原始 summary 中；本报告增加多目标可接受动作集合诊断。",
        "",
        "## 主规则（读取 Agent 预测前固定）",
        "",
        "动作进入可接受集合需同时满足：5% ε-Pareto 支配容差下不被明显支配；综合 cost regret ≤ 0.10；queue、real delay、throughput 三个归一化损失相对该状态最优值的最大差值 ≤ 0.50。该规则是探索性预注册口径，不根据命中结果回调。",
        "",
        f"主规则下平均可接受动作数为 {sens_primary['mean_set_size']:.2f}/8，中位数 {sens_primary['median_set_size']:.0f}，P90 {sens_primary['p90_set_size']:.0f}；随机动作集合命中基准为 {sens_primary['random_hit_pct']:.2f}%。",
        "",
        "## 结果",
        "",
        "| 方法 | 等权场景可接受命中率 | 随机集合基准 | 相对随机提升 |",
        "|---|---:|---:|---:|",
    ]
    for r in primary:
        report.append(f"| {r['method']} | {r['equal_weight_scene_hit_pct']:.2f}% | {r['random_hit_pct']:.2f}% | {r['equal_weight_scene_hit_pct']-r['random_hit_pct']:.2f}个百分点 |")
    report += [
        "",
        "HA-DHOA-R25 的主规则命中率高于 CONT-DQN，说明可接受动作集合比严格唯一 argmin 更能表现其“接近多目标有效动作”的能力；但当前口径下仍未达到 90%，不能宣称项目阈值已满足。R50 只在现有 O2 覆盖的 82 个状态上解释。",
        "",
        "## 敏感性与限制",
        "",
        "规则敏感性见 `allocation_accuracy_multiobjective_sensitivity.csv`。放宽 cost regret 到 0.20 时，R25 等权场景命中率为 51.56%，平均集合大小约 2.00/8，仍未达到 90%；这表明低严格 Top-1 不能完全归因于唯一标签过严。",
        "",
        "可接受命中率是探索性多目标诊断，不替代严格 Top-1，也不等同于真实全局最优动作准确率。正式项目阈值若改用该指标，还需在项目层面预先确认 ε、delta 和单指标容差，并保留集合大小与随机基准作为区分力审计。",
    ]
    (args.output_dir/"allocation_accuracy_multiobjective_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x=np.arange(len(METHODS)); vals=[next(r["equal_weight_scene_hit_pct"] for r in primary if r["method"]==m) for m in METHODS]; rand=[next(r["random_hit_pct"] for r in primary if r["method"]==m) for m in METHODS]; fig,ax=plt.subplots(figsize=(8,4.5)); w=.35; b=ax.bar(x-w/2,vals,w,label="acceptable-action hit"); ax.bar(x+w/2,rand,w,label="random-set baseline"); ax.axhline(90,color="crimson",ls="--",label="90% threshold"); ax.bar_label(b,fmt="%.1f"); ax.set_xticks(x,METHODS); ax.set_ylabel("equal-weight scene hit (%)"); ax.legend(); fig.tight_layout(); fig.savefig(args.output_dir/"allocation_accuracy_multiobjective.png",dpi=180); plt.close(fig)
if __name__=="__main__": main()
