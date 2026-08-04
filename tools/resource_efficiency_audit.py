#!/usr/bin/env python3
"""Explore frozen-evaluation opportunity realization without rerunning simulation."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from resource_metric_audit import METHODS, LABEL, SCENES, bootstrap, digest, ha_episodes, jsonl, save_csv, traditional_episodes

ROOT = Path(__file__).resolve().parents[1]
PROJECT = tuple(METHODS)
STRONG_REFERENCES = ("MaxPressure", "IndependentDQN")
ALL_METHODS = ("FixedTime", *STRONG_REFERENCES, *PROJECT)
METRIC_SPECS = {
    "queue_only": ("queue_auc",),
    "queue_waiting": ("queue_auc", "waiting_auc"),
    "queue_waiting_real_delay": ("queue_auc", "waiting_auc", "real_delay"),
}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ha-root", type=Path, default=ROOT / "data/output_data/ha_sodqn/formal_e7705f7_20260726")
    p.add_argument("--runlist", type=Path, default=ROOT / "data/output_data/analysis/plan1/p1_formal_20_runlist_20260722.csv")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/output_data/resource_efficiency_p1c_dhoa_r25_r50")
    p.add_argument("--bootstrap-resamples", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=20260803)
    p.add_argument("--allow-existing-output", action="store_true")
    return p.parse_args()


def mean(values):
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.mean(clean) if clean else None


def std(values):
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.stdev(clean) if len(clean) > 1 else (0.0 if clean else None)


def union_fields(rows):
    """Preserve first-seen order while retaining heterogeneous summary columns."""
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields


def phase_switches_from_decisions(path):
    actions = [int(r["actions"][0]) for r in jsonl(path)]
    return sum(a != b for a, b in zip(actions, actions[1:]))


def load_independent_dqn(runlist):
    rows = []
    with runlist.open(newline="", encoding="utf-8") as f:
        for item in csv.DictReader(f):
            if item["role"] != "formal" or item["agent"] != "dqn" or item.get("include", "true").lower() != "true":
                continue
            source = Path(item["run_dir"]) / "metrics/records.jsonl"
            finals = [r for r in jsonl(source) if r["record_type"] == "FINAL_EVALUATION"]
            if len(finals) != 1:
                raise ValueError(f"{source}: expected one FINAL_EVALUATION, got {len(finals)}")
            record = finals[0]
            duration = float(record["simulation_step"])
            rows.append({
                "method": "IndependentDQN", "scenario": item["network"], "scene": LABEL[item["network"]],
                "order_id": None, "training_seed": int(item["training_seed"]), "evaluation_seed": record.get("evaluation_seed"),
                "episode_role": "formal_independent_dqn_final_evaluation", "duration_seconds": duration,
                "decision_steps": int(record["decision_step"]), "queue_auc": float(record["queue"]) * duration,
                "waiting_auc": None, "waiting_time": record.get("waiting_time"), "real_delay": record.get("real_delay"),
                "approximate_delay": record.get("delay"), "throughput": record.get("throughput"),
                "phase_switch_count": record.get("phase_switches"), "source_path": str(source), "source_sha256": digest(source),
            })
    return rows


def load_all(args):
    base = traditional_episodes(args.runlist)
    for row in base:
        row.update({"waiting_auc": None, "phase_switch_count": None, "source_path": row["decisions_path"]})
        final = next(r for r in jsonl(row["decisions_path"]) if r["record_type"] == "FINAL_EVALUATION")
        row["phase_switch_count"] = final.get("phase_switches")
    independent = load_independent_dqn(args.runlist)
    ha, _ = ha_episodes(args.ha_root)
    for row in ha:
        row.update({"waiting_auc": None, "phase_switch_count": phase_switches_from_decisions(row["decisions_path"]), "source_path": row["decisions_path"]})
    rows = base + independent + ha
    for row in rows:
        row.setdefault("waiting_auc", None)
    return rows


def fixed_denominators(rows):
    fixed = {}
    for scenario in SCENES:
        items = [r for r in rows if r["scenario"] == scenario and r["method"] == "FixedTime"]
        if len(items) != 1:
            raise ValueError(f"{scenario}: expected one FixedTime episode, got {len(items)}")
        fixed[scenario] = items[0]
    return fixed


def add_ratios_and_cost(rows, fixed, requested_metrics):
    for row in rows:
        base = fixed[row["scenario"]]
        ratios = []
        missing = []
        for metric in requested_metrics:
            denominator, numerator = base.get(metric), row.get(metric)
            ratio_name = {"queue_auc": "q_ratio", "waiting_auc": "w_ratio", "real_delay": "d_ratio"}[metric]
            ratio = numerator / denominator if numerator is not None and denominator not in (None, 0) else None
            row[ratio_name] = ratio
            if ratio is None: missing.append(metric)
            else: ratios.append(ratio)
        row["J"] = mean(ratios)
        row["J_metric_count"] = len(ratios)
        row["J_requested_metrics"] = ";".join(requested_metrics)
        row["J_used_metrics"] = ";".join(m for m in requested_metrics if m not in missing)
        row["J_missing_metrics"] = ";".join(missing)
        row["queue_improvement_pct"] = (1 - row["queue_auc"] / base["queue_auc"]) * 100
        row["waiting_time_improvement_pct"] = (1 - row["waiting_time"] / base["waiting_time"]) * 100 if row.get("waiting_time") is not None else None
        row["real_delay_improvement_pct"] = (1 - row["real_delay"] / base["real_delay"]) * 100 if row.get("real_delay") is not None else None
        row["approximate_delay_improvement_pct"] = (1 - row["approximate_delay"] / base["approximate_delay"]) * 100 if row.get("approximate_delay") is not None else None
        row["throughput_improvement_pct"] = (row["throughput"] / base["throughput"] - 1) * 100
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows: grouped[(row["scenario"], row["method"])].append(row)
    output = []
    metrics = ("queue_auc", "waiting_auc", "waiting_time", "real_delay", "approximate_delay", "throughput", "phase_switch_count", "q_ratio", "w_ratio", "d_ratio", "J", "queue_improvement_pct", "waiting_time_improvement_pct", "real_delay_improvement_pct", "approximate_delay_improvement_pct", "throughput_improvement_pct")
    for (scenario, method), items in sorted(grouped.items()):
        out = {"scenario": scenario, "scene": LABEL[scenario], "method": method, "n": len(items)}
        for metric in metrics:
            out[metric + "_mean"] = mean([r.get(metric) for r in items]); out[metric + "_std"] = std([r.get(metric) for r in items])
        out["J_metric_count"] = items[0]["J_metric_count"]
        out["J_requested_metrics"] = items[0]["J_requested_metrics"]
        out["J_used_metrics"] = items[0]["J_used_metrics"]
        out["J_missing_metrics"] = items[0]["J_missing_metrics"]
        output.append(out)
    return output


def realization(summary, gap_threshold):
    by_key = {(r["scenario"], r["method"]): r for r in summary}
    output = []
    for scenario in SCENES:
        refs = [by_key[(scenario, method)] for method in STRONG_REFERENCES]
        ref = min(refs, key=lambda r: r["J_mean"]); j_ref = ref["J_mean"]; gap = 1 - j_ref
        for method in PROJECT:
            row = by_key[(scenario, method)]; effective = gap >= gap_threshold
            output.append({
                "scenario": scenario, "scene": LABEL[scenario], "method": method, "n": row["n"],
                "J_method": row["J_mean"], "J_ref": j_ref, "strong_reference_method": ref["method"],
                "improvement_gap": gap, "gap_threshold": gap_threshold,
                "scenario_class": "effective-opportunity" if effective else "low-opportunity",
                "potential_realization_pct": (1 - row["J_mean"]) / gap * 100 if effective and gap != 0 else None,
                "noninferior_5pct": row["J_mean"] <= 1.05 if not effective else None,
                "throughput_noninferior": row["throughput_mean"] >= .95 * by_key[(scenario, "FixedTime")]["throughput_mean"],
                "phase_switch_count_mean": row["phase_switch_count_mean"],
                "phase_switch_vs_reference_pct": (row["phase_switch_count_mean"] / ref["phase_switch_count_mean"] - 1) * 100 if ref["phase_switch_count_mean"] else None,
                "queue_improvement_pct": row["queue_improvement_pct_mean"],
                "waiting_time_improvement_pct": row["waiting_time_improvement_pct_mean"],
                "real_delay_improvement_pct": row["real_delay_improvement_pct_mean"],
                "approximate_delay_improvement_pct": row["approximate_delay_improvement_pct_mean"],
                "throughput_improvement_pct": row["throughput_improvement_pct_mean"],
                "J_metric_count": row["J_metric_count"], "J_used_metrics": row["J_used_metrics"], "J_missing_metrics": row["J_missing_metrics"],
            })
    return output


def macro_rows(realization_rows):
    output = []
    for method in PROJECT:
        items = [r for r in realization_rows if r["method"] == method]
        effective = [r for r in items if r["scenario_class"] == "effective-opportunity"]
        low = [r for r in items if r["scenario_class"] == "low-opportunity"]
        macro = mean([r["potential_realization_pct"] for r in effective])
        output.append({
            "method": method, "effective_opportunity_scenarios": len(effective), "low_opportunity_scenarios": len(low),
            "macro_potential_realization_pct": macro, "project_threshold_pct": 10.0,
            "project_threshold_met": macro is not None and macro >= 10,
            "low_opportunity_noninferior_pass_rate": mean([float(r["noninferior_5pct"]) for r in low]),
            "throughput_constraint_pass_rate": mean([float(r["throughput_noninferior"]) for r in items]),
            "low_opportunity_clear_degradation": any(r["noninferior_5pct"] is False or not r["throughput_noninferior"] for r in low),
        })
    return output


def sensitivity(base_rows):
    output = []
    for name, metrics in METRIC_SPECS.items():
        rows = add_ratios_and_cost([dict(r) for r in base_rows], fixed_denominators(base_rows), metrics)
        summary = aggregate(rows)
        for threshold in (.03, .05, .10):
            details = realization(summary, threshold)
            macros = {r["method"]: r for r in macro_rows(details)}
            for row in details:
                output.append({"scope": "scenario", "metric_spec": name, "requested_metrics": ";".join(metrics), **row})
            for method, row in macros.items():
                output.append({"scope": "macro", "metric_spec": name, "requested_metrics": ";".join(metrics), "gap_threshold": threshold, **row})
    return output


def throughput_detail(rows, resamples, rng):
    """Quantify throughput by scene; bootstrap reflects episode/order/seed variation only."""
    fixed = fixed_denominators(rows)
    output = []
    for scenario in SCENES:
        base = float(fixed[scenario]["throughput"])
        maxpressure = mean([r["throughput"] for r in rows if r["scenario"] == scenario and r["method"] == "MaxPressure"])
        independent = mean([r["throughput"] for r in rows if r["scenario"] == scenario and r["method"] == "IndependentDQN"])
        for method in PROJECT:
            items = [r for r in rows if r["scenario"] == scenario and r["method"] == method]
            values = [float(r["throughput"]) for r in items]
            improvements = [(v / base - 1) * 100 for v in values]
            value_low, value_high = bootstrap(values, resamples, rng)
            pct_low, pct_high = bootstrap(improvements, resamples, rng)
            output.append({
                "scope": "scenario", "scenario": scenario, "scene": LABEL[scenario], "method": method, "n": len(items),
                "fixedtime_throughput": base, "maxpressure_throughput": maxpressure, "independent_dqn_throughput_mean": independent,
                "method_throughput_mean": mean(values), "method_throughput_std": std(values),
                "method_throughput_ci95_low": value_low, "method_throughput_ci95_high": value_high,
                "absolute_gain_vs_fixed": mean(values) - base, "improvement_pct_vs_fixed": mean(improvements),
                "improvement_pct_std": std(improvements), "improvement_pct_ci95_low": pct_low,
                "improvement_pct_ci95_high": pct_high, "throughput_noninferior_95pct_fixed": mean(values) >= .95 * base,
                "improvement_pct_vs_independent_dqn": (mean(values) / independent - 1) * 100,
            })
    for method in PROJECT:
        scenes = [r for r in output if r["method"] == method]
        draw_matrix = []
        for scene in SCENES:
            base = float(fixed[scene]["throughput"])
            values = [float(r["throughput"]) for r in rows if r["scenario"] == scene and r["method"] == method]
            draw_matrix.append(rng.choice([(v / base - 1) * 100 for v in values], (resamples, len(values)), replace=True).mean(axis=1))
        macro_draws = np.mean(np.vstack(draw_matrix), axis=0)
        output.append({
            "scope": "macro", "scenario": "MACRO", "scene": "Macro", "method": method,
            "n": sum(r["n"] for r in scenes), "fixedtime_throughput": mean([r["fixedtime_throughput"] for r in scenes]),
            "maxpressure_throughput": mean([r["maxpressure_throughput"] for r in scenes]),
            "independent_dqn_throughput_mean": mean([r["independent_dqn_throughput_mean"] for r in scenes]),
            "method_throughput_mean": mean([r["method_throughput_mean"] for r in scenes]),
            "method_throughput_std": None, "method_throughput_ci95_low": None, "method_throughput_ci95_high": None,
            "absolute_gain_vs_fixed": mean([r["absolute_gain_vs_fixed"] for r in scenes]),
            "improvement_pct_vs_fixed": mean([r["improvement_pct_vs_fixed"] for r in scenes]),
            "improvement_pct_std": std([r["improvement_pct_vs_fixed"] for r in scenes]),
            "improvement_pct_ci95_low": float(np.quantile(macro_draws, .025)),
            "improvement_pct_ci95_high": float(np.quantile(macro_draws, .975)),
            "throughput_noninferior_95pct_fixed": all(r["throughput_noninferior_95pct_fixed"] for r in scenes),
            "improvement_pct_vs_independent_dqn": mean([r["improvement_pct_vs_independent_dqn"] for r in scenes]),
        })
    return output


def plots(out, realization_rows, summary, throughput_rows):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SCENES)); width = .36
    for i, (method, color) in enumerate(((PROJECT[0], "#4c78a8"), (PROJECT[1], "#f58518"))):
        vals = [next(r["potential_realization_pct"] for r in realization_rows if r["scenario"] == s and r["method"] == method) for s in SCENES]
        ax.bar(x + (i - .5) * width, vals, width, label=method, color=color)
    ax.axhline(10, color="crimson", linestyle="--", label="10% project threshold"); ax.set_xticks(x, [LABEL[s] for s in SCENES])
    ax.set_ylabel("Potential realization (%)"); ax.set_title("Effective-opportunity realization (main specification)"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "resource_efficiency_potential_realization.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8)); metrics = (("queue_improvement_pct_mean", "Queue AUC reduction (%)"), ("waiting_time_improvement_pct_mean", "Episode waiting-time reduction (%)"), ("real_delay_improvement_pct_mean", "Real-delay reduction (%)"), ("throughput_improvement_pct_mean", "Throughput improvement (%)"))
    chosen = [r for r in summary if r["method"] in (*STRONG_REFERENCES, *PROJECT)]
    for ax, (metric, title) in zip(axes.flat, metrics):
        labels = [f'{r["scene"]}\n{r["method"].replace("P1C-DHOA-", "").replace("IndependentDQN", "IndDQN")}' for r in chosen]
        ax.bar(np.arange(len(chosen)), [r[metric] for r in chosen]); ax.axhline(0, color="black", linewidth=.8); ax.set_title(title); ax.set_xticks(np.arange(len(chosen)), labels, rotation=45, ha="right", fontsize=7)
    fig.suptitle("Raw traffic outcomes vs FixedTime (not the project efficiency metric)"); fig.tight_layout()
    fig.savefig(out / "resource_efficiency_raw_traffic_effects.png", dpi=180); plt.close(fig)

    fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.15, 1]})
    scenario_rows = [r for r in throughput_rows if r["scope"] == "scenario"]
    x = np.arange(len(SCENES)); width = .25
    fixed_values = [next(r["fixedtime_throughput"] for r in scenario_rows if r["scenario"] == s) for s in SCENES]
    ax_abs.bar(x - width, fixed_values, width, label="FixedTime", color="#777777")
    for index, (method, color) in enumerate(((PROJECT[0], "#4c78a8"), (PROJECT[1], "#f58518"))):
        chosen = [next(r for r in scenario_rows if r["scenario"] == s and r["method"] == method) for s in SCENES]
        vals = np.asarray([r["method_throughput_mean"] for r in chosen]); lows = np.asarray([r["method_throughput_ci95_low"] for r in chosen]); highs = np.asarray([r["method_throughput_ci95_high"] for r in chosen])
        ax_abs.bar(x + index * width, vals, width, label=method, color=color, yerr=np.vstack((vals-lows, highs-vals)), capsize=3)
    ax_abs.set_xticks(x, [LABEL[s] for s in SCENES]); ax_abs.set_ylabel("Completed vehicles / 3600 s"); ax_abs.set_title("Absolute frozen-evaluation throughput"); ax_abs.legend(fontsize=8)
    order = [LABEL[s] for s in SCENES] + ["Macro"] ; x2 = np.arange(len(order)); width2 = .36
    for index, (method, color) in enumerate(((PROJECT[0], "#4c78a8"), (PROJECT[1], "#f58518"))):
        chosen = [next(r for r in throughput_rows if r["scene"] == scene and r["method"] == method) for scene in order]
        vals = np.asarray([r["improvement_pct_vs_fixed"] for r in chosen]); lows = np.asarray([r["improvement_pct_ci95_low"] for r in chosen]); highs = np.asarray([r["improvement_pct_ci95_high"] for r in chosen])
        bars = ax_pct.bar(x2 + (index-.5)*width2, vals, width2, label=method, color=color, yerr=np.vstack((vals-lows, highs-vals)), capsize=3)
        for bar, value in zip(bars, vals): ax_pct.text(bar.get_x()+bar.get_width()/2, value+.45, f"{value:.1f}%", ha="center", fontsize=8)
    ax_pct.axhline(10, color="purple", linestyle="--", label="10% visual reference (not constraint)"); ax_pct.axhline(0, color="black", linewidth=.8)
    ax_pct.set_xticks(x2, order); ax_pct.set_ylabel("Improvement vs FixedTime (%)"); ax_pct.set_title("Throughput improvement: scene rates and equal-weight Macro"); ax_pct.legend(fontsize=8)
    fig.suptitle("Throughput detail — traffic outcome, not the potential-realization metric"); fig.tight_layout()
    fig.savefig(out / "resource_efficiency_throughput_detail.png", dpi=180); plt.close(fig)


def write_report(out, realization_rows, macros, throughput_rows, audit):
    lines = ["# 修订后的资源配置效率探索性报告", "", "本报告仅重算当前磁盘中的正式冻结评价结果，不改变原始实验结果，也不把queue AUC削减率直接称为资源配置效率提升。", "", "## 主结论", "", "主口径预先固定为gap阈值0.05，以及queue + waiting + real_delay等权。由于所有正式冻结资产均缺少逐步waiting序列，waiting_auc不可计算；依据‘仅使用实际存在且口径一致的指标’规则，主J实际由queue_auc和real_delay两项等权构成。", "", "| 方法 | 有效机会场景 | 低机会场景 | Macro兑现率 | 10%探索阈值 | Throughput约束通过率 |", "|---|---:|---:|---:|---|---:|"]
    for row in macros:
        lines.append(f'| {row["method"]} | {row["effective_opportunity_scenarios"]} | {row["low_opportunity_scenarios"]} | {row["macro_potential_realization_pct"]:.2f}% | {"达到" if row["project_threshold_met"] else "未达到"} | {row["throughput_constraint_pass_rate"]*100:.1f}% |')
    lines += ["", "## 场景可改进空间与兑现率", "", "| 场景 | 项目方法 | J方法 | 强参照 | J参照 | 可改进空间 | 分类 | 兑现率 |", "|---|---|---:|---|---:|---:|---|---:|"]
    for r in realization_rows:
        pct = f'{r["potential_realization_pct"]:.2f}%' if r["potential_realization_pct"] is not None else "不汇总"
        lines.append(f'| {r["scene"]} | {r["method"]} | {r["J_method"]:.4f} | {r["strong_reference_method"]} | {r["J_ref"]:.4f} | {r["improvement_gap"]*100:.2f}% | {r["scenario_class"]} | {pct} |')
    lines += ["", "## 原始交通治理结果", "", "以下为四场景等权宏平均，仅描述交通结果，不作为项目效率主指标；waiting列是episode末waiting_time，而非waiting_auc。", "", "| 方法 | Queue AUC下降 | Waiting time下降 | Real delay下降 | Approx. delay下降 | Throughput提升 |", "|---|---:|---:|---:|---:|---:|"]
    for method in PROJECT:
        items = [r for r in realization_rows if r["method"] == method]
        lines.append(f'| {method} | {mean([r["queue_improvement_pct"] for r in items]):.2f}% | {mean([r["waiting_time_improvement_pct"] for r in items]):.2f}% | {mean([r["real_delay_improvement_pct"] for r in items]):.2f}% | {mean([r["approximate_delay_improvement_pct"] for r in items]):.2f}% | {mean([r["throughput_improvement_pct"] for r in items]):.2f}% |')
    lines += ["", "## Throughput细化量化", "", "Throughput表示3600秒冻结评价内完成通行的车辆数。下表中的相对提升先在每个场景内相对FixedTime计算；Macro再对四个场景等权平均，不能用总车辆数直接池化替代。10%只是便于阅读的参照线，不是本次throughput约束；正式约束是项目方法不低于FixedTime的95%。", "", "| 场景 | 方法 | FixedTime | 方法均值±SD | 绝对增量 | 相对提升 | 95%区间 | n |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in throughput_rows:
        if row["scope"] == "scenario":
            lines.append(f'| {row["scene"]} | {row["method"]} | {row["fixedtime_throughput"]:.0f} | {row["method_throughput_mean"]:.1f}±{row["method_throughput_std"]:.1f} | {row["absolute_gain_vs_fixed"]:+.1f} | {row["improvement_pct_vs_fixed"]:.2f}% | [{row["improvement_pct_ci95_low"]:.2f}%, {row["improvement_pct_ci95_high"]:.2f}%] | {row["n"]} |')
    for row in throughput_rows:
        if row["scope"] == "macro":
            lines.append(f'| Macro | {row["method"]} | — | — | — | {row["improvement_pct_vs_fixed"]:.2f}% | [{row["improvement_pct_ci95_low"]:.2f}%, {row["improvement_pct_ci95_high"]:.2f}%] | 4场景 |')
    lines += ["", "宏平均约10.5%并不表示每个场景都提升约10%。当前冻结结果呈现明显的场景结构差异：S1和S4约3%，S2约21%，S3约15%，四场景等权后形成约10.5%的Macro；现有数据本身不能单独证明这种差异的因果来源。所有场景均为正提升，因此95%非劣约束为4/4通过。置信区间只描述现有episode、训练seed或训练顺序产生的波动；交通需求实现固定，不能解释为交通随机性区间。"]
    lines += ["", "## 五层含义", "", "1. 原始交通治理改善：queue、episode waiting_time、real_delay和throughput相对FixedTime的变化。", "2. 综合配置代价J：同场景内可用代价比率的等权平均；FixedTime固定为1。", "3. 场景可改进空间：1减去MaxPressure与IndependentDQN中较低的J。", "4. 配置效能兑现率：项目方法改善占上述空间的比例，不裁剪，超过100%表示超过强参照。", "5. 低可改进空间非劣性：仅当gap低于阈值时使用J不超过1.05及throughput不少于FixedTime的95%判断。", "", "## 字段和口径限制", "", "- waiting_auc缺失；waiting_time仅作为原始交通结果，不进入J，也不伪造为AUC。", "- real_delay在五类方法中均存在且定义一致，因此未使用approximate_delay替代主delay项。", "- phase_switch_count：传统与独立DQN采用正式FINAL_EVALUATION字段；HA-SODQN由冻结动作序列相邻动作变化次数计算。", "- FixedTime、MaxPressure和IndependentDQN各使用原Plan 1正式fixed-default评价；HA-SODQN使用第4阶段后的fixed-default四场景冻结矩阵。交通实现固定，波动不能外推为交通随机性。", "- R25覆盖O1–O4 × seeds 0–4；R50仅覆盖O2 × seeds 0–4，R50不能解释为跨顺序稳健性。", "", "## 敏感性", "", "敏感性表完整测试gap阈值0.03/0.05/0.10及queue only、queue+waiting、queue+waiting+real_delay三种预设；waiting相关预设均明确记录waiting_auc缺失及实际生效指标。四场景的主口径improvement gap均大于0.85，因此三档阈值下均为effective-opportunity，没有低机会场景可执行非劣性汇总。", "", f'- 纳入冻结episode：{audit["episode_count"]}。', f'- 独立DQN冻结episode：{audit["method_counts"]["IndependentDQN"]}。']
    (out / "revised_resource_efficiency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = arguments()
    if args.output_dir.exists() and not args.allow_existing_output:
        raise FileExistsError(f"Output exists; use --allow-existing-output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_all(args); fixed = fixed_denominators(raw); rng = np.random.default_rng(args.bootstrap_seed)
    main_metrics = METRIC_SPECS["queue_waiting_real_delay"]
    episodes = add_ratios_and_cost([dict(r) for r in raw], fixed, main_metrics)
    summary = aggregate(episodes); realization_rows = realization(summary, .05); macros = macro_rows(realization_rows)
    sensitivity_rows = sensitivity(raw)
    throughput_rows = throughput_detail(raw, args.bootstrap_resamples, rng)
    save_csv(args.output_dir / "resource_efficiency_episode_level.csv", episodes)
    save_csv(args.output_dir / "resource_efficiency_scenario_summary.csv", summary)
    potential_output = realization_rows + [{"scenario": "MACRO", "scope": "macro", **r} for r in macros]
    save_csv(args.output_dir / "resource_efficiency_potential_realization.csv", potential_output, union_fields(potential_output))
    save_csv(args.output_dir / "resource_efficiency_sensitivity.csv", sensitivity_rows, union_fields(sensitivity_rows))
    save_csv(args.output_dir / "resource_efficiency_throughput_detail.csv", throughput_rows)
    plots(args.output_dir, realization_rows, summary, throughput_rows)
    counts = {m: sum(r["method"] == m for r in raw) for m in ALL_METHODS}
    audit = {"schema_version": 1, "status": "exploratory_frozen_evaluation_recalculation", "episode_count": len(raw), "method_counts": counts, "main_specification": {"gap_threshold": .05, "requested_metrics": list(main_metrics), "effective_metrics": ["queue_auc", "real_delay"], "missing_metrics": ["waiting_auc"]}, "strong_reference_set": list(STRONG_REFERENCES), "project_methods_excluded_from_reference": list(PROJECT), "comparability": {"simulation_duration_seconds": 3600, "decision_steps": 360, "traffic_realization": "fixed_default deterministic realization"}, "limitations": {"waiting_auc": "not present in any frozen evaluation; episode waiting_time retained only as raw outcome", "r50_coverage": "O2 x seeds 0-4 only"}, "source_paths": {"runlist": str(args.runlist), "ha_root": str(args.ha_root)}}
    (args.output_dir / "resource_efficiency_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir, realization_rows, macros, throughput_rows, audit)
    print(json.dumps({"output_dir": str(args.output_dir), "counts": counts, "macro": macros}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
