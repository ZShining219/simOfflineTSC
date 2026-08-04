#!/usr/bin/env python3
"""Read-only traffic resource metric audit for frozen SUMO experiment assets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENES = ("sumohz1x1_config2", "sumohz1x1", "sumohz1x1_config4", "sumohz1x1_config3")
LABEL = dict(zip(SCENES, ("S1", "S2", "S3", "S4")))
METHODS = {
    "P1C-DHOA-R25": re.compile(r"^P1C-DHOA-R25-(O[1-4])-SD([0-4])$"),
    "P1C-DHOA-R50": re.compile(r"^P1C-DHOA-R50-(O[1-4])-SD([0-4])$"),
}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ha-root", type=Path, default=ROOT / "data/output_data/ha_sodqn/formal_e7705f7_20260726")
    p.add_argument("--baseline-runlist", type=Path, default=ROOT / "data/output_data/analysis/plan1/p1_formal_20_runlist_20260722.csv")
    p.add_argument("--baseline-decisions", type=Path, default=ROOT / "data/output_data/evaluations/plan1/plan1_best_checkpoint_reevaluation_v1_20260723/records.jsonl")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/output_data/resource_metric_audit_p1c_dhoa_r25_r50")
    p.add_argument("--bootstrap-resamples", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=20260803)
    p.add_argument("--allow-existing-output", action="store_true", help="Resume/overwrite only this tool's named outputs in an existing directory")
    return p.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def stats(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not values:
        return None, None, 0
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0, len(values)


def bootstrap(values, count, rng):
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(x): return None, None
    if len(x) == 1: return float(x[0]), float(x[0])
    draws = rng.choice(x, (count, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def network_path(scene):
    cfg = load_json(ROOT / "configs/sim" / (scene + ".cfg"))
    return ROOT / "data" / cfg["roadnetFile"]


def angle(shape, inbound=True):
    pts = [tuple(map(float, p.split(","))) for p in shape.split()]
    if inbound: x, y = pts[-2][0] - pts[-1][0], pts[-2][1] - pts[-1][1]
    else: x, y = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
    result = math.atan2(x, y)
    return result if result >= 0 else result + 2 * math.pi


def phase_lane_mapping(scene):
    path = network_path(scene)
    root = ET.parse(path).getroot()
    tls = root.find("tlLogic"); tl_id = tls.attrib["id"]
    connections = [c for c in root.findall("connection") if c.get("tl") == tl_id]
    indexed, edges = {}, set()
    for c in connections:
        indexed[int(c.attrib["linkIndex"])] = c.attrib["from"] + "_" + c.attrib["fromLane"]
        edges.add(c.attrib["from"])
    lanes = {x.attrib["id"]: x for x in root.findall("./edge/lane")}
    shapes = {edge: next(x.attrib["shape"] for key, x in lanes.items() if key.startswith(edge + "_")) for edge in edges}
    state_lanes = []
    for edge in sorted(edges, key=lambda x: angle(shapes[x])):
        state_lanes.extend(sorted((x for x in lanes if x.startswith(edge + "_")), key=lambda x: int(x.rsplit("_", 1)[1])))
    phases = [p.attrib["state"] for p in tls.findall("phase") if "y" not in p.attrib["state"] and any(x in "Gg" for x in p.attrib["state"])]
    service = [sorted({indexed[i] for i, x in enumerate(state) if x in "Gg" and i in indexed}) for state in phases]
    if len(state_lanes) != 8 or len(service) != 8:
        raise ValueError(f"{scene}: expected 8 lanes/actions, got {len(state_lanes)}/{len(service)}")
    return {"scene": scene, "net_xml": str(path), "net_xml_sha256": digest(path), "state_lanes": state_lanes, "action_service_lanes": service, "green_phase_states": phases}


def capture(states, actions, mapping):
    states = np.asarray(states, float).reshape(-1, 8); actions = np.asarray(actions, int).reshape(-1)
    matrix = np.asarray([[lane in served for lane in mapping["state_lanes"]] for served in mapping["action_service_lanes"]], float)
    demand = states @ matrix.T; maxima = demand.max(axis=1)
    legal = (actions >= 0) & (actions < 8); nonempty = states.sum(axis=1) > 0
    valid = legal & nonempty & (maxima > 0); ratio = np.full(len(states), np.nan)
    idx = np.arange(len(states))[valid]; ratio[valid] = demand[idx, actions[valid]] / maxima[valid]
    selected = ratio[valid]
    return {"acc_at_0_8": float(np.mean(selected >= .8)) if len(selected) else None, "acc_at_0_9": float(np.mean(selected >= .9)) if len(selected) else None, "acc_at_1_0": float(np.mean(np.isclose(selected, 1))) if len(selected) else None, "mean_capture_ratio": float(np.mean(selected)) if len(selected) else None, "valid_decisions": int(valid.sum()), "excluded_decisions": int((~valid).sum()), "excluded_empty_demand": int((~nonempty).sum()), "excluded_illegal_action": int((~legal).sum())}


def traditional_episodes(runlist):
    rows = []
    with runlist.open(newline="", encoding="utf-8") as f:
        for item in csv.DictReader(f):
            if item["role"] != "baseline" or item["agent"] not in {"fixedtime", "maxpressure"}: continue
            record = next(r for r in jsonl(Path(item["run_dir"]) / "metrics/records.jsonl") if r["record_type"] == "FINAL_EVALUATION")
            duration = float(record["simulation_step"]); queue_auc = float(record["queue"]) * duration
            source = Path(item["run_dir"]) / "metrics/records.jsonl"
            rows.append({"method": "FixedTime" if item["agent"] == "fixedtime" else "MaxPressure", "scenario": item["network"], "scene": LABEL[item["network"]], "order_id": None, "training_seed": int(item["training_seed"]), "evaluation_seed": None, "evaluation_seed_mode": "fixed_default", "episode_role": "formal_fixed_default_baseline", "stage_index": None, "decision_steps": int(record["decision_step"]), "duration_seconds": duration, "queue_auc": queue_auc, "sum_queue": float(record["queue"]) * int(record["decision_step"]), "mean_queue": float(record["queue"]), "waiting_time": record.get("waiting_time"), "approximate_delay": record.get("delay"), "real_delay": record.get("real_delay"), "throughput": record.get("throughput"), "travel_time": record.get("travel_time"), "service_efficiency": float(record["throughput"]) / queue_auc, "accuracy_status": "not_computed_for_auxiliary_controller", "accuracy_source": None, "physical_key": None, "decisions_path": str(source), "source_sha256": digest(source)})
    return rows


def effective_attempt(logical_dir):
    manifest = load_json(logical_dir / "logical_run_manifest.json")
    item = next(x for x in manifest["attempts"] if x["attempt_id"] == manifest["effective_attempt"])
    return Path(item["attempt_dir"])


def final_aliases(logical_dir):
    for event in jsonl(effective_attempt(logical_dir) / "events.jsonl"):
        if event["event_type"] != "OPERATION_COMMITTED": continue
        key = event["payload"].get("operation_key", "")
        if re.fullmatch(r"evaluation:stage_4:local_100:(?:" + "|".join(SCENES) + r")", key):
            yield Path(event["payload"]["artifact"])


def ha_episodes(root):
    rows, identities = [], []
    for method, pattern in METHODS.items():
        for logical_dir in sorted(root.iterdir()):
            match = pattern.match(logical_dir.name)
            if not match or not (logical_dir / "logical_run_manifest.json").exists(): continue
            order, seed = match.group(1), int(match.group(2)); aliases = list(final_aliases(logical_dir))
            if len(aliases) != 4: raise ValueError(f"{logical_dir.name}: expected 4 final aliases, found {len(aliases)}")
            for alias_path in aliases:
                alias = load_json(alias_path); committed = load_json(Path(alias["physical_committed_path"]))
                summary = load_json(Path(committed["summary_path"])); decisions = list(jsonl(committed["decisions_path"]))
                queue = np.asarray([r["queue_network_sum"] for r in decisions], float)
                dt = np.asarray([r["action_interval_seconds"] for r in decisions], float)
                throughput = int(decisions[-1]["throughput_cumulative"]); queue_auc = float(np.sum(queue * dt))
                rows.append({"method": method, "scenario": summary["evaluation_network"], "scene": LABEL[summary["evaluation_network"]], "order_id": order, "training_seed": seed, "evaluation_seed": None, "evaluation_seed_mode": "fixed_default", "episode_role": "final_stage_frozen_matrix", "stage_index": 4, "decision_steps": len(decisions), "duration_seconds": float(dt.sum()), "queue_auc": queue_auc, "sum_queue": float(queue.sum()), "mean_queue": float(queue.mean()), "waiting_time": summary.get("waiting_time"), "approximate_delay": summary.get("delay"), "real_delay": summary.get("real_delay"), "throughput": throughput, "travel_time": summary.get("travel_time"), "service_efficiency": throughput / queue_auc if queue_auc > 0 else None, "accuracy_status": "not_computable_frozen_pre_action_state_missing", "accuracy_source": None, "physical_key": alias["physical_key"], "decisions_path": committed["decisions_path"], "source_sha256": digest(committed["decisions_path"])})
                identities.append({"method": method, "logical_run_id": logical_dir.name, "order_id": order, "training_seed": seed, "scenario": summary["evaluation_network"], "alias_path": str(alias_path), "physical_key": alias["physical_key"], "decisions_path": committed["decisions_path"], "decisions_sha256": digest(committed["decisions_path"])})
    return rows, identities


def training_accuracy(root, mappings):
    current = load_json(ROOT / "data/output_data/analysis/plan34/plan34_b100_current_final_v1_20260728/current_manifest.json")
    rows = []
    for method, pattern in METHODS.items():
        for logical_dir in sorted(root.iterdir()):
            match = pattern.match(logical_dir.name)
            if not match: continue
            order, seed = match.group(1), int(match.group(2))
            networks = current["scope"]["orders"][order]["networks"]
            for stage, scenario in enumerate(networks, 1):
                candidates = sorted(logical_dir.glob(f"attempts/attempt_*/trajectory/episodes/stage_{stage:02d}_episode_0100.npz"))
                if not candidates: raise FileNotFoundError(f"{logical_dir.name}: no stage {stage} episode 100 NPZ across attempts")
                path = candidates[-1]
                with np.load(path, allow_pickle=False) as data: result = capture(data["state"], data["action"], mappings[scenario])
                rows.append({"method": method, "scenario": scenario, "scene": LABEL[scenario], "order_id": order, "training_seed": seed, "stage_index": stage, "episode": 100, "accuracy_status": "training_trajectory_sensitivity_only", "accuracy_source": str(path), **result})
    return rows


PCT_METRICS = ("queue_burden_reduction_pct", "waiting_time_reduction_pct", "approximate_delay_reduction_pct", "real_delay_reduction_pct", "throughput_improvement_pct", "service_efficiency_improvement_pct")


def summarize_episodes(rows, resamples, rng):
    fixed = {r["scenario"]: r for r in rows if r["method"] == "FixedTime"}
    for row in rows:
        base = fixed[row["scenario"]]
        row["queue_burden_reduction_pct"] = (base["queue_auc"] - row["queue_auc"]) / base["queue_auc"] * 100
        row["waiting_time_reduction_pct"] = (base["waiting_time"] - row["waiting_time"]) / base["waiting_time"] * 100
        row["approximate_delay_reduction_pct"] = (base["approximate_delay"] - row["approximate_delay"]) / base["approximate_delay"] * 100
        row["real_delay_reduction_pct"] = (base["real_delay"] - row["real_delay"]) / base["real_delay"] * 100
        row["throughput_improvement_pct"] = (row["throughput"] - base["throughput"]) / base["throughput"] * 100
        row["service_efficiency_improvement_pct"] = ((row["service_efficiency"] - base["service_efficiency"]) / base["service_efficiency"] * 100) if row["service_efficiency"] is not None else None
    grouped = defaultdict(list)
    for row in rows: grouped[(row["scenario"], row["method"])].append(row)
    summary = []
    metrics = ("queue_auc", "mean_queue", "waiting_time", "approximate_delay", "real_delay", "throughput", "service_efficiency") + PCT_METRICS
    for (scenario, method), items in sorted(grouped.items()):
        out = {"scenario": scenario, "scene": LABEL[scenario], "method": method, "n": len(items)}
        for metric in metrics:
            mean, std, _ = stats([x.get(metric) for x in items]); out[metric + "_mean"] = mean; out[metric + "_std"] = std
        low, high = bootstrap([x["queue_burden_reduction_pct"] for x in items], resamples, rng)
        out["queue_burden_reduction_pct_ci95_low"] = low; out["queue_burden_reduction_pct_ci95_high"] = high
        summary.append(out)
    macro = []
    for method in ("MaxPressure", "P1C-DHOA-R25", "P1C-DHOA-R50"):
        selected = [r for r in summary if r["method"] == method]
        reductions = [r["queue_burden_reduction_pct_mean"] for r in selected]
        result = {"method": method, "comparison_baseline": "FixedTime", "scenario_count": len(selected), "macro_queue_burden_reduction_pct": statistics.mean(reductions), "scenes_meeting_queue_10pct": sum(x >= 10 for x in reductions), "scenes_with_queue_worsening": sum(x < 0 for x in reductions), "queue_efficiency_threshold_pct": 10.0, "queue_efficiency_threshold_met": statistics.mean(reductions) >= 10, "macro_acc_at_0_9": None, "accuracy_threshold_pct": 90.0, "accuracy_threshold_status": "not_computable_frozen_pre_action_state_missing" if method.startswith("P1C") else "not_applicable"}
        for metric in PCT_METRICS[1:]:
            result["macro_" + metric] = statistics.mean(r[metric + "_mean"] for r in selected)
        macro.append(result)
    return summary, macro


def summarize_accuracy(rows, resamples, rng):
    grouped = defaultdict(list)
    for row in rows: grouped[(row["scenario"], row["method"])].append(row)
    output = []
    for (scenario, method), items in sorted(grouped.items()):
        result = {"scenario": scenario, "scene": LABEL[scenario], "method": method, "accuracy_status": "training_trajectory_sensitivity_only", "n": len(items)}
        for metric in ("acc_at_0_8", "acc_at_0_9", "acc_at_1_0", "mean_capture_ratio"):
            mean, std, _ = stats([x[metric] for x in items]); result[metric + "_mean"] = mean; result[metric + "_std"] = std
        low, high = bootstrap([x["acc_at_0_9"] for x in items], resamples, rng)
        result["acc_at_0_9_ci95_low"] = low; result["acc_at_0_9_ci95_high"] = high
        result["valid_decisions"] = sum(x["valid_decisions"] for x in items); result["excluded_decisions"] = sum(x["excluded_decisions"] for x in items)
        output.append(result)
    return output


def accuracy_evidence(rows, summary, resamples, rng):
    """Build scenario-macro evidence while preserving the frozen/sensitivity distinction."""
    output = []
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["scenario"], row["method"])].append(row)
    for item in summary:
        output.append({
            "scope": "scenario", "scenario": item["scenario"], "scene": item["scene"], "method": item["method"],
            "frozen_accuracy_status": "not_computable_pre_action_state_missing",
            "evidence_role": "training_trajectory_sensitivity_only", "acc_at_0_9": item["acc_at_0_9_mean"],
            "acc_at_0_9_ci95_low": item["acc_at_0_9_ci95_low"], "acc_at_0_9_ci95_high": item["acc_at_0_9_ci95_high"],
            "threshold": .90, "sensitivity_threshold_met": item["acc_at_0_9_mean"] >= .90,
            "episode_n": item["n"], "valid_decisions": item["valid_decisions"], "excluded_decisions": item["excluded_decisions"],
            "micro_acc_at_0_9": sum(x["acc_at_0_9"] * x["valid_decisions"] for x in by_group[(item["scenario"], item["method"])]) / item["valid_decisions"],
        })
    for method in METHODS:
        scene_groups = [by_group[(scenario, method)] for scenario in SCENES]
        scene_means = [statistics.mean(x["acc_at_0_9"] for x in items) for items in scene_groups]
        draws = []
        for _ in range(resamples):
            sampled_scene_means = [statistics.mean(rng.choice([x["acc_at_0_9"] for x in items], len(items), replace=True)) for items in scene_groups]
            draws.append(statistics.mean(sampled_scene_means))
        valid = sum(x["valid_decisions"] for items in scene_groups for x in items)
        excluded = sum(x["excluded_decisions"] for items in scene_groups for x in items)
        correct = sum(x["acc_at_0_9"] * x["valid_decisions"] for items in scene_groups for x in items)
        output.append({
            "scope": "macro", "scenario": "MACRO", "scene": "Macro", "method": method,
            "frozen_accuracy_status": "not_computable_pre_action_state_missing",
            "evidence_role": "training_trajectory_sensitivity_only", "acc_at_0_9": statistics.mean(scene_means),
            "acc_at_0_9_ci95_low": float(np.quantile(draws, .025)), "acc_at_0_9_ci95_high": float(np.quantile(draws, .975)),
            "threshold": .90, "sensitivity_threshold_met": statistics.mean(scene_means) >= .90,
            "episode_n": sum(len(items) for items in scene_groups), "valid_decisions": valid,
            "excluded_decisions": excluded, "micro_acc_at_0_9": correct / valid,
        })
    return output


def baseline_curves(path):
    curves = defaultdict(list)
    for record in jsonl(path):
        if record.get("agent") == "fixedtime" and record.get("evaluation_seed") == 10000:
            curves[record["network"]].append((record["simulation_time_seconds"], record["queue_network_sum"]))
    return curves


def render_plots(out, accuracy_evidence_rows, scenario_summary, episodes, baseline_path):
    fig, ax = plt.subplots(figsize=(12, 6)); order = [LABEL[x] for x in SCENES] + ["Macro"]
    x = np.arange(len(order)); width = .36
    for index, (method, color) in enumerate((("P1C-DHOA-R25", "#4c78a8"), ("P1C-DHOA-R50", "#f58518"))):
        chosen = [next(r for r in accuracy_evidence_rows if r["scene"] == scene and r["method"] == method) for scene in order]
        values = np.asarray([r["acc_at_0_9"] * 100 for r in chosen]); lows = np.asarray([r["acc_at_0_9_ci95_low"] * 100 for r in chosen]); highs = np.asarray([r["acc_at_0_9_ci95_high"] * 100 for r in chosen])
        bars = ax.bar(x + (index - .5) * width, values, width, color=color, label=method, yerr=np.vstack((values - lows, highs - values)), capsize=3)
        for bar, value, row in zip(bars, values, chosen):
            ax.text(bar.get_x() + bar.get_width()/2, value + 1.5, f'{value:.1f}%\nn={row["episode_n"]}', ha="center", va="bottom", fontsize=8)
    ax.axhline(90, color="crimson", linestyle="--", linewidth=1.8, label="90% requirement")
    ax.set_xticks(x, order); ax.set_ylim(0, 106); ax.set_ylabel("Acc@0.9 (%)")
    ax.set_title("Resource-allocation accuracy evidence\nTraining-trajectory sensitivity only; formal frozen Acc@0.9 = N/A (pre-action state missing)")
    ax.text(.01, .98, "Formal threshold status: NOT VERIFIABLE\nSensitivity values shown below do not meet 90%", transform=ax.transAxes, va="top", color="darkred", bbox={"facecolor":"#fff3f3", "edgecolor":"darkred", "boxstyle":"round,pad=.4"})
    ax.legend(loc="upper right"); fig.tight_layout()
    fig.savefig(out / "resource_accuracy_evidence.png", dpi=180); fig.savefig(out / "resource_accuracy_acc09.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5)); chosen = [r for r in scenario_summary if r["method"] in {"MaxPressure", *METHODS}]
    labels = [f'{r["scene"]}\n{r["method"].replace("P1C-DHOA-", "")}' for r in chosen]; values = [r["queue_burden_reduction_pct_mean"] for r in chosen]
    ax.bar(labels, values); ax.axhline(10, color="crimson", linestyle="--", label="10% threshold"); ax.set_ylabel("Queue burden reduction vs FixedTime (%)"); ax.legend(); fig.tight_layout(); fig.savefig(out / "queue_burden_reduction.png", dpi=180); plt.close(fig)
    fixed = baseline_curves(baseline_path); paths = defaultdict(list)
    for row in episodes:
        if row["method"] in METHODS: paths[(row["scenario"], row["method"])].append(row["decisions_path"])
    for scenario in SCENES:
        fig, (ax, ax_auc) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]})
        if fixed[scenario]:
            times, queue = zip(*fixed[scenario]); ax.plot(times, queue, color="black", label="FixedTime seed10000 (visual reference)")
        for method, color in (("P1C-DHOA-R25", "#4c78a8"), ("P1C-DHOA-R50", "#f58518")):
            series = [[r["queue_network_sum"] for r in jsonl(path)] for path in paths[(scenario, method)]]
            ax.plot(np.arange(1, 361) * 10, np.mean(series, axis=0), color=color, label=method + " mean")
        ax.set(title=LABEL[scenario] + " queue time curve", xlabel="Simulation time (s)", ylabel="Queued vehicles"); ax.legend(fontsize=8)
        auc_rows = [r for r in scenario_summary if r["scenario"] == scenario and r["method"] in {"FixedTime", *METHODS}]
        auc_order = {"FixedTime": 0, "P1C-DHOA-R25": 1, "P1C-DHOA-R50": 2}; auc_rows.sort(key=lambda r: auc_order[r["method"]])
        auc_labels = [r["method"].replace("P1C-DHOA-", "") for r in auc_rows]
        auc_values = [r["queue_auc_mean"] for r in auc_rows]
        ax_auc.bar(auc_labels, auc_values, color=["black", "#4c78a8", "#f58518"])
        ax_auc.set_title("Formal queue AUC"); ax_auc.set_ylabel("vehicle·seconds"); ax_auc.tick_params(axis="x", rotation=30)
        fig.tight_layout(); fig.savefig(out / ("queue_curve_" + LABEL[scenario] + ".png"), dpi=180); plt.close(fig)


def write_report(out, macro, accuracy, audit):
    lines = ["# 交通资源配置探索性计算与数据审计", "", "## 结论", "", "冻结评价排队效率可以判定；冻结配置准确率不能判定，因为现有HA-SODQN决策日志未保存动作前状态。训练轨迹准确率只作敏感性诊断。", "", "| 方法 | Macro queue削减率 | 达到10%场景数 | 效率阈值 | 冻结Acc@0.9 |", "|---|---:|---:|---|---|"]
    for row in macro:
        status = "达到" if row["queue_efficiency_threshold_met"] else "未达到"
        lines.append(f'| {row["method"]} | {row["macro_queue_burden_reduction_pct"]:.2f}% | {row["scenes_meeting_queue_10pct"]}/4 | {status} | {row["accuracy_threshold_status"]} |')
    lines += ["", "## 配置准确率量化证据", "", "正式冻结Acc@0.9仍不可计算，因为冻结日志缺少动作前state。下表和resource_accuracy_evidence.png展示每阶段第100个训练episode的动作前state/action敏感性结果，不能称为冻结策略准确率或正式达标证据。", "", "| 范围 | 方法 | Acc@0.9 | 95%区间 | episode数 | 有效决策 | 排除决策 | 90%敏感性判定 |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in accuracy:
        lines.append(f'| {row["scene"]} | {row["method"]} | {row["acc_at_0_9"]*100:.2f}% | [{row["acc_at_0_9_ci95_low"]*100:.2f}%, {row["acc_at_0_9_ci95_high"]*100:.2f}%] | {row["episode_n"]} | {row["valid_decisions"]} | {row["excluded_decisions"]} | {"达到" if row["sensitivity_threshold_met"] else "未达到"} |')
    lines += ["", "四场景等权Macro敏感性Acc@0.9为R25 65.75%、R50 60.39%，均低于90%。这是一项明确的不利诊断信号；但正式结论仍是‘冻结准确率不可核验’，而不是用训练轨迹直接判定冻结评价失败。图中的95%区间仅表示现有训练episode/seed/顺序波动。"]
    lines += ["", "## 覆盖范围", "", "- R25：O1–O4 × seeds 0–4，20个逻辑运行、每场景20个最终冻结episode，覆盖完整。", "- R50：当前正式磁盘只有O2 × seeds 0–4，5个逻辑运行、每场景5个最终冻结episode；其宏平均只代表O2，不能解释为跨顺序稳健性。", "", "## 口径限制", "", "- FixedTime和MaxPressure使用原正式fixed-default episode；HA-SODQN使用第4阶段结束后的fixed-default四场景冻结矩阵。", "- FixedTime曲线使用另一个已存在的evaluation seed 10000逐步包，仅作视觉参考；图中AUC柱和阈值计算使用正式fixed-default基线。", "- queue AUC单位为车辆·秒。传统基线以正式episode平均queue乘3600秒；HA-SODQN以逐决策queue_network_sum乘10秒。", "- waiting_time和real_delay是历史累计型episode末指标；approximate_delay是episode平均归一化速度损失，均未冒充queue AUC。", "- 未重新训练、未重新运行仿真、未调整90%或10%阈值。", "", "## 审计", "", f'- 纳入HA-SODQN冻结episode：{audit["counts"]["ha_frozen_episodes"]}。', f'- 纳入传统基线episode：{audit["counts"]["traditional_episodes"]}。', f'- 训练轨迹准确率episode：{audit["counts"]["training_accuracy_episodes"]}。']
    (out / "resource_metric_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = arguments()
    if args.output_dir.exists() and not args.allow_existing_output:
        raise FileExistsError(f"Output exists; use --allow-existing-output to resume: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    mappings = {scene: phase_lane_mapping(scene) for scene in SCENES}
    mapping_rows = []
    for scene, item in mappings.items():
        for index, lane in enumerate(item["state_lanes"]):
            mapping_rows.append({"scenario": scene, "state_index": index, "lane_id": lane, "served_by_actions": ";".join(str(a) for a, lanes in enumerate(item["action_service_lanes"]) if lane in lanes), "net_xml": item["net_xml"], "net_xml_sha256": item["net_xml_sha256"]})
    save_csv(args.output_dir / "resource_phase_lane_mapping.csv", mapping_rows)
    traditional = traditional_episodes(args.baseline_runlist); ha, identities = ha_episodes(args.ha_root)
    episodes = traditional + ha; accuracy_rows = training_accuracy(args.ha_root, mappings)
    scenario_summary, macro = summarize_episodes(episodes, args.bootstrap_resamples, rng)
    accuracy_summary = summarize_accuracy(accuracy_rows, args.bootstrap_resamples, rng)
    accuracy_evidence_rows = accuracy_evidence(accuracy_rows, accuracy_summary, args.bootstrap_resamples, rng)
    save_csv(args.output_dir / "resource_metric_episode_level.csv", episodes)
    save_csv(args.output_dir / "resource_metric_scenario_summary.csv", scenario_summary)
    save_csv(args.output_dir / "resource_metric_comparison.csv", macro)
    save_csv(args.output_dir / "resource_metric_training_accuracy_sensitivity.csv", accuracy_rows)
    save_csv(args.output_dir / "resource_metric_accuracy_scenario_sensitivity.csv", accuracy_summary)
    save_csv(args.output_dir / "resource_accuracy_quantitative_evidence.csv", accuracy_evidence_rows)
    save_csv(args.output_dir / "intermediate_ha_frozen_identity_manifest.csv", identities)
    audit = {"schema_version": 1, "status": "completed_with_accuracy_and_r50_coverage_limitations", "scope": {"methods": list(METHODS), "main_baseline": "FixedTime", "auxiliary_baseline": "MaxPressure", "excluded": ["sequential_online", "independent_online_dqn"]}, "thresholds": {"macro_acc_at_0_9": .90, "macro_queue_burden_reduction_pct": 10.0}, "counts": {"ha_frozen_episodes": len(ha), "traditional_episodes": len(traditional), "training_accuracy_episodes": len(accuracy_rows)}, "coverage": {"P1C-DHOA-R25": {"orders": ["O1", "O2", "O3", "O4"], "training_seeds": [0, 1, 2, 3, 4], "logical_runs": 20, "frozen_episodes": 80, "complete_requested_matrix": True}, "P1C-DHOA-R50": {"orders": ["O2"], "training_seeds": [0, 1, 2, 3, 4], "logical_runs": 5, "frozen_episodes": 20, "complete_requested_matrix": False, "limitation": "No formal O1/O3/O4 directories exist on current disk"}}, "accuracy": {"frozen_status": "not_computable", "reason": "HA frozen decisions contain post-action lane counts but no pre-action raw_state", "sensitivity_source": "stage-specific episode_0100 training NPZ"}, "efficiency": {"queue_auc_unit": "vehicle_seconds", "ha_formula": "sum(queue_network_sum * action_interval_seconds)", "traditional_formula": "episode_mean_queue * duration_seconds", "traffic_seed_mode": "fixed_default"}, "bootstrap": {"resamples": args.bootstrap_resamples, "seed": args.bootstrap_seed, "unit": "episode/order/training-seed within scenario"}, "source_paths": {"ha_root": str(args.ha_root), "baseline_runlist": str(args.baseline_runlist), "baseline_decisions_visual_only": str(args.baseline_decisions)}, "phase_lane_mappings": mappings}
    (args.output_dir / "resource_metric_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_plots(args.output_dir, accuracy_evidence_rows, scenario_summary, episodes, args.baseline_decisions)
    write_report(args.output_dir, macro, accuracy_evidence_rows, audit)
    print(json.dumps({"output_dir": str(args.output_dir), "counts": audit["counts"], "macro": macro}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
