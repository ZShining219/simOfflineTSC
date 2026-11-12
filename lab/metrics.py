"""台账指标的唯一口径：最佳 checkpoint 评估（用户 2026-11-12 裁定）。

最佳 checkpoint 沿用 trainer 自己写进 evaluation/summary.json 的选择：
travel_time 最小、并列取最早 episode（在线）/ update（纯离线）。
注意：该值来自训练期逐回合评估，选择与报告用的是同一组评估，数值偏乐观；
需要无偏比较时应对最佳 checkpoint 另做冻结复评（另行登记为 eval 行）。

metric_source 取值：
    best_ckpt        按上述口径从 run 目录复算，台账 metrics 即标准值
    eval_attempt     eval 行：对某个 checkpoint 的一次冻结评估，不做"选最佳"
    legacy:<公式>    无法按标准复算，保留旧值并标明旧公式（eval_last20 / mixed_last20 /
                     final_eval / unknown）；mixed_last20 = 末 20 条记录混入了训练回合
"""
from __future__ import annotations

import json
import os
import statistics

METRIC_KEYS = ("travel_time", "throughput", "unfinished_vehicles", "queue", "delay")
STANDARD = "best_ckpt"
EVAL_ATTEMPT = "eval_attempt"
LEGACY_PREFIX = "legacy:"
TOLERANCE = 1e-6
# 识别旧公式时的容差：部分回填脚本把指标四舍五入到 3 位小数。
LEGACY_TOLERANCE = 6e-4


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def best_checkpoint(run_dir):
    """按 trainer 的最佳 checkpoint 选择返回 (metrics, detail)；没有可用 summary 时返回 None。"""
    if not run_dir:
        return None
    summary = _read_json(os.path.join(run_dir, "evaluation", "summary.json"))
    if not summary or not summary.get("evaluations"):
        return None
    step_key = "training_update" if "best_update" in summary else "episode"
    best_step = summary.get("best_update", summary.get("best_episode"))
    entries = [e for e in summary["evaluations"] if e.get(step_key) == best_step]
    if len(entries) != 1:
        return None
    entry = entries[0]
    if abs(entry.get("travel_time", float("nan")) - summary["best_travel_time"]) > TOLERANCE:
        return None
    metrics = {key: entry.get(key) for key in METRIC_KEYS}
    detail = {"step_key": step_key, "step": best_step,
              "checkpoint": summary.get("best_checkpoint"),
              "selection_rule": summary.get("selection_rule"),
              "evaluations": len(summary["evaluations"])}
    return metrics, detail


def _records(run_dir):
    path = os.path.join(run_dir, "metrics", "records.jsonl")
    try:
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except (OSError, ValueError):
        return None


def _close(value, expected, tolerance=TOLERANCE):
    return value is not None and expected is not None and abs(value - expected) <= tolerance


def legacy_formula(travel_time, run_dir):
    """识别旧台账 travel_time 用的是哪种公式；识别不了返回 'unknown'。"""
    if travel_time is None or not run_dir:
        return "unknown"

    def close(value):
        return _close(value, travel_time, LEGACY_TOLERANCE)

    best = best_checkpoint(run_dir)
    if best and close(best[0]["travel_time"]):
        return STANDARD
    summary = _read_json(os.path.join(run_dir, "evaluation", "summary.json")) or {}
    records = _records(run_dir) or []
    evaluations = [r["travel_time"] for r in records
                   if r.get("record_type") == "EVALUATION" and r.get("travel_time") is not None]
    finals = [r["travel_time"] for r in records
              if r.get("record_type") == "FINAL_EVALUATION" and r.get("travel_time") is not None]
    if close(summary.get("final_travel_time")) or (finals and close(finals[-1])):
        return "final_eval"
    if evaluations and close(statistics.mean(evaluations[-20:])):
        return "eval_last20"
    mixed = [r for r in records
             if r.get("travel_time") is not None and r.get("throughput") is not None][-20:]
    if mixed and close(statistics.mean(r["travel_time"] for r in mixed)):
        has_train = any(r.get("record_type") == "TRAIN" for r in mixed)
        return "mixed_last20" if has_train else "eval_last20"
    return "unknown"


def standardize(row, run_dir):
    """返回按标准口径更新后的台账字段（只返回需要写入的键），不修改传入的 row。"""
    old = row.get("metrics") or {}
    has_old = any(old.get(key) is not None for key in METRIC_KEYS)
    if row.get("run_kind") == "eval":
        return {"metric_source": EVAL_ATTEMPT} if has_old else {}
    best = best_checkpoint(run_dir) if row.get("status") == "DONE" else None
    if best:
        metrics, detail = best
        update = {"metrics": metrics, "metric_source": STANDARD, "metric_detail": detail}
        if has_old and not _close(old.get("travel_time"), metrics["travel_time"], LEGACY_TOLERANCE):
            update["metrics_legacy"] = old
            update["metric_source_legacy"] = LEGACY_PREFIX + legacy_formula(
                old.get("travel_time"), run_dir)
        return update
    if has_old:
        return {"metric_source": LEGACY_PREFIX + legacy_formula(old.get("travel_time"), run_dir)}
    return {}


def verify(row, run_dir):
    """标准口径行：按 run 目录重算，与台账一致返回 True（无法重算时也返回 False）。"""
    best = best_checkpoint(run_dir)
    if not best:
        return False
    recorded = row.get("metrics") or {}
    return all(_close(recorded.get(key), value) for key, value in best[0].items()
               if value is not None)
