"""台账结构不变量：只读入库的 ledger/runs.jsonl，不依赖本机产物，任何环境可跑。"""
import os
from collections import Counter

from lab import ledger, metrics

ROWS = ledger.read_lines()


def _has_metrics(row):
    return ledger.has_metrics(row)


def test_every_metric_has_a_declared_source():
    allowed = {metrics.STANDARD, metrics.EVAL_ATTEMPT}
    bad = [r["run_id"] for r in ROWS if _has_metrics(r) and not (
        r.get("metric_source") in allowed
        or str(r.get("metric_source", "")).startswith(metrics.LEGACY_PREFIX))]
    assert not bad, f"有指标但未声明口径（请用 python -m lab metrics 回填）：{bad[:10]}"


def test_standard_rows_carry_selection_detail():
    bad = [r["run_id"] for r in ROWS if r.get("metric_source") == metrics.STANDARD
           and not (r.get("metric_detail") or {}).get("checkpoint")]
    assert not bad, f"标准口径行缺少最佳 checkpoint 信息：{bad[:10]}"


def test_replaced_values_are_preserved():
    bad = [r["run_id"] for r in ROWS
           if r.get("metrics_legacy") and not r.get("metric_source_legacy")]
    assert not bad, f"保留了旧值但未注明旧公式：{bad[:10]}"


def test_run_dirs_are_repo_relative():
    bad = [r["run_id"] for r in ROWS if os.path.isabs(r.get("run_dir") or "")]
    assert not bad, f"run_dir 须为仓内相对路径（GOVERNANCE §10）：{bad[:10]}"


def test_run_ids_are_unique():
    # 2026-11-12 清理了 20 处逐字段相同的重复追加；此后一律不允许重复。
    duplicates = [run_id for run_id, n in Counter(r["run_id"] for r in ROWS).items() if n > 1]
    assert not duplicates, f"run_id 重复登记（追加前先查重，或用 lab 统一登记）：{duplicates[:10]}"
