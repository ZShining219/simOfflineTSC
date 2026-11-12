"""台账（ledger/runs.jsonl）的唯一读取口径。

各登记脚本历史上各写一套字段，这里集中做两件事：
1. 规范化同义值（如 run_kind 的 collect/collection），不改写台账原文；
2. 给出每行"代码身份"的可信等级，避免把常量或登记时提交号当成运行时版本。
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from .provenance import REPO_ROOT

LEDGER_PATH = REPO_ROOT / "ledger" / "runs.jsonl"
SIM_CONFIG_DIR = REPO_ROOT / "configs" / "sim"

# utils/logger.py 中写死的常量；manifest/台账里出现它不代表真实运行版本。
CONSTANT_BASELINE_COMMIT = "73d860bb3924ec15c30433a8f8b7af17787baeff"

RUN_KIND_ALIASES = {"collect": "collection"}
STATUSES = ("DONE", "FAILED", "ABORTED", "SUPERSEDED", "REGISTERED",
            "SCOPE_DISCARD", "LEGACY_UNCLEAR", "RUNNING")
METRIC_KEYS = ("travel_time", "throughput", "unfinished_vehicles", "delay", "queue")

# 代码身份可信等级（由强到弱）
CODE_VERIFIED = "verified"        # manifest 记录了运行时 HEAD（新 schema）
CODE_REGISTERED = "registered"    # 仅台账登记时的提交号，未在运行时核对
CODE_CONSTANT = "constant"        # 只有写死的 BASELINE_COMMIT，等于没有
CODE_NONE = "none"
CODE_LEVELS = (CODE_VERIFIED, CODE_REGISTERED, CODE_CONSTANT, CODE_NONE)


def load(path=None, duplicates=None):
    """按原顺序读取台账，返回 run_id → 行（后出现的同 run_id 覆盖先出现的）。

    传入 list 作为 duplicates 时，被覆盖的重复 run_id 行号会以 (run_id, 先前行号, 当前行号) 追加进去。
    """
    rows = OrderedDict()
    with open(path or LEDGER_PATH, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = number
            kind = row.get("run_kind")
            row["run_kind"] = RUN_KIND_ALIASES.get(kind, kind)
            if duplicates is not None and row["run_id"] in rows:
                duplicates.append((row["run_id"], rows[row["run_id"]]["_line_no"], number))
            rows[row["run_id"]] = row
    return rows


def has_metrics(row):
    metrics = row.get("metrics") or {}
    return any(metrics.get(key) is not None for key in METRIC_KEYS)


def config_mismatch(row, manifest):
    """台账 config_sha 与 run_manifest.config_hash 都存在且不同时返回 True。"""
    registered = row.get("config_sha")
    actual = (manifest or {}).get("config_hash")
    return bool(registered and actual and registered != actual)


def code_level(row, manifest=None):
    """综合台账行与 run_manifest 判定代码身份可信等级，返回 (等级, 提交号或 None)。"""
    code = (manifest or {}).get("code") or {}
    if code.get("commit"):
        return CODE_VERIFIED, code["commit"]
    commit = row.get("code_commit") or ""
    if commit and not CONSTANT_BASELINE_COMMIT.startswith(commit[:7]):
        return CODE_REGISTERED, commit
    if commit or (manifest or {}).get("baseline_commit"):
        return CODE_CONSTANT, None
    return CODE_NONE, None


def sim_config(net):
    """台账 net → configs/sim/<net>.cfg；持续学习的阶段序列（含 ':'）等无单一配置时返回 None。"""
    path = SIM_CONFIG_DIR / f"{net}.cfg"
    return path if net and path.exists() else None
