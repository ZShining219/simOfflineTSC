"""台账统一登记与回填：取代按 campaign 各写一份的 register_/backfill_ 脚本。

    python -m lab register artifacts/<campaign>/run_queue.json [--line L --kind train ...] [--apply]
    python -m lab harvest --campaign <campaign> [--apply]

register：按队列任务追加 REGISTERED 行（幂等，已有 run_id 跳过）。code_commit / config_sha 此时留空——
登记时的提交号不等于运行时代码，运行时版本由 run_manifest 如实记录，harvest 时回填。
harvest：读 run 目录的 run_status.json / run_manifest.json，回填状态、起止时间、运行时代码版本（verified）、
配置哈希、数据指纹、机器代号与标准口径指标（lab.metrics）。默认只预览，--apply 才写台账。
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from . import evidence, ledger, metrics
from .paths import output_root
from .provenance import REPO_ROOT

# 台账字段的规范顺序（GOVERNANCE §4.1）；新行按此顺序写出，缺省为 None。
FIELDS = ("run_id", "line", "campaign", "arm", "alias_of", "net", "seed", "episodes_total",
          "episodes_increment", "resume_from", "ngpu", "status", "plan_id", "code_commit", "dirty",
          "queue_file", "run_dir", "tmux", "started", "eval_mode", "behavior_source", "metrics",
          "abort_reason", "error", "evidence_lost", "notes", "run_kind", "config_sha",
          "sumo_seed_mode", "sumo_seed", "eval_seed", "condition", "superseded_by", "runtime_env",
          "end_time", "machine", "data_fingerprint")
STATUS_MAP = {"已完成": "DONE", "失败": "FAILED", "运行中": "RUNNING"}


def _relative(path):
    if not path:
        return None
    path = str(path)
    root = str(REPO_ROOT) + os.sep
    return path[len(root):] if path.startswith(root) else path


def _flag(command, *names):
    for name in names:
        if name in command:
            index = command.index(name)
            if index + 1 < len(command):
                return command[index + 1]
    return None


def _run_dir_from_command(command):
    world, agent = _flag(command, "-w", "--world"), _flag(command, "-a", "--agent")
    net, prefix = _flag(command, "-n", "--network"), _flag(command, "--prefix")
    task = _flag(command, "-t", "--task") or "tsc"
    if world and agent and net and prefix:
        return _relative(output_root() / task / f"{world}_{agent}" / net / prefix)
    return None


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def rows_from_queue(queue_path, campaign=None, line=None, run_kind="train",
                    eval_mode="train_frozen_eval", notes=""):
    """把队列任务转成 REGISTERED 台账行（不写盘）。"""
    queue_path = Path(queue_path)
    spec = json.loads(queue_path.read_text(encoding="utf-8"))
    tasks = spec["tasks"] if isinstance(spec, dict) else spec
    campaign = campaign or (spec.get("campaign") if isinstance(spec, dict) else None)
    if not campaign:
        raise ValueError("队列未声明 campaign，请用 --campaign 指定")
    today = date.today().isoformat()
    rows = []
    for task in tasks:
        command = [str(part) for part in task.get("command", [])]
        row = dict.fromkeys(FIELDS)
        row.update({
            "run_id": task["run_id"], "line": line or campaign, "campaign": campaign,
            "arm": task.get("arm") or task.get("method") or _flag(command, "-a", "--agent"),
            "net": task.get("net") or task.get("scene") or _flag(command, "-n", "--network"),
            "seed": _int(task.get("seed", task.get("training_seed", _flag(command, "--seed")))),
            "status": "REGISTERED", "plan_id": campaign, "evidence_lost": False,
            "queue_file": _relative(queue_path.resolve()),
            "run_dir": _relative(task.get("output_path")) or _run_dir_from_command(command),
            "eval_mode": eval_mode, "run_kind": run_kind,
            "sumo_seed": _int(_flag(command, "--sumo_seed")),
            "condition": task.get("condition"),
            "runtime_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "notes": f"REGISTERED {today} via lab register" + (f"：{notes}" if notes else ""),
        })
        rows.append(row)
    return rows


def register(queue_path, apply=False, ledger_path=None, **options):
    """返回 (新增行, 跳过的 run_id)；apply=True 时追加写入台账。"""
    existing = ledger.load(ledger_path)
    candidates = rows_from_queue(queue_path, **options)
    new = [row for row in candidates if row["run_id"] not in existing]
    skipped = [row["run_id"] for row in candidates if row["run_id"] in existing]
    if apply and new:
        rows = ledger.read_lines(ledger_path) + new
        ledger.write_lines(rows, ledger_path)
    return new, skipped


def harvest_update(row, run_dir):
    """按 run 目录推导台账应更新的字段（只返回有变化的键）。"""
    if run_dir is None:
        return {}
    files = evidence.run_files(run_dir)
    status, manifest = files["run_status.json"] or {}, files["run_manifest.json"] or {}
    update = {}
    mapped = STATUS_MAP.get(status.get("status"))
    if mapped and row.get("status") in ("REGISTERED", "RUNNING", mapped):
        update["status"] = mapped
        if mapped == "FAILED":
            update["error"] = status.get("error_message") or status.get("error_type")
    update["started"] = status.get("started_at_utc") or row.get("started")
    update["end_time"] = status.get("finished_at_utc") or row.get("end_time")
    code = manifest.get("code") or {}
    if code.get("commit"):
        update["code_commit"], update["dirty"] = code["commit"], code.get("dirty")
    if manifest.get("machine"):
        update["machine"] = manifest["machine"]
    if (manifest.get("data") or {}).get("fingerprint"):
        update["data_fingerprint"] = manifest["data"]["fingerprint"]
    if manifest.get("config_hash") and row.get("config_sha") != manifest["config_hash"]:
        if row.get("config_sha"):
            update["config_sha_registered"] = row["config_sha"]
        update["config_sha"] = manifest["config_hash"]
    merged = {**row, **update}
    update.update(metrics.standardize(merged, str(run_dir)))
    return {key: value for key, value in update.items() if row.get(key) != value}


def harvest(campaign, apply=False, ledger_path=None):
    """返回 {run_id: update}；apply=True 时写回台账。"""
    rows = ledger.read_lines(ledger_path)
    changes = {}
    for row in rows:
        if row.get("campaign") != campaign:
            continue
        _, path = evidence.locate(row.get("run_dir"))
        update = harvest_update(row, path)
        if update:
            changes[row["run_id"]] = update
            row.update(update)
    if apply and changes:
        ledger.write_lines(rows, ledger_path)
    return changes
