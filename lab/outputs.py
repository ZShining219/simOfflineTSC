"""产物地图：各产物区每个目录是什么、被台账引用多少、占多大、最近何时写入。

回答"这个目录对应哪个实验、能不能动"。只读扫描，不移动不删除任何文件。
"""
from __future__ import annotations

import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import ledger
from .provenance import REPO_ROOT

ROOTS = ("data/output_data", "artifacts", "analysis", "final_result")


def _sizes(root):
    """du 一次取出 root 下各一级目录字节数（大目录较慢，结果仅供盘点）。"""
    try:
        out = subprocess.run(["du", "-b", "--max-depth=1", str(root)], capture_output=True,
                             text=True, timeout=1800).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    sizes = {}
    for line in out.splitlines():
        size, path = line.split("\t", 1)
        sizes[Path(path).name] = int(size)
    return sizes


def _latest_mtime(path, limit=2000):
    """一级与二级条目中最新的修改时间（不深扫，避免在百 G 目录上遍历）。"""
    latest = path.stat().st_mtime
    seen = 0
    for entry in os.scandir(path):
        latest = max(latest, entry.stat(follow_symlinks=False).st_mtime)
        seen += 1
        if entry.is_dir(follow_symlinks=False) and seen < limit:
            for child in os.scandir(entry.path):
                latest = max(latest, child.stat(follow_symlinks=False).st_mtime)
                seen += 1
                if seen >= limit:
                    break
    return datetime.fromtimestamp(latest, tz=timezone.utc).strftime("%Y-%m-%d")


def _references(rows, prefix):
    """台账中 run_dir/queue_file/behavior_source 落在 prefix 下的行，按 campaign 计数。"""
    campaigns = Counter()
    for row in rows:
        for key in ("run_dir", "queue_file", "behavior_source"):
            value = row.get(key) or ""
            if value == prefix or value.startswith(prefix + "/"):
                campaigns[row.get("campaign")] += 1
                break
    return campaigns


def _writers(name):
    """在已跟踪代码中搜索目录名，返回引用它的脚本（用于推断未登记产物的来源）。"""
    try:
        out = subprocess.run(["git", "grep", "-l", "-F", name, "--", "*.py", "*.sh"],
                             cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line for line in out.splitlines() if not line.startswith("tests/")][:3]


def scan(roots=ROOTS):
    rows = list(ledger.load().values())
    result = []
    for root in roots:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        sizes = _sizes(base)
        for entry in sorted(base.iterdir()):
            if not entry.is_dir() or entry.is_symlink():
                continue
            relative = f"{root}/{entry.name}"
            refs = _references(rows, relative)
            result.append({
                "path": relative,
                "bytes": sizes.get(entry.name),
                "latest": _latest_mtime(entry),
                "ledger_rows": sum(refs.values()),
                "campaigns": dict(refs.most_common(4)),
                "writers": [] if refs else _writers(entry.name),
            })
    return result
