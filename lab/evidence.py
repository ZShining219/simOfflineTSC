"""定位台账 run_dir 在本机的实际位置，并读取 run 目录里的溯源文件。

查找顺序：主树 → 本仓其他 git 工作树（自动发现，不写死路径）→ 台账里的绝对路径。
找不到即 missing：可能在另一台机器上，也可能已丢失——本机无法区分，需到对应机器复核。
"""
from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

from .provenance import REPO_ROOT

HERE = "here"
WORKTREE = "worktree"
ABSOLUTE = "absolute"
MISSING = "missing"
NO_RUN_DIR = "no_run_dir"
LOCATIONS = (HERE, WORKTREE, ABSOLUTE, MISSING, NO_RUN_DIR)

RUN_FILES = ("run_manifest.json", "run_status.json",
             "config/config_hashes.json", "evaluation/summary.json")


@lru_cache(maxsize=1)
def worktree_roots():
    try:
        out = subprocess.run(["git", "worktree", "list", "--porcelain"],
                             cwd=str(REPO_ROOT), capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    roots = [Path(line.split(" ", 1)[1]) for line in out.splitlines()
             if line.startswith("worktree ")]
    return tuple(root for root in roots if root.resolve() != REPO_ROOT.resolve())


def locate(run_dir):
    """返回 (位置类别, 本机绝对路径或 None)。"""
    if not run_dir:
        return NO_RUN_DIR, None
    path = Path(run_dir)
    if path.is_absolute():
        return (ABSOLUTE, path) if path.exists() else (MISSING, None)
    if (REPO_ROOT / path).exists():
        return HERE, REPO_ROOT / path
    for root in worktree_roots():
        if (root / path).exists():
            return WORKTREE, root / path
    return MISSING, None


def read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def run_files(directory):
    """读取 run 目录中的标准溯源文件，缺失的键为 None。"""
    if directory is None:
        return {name: None for name in RUN_FILES}
    return {name: read_json(os.path.join(directory, name)) for name in RUN_FILES}
