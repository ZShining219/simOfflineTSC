"""运行时代码身份：记录一次 run 实际使用的 commit、分支与未提交改动。

``utils/logger.py`` 的 ``BASELINE_COMMIT`` 是写死的常量，不能说明 run 用了哪版代码；
新 run 的 manifest 以本模块的 ``code_identity()`` 为准。任何 git 异常都只降级为
``None`` 字段，绝不让溯源失败中断训练。
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MACHINE_ENV = "TSC_MACHINE"
MAX_LISTED_FILES = 50


def _git(args, cwd):
    try:
        result = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def code_identity(root=None) -> dict:
    """返回 root 所在仓库的代码身份；不在 git 仓库内时各字段为 None。"""
    root = Path(root) if root else REPO_ROOT
    head = _git(["rev-parse", "HEAD"], root)
    if head is None:
        return {"commit": None, "branch": None, "dirty": None,
                "dirty_files": [], "dirty_file_count": None,
                "diff_sha256": None}
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    status = _git(["status", "--porcelain", "--untracked-files=normal"], root) or b""
    dirty_files = [line[3:] for line in status.decode("utf-8", "replace").splitlines()
                   if line.strip()]
    diff = _git(["diff", "HEAD", "--binary"], root) or b""
    return {
        "commit": head.decode().strip(),
        "branch": branch.decode().strip() if branch else None,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files[:MAX_LISTED_FILES],
        "dirty_file_count": len(dirty_files),
        # 已跟踪文件的改动指纹；同 commit + 同指纹 = 同一份代码（未跟踪文件只列名）。
        "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
    }


def machine_alias():
    """机器代号（如 34/73），由环境变量 TSC_MACHINE 提供；入库文件只用代号不写主机名。"""
    return os.environ.get(MACHINE_ENV) or None
