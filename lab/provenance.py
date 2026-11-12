"""运行时代码身份：记录一次 run 实际使用的 commit、分支与未提交改动。

``utils/logger.py`` 的 ``BASELINE_COMMIT`` 是写死的常量，不能说明 run 用了哪版代码；
新 run 的 manifest 以本模块的 ``code_identity()`` 为准。任何 git 异常都只降级为
``None`` 字段，绝不让溯源失败中断训练。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
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


SIM_FILE_KEYS = ("roadnetFile", "flowFile", "combined_file")
SUMOCFG_INPUT_TAGS = ("net-file", "route-files", "additional-files")


def _sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _sumocfg_inputs(path):
    """解析 .sumocfg 中引用的 net/route/additional 文件（相对 sumocfg 所在目录）。"""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return []
    found = []
    for tag in SUMOCFG_INPUT_TAGS:
        for element in root.iter(tag):
            for name in (element.get("value") or "").split(","):
                if name.strip():
                    found.append(os.path.normpath(os.path.join(os.path.dirname(path), name.strip())))
    return found


def data_identity(sim_cfg_path, root=None) -> dict:
    """返回仿真输入数据的文件指纹；同一路网名在不同时期内容可能不同，只认指纹。

    sim_cfg_path 为 run 归档的 simulator_resolved.cfg（JSON）；路径相对仓库根解析。
    """
    root = Path(root) if root else REPO_ROOT
    try:
        with open(sim_cfg_path, encoding="utf-8") as handle:
            cfg = json.load(handle)
    except (OSError, ValueError):
        return {"files": {}, "fingerprint": None}
    base = root / cfg.get("dir", "")
    paths = [str(base / cfg[key]) for key in SIM_FILE_KEYS if cfg.get(key)]
    for sumocfg in [path for path in paths if path.endswith(".sumocfg")]:
        paths.extend(_sumocfg_inputs(sumocfg))
    files = {}
    for path in dict.fromkeys(os.path.normpath(p) for p in paths):
        rel = os.path.relpath(path, root)
        files[rel] = _sha256_file(path) if os.path.isfile(path) else None
    fingerprint = hashlib.sha256("\n".join(
        f"{rel}:{sha}" for rel, sha in sorted(files.items())).encode()).hexdigest()
    return {"files": files, "fingerprint": fingerprint}
