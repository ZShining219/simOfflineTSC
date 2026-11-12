"""提交闸门：把 GOVERNANCE §8/§10 与 AGENTS.md 中的入库纪律落成可执行检查。

由 .githooks/pre-commit 与 .githooks/commit-msg 调用（启用：git config core.hooksPath .githooks）；
也可手动运行 ``python -m lab check --staged`` / ``--all``。
只检查本次提交新增或修改的内容，存量文件不追溯；确需例外时在此登记并写明理由。
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

# 仓根白名单：新增仓根条目必须先在这里登记（防止散件堆在仓根）。
ROOT_ALLOWLIST = {
    ".github", ".githooks", ".gitignore",
    "AGENTS.md", "EXPERIMENTS.md", "GOVERNANCE.md", "README.md", "THIRD_PARTY_NOTICES.md",
    "EXPERIMENTS", "agent", "arterial", "bench", "common", "configs", "data", "dev",
    "docs", "generator", "hitl", "lab", "ledger", "plan5", "scene_attention",
    "sequential", "task", "tests", "tools", "trainer", "utils", "world",
    "run.py", "environment.py", "arterial_run.py", "offline_run.py", "sequential_run.py",
    "requirements.txt", "requirements-colight.txt", "requirements-dashboard.txt",
    "requirements-offline.txt",
    "artifacts",  # 仅 artifacts/wiki_claims/README.md 入库，其余由 FORBIDDEN_PREFIXES 拦截
}

# 产物区与产物类型：任何情况下都不入库（.gitignore 可被 git add -f 绕过，这里兜底）。
FORBIDDEN_PREFIXES = ("data/output_data/", "artifacts/", "analysis/", "final_result/",
                      "outputs/", "runs/")
FORBIDDEN_EXCEPTIONS = {"artifacts/wiki_claims/README.md", "final_result/.gitkeep"}
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".ckpt", ".npz", ".npy", ".pkl", ".pickle", ".h5",
                      ".hdf5", ".onnx", ".log", ".sqlite", ".mdb")

MAX_BYTES = 2 * 1024 * 1024
# 输入路网/车流允许更大，但不超过 GitHub 单文件硬上限；台账是事实源，不设限。
SIZE_RULES = (("data/raw_data/", 95 * 1024 * 1024), ("ledger/runs.jsonl", None))

# 公网仓库：新增内容不得出现内网地址与本机绝对路径（GOVERNANCE §10）。
SENSITIVE_PATTERNS = (
    ("内网 IP", re.compile(r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}"
                         r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b")),
    ("本机绝对路径", re.compile(r"(?<![\w.])/(?:home/[\w.-]+|projects)/")),
)
SENSITIVE_EXEMPT = {"lab/checks.py", "tests/test_lab_checks.py"}

CJK = re.compile(r"[一-鿿]")


def _git(repo, *args):
    return subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=str(repo),
                          capture_output=True, check=True).stdout


def _staged(repo):
    out = _git(repo, "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")
    return [p for p in out.decode("utf-8").split("\0") if p]


def _size_limit(path):
    for prefix, limit in SIZE_RULES:
        if path == prefix or path.startswith(prefix):
            return limit
    return MAX_BYTES


def check_symlinks(repo):
    out = _git(repo, "ls-files", "-s", "-z").decode("utf-8")
    return [f"符号链接条目（§8.5 零容忍）：{entry.split(chr(9), 1)[1]}"
            for entry in out.split("\0") if entry.startswith("120000 ")]


def check_paths(paths):
    problems = []
    for path in paths:
        root = path.split("/", 1)[0]
        if root not in ROOT_ALLOWLIST:
            problems.append(f"仓根新增条目未登记：{root}（放进对应目录，或在 lab/checks.py 白名单登记理由）")
        if path in FORBIDDEN_EXCEPTIONS:
            continue
        if path.startswith(FORBIDDEN_PREFIXES):
            problems.append(f"产物区文件不得入库：{path}")
        elif path.lower().endswith(FORBIDDEN_SUFFIXES):
            problems.append(f"产物类型文件不得入库：{path}")
    return problems


def check_sizes(repo, paths):
    problems = []
    for path in paths:
        limit = _size_limit(path)
        if limit is None:
            continue
        size = int(_git(repo, "cat-file", "-s", f":{path}").strip())
        if size > limit:
            problems.append(f"文件过大（{size / 1048576:.1f} MiB > {limit / 1048576:.0f} MiB）：{path}")
    return problems


def check_sensitive(repo, paths):
    problems = []
    for path in paths:
        if path in SENSITIVE_EXEMPT:
            continue
        diff = _git(repo, "diff", "--cached", "-U0", "--no-color", "--", path)
        for line in diff.decode("utf-8", "replace").splitlines():
            if not line.startswith("+") or line.startswith("+++"):
                continue
            for label, pattern in SENSITIVE_PATTERNS:
                match = pattern.search(line)
                if match:
                    problems.append(f"{label}「{match.group(0)}」：{path}")
    return problems


def check_staged(repo):
    paths = _staged(repo)
    return (check_symlinks(repo) + check_paths(paths)
            + check_sizes(repo, paths) + check_sensitive(repo, paths))


def check_all(repo):
    """全量结构检查（CI 用）：符号链接与产物区；仓根与大小只对新增内容生效，不在此追溯。"""
    tracked = [p for p in _git(repo, "ls-files", "-z").decode("utf-8").split("\0") if p]
    forbidden = [p for p in check_paths(tracked) if not p.startswith("仓根")]
    return check_symlinks(repo) + forbidden


def check_commit_message(text):
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        return ["提交信息为空"]
    if not CJK.search(lines[0]):
        return [f"提交信息首行须用中文说明主要修改（AGENTS.md）：{lines[0][:60]}"]
    return []


def report(problems, header):
    if problems:
        print(f"✗ {header}未通过（{len(problems)} 项）：")
        for problem in problems:
            print("  -", problem)
        return 1
    print(f"✓ {header}通过")
    return 0
