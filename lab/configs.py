"""configs/tsc 索引：区分"智能体默认配置"与"实验 overlay"，并列出引用位置。

run.py 按 configs/<task>/<智能体名>.yml 自动加载智能体默认配置（utils/logger.py），这些文件必须留在根目录；
其余根目录 yml 是经 --experiment-config 显式传入的实验 overlay，为历史存量（只许减少）。
新 overlay 一律放 configs/tsc/<campaign>/。`python -m lab configs --write` 重新生成 configs/tsc/README.md。
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .provenance import REPO_ROOT

TSC_DIR = REPO_ROOT / "configs" / "tsc"
README = TSC_DIR / "README.md"
# 引用统计排除历史记录（台账、实验记录、归档），只看仍在用的代码与文档。
HISTORY_EXCLUDES = (":!ledger", ":!EXPERIMENTS", ":!EXPERIMENTS.md", ":!docs/done",
                    ":!docs/experiments", ":!dev/_legacy", ":!configs/tsc/README.md")


def agent_names():
    """仓库中所有注册的智能体名（含 tarl_{method} 这类循环注册）。"""
    out = subprocess.run(["git", "grep", "-hoE", r"register_model\(f?['\"][a-z0-9_{}]+['\"]", "--", "*.py"],
                         cwd=REPO_ROOT, capture_output=True, text=True).stdout
    names = {re.sub(r"register_model\(f?['\"]|['\"]$", "", line) for line in out.split()}
    tarl = REPO_ROOT / "agent" / "tarl.py"
    if "tarl_{_method}" in names and tarl.exists():
        methods = re.search(r"for _method in \(([^)]*)\)", tarl.read_text(encoding="utf-8"), re.S)
        names |= {f"tarl_{m}" for m in re.findall(r"'([a-z_]+)'", methods.group(1))} if methods else set()
    return {name for name in names if "{" not in name} | {"base"}


def _references(relatives):
    """一次 git grep 取出所有文件被哪些代码/文档引用：{relative: [file, ...]}。"""
    args = ["git", "grep", "-o", "-F"]
    for relative in relatives:
        args += ["-e", relative]
    out = subprocess.run([*args, "--", *HISTORY_EXCLUDES], cwd=REPO_ROOT,
                         capture_output=True, text=True).stdout
    found = {relative: [] for relative in relatives}
    for line in out.splitlines():
        source, _, match = line.partition(":")
        if match in found and source not in found[match]:
            found[match].append(source)
    return found


def index():
    agents = agent_names()
    paths = sorted(TSC_DIR.glob("*.yml"))
    references = _references([f"configs/tsc/{path.name}" for path in paths])
    rows = []
    for path in paths:
        kind = "智能体默认配置" if path.stem in agents else "实验 overlay（存量）"
        rows.append((path.name, kind, sorted(references[f"configs/tsc/{path.name}"])))
    subdirs = sorted((d.name, len(list(d.rglob("*.yml")))) for d in TSC_DIR.iterdir() if d.is_dir())
    return rows, subdirs


def render():
    rows, subdirs = index()
    lines = ["# configs/tsc 索引", "",
             "> 由 `python -m lab configs --write` 生成，勿手改；`tests/test_architecture.py` 校验其与实际文件一致。", "",
             "- **智能体默认配置**：run.py 按 `configs/tsc/<智能体名>.yml` 自动加载，必须留在本目录根部。",
             "- **实验 overlay**：经 `--experiment-config` 显式传入。根目录的是历史存量（只许减少）；"
             "**新 overlay 一律放 `configs/tsc/<campaign>/`**。",
             "- run 实际使用的配置以 run 目录 `config/` 下的快照为准（`python -m lab show <run_id>`）。", "",
             "## 子目录（按 campaign）", "", "| 目录 | yml 数 |", "|---|---|"]
    lines += [f"| `{name}/` | {count} |" for name, count in subdirs]
    lines += ["", "## 根目录文件", "", "| 文件 | 类型 | 仍在引用它的代码/文档（至多 3 处） |", "|---|---|---|"]
    for name, kind, refs in rows:
        shown = "、".join(f"`{r}`" for r in refs[:3]) + (f" 等 {len(refs)} 处" if len(refs) > 3 else "")
        lines.append(f"| `{name}` | {kind} | {shown or '无'} |")
    return "\n".join(lines) + "\n"
