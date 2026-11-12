"""产物落点的唯一来源。新代码一律经此取路径，不再硬编码 data/output_data 等字面量。

产物分区（均被 .gitignore 排除）：
    <output_root>/<task>/<world>_<agent>/<net>/<prefix>/   run 产物（run.py 既有布局，台账依赖，不改）
    <output_root>/reports/<topic>/                         分析表、图、汇报材料（取代仓根 analysis/、final_result/）
    artifacts/<campaign>/                                  队列与运行时状态
仓根 analysis/、final_result/ 与 data/output_data 下的历史散目录冻结只读，不再写入。

环境变量 TSC_OUTPUT_ROOT 可把产物根指向同一处，避免多个工作树各自堆产物；
未设置时保持 <仓库>/data/output_data，与历史路径完全一致。
"""
from __future__ import annotations

import os
from pathlib import Path

from .provenance import REPO_ROOT

OUTPUT_ENV = "TSC_OUTPUT_ROOT"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "output_data"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"


def output_root() -> Path:
    override = os.environ.get(OUTPUT_ENV)
    return Path(override).expanduser() if override else DEFAULT_OUTPUT_ROOT


def report_dir(topic, create=True) -> Path:
    """分析/报告产物目录：<output_root>/reports/<topic>/。"""
    if not topic or "/" in topic or topic.startswith("."):
        raise ValueError(f"report topic 须为单级目录名：{topic!r}")
    path = output_root() / "reports" / topic
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def queue_dir(campaign, create=True) -> Path:
    """队列与运行时状态目录：artifacts/<campaign>/。"""
    path = ARTIFACTS_ROOT / campaign
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
