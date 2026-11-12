"""黄金 smoke：固定种子的小规模运行，改代码后指标必须逐条一致。

用例在 bench/golden/<name>.json：
    {"description": ..., "steps": [{"id": "st1", "args": [...run.py 参数...], "overlay": {...}}, ...],
     "expected": {"st1": [[record_type, episode, travel_time, ...], ...], ...}}
后续步骤的 overlay 可用 "{run:<step_id>}" 引用前一步的 run 目录（例如历史归档、stage checkpoint）。
产物经 TSC_OUTPUT_ROOT 写到临时目录，跑完即删，不进 data/output_data、不进台账。

    python -m lab golden                 运行全部用例并比对
    python -m lab golden NAME --update   重新生成期望值（仅在有意改变行为时使用，提交时写明原因）
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .provenance import REPO_ROOT

GOLDEN_DIR = REPO_ROOT / "bench" / "golden"
RECORD_FIELDS = ("record_type", "episode", "travel_time", "throughput", "unfinished_vehicles",
                 "queue", "delay", "reward_sum", "loss_mean", "gradient_updates",
                 "train_diagnostics")


def cases():
    return sorted(path.stem for path in GOLDEN_DIR.glob("*.json"))


def load(name):
    return json.loads((GOLDEN_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _arg(args, flag):
    return args[args.index(flag) + 1]


def _substitute(value, runs):
    if isinstance(value, str):
        for step_id, path in runs.items():
            value = value.replace(f"{{run:{step_id}}}", str(path))
        return value
    if isinstance(value, list):
        return [_substitute(item, runs) for item in value]
    if isinstance(value, dict):
        return {key: _substitute(item, runs) for key, item in value.items()}
    return value


def _records(run_dir):
    rows = []
    with open(run_dir / "metrics" / "records.jsonl", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            rows.append([record.get(field) for field in RECORD_FIELDS])
    return rows


def run_case(name, python=None, keep=False, repo=None):
    """运行一个用例，返回 {step_id: records}；失败抛 RuntimeError（附日志尾部）。

    repo 指定在哪份代码上运行（默认本仓库），可用于与另一工作树（如重构前锚点）对照。
    """
    spec = load(name)
    repo = Path(repo) if repo else REPO_ROOT
    workdir = Path(tempfile.mkdtemp(prefix=f"golden_{name}_"))
    env = {**os.environ, "TSC_OUTPUT_ROOT": str(workdir / "out")}
    runs, observed = {}, {}
    try:
        for step in spec["steps"]:
            prefix = f"golden_{name}_{step['id']}"
            overlay = workdir / f"{step['id']}.yml"
            # JSON 是 YAML 的子集，run.py 的 yaml.safe_load 可直接读取。
            overlay.write_text(json.dumps(_substitute(step["overlay"], runs)), encoding="utf-8")
            args = step["args"]
            command = [python or sys.executable, "run.py", *args, "--prefix", prefix,
                       "--experiment-config", str(overlay)]
            log = workdir / f"{step['id']}.log"
            with open(log, "w", encoding="utf-8") as handle:
                code = subprocess.call(command, cwd=repo, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT)
            if code != 0:
                tail = log.read_text(encoding="utf-8", errors="replace")[-3000:]
                raise RuntimeError(f"{name}/{step['id']} 退出码 {code}\n{tail}")
            relative = Path("tsc") / f"{_arg(args, '-w')}_{_arg(args, '-a')}" / _arg(args, "-n") / prefix
            run_dir = workdir / "out" / relative
            if not run_dir.exists():
                # 不认识 TSC_OUTPUT_ROOT 的旧代码会写到自身的 data/output_data，搬进临时目录后再用。
                legacy = repo / "data" / "output_data" / relative
                if legacy.exists():
                    run_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(legacy), str(run_dir))
            runs[step["id"]] = run_dir
            observed[step["id"]] = _records(run_dir)
        return observed
    finally:
        if keep:
            print(f"保留临时目录：{workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def compare(expected, observed):
    """返回差异描述列表；空列表表示逐条一致。"""
    problems = []
    for step_id, rows in expected.items():
        got = observed.get(step_id)
        if got is None:
            problems.append(f"{step_id}: 未产生记录")
            continue
        if len(got) != len(rows):
            problems.append(f"{step_id}: 记录数 {len(got)} ≠ 期望 {len(rows)}")
        for index, (want, have) in enumerate(zip(rows, got)):
            if want != have:
                diff = {field: (w, h) for field, w, h in zip(RECORD_FIELDS, want, have) if w != h}
                problems.append(f"{step_id} 第 {index} 条（{want[0]} ep={want[1]}）：{diff}")
                break
    return problems


def update(name, observed):
    spec = load(name)
    spec["expected"] = observed
    (GOLDEN_DIR / f"{name}.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
