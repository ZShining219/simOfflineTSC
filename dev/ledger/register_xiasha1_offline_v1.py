#!/usr/bin/env python3
"""Register campaign ``xiasha1_offline_v1`` in ledger/runs.jsonl.

Reads ``artifacts/xiasha1_offline_v1/run_queue.json`` and appends one
REGISTERED train row per task (126 total).  Idempotent: skips run_ids
already present.  config_sha is a deterministic pre-registration hash over
offline base.yml + agent yml + xiasha1_v1 overlay + dataset manifest +
eval-network sim cfg; on DONE it should be replaced by
run_manifest.config_hash.

Usage:
    python dev/ledger/register_xiasha1_offline_v1.py --dry-run
    python dev/ledger/register_xiasha1_offline_v1.py --code-commit <sha>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "ledger" / "runs.jsonl"
QUEUE = ROOT / "artifacts" / "xiasha1_offline_v1" / "run_queue.json"
CAMPAIGN = "xiasha1_offline_v1"
EVAL_NET = "xiasha1_asym_ns"


def _config_sha(task):
    digest = hashlib.sha256()
    relatives = [
        "configs/offline_tsc/base.yml",
        f"configs/offline_tsc/{task['agent']}.yml",
        "configs/offline_tsc/xiasha1_v1.yml",
        task["command"][task["command"].index("--dataset-manifest") + 1],
        "configs/sim/xiasha1_asym_ns.cfg",
    ]
    for relative in relatives:
        path = ROOT / relative if not relative.startswith("/") else Path(relative)
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _row(task, code_commit):
    spec = task["dataset_spec"]
    return {
        "run_id": task["run_id"],
        "line": "pure_offline",
        "campaign": CAMPAIGN,
        "arm": task["agent"],
        "alias_of": None,
        "net": EVAL_NET,
        "seed": int(task["seed"]),
        "episodes_total": 144000,
        "episodes_increment": 144000,
        "resume_from": None,
        "ngpu": -1,
        "status": "REGISTERED",
        "plan_id": CAMPAIGN,
        "code_commit": code_commit,
        "dirty": False,
        "queue_file": "artifacts/xiasha1_offline_v1/run_queue.json",
        "run_dir": task["output_path"].replace(str(ROOT) + "/", ""),
        "tmux": None,
        "started": None,
        "eval_mode": "offline_periodic_eval",
        "behavior_source": f"v1_hetero/{spec}",
        "metrics": None,
        "abort_reason": None,
        "error": None,
        "evidence_lost": False,
        "notes": (
            f"REGISTERED 2026-10-22: 异构历史经验离线矩阵 {spec} "
            f"(experiment={task['experiment']}, pool_replica="
            f"{task['pool_replica']}); 固定池均匀采样, 全窗口无质量筛选; "
            f"144k updates + 81 评估节点(1800 间隔); native backend; "
            f"DONE 时 config_sha 替换为 run_manifest.config_hash"
        ),
        "run_kind": "train",
        "config_sha": _config_sha(task),
        "sumo_seed_mode": "fixed_default",
        "sumo_seed": None,
        "eval_seed": None,
        "condition": spec,
        "superseded_by": None,
        "runtime_env": "colight",
        "end_time": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--code-commit", default="PENDING")
    args = parser.parse_args()

    spec = json.loads(QUEUE.read_text(encoding="utf-8"))
    existing = set()
    if LEDGER.is_file():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing.add(row.get("run_id"))
    fresh = [_row(task, args.code_commit) for task in spec["tasks"]
             if task["run_id"] not in existing]
    print(f"campaign={CAMPAIGN} tasks={len(spec['tasks'])} "
          f"new={len(fresh)} skipped={len(spec['tasks']) - len(fresh)}")
    for row in fresh[:20]:
        print(" ", row["run_id"], row["condition"])
    if len(fresh) > 20:
        print(f"  ... {len(fresh) - 20} more")
    if args.dry_run:
        print("dry-run: ledger not written")
        return
    if fresh:
        with LEDGER.open("a", encoding="utf-8") as handle:
            for row in fresh:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"appended {len(fresh)} rows to {LEDGER.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
