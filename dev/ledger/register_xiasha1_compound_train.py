#!/usr/bin/env python3
"""Register campaign ``xiasha1_compound_train_v1`` in ledger/runs.jsonl.

Reads ``artifacts/xiasha1_compound_train/run_queue.json`` and appends one
REGISTERED train row per task (30 total).  Idempotent: skips run_ids
already present.  config_sha is a deterministic pre-registration hash over
base.yml + canonical agent yml + v2 overlay + pack sim cfg; on DONE it
should be replaced by run_manifest.config_hash.

Usage:
    python dev/ledger/register_xiasha1_compound_train.py --dry-run
    python dev/ledger/register_xiasha1_compound_train.py --code-commit <sha>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "ledger" / "runs.jsonl"
QUEUE = ROOT / "artifacts" / "xiasha1_compound_train" / "run_queue.json"
CAMPAIGN = "xiasha1_compound_train_v1"
NATIVE_EXPLORATION = {"sac", "mplight"}


def _config_sha(agent, overlay, pack):
    digest = hashlib.sha256()
    for relative in ("configs/tsc/base.yml", f"configs/tsc/{agent}.yml",
                     overlay, f"configs/sim/xiasha1_{pack}.cfg"):
        path = ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _row(task, code_commit):
    arm = task["arm"]
    pack = task["pack"]
    seed = int(task["seed"])
    overlay = task["command"][task["command"].index(
        "--experiment-config") + 1]
    agent = "dqn" if arm in {"dqn", "ddqn", "dueling", "d3qn"} else arm
    condition = "native_exploration" if arm in NATIVE_EXPLORATION else "full"
    return {
        "run_id": task["run_id"],
        "line": "online_baseline",
        "campaign": CAMPAIGN,
        "arm": arm,
        "alias_of": None,
        "net": f"xiasha1_{pack}",
        "seed": seed,
        "episodes_total": 200,
        "episodes_increment": 200,
        "resume_from": None,
        "ngpu": -1,
        "status": "REGISTERED",
        "plan_id": CAMPAIGN,
        "code_commit": code_commit,
        "dirty": False,
        "queue_file": "artifacts/xiasha1_compound_train/run_queue.json",
        "run_dir": task["output_path"].replace(str(ROOT) + "/", ""),
        "tmux": None,
        "started": None,
        "eval_mode": "train_frozen_eval",
        "behavior_source": None,
        "metrics": None,
        "abort_reason": None,
        "error": None,
        "evidence_lost": False,
        "notes": (
            f"REGISTERED 2026-10-22: 复合需求包训练探测——验证 v2 ~ep30 早收敛"
            f"是否源于单轴需求包过易；pack={pack}；condition={condition}；"
            f"200ep+每ep贪心评估；legacy pseudo_step；saveReplay=false；"
            f"DONE 时 config_sha 替换为 run_manifest.config_hash"
        ),
        "run_kind": "train",
        "config_sha": _config_sha(agent, overlay, pack),
        "sumo_seed_mode": "fixed_default",
        "sumo_seed": None,
        "eval_seed": None,
        "condition": condition,
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
        print(" ", row["run_id"], row["arm"], row["condition"], row["net"])
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
