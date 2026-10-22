#!/usr/bin/env python3
"""Register campaign ``xiasha1_cross_eval_v1`` in ledger/runs.jsonl.

Reads ``artifacts/xiasha1_cross_eval_v1/run_queue.json`` and appends one
REGISTERED eval row per task (384 total).  Idempotent: skips run_ids
already present.  config_sha is the hash of the evaluation overlay plus the
target pack simulator cfg; checkpoint identity is recorded in notes.

Usage:
    python dev/ledger/register_xiasha1_cross_eval.py --dry-run
    python dev/ledger/register_xiasha1_cross_eval.py --code-commit <sha>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "ledger" / "runs.jsonl"
QUEUE = ROOT / "artifacts" / "xiasha1_cross_eval_v1" / "run_queue.json"
CAMPAIGN = "xiasha1_cross_eval_v1"
EVAL_CONFIG = "configs/tsc/eval_baseline_checkpoint.yml"
EVAL_CONFIGS = {
    "dqn": EVAL_CONFIG,
    "ddqn": "configs/tsc/eval_baseline_checkpoint_ddqn.yml",
    "dueling": "configs/tsc/eval_baseline_checkpoint_dueling.yml",
    "d3qn": "configs/tsc/eval_baseline_checkpoint_d3qn.yml",
    "presslight": EVAL_CONFIG,
    "frap": EVAL_CONFIG,
    "mplight": EVAL_CONFIG,
    "sac": EVAL_CONFIG,
}


def _config_sha(target_pack, arm):
    digest = hashlib.sha256()
    for relative in (EVAL_CONFIGS[arm],
                     f"configs/sim/xiasha1_{target_pack}.cfg"):
        path = ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _row(task, code_commit):
    checkpoint = Path(task["checkpoint"])
    try:
        checkpoint_rel = str(checkpoint.resolve().relative_to(ROOT))
    except ValueError:
        checkpoint_rel = str(checkpoint)
    return {
        "run_id": task["run_id"],
        "line": "online_baseline",
        "campaign": CAMPAIGN,
        "arm": task["arm"],
        "alias_of": None,
        "net": f"xiasha1_{task['target_pack']}",
        "seed": int(task["train_seed"]),
        "episodes_total": 1,
        "episodes_increment": 1,
        "resume_from": None,
        "ngpu": -1,
        "status": "REGISTERED",
        "plan_id": CAMPAIGN,
        "code_commit": code_commit,
        "dirty": False,
        "queue_file": "artifacts/xiasha1_cross_eval_v1/run_queue.json",
        "run_dir": task["output_path"].replace(str(ROOT) + "/", ""),
        "tmux": None,
        "started": None,
        "eval_mode": "train_frozen_eval",
        "behavior_source": "frozen_best_checkpoint",
        "metrics": None,
        "abort_reason": None,
        "error": None,
        "evidence_lost": False,
        "notes": (
            f"REGISTERED 2026-10-22: cross-scenario best-ckpt eval；"
            f"source={task['source_pack']}；target={task['target_pack']}；"
            f"train_seed={task['train_seed']}；sumo_seed={task['sumo_seed']}；"
            f"checkpoint={checkpoint_rel}；best_episode={task.get('best_episode')}"
        ),
        "run_kind": "eval",
        "config_sha": _config_sha(task["target_pack"], task["arm"]),
        "sumo_seed_mode": "explicit",
        "sumo_seed": int(task["sumo_seed"]),
        "eval_seed": int(task["sumo_seed"]),
        "condition": f"{task['source_pack']}->{task['target_pack']}",
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
                existing.add(json.loads(line).get("run_id"))
    fresh = [_row(task, args.code_commit) for task in spec["tasks"]
             if task["run_id"] not in existing]
    print(f"campaign={CAMPAIGN} tasks={len(spec['tasks'])} "
          f"new={len(fresh)} skipped={len(spec['tasks']) - len(fresh)}")
    for row in fresh[:10]:
        print(" ", row["run_id"], row["arm"], row["condition"])
    if len(fresh) > 10:
        print(f"  ... {len(fresh) - 10} more")
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
