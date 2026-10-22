#!/usr/bin/env python3
"""Register campaign ``xiasha1_v2_bestckpt_ts_v1`` in ledger/runs.jsonl.

Reads ``artifacts/xiasha1_v2_bestckpt_ts/run_queue.json`` and appends one
REGISTERED eval row per task (96 total).  Idempotent: skips run_ids
already present.  config_sha is the hash of the per-arm eval overlay plus
the target pack simulator cfg; checkpoint identity is recorded in notes.

Usage:
    python dev/ledger/register_xiasha1_v2_bestckpt_ts.py --dry-run
    python dev/ledger/register_xiasha1_v2_bestckpt_ts.py --code-commit <sha>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "ledger" / "runs.jsonl"
QUEUE = ROOT / "artifacts" / "xiasha1_v2_bestckpt_ts" / "run_queue.json"
CAMPAIGN = "xiasha1_v2_bestckpt_ts_v1"


def _config_sha(pack, eval_overlay):
    digest = hashlib.sha256()
    for relative in (eval_overlay, f"configs/sim/xiasha1_{pack}.cfg"):
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
    source_run = Path(task["source_run_dir"])
    try:
        source_rel = str(source_run.resolve().relative_to(ROOT))
    except ValueError:
        source_rel = str(source_run)
    return {
        "run_id": task["run_id"],
        "line": "online_baseline",
        "campaign": CAMPAIGN,
        "arm": task["arm"],
        "alias_of": None,
        "net": f"xiasha1_{task['pack']}",
        "seed": int(task["eval_seed"]),
        "episodes_total": 1,
        "episodes_increment": 1,
        "resume_from": None,
        "ngpu": -1,
        "status": "REGISTERED",
        "plan_id": CAMPAIGN,
        "code_commit": code_commit,
        "dirty": False,
        "queue_file": "artifacts/xiasha1_v2_bestckpt_ts/run_queue.json",
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
            f"REGISTERED 2026-10-22: in-scenario frozen best-ckpt replay "
            f"with eval_step_log(1s)；pack={task['pack']}；"
            f"rep_train_seed={task['rep_train_seed']}；eval_seed=0；"
            f"sumo_seed=fixed_default；"
            f"best_episode={task.get('best_episode')}；"
            f"checkpoint={checkpoint_rel}；source_run={source_rel}"
        ),
        "run_kind": "eval",
        "config_sha": _config_sha(task["pack"], task["eval_overlay"]),
        "sumo_seed_mode": "fixed_default",
        "sumo_seed": None,
        "eval_seed": int(task["eval_seed"]),
        "condition": f"in_scene:{task['pack']}",
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
