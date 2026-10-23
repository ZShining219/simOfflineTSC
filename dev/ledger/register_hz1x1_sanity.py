#!/usr/bin/env python3
"""Register campaign ``hz1x1_sanity_v1`` in ledger/runs.jsonl.

Reads ``artifacts/hz1x1_sanity_v1/run_queue.json`` and appends one
REGISTERED train row per task (48 total; hz1x1 原始真实包口径体检).
Idempotent: skips run_ids already present.

Usage:
    python dev/ledger/register_hz1x1_sanity.py --dry-run
    python dev/ledger/register_hz1x1_sanity.py --code-commit <sha>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "ledger" / "runs.jsonl"
QUEUE = ROOT / "artifacts" / "hz1x1_sanity_v1" / "run_queue.json"
CAMPAIGN = "hz1x1_sanity_v1"
SIMCFG = {"bctyc": "sumohz1x1", "qcyn": "sumohz1x1_config2",
          "knhz": "sumohz1x1_config3", "sbsx": "sumohz1x1_config4"}
NATIVE_EXPLORATION = {"sac", "mplight"}


def _config_sha(agent, overlay, pack):
    digest = hashlib.sha256()
    for relative in ("configs/tsc/base.yml", f"configs/tsc/{agent}.yml",
                     overlay, f"configs/sim/{SIMCFG[pack]}.cfg"):
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
        "net": f"hz1x1_{pack}",
        "seed": seed,
        "episodes_total": 200,
        "episodes_increment": 200,
        "resume_from": None,
        "ngpu": -1,
        "status": "REGISTERED",
        "plan_id": CAMPAIGN,
        "code_commit": code_commit,
        "dirty": None,
        "queue_file": str(QUEUE.relative_to(ROOT)),
        "run_dir": str(Path(task["output_path"]).relative_to(ROOT)),
        "tmux": None,
        "started": None,
        "eval_mode": "train_frozen_eval",
        "behavior_source": None,
        "metrics": None,
        "abort_reason": None,
        "error": None,
        "evidence_lost": False,
        "notes": f"hz1x1 口径体检：原始真实包 {task['network']}；"
                 f"协议与 xiasha1_online_v3 逐位一致；condition={condition}",
        "run_kind": "train",
        "config_sha": _config_sha(agent, overlay, pack),
        "sumo_seed_mode": "fixed_default",
        "sumo_seed": None,
        "eval_seed": None,
        "condition": condition,
        "superseded_by": None,
        "runtime_env": "colight",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-commit", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    queue = json.loads(QUEUE.read_text(encoding="utf-8"))
    tasks = queue["tasks"] if isinstance(queue, dict) else queue
    existing = set()
    if LEDGER.is_file():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.add(json.loads(line)["run_id"])
    rows = []
    for task in tasks:
        if task["run_id"] in existing:
            continue
        rows.append(_row(task, args.code_commit))
        print(" ", task["run_id"], task["arm"], task["pack"])
    if not args.dry_run and rows:
        with open(LEDGER, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"appended {len(rows)} rows to {LEDGER.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
