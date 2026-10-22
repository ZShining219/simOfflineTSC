#!/usr/bin/env python3
"""Backfill campaign ``xiasha1_cross_eval_v1`` after the 384 evals finish.

For each REGISTERED eval row, read the run's baseline_eval_summary.json and
write DONE with metrics, actual config hash, times and a provenance note.
Idempotent; use --dry-run first.

Usage:
    python dev/ledger/backfill_xiasha1_cross_eval.py --dry-run
    python dev/ledger/backfill_xiasha1_cross_eval.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "ledger" / "runs.jsonl"
CAMPAIGN = "xiasha1_cross_eval_v1"
FINAL = {"DONE", "FAILED", "ABORTED"}
EVAL_CONFIGS = {
    "dqn": "configs/tsc/eval_baseline_checkpoint.yml",
    "ddqn": "configs/tsc/eval_baseline_checkpoint_ddqn.yml",
    "dueling": "configs/tsc/eval_baseline_checkpoint_dueling.yml",
    "d3qn": "configs/tsc/eval_baseline_checkpoint_d3qn.yml",
    "presslight": "configs/tsc/eval_baseline_checkpoint.yml",
    "frap": "configs/tsc/eval_baseline_checkpoint.yml",
    "mplight": "configs/tsc/eval_baseline_checkpoint.yml",
    "sac": "configs/tsc/eval_baseline_checkpoint.yml",
}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _config_sha(target_pack, arm):
    digest = hashlib.sha256()
    for relative in (EVAL_CONFIGS[arm],
                     f"configs/sim/xiasha1_{target_pack}.cfg"):
        path = ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _metrics(summary):
    return {
        "travel_time": round(float(summary["travel_time"]), 3),
        "throughput": int(summary["throughput"]),
        "unfinished_vehicles": int(summary["unfinished_vehicles"]),
        "queue": round(float(summary.get("queue", 0.0)), 3),
        "delay": round(float(summary.get("delay", 0.0)), 3),
    }


def _backfill(row):
    run_dir = ROOT / row["run_dir"]
    status_path = run_dir / "run_status.json"
    summary_path = run_dir / "baseline_eval_summary.json"
    if not status_path.is_file() or not summary_path.is_file():
        return None, "missing run_status/baseline_eval_summary"
    status = _json(status_path)
    summary = _json(summary_path)
    if status.get("status") != "已完成" or status.get("exit_code") != 0:
        return "FAILED", {
            "error": status.get("error_message") or "eval run failed",
            "started": status.get("started_at_utc"),
            "end_time": status.get("finished_at_utc"),
            "note": "FAILED 2026-10-22: 跨场景 eval 失败",
        }
    target = summary["network"].replace("xiasha1_", "", 1)
    return "DONE", {
        "metrics": _metrics(summary),
        "episodes_total": 1,
        "episodes_increment": 1,
        "started": status.get("started_at_utc"),
        "end_time": status.get("finished_at_utc"),
        "config_sha": _config_sha(target, row["arm"]),
        "note": (
            f"DONE 2026-10-22: cross-scenario eval {row.get('condition')}；"
            f"train_seed={row.get('seed')}；sumo_seed={row.get('sumo_seed')}；"
            f"TT={summary.get('travel_time'):.3f} "
            f"thr={summary.get('throughput')} "
            f"unf={summary.get('unfinished_vehicles')} "
            f"queue={summary.get('queue'):.3f} "
            f"real_delay={summary.get('real_delay'):.3f}"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lines = LEDGER.read_text(encoding="utf-8").splitlines()
    counts = Counter()
    changed = 0
    missing = []
    out = []
    for line in lines:
        if not line.strip():
            out.append(line)
            continue
        row = json.loads(line)
        if row.get("campaign") != CAMPAIGN:
            out.append(line)
            continue
        if row.get("status") in FINAL:
            counts[row["status"]] += 1
            out.append(line)
            continue
        new_status, payload = _backfill(row)
        if new_status is None:
            missing.append((row.get("run_id"), payload))
            out.append(line)
            continue
        row["status"] = new_status
        for key in ("metrics", "episodes_total", "episodes_increment",
                    "started", "end_time", "config_sha", "error"):
            if key in payload and payload[key] is not None:
                row[key] = payload[key]
        notes = (row.get("notes") or "").strip()
        row["notes"] = (f"{notes}; {payload['note']}" if notes
                        else payload["note"])
        counts[new_status] += 1
        changed += 1
        out.append(json.dumps(row, ensure_ascii=False))

    print(f"campaign={CAMPAIGN} statuses={dict(counts)} changed={changed}")
    if missing:
        print(f"missing={len(missing)}")
        for item in missing[:20]:
            print("  ", item)
    if args.dry_run:
        print("dry-run: ledger not written")
        return
    LEDGER.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {LEDGER.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
