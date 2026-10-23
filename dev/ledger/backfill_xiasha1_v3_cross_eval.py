#!/usr/bin/env python3
"""Backfill xiasha1_v3_cross_eval_v1 ledger rows: REGISTERED -> DONE.

从每个 eval run 目录读 baseline_eval_summary.json，写回 metrics。
幂等：已是 DONE 的行跳过。

用法: python dev/ledger/backfill_xiasha1_v3_cross_eval.py [--dry-run]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "ledger" / "runs.jsonl"
CAMPAIGN = "xiasha1_v3_cross_eval_v1"
KEYS = ("travel_time", "throughput", "unfinished_vehicles", "queue",
        "delay", "real_delay")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(LEDGER)]
    n = miss = 0
    for r in rows:
        if r.get("campaign") != CAMPAIGN or r.get("status") == "DONE":
            continue
        p = ROOT / r["run_dir"] / "baseline_eval_summary.json"
        if not p.is_file():
            miss += 1
            continue
        s = json.loads(p.read_text())
        r["status"] = "DONE"
        r["metrics"] = {k: s.get(k) for k in KEYS}
        n += 1
    print(f"backfilled {n} DONE rows; {miss} missing summary")
    if not args.dry_run and n:
        with open(LEDGER, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
