#!/usr/bin/env python3
"""Backfill xiasha1_online_v3 ledger rows: REGISTERED -> DONE + metrics.

从每个 run 目录读 metrics/records.jsonl，取末尾 20 条评估记录均值
（travel_time/throughput/unfinished_vehicles/queue/delay），写回台账。
幂等：已是 DONE 且有 metrics 的行跳过。

用法: python dev/ledger/backfill_xiasha1_online_v3.py [--dry-run]
"""
import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "ledger" / "runs.jsonl"
CAMPAIGN = "xiasha1_online_v3"
KEYS = ("travel_time", "throughput", "unfinished_vehicles", "queue", "delay")


def run_metrics(run_dir):
    recs = [json.loads(l) for l in open(run_dir / "metrics" / "records.jsonl")]
    ev = [r for r in recs if r.get("travel_time") is not None
          and r.get("throughput") is not None][-20:]
    if not ev:
        return None
    return {k: st.mean(r.get(k, 0) for r in ev) for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(LEDGER)]
    n = miss = 0
    for r in rows:
        if r.get("campaign") != CAMPAIGN or r.get("status") == "DONE":
            continue
        m = run_metrics(ROOT / r["run_dir"])
        if m is None:
            miss += 1
            continue
        r["status"] = "DONE"
        r["metrics"] = m
        r["episodes_total"] = 200
        n += 1
    print(f"backfilled {n} DONE rows; {miss} missing metrics")
    if not args.dry_run and n:
        with open(LEDGER, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
