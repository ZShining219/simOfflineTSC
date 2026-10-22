#!/usr/bin/env python3
"""Backfill campaign ``xiasha1_compound_probe_v1`` after the 22 evals finish.

Learned rows read ``baseline_eval_summary.json``；rule 行读
``metrics/records.jsonl`` 末条 FINAL_EVALUATION。幂等；先 --dry-run。

Usage:
    python dev/ledger/backfill_xiasha1_compound_probe.py --dry-run
    python dev/ledger/backfill_xiasha1_compound_probe.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "ledger" / "runs.jsonl"
CAMPAIGN = "xiasha1_compound_probe_v1"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _round(v):
    return round(float(v), 3) if v is not None else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lines = LEDGER.read_text(encoding="utf-8").splitlines()
    out, done, failed, missing = [], 0, 0, 0
    for ln in lines:
        if not ln.strip():
            out.append(ln)
            continue
        row = json.loads(ln)
        if row.get("campaign") != CAMPAIGN or row.get("status") != "REGISTERED":
            out.append(ln)
            continue
        run_dir = ROOT / row["run_dir"]
        status_path = run_dir / "run_status.json"
        if not status_path.is_file():
            missing += 1
            out.append(ln)
            continue
        status = _json(status_path)
        row["started"] = status.get("started_at_utc")
        row["end_time"] = status.get("finished_at_utc")
        if status.get("status") != "已完成" or status.get("exit_code") != 0:
            row["status"] = "FAILED"
            row["error"] = status.get("error_message") or "eval run failed"
            failed += 1
            out.append(json.dumps(row, ensure_ascii=False))
            continue
        summ_path = run_dir / "baseline_eval_summary.json"
        if summ_path.is_file():
            m = _json(summ_path)
        else:
            recs = [json.loads(l) for l in
                    (run_dir / "metrics" / "records.jsonl")
                    .read_text(encoding="utf-8").splitlines() if l.strip()]
            m = [r for r in recs if r.get("record_type")
                 in ("EVALUATION", "FINAL_EVALUATION")][-1]
        row["status"] = "DONE"
        row["metrics"] = {
            "travel_time": _round(m.get("travel_time")),
            "throughput": m.get("throughput"),
            "unfinished_vehicles": m.get("unfinished_vehicles"),
            "queue": _round(m.get("queue")),
            "delay": _round(m.get("delay") or m.get("real_delay")),
        }
        row["notes"] = (
            f"DONE 2026-10-22: compound-pack probe {row.get('condition')}；"
            f"eval_step_log=1s；TT={m.get('travel_time'):.3f} "
            f"thr={m.get('throughput')} unf={m.get('unfinished_vehicles')}"
        )
        done += 1
        out.append(json.dumps(row, ensure_ascii=False))
    print(f"done={done} failed={failed} missing_run_status={missing}")
    if args.dry_run:
        print("dry-run: ledger not written")
        return
    LEDGER.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("ledger updated")


if __name__ == "__main__":
    main()
