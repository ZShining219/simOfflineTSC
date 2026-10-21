#!/usr/bin/env python3
"""Backfill campaign ``xiasha1_explore_v1`` after the 12 runs finish.

For each REGISTERED row:
* completed run -> DONE, final-eval three metrics, actual run config hash,
  started/end times, final-20 summary in notes;
* failed run -> FAILED with error;
* residual 运行中 / incomplete run -> ABORTED with reason and written count.

Idempotent: rows already in a terminal state are skipped.  Use --dry-run
first.  Only rows of this campaign are rewritten.

Usage:
    python dev/ledger/backfill_xiasha1_explore.py --dry-run
    python dev/ledger/backfill_xiasha1_explore.py
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "ledger" / "runs.jsonl"
CAMPAIGN = "xiasha1_explore_v1"
FINAL = {"DONE", "FAILED", "ABORTED"}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _records(run_dir):
    path = run_dir / "metrics" / "records.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def _evals(rows):
    return sorted(
        (row for row in rows
         if row.get("record_type") in {"EVALUATION", "FINAL_EVALUATION"}),
        key=lambda row: int(row["episode"]),
    )


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return statistics.mean(values) if values else None


def _metrics(row):
    if not row:
        return None
    return {
        "travel_time": round(float(row["travel_time"]), 3),
        "throughput": int(row["throughput"]),
        "unfinished_vehicles": int(row["unfinished_vehicles"]),
        "queue": round(float(row.get("queue", 0.0)), 3),
        "delay": round(float(row.get("delay", 0.0)), 3),
    }


def _final20_summary(evals, loaded):
    last = evals[-20:]
    if len(last) < 20:
        return None
    tt = _mean(row.get("travel_time") for row in last)
    throughput = _mean(row.get("throughput") for row in last)
    unfinished = _mean(row.get("unfinished_vehicles") for row in last)
    uninserted = (
        None if throughput is None or unfinished is None
        else loaded - throughput - unfinished
    )
    return {
        "travel_time": tt,
        "throughput": throughput,
        "unfinished_vehicles": unfinished,
        "uninserted": uninserted,
    }


def _loaded(pack):
    path = ROOT / "data" / "raw_data" / "xiasha1_demand_packs.json"
    payload = _json(path)
    return {item["pack"]: int(item["n_vehicles"])
            for item in payload["packs"]}[pack]


def _backfill(row, abort_reason, abort_note):
    run_dir = ROOT / row["run_dir"]
    status_path = run_dir / "run_status.json"
    if not status_path.is_file():
        return None, "missing run_status"
    status = _json(status_path)
    status_zh = status.get("status")
    rows = _records(run_dir)
    evals = _evals(rows)
    train_rows = [item for item in rows if item.get("record_type") == "TRAIN"]
    manifest_path = run_dir / "run_manifest.json"
    config_sha = (_json(manifest_path).get("config_hash")
                  if manifest_path.is_file() else None)
    pack = row["net"].replace("xiasha1_", "", 1)
    final20 = _final20_summary(evals, _loaded(pack))
    summary_note = ""
    if final20:
        summary_note = (
            f"；final-20 TT={final20['travel_time']:.2f} "
            f"thr={final20['throughput']:.0f} "
            f"unf={final20['unfinished_vehicles']:.1f} "
            f"unins={final20['uninserted']:.1f}"
        )

    if status_zh == "已完成" and status.get("exit_code") == 0:
        return "DONE", {
            "metrics": _metrics(evals[-1] if evals else None),
            "episodes_total": int(evals[-1]["episode"]) if evals else 0,
            "episodes_increment": int(evals[-1]["episode"]) if evals else 0,
            "started": status.get("started_at_utc"),
            "end_time": status.get("finished_at_utc"),
            "config_sha": config_sha,
            "note": f"DONE 2026-10-21: 200ep 完成{summary_note}",
        }
    if status_zh == "失败" or status.get("exit_code") not in (None, 0):
        return "FAILED", {
            "error": status.get("error_message") or "run_status=失败",
            "started": status.get("started_at_utc"),
            "end_time": status.get("finished_at_utc"),
            "config_sha": config_sha,
            "note": "FAILED 2026-10-21: 运行失败，见 run_status.error_message",
        }
    return "ABORTED", {
        "abort_reason": abort_reason,
        "episodes_total": int(evals[-1]["episode"]) if evals else 0,
        "episodes_increment": int(evals[-1]["episode"]) if evals else 0,
        "started": status.get("started_at_utc"),
        "end_time": status.get("finished_at_utc"),
        "config_sha": config_sha,
        "note": (
            f"ABORTED 2026-10-21: 最后 eval ep"
            f"{int(evals[-1]['episode']) if evals else 0}；已写训练记录 "
            f"{len(train_rows)} 条{summary_note}；{abort_note}"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--abort-reason",
        default="用户暂停（2026-10-21 转 xiasha1_online_v2）")
    parser.add_argument(
        "--abort-note",
        default="用户暂停当前 explore 验证；partial 目录保留；由 "
                "xiasha1_online_v2 替代")
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
        new_status, payload = _backfill(
            row, args.abort_reason, args.abort_note)
        if new_status is None:
            missing.append((row.get("run_id"), payload))
            out.append(line)
            continue
        row["status"] = new_status
        for key in ("metrics", "episodes_total", "episodes_increment",
                    "started", "end_time", "config_sha", "abort_reason",
                    "error"):
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
        print(f"missing/incomplete dirs={len(missing)}")
        for item in missing:
            print("  ", item)
    if args.dry_run:
        print("dry-run: ledger not written")
        return
    LEDGER.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {LEDGER.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
