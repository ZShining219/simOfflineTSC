#!/usr/bin/env python3
"""Close campaign ``xiasha1_online_v1`` in ledger/runs.jsonl (2026-10-21).

The 288-run matrix was stopped on 2026-10-21 and will not be resumed.
This script performs the one-time reconciliation:

* 24 runs with ``run_status.status == 已完成`` -> DONE, with final-eval
  three metrics and the actual run config hash;
* 12 runs with residual ``运行中`` and no live process -> ABORTED, with
  last-eval metrics (partial evidence only) and the number of written
  training records;
* 252 never-started runs -> SCOPE_DISCARD (queue dropped).

Idempotent: rows already in a terminal closure state are skipped.  Use
``--dry-run`` first.  Only rows of this campaign are rewritten; all other
ledger lines are preserved byte-for-byte.

Usage:
    python dev/ledger/close_xiasha1_online.py --dry-run
    python dev/ledger/close_xiasha1_online.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "ledger" / "runs.jsonl"
AUDIT = ROOT / "artifacts" / "xiasha1_online_v1" / "stop_audit.csv"
CAMPAIGN = "xiasha1_online_v1"
CLOSED = {"DONE", "ABORTED", "SCOPE_DISCARD"}
ABORT_REASON = "进程外部终止/状态残留（2026-10-21 11:03 人工停止；"\
               "11:14 同前缀意外重启覆盖原 partial 记录后再次停止）"
STOP_TIME = "2026-10-21T11:03:28Z"  # run_manifest.updated_at at the stop


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _records(run_dir):
    path = run_dir / "metrics" / "records.jsonl"
    if not path.is_file():
        return [], path
    rows = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    return rows, path


def _evaluations(rows):
    return sorted(
        (row for row in rows
         if row.get("record_type") in {"EVALUATION", "FINAL_EVALUATION"}),
        key=lambda row: int(row.get("episode", -1)),
    )


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


def _utc_from_mtime(path):
    stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_note(row, note):
    current = (row.get("notes") or "").strip()
    if note in current:
        return
    row["notes"] = f"{current}; {note}" if current else note


def _audit_map():
    import csv
    if not AUDIT.is_file():
        return {}
    with AUDIT.open(encoding="utf-8") as handle:
        return {row["run_id"]: row for row in csv.DictReader(handle)}


def _repair_aborted(row, audit):
    """Restore pre-restart audit extent and flag the lost raw evidence."""
    if not audit:
        return False
    last_episode = int(audit["last_episode"])
    records = int(audit["records"])
    notes = (row.get("notes") or "").split("; ABORTED ", 1)[0]
    row["episodes_total"] = last_episode
    row["episodes_increment"] = last_episode
    row["metrics"] = None
    row["started"] = None
    row["end_time"] = STOP_TIME
    row["abort_reason"] = ABORT_REASON
    row["evidence_lost"] = True
    row["notes"] = (
        f"{notes}; ABORTED 2026-10-21: 停止前审计 records={records}, "
        f"last eval ep={last_episode}（artifacts/xiasha1_online_v1/"
        f"stop_audit.csv）；11:14 同前缀意外重启，run.py 清空目录，原 "
        f"partial raw records 丢失；重启段停在 ep7；原始 metrics 不可恢复；"
        f"campaign 不再恢复"
    )
    return True


def _classify(row):
    """Return (new_status, payload) for one REGISTERED campaign row."""
    run_dir = ROOT / row["run_dir"]
    status_path = run_dir / "run_status.json"
    manifest_path = run_dir / "run_manifest.json"
    rows, records_path = _records(run_dir)
    evaluations = _evaluations(rows)
    last_eval = evaluations[-1] if evaluations else None
    last_episode = int(last_eval["episode"]) if last_eval else None
    config_sha = None
    if manifest_path.is_file():
        config_sha = _json(manifest_path).get("config_hash")

    if status_path.is_file():
        status = _json(status_path)
        status_zh = status.get("status")
        if status_zh == "已完成" and status.get("exit_code") == 0:
            return "DONE", {
                "metrics": _metrics(last_eval),
                "episodes_total": last_episode or 200,
                "episodes_increment": last_episode or 200,
                "started": status.get("started_at_utc"),
                "end_time": status.get("finished_at_utc"),
                "config_sha": config_sha,
                "note": (
                    f"DONE 2026-10-21: 200ep 完成；final eval ep{last_episode} "
                    f"三指标已回填；campaign 收口（24/288 完成）"
                ),
            }
        if status_zh == "运行中":
            written = sum(1 for item in rows
                          if item.get("record_type") == "TRAIN")
            end_time = (_utc_from_mtime(records_path)
                        if records_path.is_file() else None)
            return "ABORTED", {
                "metrics": _metrics(last_eval),
                "episodes_total": last_episode,
                "episodes_increment": last_episode,
                "started": status.get("started_at_utc"),
                "end_time": end_time,
                "config_sha": config_sha,
                "abort_reason": ABORT_REASON,
                "note": (
                    f"ABORTED 2026-10-21: 停止时最后 eval ep{last_episode}；"
                    f"已写训练记录 {written} 条；partial 目录保留；metrics "
                    f"仅为中断前证据，不作最终性能结论；campaign 不再恢复"
                ),
            }
        return None, {"reason": f"unexpected run_status={status_zh}"}

    return "SCOPE_DISCARD", {
        "note": (
            "SCOPE_DISCARD 2026-10-21: 队列 DROPPED，不恢复原 288-run 矩阵；"
            "未启动，无产物；由 xiasha1_explore_v1（slow/full 探索验证）替代"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--repair-aborted", action="store_true",
        help="rewrite ABORTED rows from the pre-restart stop_audit.csv")
    args = parser.parse_args()

    audit = _audit_map() if args.repair_aborted else {}
    lines = LEDGER.read_text(encoding="utf-8").splitlines()
    counts = Counter()
    changed = 0
    skipped = []
    out = []
    for line in lines:
        if not line.strip():
            out.append(line)
            continue
        row = json.loads(line)
        if row.get("campaign") != CAMPAIGN:
            out.append(line)
            continue
        if args.repair_aborted and row.get("status") == "ABORTED":
            if _repair_aborted(row, audit.get(row.get("run_id"))):
                counts["ABORTED_REPAIRED"] += 1
                changed += 1
                out.append(json.dumps(row, ensure_ascii=False))
            else:
                out.append(line)
            continue
        if row.get("status") in CLOSED:
            counts[row["status"]] += 1
            out.append(line)
            continue
        if row.get("status") != "REGISTERED":
            skipped.append((row.get("run_id"), row.get("status")))
            out.append(line)
            continue
        new_status, payload = _classify(row)
        if new_status is None:
            skipped.append((row.get("run_id"), payload.get("reason")))
            out.append(line)
            continue
        row["status"] = new_status
        for key in ("metrics", "episodes_total", "episodes_increment",
                    "started", "end_time", "config_sha", "abort_reason"):
            if key in payload and payload[key] is not None:
                row[key] = payload[key]
        _append_note(row, payload["note"])
        counts[new_status] += 1
        changed += 1
        out.append(json.dumps(row, ensure_ascii=False))

    print(f"campaign={CAMPAIGN} rows={sum(counts.values())} "
          f"changed={changed} statuses={dict(counts)}")
    if skipped:
        print(f"skipped/unclassified={len(skipped)}")
        for item in skipped[:20]:
            print("  ", item)

    if args.dry_run:
        print("dry-run: ledger not written")
        return
    LEDGER.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {LEDGER.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
