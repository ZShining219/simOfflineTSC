#!/usr/bin/env python3
"""Backfill campaign ``xiasha1_online_v2`` after the 288 runs finish.

For each REGISTERED row:
* completed run -> DONE, final-eval metrics, actual run config hash,
  started/end times, final-20 / queue / convergence summary in notes;
* failed run -> FAILED with error;
* residual 运行中 -> ABORTED with reason and written count.

Idempotent: rows already in a terminal state are skipped.  Use --dry-run
first.  Only rows of this campaign are rewritten.

Usage:
    python dev/ledger/backfill_xiasha1_online_v2.py --dry-run
    python dev/ledger/backfill_xiasha1_online_v2.py
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "ledger" / "runs.jsonl"
CAMPAIGN = "xiasha1_online_v2"
FINAL = {"DONE", "FAILED", "ABORTED"}
EPS_MIN = {"dqn", "ddqn", "dueling", "d3qn", "presslight", "frap"}


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


def _convergence(evals, tolerance=0.05):
    points = [(int(row["episode"]), float(row["travel_time"]))
              for row in evals if row.get("travel_time") is not None]
    if len(points) < 30:
        return None
    episodes = [item[0] for item in points]
    values = [item[1] for item in points]
    reference = statistics.mean(values[-20:])
    finite = [abs(value) for value in values if math.isfinite(value)]
    scale = statistics.median(finite) if finite else 0.0
    denominator = max(abs(reference), 0.01 * scale, 1e-8)
    moving = [
        None if index < 9 else statistics.mean(values[index - 9:index + 1])
        for index in range(len(values))
    ]
    for index in range(9, len(values) - 19):
        if all(moving[offset] is not None
               and abs(moving[offset] - reference) / denominator <= tolerance
               for offset in range(index, index + 20)):
            return episodes[index]
    return None


def _eps_min_episode(train_rows, epsilon_min=0.05):
    for row in sorted(train_rows, key=lambda item: int(item["episode"])):
        epsilon = row.get("epsilon")
        if epsilon is not None and float(epsilon) <= epsilon_min + 1e-9:
            return int(row["episode"])
    return None


def _loaded(pack):
    path = ROOT / "data" / "raw_data" / "xiasha1_demand_packs.json"
    payload = _json(path)
    return {item["pack"]: int(item["n_vehicles"])
            for item in payload["packs"]}[pack]


def _summary(evals, train_rows, loaded, arm):
    last = evals[-20:]
    if len(last) < 20:
        return ""
    tt = _mean(row.get("travel_time") for row in last)
    throughput = _mean(row.get("throughput") for row in last)
    unfinished = _mean(row.get("unfinished_vehicles") for row in last)
    queue = _mean(row.get("queue") for row in last)
    uninserted = loaded - throughput - unfinished
    conv = _convergence(evals)
    eps_ep = (_eps_min_episode(train_rows) if arm in EPS_MIN else None)
    parts = [
        f"final-20 TT={tt:.2f} thr={throughput:.0f} unf={unfinished:.1f} "
        f"unins={uninserted:.1f} queue={queue:.2f}",
        f"conv5={conv if conv is not None else '—'}",
    ]
    if eps_ep is not None:
        parts.append(f"eps-min ep={eps_ep}")
    return "；".join(parts)


def _backfill(row):
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
    arm = row["arm"]
    summary = _summary(evals, train_rows, _loaded(pack), arm)

    if status_zh == "已完成" and status.get("exit_code") == 0:
        return "DONE", {
            "metrics": _metrics(evals[-1] if evals else None),
            "episodes_total": int(evals[-1]["episode"]) if evals else 0,
            "episodes_increment": int(evals[-1]["episode"]) if evals else 0,
            "started": status.get("started_at_utc"),
            "end_time": status.get("finished_at_utc"),
            "config_sha": config_sha,
            "note": f"DONE 2026-10-22: 200ep 完成；{summary}",
        }
    if status_zh == "失败" or status.get("exit_code") not in (None, 0):
        return "FAILED", {
            "error": status.get("error_message") or "run_status=失败",
            "started": status.get("started_at_utc"),
            "end_time": status.get("finished_at_utc"),
            "config_sha": config_sha,
            "note": "FAILED 2026-10-22: 运行失败，见 run_status.error_message",
        }
    return "ABORTED", {
        "abort_reason": "进程外部终止/状态残留（2026-10-22 v2 收口）",
        "episodes_total": int(evals[-1]["episode"]) if evals else 0,
        "episodes_increment": int(evals[-1]["episode"]) if evals else 0,
        "started": status.get("started_at_utc"),
        "end_time": status.get("finished_at_utc"),
        "config_sha": config_sha,
        "note": (
            f"ABORTED 2026-10-22: 最后 eval ep"
            f"{int(evals[-1]['episode']) if evals else 0}；已写训练记录 "
            f"{len(train_rows)} 条；{summary}"
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
        for item in missing[:20]:
            print("  ", item)
    if args.dry_run:
        print("dry-run: ledger not written")
        return
    LEDGER.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {LEDGER.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
