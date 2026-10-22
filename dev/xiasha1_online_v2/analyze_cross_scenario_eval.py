#!/usr/bin/env python3
"""Analyze xiasha1_cross_eval_v1 (384 best-checkpoint cross-scenario evals).

For each target pack, compare the in-domain model (trained on that target)
against models trained on the other source packs.  This target-relative
comparison controls for the large difficulty differences between packs.

Outputs:
    artifacts/xiasha1_cross_eval_v1/analysis/eval_metrics.csv
    artifacts/xiasha1_cross_eval_v1/analysis/summary.md

Usage:
    python dev/xiasha1_online_v2/analyze_cross_scenario_eval.py
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "artifacts" / "xiasha1_cross_eval_v1" / "run_queue.json"
DEMAND = ROOT / "data" / "raw_data" / "xiasha1_demand_packs.json"
OUT = ROOT / "artifacts" / "xiasha1_cross_eval_v1" / "analysis"
ARMS = ["dqn", "ddqn", "dueling", "d3qn", "presslight", "frap", "mplight", "sac"]
PACKS = ["asym_nonly", "asym_ns", "dense_extreme", "fluct_burst"]


def _mean(values):
    values = [float(v) for v in values if v is not None]
    return statistics.mean(values) if values else None


def _fmt(value, digits=2):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _fmt_delta(value, digits=2):
    if value is None:
        return "—"
    return f"{float(value):+.{digits}f}"


def _load():
    loaded = {item["pack"]: int(item["n_vehicles"])
              for item in json.loads(DEMAND.read_text())["packs"]}
    spec = json.loads(QUEUE.read_text())
    rows = []
    missing = []
    for task in spec["tasks"]:
        run_dir = Path(task["output_path"])
        status_path = run_dir / "run_status.json"
        summary_path = run_dir / "baseline_eval_summary.json"
        if not status_path.is_file() or not summary_path.is_file():
            missing.append(task["run_id"])
            continue
        status = json.loads(status_path.read_text())
        summary = json.loads(summary_path.read_text())
        if status.get("status") != "已完成" or status.get("exit_code") != 0:
            missing.append(task["run_id"])
            continue
        target = task["target_pack"]
        throughput = summary.get("throughput")
        unfinished = summary.get("unfinished_vehicles")
        uninserted = (loaded[target] - throughput - unfinished
                      if throughput is not None and unfinished is not None
                      else None)
        rows.append({
            "arm": task["arm"],
            "source": task["source_pack"],
            "target": target,
            "train_seed": task["train_seed"],
            "sumo_seed": task["sumo_seed"],
            "in_domain": task["source_pack"] == target,
            "travel_time": summary.get("travel_time"),
            "throughput": throughput,
            "unfinished_vehicles": unfinished,
            "uninserted": uninserted,
            "queue": summary.get("queue"),
            "real_delay": summary.get("real_delay"),
            "phase_switches": summary.get("phase_switches"),
        })
    return rows, missing


def _write_csv(rows, path):
    fields = ["arm", "source", "target", "train_seed", "sumo_seed",
              "in_domain", "travel_time", "throughput", "unfinished_vehicles",
              "uninserted", "queue", "real_delay", "phase_switches"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _target_stats(rows, arm, target):
    """In-domain (train on target) vs cross (train on other sources) for one target."""
    in_rows = [r for r in rows if r["arm"] == arm and r["target"] == target
               and r["source"] == target]
    cross_rows = [r for r in rows if r["arm"] == arm and r["target"] == target
                  and r["source"] != target]
    return {
        "in": {
            "tt": _mean(r["travel_time"] for r in in_rows),
            "un": _mean(r["uninserted"] for r in in_rows),
            "q": _mean(r["queue"] for r in in_rows),
        },
        "cross": {
            "tt": _mean(r["travel_time"] for r in cross_rows),
            "un": _mean(r["uninserted"] for r in cross_rows),
            "q": _mean(r["queue"] for r in cross_rows),
        },
    }


def _summary(rows, missing):
    lines = [
        "# xiasha1_cross_eval_v1: best-checkpoint cross-scenario evaluation",
        "",
        "Comparison is **target-relative**: for each target pack T, the "
        "in-domain model is trained on T and evaluated on T; the cross models "
        "are trained on the other 3 source packs and evaluated on T. "
        "This controls for the large demand/difficulty differences between "
        "packs. Values are means over train seeds and source packs.",
        "",
        "## Per arm: in-domain vs cross-source on the same target",
        "",
        "| arm | in TT | cross TT | ΔTT | in unins | cross unins | Δunins | in queue | cross queue | Δqueue | mean TT gap % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    per_arm_target = {}
    for arm in ARMS:
        stats = {target: _target_stats(rows, arm, target) for target in PACKS}
        per_arm_target[arm] = stats
        in_tt = _mean(stats[t]["in"]["tt"] for t in PACKS)
        cross_tt = _mean(stats[t]["cross"]["tt"] for t in PACKS)
        in_un = _mean(stats[t]["in"]["un"] for t in PACKS)
        cross_un = _mean(stats[t]["cross"]["un"] for t in PACKS)
        in_q = _mean(stats[t]["in"]["q"] for t in PACKS)
        cross_q = _mean(stats[t]["cross"]["q"] for t in PACKS)
        gaps = []
        for t in PACKS:
            inside = stats[t]["in"]["tt"]
            cross = stats[t]["cross"]["tt"]
            if inside and cross is not None:
                gaps.append((cross - inside) / inside * 100)
        lines.append(
            "| {arm} | {itt} | {ctt} | {dtt} | {iun} | {cun} | {dun} | {iq} | {cq} | {dq} | {gap} |".format(
                arm=arm,
                itt=_fmt(in_tt), ctt=_fmt(cross_tt),
                dtt=_fmt_delta(cross_tt - in_tt if cross_tt is not None and in_tt is not None else None),
                iun=_fmt(in_un, 1), cun=_fmt(cross_un, 1),
                dun=_fmt_delta(cross_un - in_un if cross_un is not None and in_un is not None else None, 1),
                iq=_fmt(in_q, 2), cq=_fmt(cross_q, 2),
                dq=_fmt_delta(cross_q - in_q if cross_q is not None and in_q is not None else None, 2),
                gap=_fmt(_mean(gaps), 1),
            )
        )

    lines += [
        "",
        "## Per arm × target: cross-source vs in-domain",
        "",
        "ΔTT / Δuninserted = cross-source mean − in-domain mean for the same "
        "target (negative TT = cross is faster; negative uninserted = cross "
        "leaves fewer vehicles uninserted).",
        "",
        "| arm | target | in TT | cross TT | ΔTT | in unins | cross unins | Δunins |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        for target in PACKS:
            stats = per_arm_target[arm][target]
            inside, cross = stats["in"], stats["cross"]
            lines.append(
                "| {arm} | {target} | {itt} | {ctt} | {dtt} | {iun} | {cun} | {dun} |".format(
                    arm=arm, target=target,
                    itt=_fmt(inside["tt"]), ctt=_fmt(cross["tt"]),
                    dtt=_fmt_delta(cross["tt"] - inside["tt"] if cross["tt"] is not None and inside["tt"] is not None else None),
                    iun=_fmt(inside["un"], 1), cun=_fmt(cross["un"], 1),
                    dun=_fmt_delta(cross["un"] - inside["un"] if cross["un"] is not None and inside["un"] is not None else None, 1),
                )
            )

    lines += [
        "",
        "## Source→target transfer matrix: ΔTT / Δuninserted vs target in-domain",
        "",
        "Cell = model trained on source S, evaluated on target T, minus the "
        "in-domain model trained on T (mean over arms and train seeds). "
        "Target-relative; diagonal = 0. Positive ΔTT = worse than training on "
        "the target itself.",
        "",
        "| source \\ target | " + " | ".join(PACKS) + " |",
        "| --- | " + " | ".join("---:" for _ in PACKS) + " |",
    ]
    target_in = {}
    for target in PACKS:
        target_in[target] = (
            _mean(r["travel_time"] for r in rows
                  if r["target"] == target and r["source"] == target),
            _mean(r["uninserted"] for r in rows
                  if r["target"] == target and r["source"] == target),
        )
    for source in PACKS:
        cells = []
        for target in PACKS:
            if source == target:
                cells.append("0.00 / 0.0")
                continue
            group = [r for r in rows
                     if r["source"] == source and r["target"] == target]
            tt = _mean(r["travel_time"] for r in group)
            un = _mean(r["uninserted"] for r in group)
            in_tt, in_un = target_in[target]
            cells.append(
                f"{_fmt_delta(tt - in_tt if tt is not None and in_tt is not None else None)}"
                f" / {_fmt_delta(un - in_un if un is not None and in_un is not None else None, 1)}")
        lines.append(f"| {source} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Cross-source ranking on the same target set",
        "",
        "Lower cross TT / uninserted / queue is better; all arms use the same "
        "4 targets and 3 source packs, so this ranking is comparable across "
        "arms.",
        "",
        "| rank | arm | cross TT | cross unins | cross queue | mean TT gap % |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    ranked = []
    for arm in ARMS:
        stats = per_arm_target[arm]
        cross_tt = _mean(stats[t]["cross"]["tt"] for t in PACKS)
        cross_un = _mean(stats[t]["cross"]["un"] for t in PACKS)
        cross_q = _mean(stats[t]["cross"]["q"] for t in PACKS)
        gaps = []
        for t in PACKS:
            inside = stats[t]["in"]["tt"]
            cross = stats[t]["cross"]["tt"]
            if inside and cross is not None:
                gaps.append((cross - inside) / inside * 100)
        ranked.append((arm, cross_tt, cross_un, cross_q, _mean(gaps)))
    ranked.sort(key=lambda item: item[1] if item[1] is not None else 1e9)
    for index, (arm, tt, un, q, gap) in enumerate(ranked, 1):
        lines.append(f"| {index} | {arm} | {_fmt(tt)} | {_fmt(un, 1)} | {_fmt(q, 2)} | {_fmt(gap, 1)} |")

    if missing:
        lines += ["", "## Missing / failed", ""]
        for run_id in missing:
            lines.append(f"- `{run_id}`")
    lines.append("")
    return "\n".join(lines)


def main():
    rows, missing = _load()
    OUT.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, OUT / "eval_metrics.csv")
    summary = _summary(rows, missing)
    (OUT / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"\nwrote {OUT / 'eval_metrics.csv'}")
    print(f"wrote {OUT / 'summary.md'}")
    print(f"rows={len(rows)} missing={len(missing)}")


if __name__ == "__main__":
    main()
