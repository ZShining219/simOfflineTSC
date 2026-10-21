#!/usr/bin/env python3
"""Compare native / slow / full epsilon schedules on xiasha1 DQN.

Runs:
    native  xi1on_dqn_<pack>_s<seed>        (epsilon 0.1 -> 0.01, min ~ep4)
    slow    xi1exp_dqn_slow_<pack>_s<seed>  (epsilon 1.0 -> 0.05, min ~ep118)
    full    xi1exp_dqn_full_<pack>_s<seed>  (epsilon 1.0 -> 0.05, min ~ep180)

Metrics per run (last 20 EVALUATION/FINAL_EVALUATION rows):
    travel_time, throughput, unfinished_vehicles, queue,
    uninserted = loaded - throughput - unfinished,
    phase switches, max action share,
    convergence episode at +/-5% and +/-3%,
    first-100 AUC/mean, epsilon-min episode, best checkpoint episode/TT.

Outputs:
    artifacts/xiasha1_explore_v1/analysis/pilot_metrics.csv
    artifacts/xiasha1_explore_v1/analysis/pilot_summary.md

Usage (after the 12 new runs finish):
    python dev/xiasha1_online/analyze_schedule_pilot.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKS = ("asym_nonly", "asym_ns")
SEEDS = (0, 1, 2)
QUEUE = ROOT / "artifacts" / "xiasha1_explore_v1" / "run_queue.json"
SCHEDULES = {
    "native": "xi1on_dqn_{pack}_s{seed}",
    "slow": "xi1exp_dqn_slow_{pack}_s{seed}",
    "full": "xi1exp_dqn_full_{pack}_s{seed}",
}
DEFAULT_EPS_MIN = {"native": 0.01, "slow": 0.05, "full": 0.05}
DEFAULT_OUT = ROOT / "artifacts" / "xiasha1_explore_v1" / "analysis"


def _jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def _loaded_map():
    path = ROOT / "data" / "raw_data" / "xiasha1_demand_packs.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["pack"]: int(item["n_vehicles"]) for item in payload["packs"]}


def _eps_min_map():
    values = dict(DEFAULT_EPS_MIN)
    if QUEUE.is_file():
        try:
            spec = json.loads(QUEUE.read_text(encoding="utf-8"))
            for name, meta in (spec.get("schedules") or {}).items():
                if "epsilon_min" in meta:
                    values[name] = float(meta["epsilon_min"])
        except (json.JSONDecodeError, OSError):
            pass
    return values


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return statistics.mean(values) if values else None


def _stdev(values):
    values = [float(value) for value in values if value is not None]
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _convergence_episode(evaluations, tolerance):
    """10-episode moving average stable for 20 evaluations within tolerance."""
    points = [(int(row["episode"]), float(row["travel_time"]))
              for row in evaluations
              if row.get("travel_time") is not None]
    if len(points) < 30:
        return None
    episodes = [item[0] for item in points]
    values = [item[1] for item in points]
    final_reference = statistics.mean(values[-20:])
    finite = [abs(value) for value in values if math.isfinite(value)]
    scale = statistics.median(finite) if finite else 0.0
    denominator = max(abs(final_reference), 0.01 * scale, 1e-8)
    moving = [
        None if index < 9 else statistics.mean(values[index - 9:index + 1])
        for index in range(len(values))
    ]
    for index in range(9, len(values) - 19):
        if all(
            moving[offset] is not None
            and abs(moving[offset] - final_reference) / denominator <= tolerance
            for offset in range(index, index + 20)
        ):
            return episodes[index]
    return None


def _first100_auc(evaluations):
    points = sorted(
        (int(row["episode"]), float(row["travel_time"]))
        for row in evaluations
        if row.get("travel_time") is not None and int(row["episode"]) <= 100
    )
    if len(points) < 2:
        return None
    area = 0.0
    for (episode_a, value_a), (episode_b, value_b) in zip(
            points, points[1:]):
        area += (episode_b - episode_a) * (value_a + value_b) / 2.0
    return area


def _epsilon_min_episode(train_rows, epsilon_min):
    for row in sorted(train_rows, key=lambda item: int(item["episode"])):
        epsilon = row.get("epsilon")
        if epsilon is not None and float(epsilon) <= epsilon_min + 1e-9:
            return int(row["episode"]), float(epsilon)
    return None, None


def _run_metrics(run_dir, loaded, epsilon_min):
    run_label = str(run_dir.relative_to(ROOT))
    status_path = run_dir / "run_status.json"
    if not status_path.is_file():
        return {"status": "missing", "run_dir": run_label}
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "已完成" or status.get("exit_code") != 0:
        return {
            "status": "incomplete",
            "run_dir": run_label,
            "run_status": status.get("status"),
            "exit_code": status.get("exit_code"),
        }
    records = _jsonl(run_dir / "metrics" / "records.jsonl")
    evaluations = sorted(
        (row for row in records
         if row.get("record_type") in {"EVALUATION", "FINAL_EVALUATION"}),
        key=lambda row: int(row["episode"]),
    )
    train_rows = [row for row in records if row.get("record_type") == "TRAIN"]
    if len(evaluations) < 20:
        return {
            "status": "short",
            "run_dir": run_label,
            "evaluation_rows": len(evaluations),
        }
    last = evaluations[-20:]
    final_tt = _mean(row.get("travel_time") for row in last)
    final_thr = _mean(row.get("throughput") for row in last)
    final_unf = _mean(row.get("unfinished_vehicles") for row in last)
    final_queue = _mean(row.get("queue") for row in last)
    final_switches = _mean(row.get("phase_switches") for row in last)
    action_shares = []
    for row in last:
        distribution = row.get("action_distribution") or {}
        if distribution:
            action_shares.append(max(float(value)
                                     for value in distribution.values()))
    uninserted = (
        None if final_thr is None or final_unf is None
        else loaded - final_thr - final_unf
    )
    eps_ep, eps_value = _epsilon_min_episode(train_rows, epsilon_min)
    summary = {}
    summary_path = run_dir / "evaluation" / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "status": "ok",
        "run_dir": run_label,
        "final_tt": final_tt,
        "final_throughput": final_thr,
        "final_unfinished": final_unf,
        "final_uninserted": uninserted,
        "final_queue": final_queue,
        "final_phase_switches": final_switches,
        "max_action_share": _mean(action_shares),
        "convergence_5pct": _convergence_episode(evaluations, 0.05),
        "convergence_3pct": _convergence_episode(evaluations, 0.03),
        "first100_auc": _first100_auc(evaluations),
        "first100_mean": _mean(
            row.get("travel_time") for row in evaluations
            if int(row["episode"]) <= 100),
        "epsilon_min_episode": eps_ep,
        "epsilon_at_min": eps_value,
        "best_episode": summary.get("best_episode"),
        "best_travel_time": summary.get("best_travel_time"),
        "evaluation_rows": len(evaluations),
        "train_rows": len(train_rows),
    }


def _collect():
    loaded = _loaded_map()
    eps_min = _eps_min_map()
    rows = []
    for pack in PACKS:
        for schedule, template in SCHEDULES.items():
            for seed in SEEDS:
                prefix = template.format(pack=pack, seed=seed)
                run_dir = (ROOT / "data" / "output_data" / "tsc" / "sumo_dqn"
                           / f"xiasha1_{pack}" / prefix)
                metrics = _run_metrics(run_dir, loaded[pack],
                                       eps_min[schedule])
                rows.append({
                    "pack": pack,
                    "schedule": schedule,
                    "seed": seed,
                    **metrics,
                })
    return rows


def _write_csv(rows, path):
    fields = [
        "pack", "schedule", "seed", "status", "run_dir",
        "final_tt", "final_throughput", "final_unfinished",
        "final_uninserted", "final_queue", "final_phase_switches",
        "max_action_share", "convergence_5pct", "convergence_3pct",
        "first100_auc", "first100_mean", "epsilon_min_episode",
        "epsilon_at_min", "best_episode", "best_travel_time",
        "evaluation_rows", "train_rows",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value, digits=2):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _group(rows, pack, schedule):
    return [row for row in rows
            if row["pack"] == pack and row["schedule"] == schedule
            and row["status"] == "ok"]


def _summary(rows):
    lines = [
        "# xiasha1 sufficient-exploration validation",
        "",
        "Native = `xi1on_dqn_*` (epsilon min ~ep4); "
        "slow = `xi1exp_dqn_slow_*` (epsilon min ~ep118); "
        "full = `xi1exp_dqn_full_*` (epsilon min ~ep180).",
        "All values are last-20 evaluation means unless noted; "
        "`uninserted = loaded - throughput - unfinished`.",
        "",
        "## Per pack / schedule",
        "",
        "| pack | schedule | n | final TT | throughput | unfinished | uninserted | queue | conv 5% ep | conv 3% ep | eps-min ep | first-100 AUC | max action share | best TT (ep) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pack in PACKS:
        for schedule in SCHEDULES:
            group = _group(rows, pack, schedule)
            conv5 = [row["convergence_5pct"] for row in group
                     if row["convergence_5pct"] is not None]
            conv3 = [row["convergence_3pct"] for row in group
                     if row["convergence_3pct"] is not None]
            eps_ep = [row["epsilon_min_episode"] for row in group
                      if row["epsilon_min_episode"] is not None]
            best_tt = [row["best_travel_time"] for row in group
                       if row["best_travel_time"] is not None]
            best_ep = [row["best_episode"] for row in group
                       if row["best_episode"] is not None]
            lines.append(
                "| {pack} | {schedule} | {n} | {tt} | {thr} | {unf} | {unins} | {queue} | {c5} | {c3} | {eps} | {auc} | {share} | {best} ({best_ep}) |".format(
                    pack=pack,
                    schedule=schedule,
                    n=len(group),
                    tt=_fmt(_mean(row["final_tt"] for row in group)),
                    thr=_fmt(_mean(row["final_throughput"] for row in group), 0),
                    unf=_fmt(_mean(row["final_unfinished"] for row in group), 1),
                    unins=_fmt(_mean(row["final_uninserted"] for row in group), 1),
                    queue=_fmt(_mean(row["final_queue"] for row in group)),
                    c5=_fmt(_mean(conv5), 1),
                    c3=_fmt(_mean(conv3), 1),
                    eps=_fmt(_mean(eps_ep), 1),
                    auc=_fmt(_mean(row["first100_auc"] for row in group), 1),
                    share=_fmt(_mean(row["max_action_share"] for row in group), 2),
                    best=_fmt(_mean(best_tt)),
                    best_ep=_fmt(_mean(best_ep), 0),
                )
            )

    lines += [
        "",
        "## Paired deltas vs native (same pack/seed)",
        "",
        "| schedule | pack | n pairs | ΔTT | Δthroughput | Δuninserted | Δconv 5% |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for schedule in ("slow", "full"):
        for pack in PACKS:
            pairs = []
            for seed in SEEDS:
                native = next(
                    (row for row in rows if row["pack"] == pack
                     and row["schedule"] == "native" and row["seed"] == seed
                     and row["status"] == "ok"), None)
                variant = next(
                    (row for row in rows if row["pack"] == pack
                     and row["schedule"] == schedule and row["seed"] == seed
                     and row["status"] == "ok"), None)
                if native and variant:
                    pairs.append((native, variant))
            if not pairs:
                lines.append(f"| {schedule} | {pack} | 0 | — | — | — | — |")
                continue
            def delta(key):
                native_mean = _mean(row[0][key] for row in pairs)
                variant_mean = _mean(row[1][key] for row in pairs)
                if native_mean is None or variant_mean is None:
                    return None
                return variant_mean - native_mean
            lines.append(
                "| {schedule} | {pack} | {n} | {tt} | {thr} | {unins} | {conv} |".format(
                    schedule=schedule,
                    pack=pack,
                    n=len(pairs),
                    tt=_fmt(delta("final_tt")),
                    thr=_fmt(delta("final_throughput"), 0),
                    unins=_fmt(delta("final_uninserted"), 1),
                    conv=_fmt(delta("convergence_5pct"), 1),
                )
            )

    lines += [
        "",
        "## Early-convergence check",
        "",
        "| schedule | n converged | median conv 5% ep | median eps-min ep | converged before eps-min | converged before 50% of eps-min ep |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for schedule in SCHEDULES:
        group = [row for row in rows if row["schedule"] == schedule
                 and row["status"] == "ok"]
        conv = [row["convergence_5pct"] for row in group
                if row["convergence_5pct"] is not None]
        eps = [row["epsilon_min_episode"] for row in group
               if row["epsilon_min_episode"] is not None]
        before = sum(
            1 for row in group
            if row["convergence_5pct"] is not None
            and row["epsilon_min_episode"] is not None
            and row["convergence_5pct"] < row["epsilon_min_episode"])
        before_half = sum(
            1 for row in group
            if row["convergence_5pct"] is not None
            and row["epsilon_min_episode"] is not None
            and row["convergence_5pct"] < 0.5 * row["epsilon_min_episode"])
        lines.append(
            "| {schedule} | {n} / {total} | {conv} | {eps} | {before} | {before_half} |".format(
                schedule=schedule,
                n=len(conv),
                total=len(group),
                conv=_fmt(statistics.median(conv) if conv else None, 1),
                eps=_fmt(statistics.median(eps) if eps else None, 1),
                before=before,
                before_half=before_half,
            )
        )

    lines += [
        "",
        "## Per-seed paired comparison",
        "",
        "| pack | seed | native TT | slow TT | full TT | native unins | slow unins | full unins | native conv | slow conv | full conv |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pack in PACKS:
        for seed in SEEDS:
            values = {}
            for schedule in SCHEDULES:
                values[schedule] = next(
                    (row for row in rows if row["pack"] == pack
                     and row["schedule"] == schedule and row["seed"] == seed),
                    None)
            def cell(schedule, key, digits=2):
                row = values[schedule]
                if not row or row.get("status") != "ok":
                    return "—"
                return _fmt(row.get(key), digits)
            lines.append(
                "| {pack} | {seed} | {nt} | {st} | {ft} | {nu} | {su} | {fu} | {nc} | {sc} | {fc} |".format(
                    pack=pack,
                    seed=seed,
                    nt=cell("native", "final_tt"),
                    st=cell("slow", "final_tt"),
                    ft=cell("full", "final_tt"),
                    nu=cell("native", "final_uninserted", 1),
                    su=cell("slow", "final_uninserted", 1),
                    fu=cell("full", "final_uninserted", 1),
                    nc=cell("native", "convergence_5pct", 1),
                    sc=cell("slow", "convergence_5pct", 1),
                    fc=cell("full", "convergence_5pct", 1),
                )
            )

    missing = [row for row in rows if row["status"] != "ok"]
    if missing:
        lines += ["", "## Missing / incomplete", ""]
        for row in missing:
            lines.append(
                f"- `{row['pack']}` {row['schedule']} seed {row['seed']}: "
                f"{row['status']} ({row['run_dir']})")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    output = Path(args.output_dir)
    rows = _collect()
    _write_csv(rows, output / "pilot_metrics.csv")
    summary = _summary(rows)
    (output / "pilot_summary.md").write_text(summary + "\n",
                                            encoding="utf-8")
    print(summary)
    print(f"\nwrote {output / 'pilot_metrics.csv'}")
    print(f"wrote {output / 'pilot_summary.md'}")


if __name__ == "__main__":
    main()
