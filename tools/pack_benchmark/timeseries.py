"""Within-episode replay collection and plotting.

The canonical performance table is end-of-episode.  This module adds a
replay with `trainer.eval_step_log: true` so the same controller/config is
sampled inside the episode.  The replay is validated against the canonical
cell before its curves are plotted.
"""
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import __version__
from .collect import collect_dataset
from .metrics import METRICS, get_metric
from .plot import _arm_label, _arm_order, _pack_order, _save, load_dataset
from .spec import REPO_ROOT, run_dir_for, sha256_file


TS_METRICS = (
    ("queue_veh", "Queue (in-lane waiting vehicles)", "veh",
     "Instantaneous sum of lane_waiting_count on the controlled approach lanes."),
    ("throughput_arrived_veh", "Cumulative arrived vehicles", "veh",
     "Number of vehicles that have arrived by this simulation time."),
    ("backlog_total_veh", "Total backlog (uninserted + in-net unfinished)", "veh",
     "Uninserted loaded demand plus departed-but-not-arrived vehicles."),
    ("att_arrived_s", "Cumulative mean travel time (arrived)", "s",
     "Running mean depart-to-arrive duration of vehicles arrived so far."),
    ("waiting_end_in_net_s", "Mean accumulated waiting of in-net vehicles", "s",
     "Instantaneous mean accumulated waiting over vehicles currently in the network."),
)
TS_FIELDS = (
    "benchmark_id", "analysis_id", "pack_id", "arm_id", "control_semantics",
    "sim_time_s", "queue_veh", "throughput_arrived_veh",
    "unfinished_departed_veh", "inserted_veh", "uninserted_loaded_veh",
    "backlog_total_veh", "att_arrived_s", "waiting_end_in_net_s",
)


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _short(agent):
    return {"fixedtime": "ft", "sotl": "sotl", "maxpressure": "mp"}[agent]


def _env():
    env = {
        key: os.environ[key]
        for key in ("LD_PRELOAD", "SUMO_HOME", "CONDA_PREFIX")
        if key in os.environ
    }
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "lib" / "libstdc++.so.6"
        if candidate.is_file():
            env["LD_PRELOAD"] = str(candidate)
    env.update({
        "CUDA_VISIBLE_DEVICES": "-1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    })
    return env


def write_replay_queue(spec, queue_path, overlay_path):
    overlay_path = Path(overlay_path).resolve()
    if not overlay_path.is_file():
        raise FileNotFoundError(f"Replay overlay does not exist: {overlay_path}")
    tasks = []
    for pack in spec["_packs"]:
        for arm in spec["_arms"]:
            if arm["kind"] != "run_dir":
                continue
            run_id = f"xi1ts_{_short(arm['agent'])}_{pack['pack_id']}"
            output_path = (
                REPO_ROOT / "data" / "output_data" / "tsc"
                / f"sumo_{arm['agent']}" / pack["network"] / run_id
            )
            command = [
                sys.executable, "run.py", "-w", "sumo",
                "-a", arm["agent"], "-n", pack["network"],
                "--seed", str(spec["_seed"]), "--interface", "libsumo",
                "--prefix", run_id,
                "--experiment-config", str(overlay_path),
            ]
            tasks.append({
                "run_id": run_id,
                "command": command,
                "cwd": str(REPO_ROOT),
                "env": _env(),
                "output_path": str(output_path),
                "completion": {
                    "path": str(output_path / "run_status.json"),
                    "json_field": "status",
                    "equals": "已完成",
                },
            })
    payload = {
        "schema_version": 1,
        "plan": "xiasha1_notrain_timeseries_v1",
        "expected_tasks": len(tasks),
        "note": (
            "Replay of the 36 rule-controller runs with 1 s eval_step_log; "
            "same controller/config/seed, step logging only."
        ),
        "tasks": tasks,
    }
    queue_path = Path(queue_path)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return queue_path


def _read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_step_log(run_dir):
    files = sorted((Path(run_dir) / "evaluation").glob("steps_ep*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No evaluation/steps_ep*.jsonl under {run_dir}")
    return files[-1], _read_jsonl(files[-1])


def _read_orig_samples(raw_dir):
    path = Path(raw_dir) / "queue_samples.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing orig samples: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_reference(reference_dataset):
    dataset = load_dataset(reference_dataset)
    return {(cell["pack_id"], cell["arm_id"]): cell for cell in dataset["cells"]}


def _compare_replay(replay_dataset, reference_cells):
    mismatches = []
    checked = 0
    fields = (
        "att_arrived_s", "throughput_arrived_veh", "unfinished_departed_veh",
        "queue_decision_mean_veh", "waiting_end_in_net_s",
        "real_delay_world_s", "lane_delay_index", "phase_switches",
    )
    for cell in replay_dataset["cells"]:
        if cell["source_kind"] != "run_dir":
            continue
        reference = reference_cells.get((cell["pack_id"], cell["arm_id"]))
        if reference is None:
            mismatches.append({
                "pack_id": cell["pack_id"], "arm_id": cell["arm_id"],
                "field": "cell", "reference": None,
                "replay": "present",
            })
            continue
        for field in fields:
            checked += 1
            reference_value = (reference.get("metrics") or {}).get(field)
            replay_value = (cell.get("metrics") or {}).get(field)
            if reference_value is None and replay_value is None:
                continue
            if isinstance(reference_value, int) and isinstance(replay_value, int):
                equal = reference_value == replay_value
            else:
                try:
                    equal = abs(float(reference_value) - float(replay_value)) <= 1e-6
                except (TypeError, ValueError):
                    equal = reference_value == replay_value
            if not equal:
                mismatches.append({
                    "pack_id": cell["pack_id"], "arm_id": cell["arm_id"],
                    "field": field, "reference": reference_value,
                    "replay": replay_value,
                })
    return {"checked_values": checked, "mismatch_count": len(mismatches),
            "mismatches": mismatches}


def collect_timeseries(spec, output_dir, reference_dataset, orig_mode="rerun",
                       sumo_binary=None):
    output_dir = Path(output_dir)
    reference_cells = _load_reference(reference_dataset)
    replay_dataset, output_dir = collect_dataset(
        spec, output_dir, orig_mode=orig_mode, sumo_binary=sumo_binary
    )
    replay_validation = _compare_replay(replay_dataset, reference_cells)
    timeseries_path, rows, step_files = _timeseries_table(
        replay_dataset, output_dir
    )
    manifest = {
        "schema_version": 1,
        "tool": {"name": "tools.pack_benchmark", "version": __version__},
        "benchmark_id": replay_dataset["benchmark_id"],
        "analysis_id": replay_dataset["analysis_id"],
        "created_at_utc": utc_now(),
        "reference_dataset": {
            "path": str(Path(reference_dataset).resolve().relative_to(REPO_ROOT)),
            "sha256": sha256_file(reference_dataset),
        },
        "replay_dataset": {
            "path": "benchmark.json",
            "sha256": sha256_file(output_dir / "benchmark.json"),
        },
        "replay_validation": replay_validation,
        "rows": len(rows),
        "step_log_files": step_files,
        "resolution": {
            "run_dir": "1 s (eval_step_log_interval=1)",
            "direct_sumo": "10 s (decision samples)",
        },
    }
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests" / "timeseries_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "timeseries_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return replay_dataset, timeseries_path, manifest


def _timeseries_table(replay_dataset, output_dir):
    """Assemble timeseries_long.csv from per-cell step logs.

    Shared by collect_timeseries and collect_ckpt_timeseries; returns the
    CSV path, the row list and the per-cell step-log file map.
    """
    output_dir = Path(output_dir)
    loaded_by_pack = {
        pack["pack_id"]: pack.get("loaded") for pack in replay_dataset["packs"]
    }
    rows = []
    step_files = {}
    for cell in replay_dataset["cells"]:
        pack_id = cell["pack_id"]
        arm_id = cell["arm_id"]
        loaded = loaded_by_pack.get(pack_id)
        if loaded is None:
            loaded = (cell.get("metrics") or {}).get("loaded")
        if cell["source_kind"] == "run_dir":
            run_dir = REPO_ROOT / cell["run_dir"]
            step_path, records = _read_step_log(run_dir)
            step_files[f"{pack_id}__{arm_id}"] = str(step_path.relative_to(REPO_ROOT))
            for record in records:
                inserted = record.get("inserted_vehicles")
                unfinished = record.get("unfinished_vehicles")
                uninserted = (
                    loaded - inserted
                    if loaded is not None and inserted is not None else None
                )
                backlog = (
                    unfinished + uninserted
                    if unfinished is not None and uninserted is not None else None
                )
                rows.append({
                    "benchmark_id": replay_dataset["benchmark_id"],
                    "analysis_id": replay_dataset["analysis_id"],
                    "pack_id": pack_id,
                    "arm_id": arm_id,
                    "control_semantics": cell["control_semantics"],
                    "sim_time_s": record.get("sim_time"),
                    "queue_veh": record.get("queue"),
                    "throughput_arrived_veh": record.get("throughput"),
                    "unfinished_departed_veh": unfinished,
                    "inserted_veh": inserted,
                    "uninserted_loaded_veh": uninserted,
                    "backlog_total_veh": backlog,
                    "att_arrived_s": record.get("travel_time"),
                    "waiting_end_in_net_s": record.get("waiting_time"),
                })
        else:
            raw_dir = output_dir / "raw" / "orig_prog" / pack_id
            for record in _read_orig_samples(raw_dir):
                inserted = int(float(record["inserted_veh"]))
                unfinished = int(float(record["unfinished_departed_veh"]))
                uninserted = loaded - inserted if loaded is not None else None
                backlog = (
                    unfinished + uninserted if uninserted is not None else None
                )
                rows.append({
                    "benchmark_id": replay_dataset["benchmark_id"],
                    "analysis_id": replay_dataset["analysis_id"],
                    "pack_id": pack_id,
                    "arm_id": arm_id,
                    "control_semantics": cell["control_semantics"],
                    "sim_time_s": float(record["time_s"]),
                    "queue_veh": float(record["queue_veh"]),
                    "throughput_arrived_veh": int(float(record["throughput_arrived_veh"])),
                    "unfinished_departed_veh": unfinished,
                    "inserted_veh": inserted,
                    "uninserted_loaded_veh": uninserted,
                    "backlog_total_veh": backlog,
                    "att_arrived_s": (
                        float(record["att_arrived_s"])
                        if record.get("att_arrived_s") not in (None, "") else None
                    ),
                    "waiting_end_in_net_s": (
                        float(record["waiting_end_in_net_s"])
                        if record.get("waiting_end_in_net_s") not in (None, "") else None
                    ),
                })
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    timeseries_path = tables / "timeseries_long.csv"
    with timeseries_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return timeseries_path, rows, step_files


def _read_timeseries(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_cell = {}
    for row in rows:
        key = (row["pack_id"], row["arm_id"])
        by_cell.setdefault(key, []).append(row)
    return by_cell


def _float(value):
    if value in (None, ""):
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def plot_timeseries(dataset, output_dir, dpi=160):
    output_dir = Path(output_dir)
    timeseries_path = output_dir / "tables" / "timeseries_long.csv"
    by_cell = _read_timeseries(timeseries_path)
    packs = _pack_order(dataset)
    arms = _arm_order(dataset)
    roles = {arm["arm_id"]: arm.get("role") for arm in dataset["arms"]}
    ts_palette = (
        "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00",
        "#F0E442", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    )
    colors = {
        arm: ts_palette[index % len(ts_palette)] for index, arm in enumerate(arms)
    }
    figures = []
    for figure_index, (metric_id, label, unit, definition) in enumerate(TS_METRICS, 6):
        fig, axes = plt.subplots(4, 3, figsize=(18, 14), constrained_layout=True)
        flat = axes.flat
        for axis, pack in zip(flat, packs):
            for arm in arms:
                rows = by_cell.get((pack, arm), [])
                if not rows:
                    continue
                selected = [
                    row for row in rows
                    if int(round(_float(row["sim_time_s"]))) % 10 == 0
                ]
                times = np.array([_float(row["sim_time_s"]) / 60.0 for row in selected])
                values = np.array([_float(row[metric_id]) for row in selected])
                valid = np.isfinite(times) & np.isfinite(values)
                if not valid.any():
                    continue
                reference = roles.get(arm) == "reference"
                axis.plot(
                    times[valid], values[valid],
                    linewidth=0.9 if reference else 1.1,
                    linestyle="--" if reference else "-",
                    alpha=0.85 if reference else 1.0,
                    color=colors[arm], label=_arm_label(dataset, arm),
                )
            axis.set_title(pack, fontsize=9)
            axis.set_xlabel("simulation time (min)", fontsize=8)
            axis.set_ylabel(f"{label} ({unit})", fontsize=8)
            axis.grid(alpha=0.2)
            axis.tick_params(labelsize=7)
        for axis in list(flat)[len(packs):]:
            axis.set_visible(False)
        handles, labels = flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels, loc="lower center", ncol=4, fontsize=8,
                bbox_to_anchor=(0.5, -0.015),
            )
        fig.suptitle(f"Within-episode performance: {label}", fontsize=13)
        figure_id = f"fig{figure_index:02d}_timeseries_{metric_id}"
        paths = _save(fig, output_dir / "figures" / figure_id, dpi)
        figures.append({
            "figure_id": figure_id,
            "title": f"Within-episode {label}",
            "metric_ids": [metric_id],
            "paths": [str(path.relative_to(output_dir)) for path in paths],
            "caption": (
                f"## {figure_id}: within-episode {label}\n\n"
                f"Replay of the same controller/config/seed; run_dir curves are "
                f"sampled at 1 s and plotted every 10 s; orig is sampled at 10 s. "
                f"Metric: {label} ({unit}). {definition}\n"
            ),
        })
    captions = "\n".join(figure["caption"] for figure in figures)
    (output_dir / "captions_timeseries.md").write_text(captions, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "benchmark_id": dataset["benchmark_id"],
        "analysis_id": dataset["analysis_id"],
        "dpi": dpi,
        "figures": figures,
    }
    (output_dir / "figures_timeseries_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


# ---------------------------------------------------------------------------
# Focus views: learned-arm zoom, delta-vs-maxpressure, discriminability summary
# ---------------------------------------------------------------------------

SMOOTH_WINDOWS = {"queue_veh": 6}  # 6 x 10 s-decimated samples = 60 s mean

SUMMARY_METRICS = (
    ("att_arrived_s", "Cumulative mean travel time (arrived)", "s"),
    ("queue_p90_veh", "Queue P90 (10 s decision samples)", "veh"),
    ("waiting_end_in_net_s", "Mean accumulated waiting of in-net vehicles", "s"),
    ("uninserted_loaded_veh", "Uninserted loaded demand", "veh"),
)


def _rolling_mean(values, window):
    n = len(values)
    half = window // 2
    out = np.empty(n)
    for i in range(n):
        out[i] = values[max(0, i - half):min(n, i + half + 1)].mean()
    return out


def _decimated_series(rows, metric_id, smooth=True):
    """10 s-decimated (sim_time_min, value) series for one cell's step rows."""
    selected = [
        row for row in rows
        if int(round(_float(row["sim_time_s"]))) % 10 == 0
    ]
    times = np.array([_float(row["sim_time_s"]) / 60.0 for row in selected])
    values = np.array([_float(row[metric_id]) for row in selected])
    valid = np.isfinite(times) & np.isfinite(values)
    times, values = times[valid], values[valid]
    if smooth and metric_id in SMOOTH_WINDOWS and len(values):
        values = _rolling_mean(values, SMOOTH_WINDOWS[metric_id])
    return times, values


def plot_timeseries_focus(dataset, output_dir, dpi=160):
    """Zoomed learned-band curves + delta-vs-maxpressure + summary heatmaps.

    The plain timeseries figures share one wide y-range with the orig/rule
    references, which visually compresses the learned arms.  These views
    re-render the same timeseries_long.csv with (a) per-subplot zoom on the
    learned-arm band, (b) per-pack delta against the maxpressure curve, and
    (c) a pack x arm heatmap of the metrics that actually discriminate arms.
    """
    output_dir = Path(output_dir)
    timeseries_path = output_dir / "tables" / "timeseries_long.csv"
    by_cell = _read_timeseries(timeseries_path)
    packs = _pack_order(dataset)
    arms = _arm_order(dataset)
    roles = {arm["arm_id"]: arm.get("role") for arm in dataset["arms"]}
    learned = [arm for arm in arms if roles.get(arm) != "reference"]
    reference = [arm for arm in arms if roles.get(arm) == "reference"]
    palette = (
        "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00",
        "#F0E442", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    )
    colors = {
        arm: palette[index % len(palette)] for index, arm in enumerate(arms)
    }

    def series(pack, arm, metric_id):
        return _decimated_series(by_cell.get((pack, arm), []), metric_id)

    figures = []

    # -- (a) zoomed curves: reference arms clipped, learned band decides ylim
    for figure_index, (metric_id, label, unit, definition) in enumerate(
            TS_METRICS, 11):
        fig, axes = plt.subplots(4, 3, figsize=(18, 14), constrained_layout=True)
        flat = axes.flat
        for axis, pack in zip(flat, packs):
            y_hi, y_lo = -np.inf, np.inf
            for arm in reference + learned:
                times, values = series(pack, arm, metric_id)
                if not len(times):
                    continue
                is_ref = arm in reference
                axis.plot(
                    times, values,
                    linewidth=0.9 if is_ref else 1.2,
                    linestyle="--" if is_ref else "-",
                    alpha=0.55 if is_ref else 1.0,
                    color=colors[arm], label=_arm_label(dataset, arm),
                    zorder=1 if is_ref else 2,
                )
                if not is_ref:
                    y_hi = max(y_hi, float(values.max()))
                    y_lo = min(y_lo, float(values.min()))
            if np.isfinite(y_hi):
                span = max(y_hi - y_lo, 1e-6)
                axis.set_ylim(min(0.0, y_lo - 0.05 * span), y_hi + 0.08 * span)
            axis.set_title(pack, fontsize=9)
            axis.set_xlabel("simulation time (min)", fontsize=8)
            ylabel = label + (" (60 s mean)" if metric_id in SMOOTH_WINDOWS else "")
            axis.set_ylabel(f"{ylabel} ({unit})", fontsize=8)
            axis.grid(alpha=0.2)
            axis.tick_params(labelsize=7)
        for axis in list(flat)[len(packs):]:
            axis.set_visible(False)
        handles, labels = flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
                       bbox_to_anchor=(0.5, -0.015))
        fig.suptitle(
            f"Within-episode (zoomed to learned band): {label}", fontsize=13
        )
        figure_id = f"fig{figure_index:02d}_zoom_{metric_id}"
        paths = _save(fig, output_dir / "figures" / figure_id, dpi)
        figures.append({
            "figure_id": figure_id,
            "title": f"Zoomed within-episode {label}",
            "metric_ids": [metric_id],
            "paths": [str(path.relative_to(output_dir)) for path in paths],
            "caption": (
                f"## {figure_id}: zoomed within-episode {label}\n\n"
                f"Y-range is set by the learned arms; reference arms are "
                f"dashed and may clip outside the band. Queue is a 60 s "
                f"rolling mean of the 1 s step log. {definition}\n"
            ),
        })

    # -- (b) delta vs maxpressure: learned arms only, y=0 is the MP curve
    for figure_index, (metric_id, label, unit, definition) in enumerate(
            TS_METRICS, 21):
        fig, axes = plt.subplots(4, 3, figsize=(18, 14), constrained_layout=True)
        flat = axes.flat
        for axis, pack in zip(flat, packs):
            ref_times, ref_values = series(pack, "maxpressure", metric_id)
            ref_map = dict(zip(np.round(ref_times, 6), ref_values))
            y_lim = 0.0
            for arm in learned:
                times, values = series(pack, arm, metric_id)
                keys = np.round(times, 6)
                common = np.array([key in ref_map for key in keys], dtype=bool)
                if not common.any():
                    continue
                delta = values[common] - np.array(
                    [ref_map[key] for key in keys[common]]
                )
                axis.plot(times[common], delta, linewidth=1.2,
                          color=colors[arm], label=_arm_label(dataset, arm))
                y_lim = max(y_lim, float(np.abs(delta).max()))
            axis.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
            if y_lim > 0:
                axis.set_ylim(-0.15 * y_lim, 1.08 * y_lim)
            axis.set_title(pack, fontsize=9)
            axis.set_xlabel("simulation time (min)", fontsize=8)
            ylabel = label + (" (60 s mean)" if metric_id in SMOOTH_WINDOWS else "")
            axis.set_ylabel(f"Δ {ylabel} vs maxpressure ({unit})", fontsize=8)
            axis.grid(alpha=0.2)
            axis.tick_params(labelsize=7)
        for axis in list(flat)[len(packs):]:
            axis.set_visible(False)
        handles, labels = flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
                       bbox_to_anchor=(0.5, -0.015))
        fig.suptitle(
            f"Within-episode delta vs maxpressure: {label}", fontsize=13
        )
        figure_id = f"fig{figure_index:02d}_delta_mp_{metric_id}"
        paths = _save(fig, output_dir / "figures" / figure_id, dpi)
        figures.append({
            "figure_id": figure_id,
            "title": f"Delta vs maxpressure: {label}",
            "metric_ids": [metric_id],
            "paths": [str(path.relative_to(output_dir)) for path in paths],
            "caption": (
                f"## {figure_id}: within-episode Δ {label} vs maxpressure\n\n"
                f"Each learned arm's curve minus the maxpressure curve of "
                f"the same pack; y=0 is maxpressure itself. Negative is "
                f"better for ATT/queue/backlog/waiting. {definition}\n"
            ),
        })

    # -- (c) discriminability summary: pack x arm heatmaps of endpoint/P90 metrics
    cells = {(cell["pack_id"], cell["arm_id"]): cell for cell in dataset["cells"]}
    value_by_metric = {}
    for metric_id, label, unit in SUMMARY_METRICS:
        grid = np.full((len(packs), len(learned)), np.nan)
        for pi, pack in enumerate(packs):
            for ai, arm in enumerate(learned):
                if metric_id == "queue_p90_veh":
                    _, values = _decimated_series(
                        by_cell.get((pack, arm), []), "queue_veh",
                        smooth=False,
                    )
                    grid[pi, ai] = (
                        float(np.percentile(values, 90)) if len(values)
                        else np.nan
                    )
                else:
                    cell = cells.get((pack, arm)) or {}
                    value = (cell.get("metrics") or {}).get(metric_id)
                    grid[pi, ai] = (
                        float(value) if value is not None else np.nan
                    )
        value_by_metric[metric_id] = grid

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    for axis, (metric_id, label, unit) in zip(axes.flat, SUMMARY_METRICS):
        grid = value_by_metric[metric_id]
        image = axis.imshow(grid, aspect="auto", cmap="RdYlGn_r")
        axis.set_xticks(range(len(learned)))
        axis.set_xticklabels(learned, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(range(len(packs)))
        axis.set_yticklabels(packs, fontsize=8)
        axis.set_title(f"{label} ({unit})", fontsize=10)
        finite = grid[np.isfinite(grid)]
        fmt = ".0f" if finite.size and finite.max() >= 100 else ".1f"
        for pi in range(len(packs)):
            for ai in range(len(learned)):
                if np.isfinite(grid[pi, ai]):
                    axis.text(ai, pi, format(grid[pi, ai], fmt),
                              ha="center", va="center", fontsize=6)
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.suptitle(
        "Learned-arm discriminability (endpoint / decision-level metrics)",
        fontsize=13,
    )
    figure_id = "fig31_discrim_summary"
    paths = _save(fig, output_dir / "figures" / figure_id, dpi)
    figures.append({
        "figure_id": figure_id,
        "title": "Learned-arm discriminability summary",
        "metric_ids": [metric[0] for metric in SUMMARY_METRICS],
        "paths": [str(path.relative_to(output_dir)) for path in paths],
        "caption": (
            "## fig31: pack x arm heatmaps\n\n"
            "Cells are the endpoint value (att / waiting / uninserted from "
            "the canonical cells) or the decision-level queue P90 computed "
            "from the 10 s-decimated step series. Redder = worse.\n"
        ),
    })

    captions = "\n".join(figure["caption"] for figure in figures)
    (output_dir / "captions_timeseries_focus.md").write_text(
        captions, encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "benchmark_id": dataset["benchmark_id"],
        "analysis_id": dataset["analysis_id"],
        "dpi": dpi,
        "figures": figures,
    }
    (output_dir / "figures_timeseries_focus_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
