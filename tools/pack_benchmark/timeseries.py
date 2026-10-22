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
