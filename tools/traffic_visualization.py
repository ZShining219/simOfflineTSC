"""Simple, unified dynamic comparison of three SUMO controllers.

The utility consumes decision-level decisions.jsonl files produced by the
existing evaluation runners. It does not start SUMO, alter experiment data,
smooth values, or choose checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "simofflinetsc-matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter


METHODS = ("online_dqn", "hadhoa", "fixedtime")
METHOD_LABELS = {
    "online_dqn": "Online DQN",
    "hadhoa": "HADHOA",
    "fixedtime": "FixedTime",
}
METHOD_COLORS = {
    "online_dqn": "#1f77b4",
    "hadhoa": "#d62728",
    "fixedtime": "#2ca02c",
}
REQUIRED_FIELDS = ("record_type", "network", "decision_step", "simulation_time_seconds")


def _records_path(value):
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "decisions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Decision record file does not exist: {path}")
    return path


def load_decisions(value, controller_id=None, evaluation_seed=None, training_seed=None):
    """Load and validate one decision-level JSONL stream."""
    path = _records_path(value)
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank decision record at {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            missing = [field for field in REQUIRED_FIELDS if field not in record]
            if missing:
                raise ValueError(
                    f"{path}:{line_number} is missing required field(s): {', '.join(missing)}"
                )
            if record["record_type"] != "DECISION_METRICS":
                raise ValueError(f"{path}:{line_number} is not a DECISION_METRICS record")
            records.append(record)
    if controller_id is not None:
        records = [record for record in records if record.get("controller_id") == controller_id]
    if evaluation_seed is not None:
        records = [record for record in records if record.get("evaluation_seed") == evaluation_seed]
    if training_seed is not None:
        records = [record for record in records if record.get("training_seed") == training_seed]
    if not records:
        filters = {
            "controller_id": controller_id,
            "evaluation_seed": evaluation_seed,
            "training_seed": training_seed,
        }
        raise ValueError(f"No decision records matched filters for {path}: {filters}")
    if not records:
        raise ValueError(f"Decision record file is empty: {path}")
    return path, records


def _metric(record, field):
    if field in record:
        return float(record[field])
    if field == "queue_network_sum" and "queue_lanes" in record:
        return float(sum(float(value) for value in record["queue_lanes"].values()))
    if field == "queue_network_sum" and "queue" in record:
        return float(record["queue"])
    if field == "throughput_cumulative" and "throughput" in record:
        return float(record["throughput"])
    raise ValueError(f"Decision record has no supported metric field: {field}")


def _normalise(records, method, scene, start, end):
    rows = []
    for record in records:
        if str(record["network"]) != scene:
            raise ValueError(
                f"{method} record network {record['network']!r} does not match scene {scene!r}"
            )
        time_seconds = float(record["simulation_time_seconds"])
        if not math.isfinite(time_seconds):
            raise ValueError(f"{method} has a non-finite simulation time")
        if start <= time_seconds <= end:
            rows.append({
                "method": method,
                "decision_step": int(record["decision_step"]),
                "simulation_time_seconds": time_seconds,
                "queue_network_sum": _metric(record, "queue_network_sum"),
                "throughput_cumulative": _metric(record, "throughput_cumulative"),
            })
    rows.sort(key=lambda row: (row["simulation_time_seconds"], row["decision_step"]))
    if not rows:
        raise ValueError(f"{method} has no records in [{start}, {end}] seconds")
    times = [row["simulation_time_seconds"] for row in rows]
    if len(set(times)) != len(times):
        raise ValueError(f"{method} contains duplicate timestamps in the selected window")
    return rows


def align_three_methods(inputs, scene, start=0.0, end=None, selectors=None):
    """Align three streams on their common timestamps without interpolation."""
    if set(inputs) != set(METHODS):
        raise ValueError(f"inputs must contain exactly {METHODS}")
    if start < 0 or (end is not None and end <= start):
        raise ValueError("Require 0 <= start < end")
    selectors = selectors or {}
    loaded = {
        method: load_decisions(inputs[method], **selectors.get(method, {}))
        for method in METHODS
    }
    if end is None:
        end = min(
            max(float(record["simulation_time_seconds"]) for record in records)
            for _, records in loaded.values()
        )
    prepared = {
        method: _normalise(records, method, scene, start, end)
        for method, (_, records) in loaded.items()
    }
    time_sets = {
        method: {row["simulation_time_seconds"] for row in rows}
        for method, rows in prepared.items()
    }
    common_times = sorted(set.intersection(*time_sets.values()))
    if len(common_times) < 2:
        raise ValueError("At least two common simulation timestamps are required")
    by_time = {
        method: {row["simulation_time_seconds"]: row for row in rows}
        for method, rows in prepared.items()
    }
    aligned = []
    for time_seconds in common_times:
        row = {"simulation_time_seconds": time_seconds}
        for method in METHODS:
            source = by_time[method][time_seconds]
            row[f"{method}_queue_network_sum"] = source["queue_network_sum"]
            row[f"{method}_throughput_cumulative"] = source["throughput_cumulative"]
        aligned.append(row)
    return {method: loaded[method][0] for method in METHODS}, aligned


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_comparison(rows, output, scene, mode="line", fps=10, dpi=120):
    """Render line, bar, or both forms of a two-panel comparison."""
    if mode not in {"line", "bar", "both"}:
        raise ValueError("mode must be line, bar, or both")
    output_dir = Path(output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "aligned_metrics.csv"
    _write_csv(csv_path, rows)
    times = [row["simulation_time_seconds"] for row in rows]
    outputs = [str(csv_path)]

    def render(kind):
        figure, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
        queue_axis, throughput_axis = axes
        queue_axis.set_ylabel("Queued vehicles")
        throughput_axis.set_ylabel("Completed vehicles")
        throughput_axis.set_xlabel("Simulation time (s)")
        queue_axis.set_title(f"{scene}: three-controller dynamic comparison")
        queue_axis.grid(alpha=0.2)
        throughput_axis.grid(alpha=0.2)
        queue_axis.set_xlim(times[0], times[-1])
        throughput_axis.set_xlim(times[0], times[-1])
        queue_max = max(row[f"{method}_queue_network_sum"] for row in rows for method in METHODS)
        throughput_max = max(row[f"{method}_throughput_cumulative"] for row in rows for method in METHODS)
        queue_axis.set_ylim(0, max(1.0, queue_max * 1.08))
        throughput_axis.set_ylim(0, max(1.0, throughput_max * 1.08))
        lines = {}
        positions = list(range(len(METHODS)))
        for method in METHODS:
            label = METHOD_LABELS[method]
            color = METHOD_COLORS[method]
            if kind == "line":
                lines[method] = (
                    queue_axis.plot([], [], color=color, label=label, linewidth=2)[0],
                    throughput_axis.plot([], [], color=color, label=label, linewidth=2)[0],
                )
        if kind == "bar":
            queue_bars = queue_axis.bar(
                positions, [0] * len(METHODS),
                color=[METHOD_COLORS[method] for method in METHODS], alpha=0.8,
            )
            throughput_bars = throughput_axis.bar(
                positions, [0] * len(METHODS),
                color=[METHOD_COLORS[method] for method in METHODS], alpha=0.8,
            )
            queue_axis.legend(
                queue_bars, [METHOD_LABELS[method] for method in METHODS],
                loc="upper left",
            )
        if kind == "line":
            queue_axis.legend(loc="upper left")

        def update(index):
            current_time = times[index]
            if kind == "line":
                current_times = times[:index + 1]
                for method, (queue_line, throughput_line) in lines.items():
                    queue_line.set_data(
                        current_times,
                        [row[f"{method}_queue_network_sum"] for row in rows[:index + 1]],
                    )
                    throughput_line.set_data(
                        current_times,
                        [row[f"{method}_throughput_cumulative"] for row in rows[:index + 1]],
                    )
                artists = [line for pair in lines.values() for line in pair]
            else:
                for method_index, method in enumerate(METHODS):
                    queue_bars[method_index].set_height(
                        rows[index][f"{method}_queue_network_sum"]
                    )
                    throughput_bars[method_index].set_height(
                        rows[index][f"{method}_throughput_cumulative"]
                    )
                labels = [METHOD_LABELS[method] for method in METHODS]
                queue_axis.set_xticks(positions, labels)
                throughput_axis.set_xticks(positions, labels)
                artists = list(queue_bars) + list(throughput_bars)
            figure.suptitle(f"{scene} | simulation time = {current_time:.0f} s")
            return artists

        animation = FuncAnimation(
            figure, update, frames=len(rows), interval=1000 / fps, blit=False,
        )
        target = output_dir / f"comparison_{kind}.mp4"
        try:
            animation.save(target, writer=FFMpegWriter(fps=fps), dpi=dpi)
        except (FileNotFoundError, RuntimeError) as error:
            target = output_dir / f"comparison_{kind}.gif"
            animation.save(target, writer=PillowWriter(fps=fps), dpi=dpi)
            (output_dir / f"comparison_{kind}.fallback.json").write_text(
                json.dumps({"reason": str(error), "fallback": str(target)}, indent=2) + "\n",
                encoding="utf-8",
            )
        plt.close(figure)
        return target

    for kind in (("line", "bar") if mode == "both" else (mode,)):
        outputs.append(str(render(kind)))
    manifest = {
        "schema_version": 1,
        "scene": scene,
        "methods": list(METHODS),
        "method_labels": METHOD_LABELS,
        "metric_semantics": {
            "queue_network_sum": "queued vehicles summed over recorded lanes",
            "throughput_cumulative": "cumulative completed vehicles",
        },
        "timestamps": {"count": len(times), "start": times[0], "end": times[-1]},
        "interpolation": False,
        "outputs": outputs,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render", help="render a unified three-method comparison")
    render.add_argument("--scene", required=True)
    render.add_argument("--online-dqn", required=True)
    render.add_argument("--hadhoa", required=True)
    render.add_argument("--fixedtime", required=True)
    for method in METHODS:
        render.add_argument(f"--{method.replace('_', '-')}-controller-id")
        render.add_argument(
            f"--{method.replace('_', '-')}-evaluation-seed", type=int,
        )
        render.add_argument(
            f"--{method.replace('_', '-')}-training-seed", type=int,
        )
    render.add_argument("--output", required=True)
    render.add_argument("--start", type=float, default=0.0)
    render.add_argument("--end", type=float, default=None)
    render.add_argument("--mode", choices=("line", "bar", "both"), default="line")
    render.add_argument("--fps", type=int, default=10)
    render.add_argument("--dpi", type=int, default=120)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command != "render":
        raise ValueError(f"Unsupported command: {args.command}")
    paths, rows = align_three_methods(
        {
            "online_dqn": args.online_dqn,
            "hadhoa": args.hadhoa,
            "fixedtime": args.fixedtime,
        },
        scene=args.scene,
        start=args.start,
        end=args.end,
        selectors={
            method: {
                "controller_id": getattr(args, f"{method}_controller_id"),
                "evaluation_seed": getattr(args, f"{method}_evaluation_seed"),
                "training_seed": getattr(args, f"{method}_training_seed"),
            }
            for method in METHODS
        },
    )
    manifest = render_comparison(rows, args.output, args.scene, args.mode, args.fps, args.dpi)
    manifest["inputs"] = {method: str(path) for method, path in paths.items()}
    Path(args.output, "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
