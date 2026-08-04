"""Supplemental Plan 1 final-checkpoint decision-level figures.

This module consumes the completed, immutable final-checkpoint cross-scene
evaluation package.  It does not run SUMO or modify upstream experiment data.
"""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from . import __version__
from .loaders import load_evaluation_records
from .plotting import _save, BOOTSTRAP_SAMPLES
from utils.logger import validate_evaluation_package


SCENES = ("S1", "S2", "S3", "S4")
SCENE_COLORS = {
    "S1": "#0072B2",
    "S2": "#E69F00",
    "S3": "#009E73",
    "S4": "#CC79A7",
}
BOOTSTRAP_SEED = 20260728

METRICS = (
    (
        "controller_reward_mean_trend",
        "controller_reward_mean",
        "Cumulative mean DQN reward",
        "DQN reward (higher is better)",
    ),
    (
        "queue_network_mean_trend",
        "queue_network_mean",
        "Mean lane queue (60-s moving mean)",
        "Queued vehicles (lower is better)",
    ),
    (
        "delay_network_weighted_mean_trend",
        "delay_weighted_mean",
        "Cumulative vehicle-weighted delay",
        "Weighted delay (lower is better)",
    ),
    (
        "throughput_interval_trend",
        "throughput_interval",
        "Completed vehicles per 60-s window",
        "Completed vehicles (higher is better)",
    ),
    (
        "throughput_cumulative_trend",
        "throughput_cumulative",
        "Cumulative completed vehicles",
        "Completed vehicles (higher is better)",
    ),
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _controller_reward(value):
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("controller_reward_agents must contain finite values")
    return float(np.mean(array))


def _prepare_records(records, controller_metadata, smoothing_window_seconds):
    if smoothing_window_seconds <= 0:
        raise ValueError("smoothing_window_seconds must be positive")
    rows = []
    for record in records:
        if record.get("record_type") != "DECISION_METRICS":
            raise ValueError("Final-checkpoint package contains a non-decision record")
        controller_id = record["controller_id"]
        if controller_id not in controller_metadata:
            raise ValueError(f"Unknown controller_id in decision records: {controller_id}")
        identity = controller_metadata[controller_id]
        rows.append({
            "controller_id": controller_id,
            "source_scene": identity["source_scene"],
            "target_scene": identity["target_scene"],
            "source_network": identity["source_network"],
            "target_network": identity["target_network"],
            "training_seed": int(record["training_seed"]),
            "evaluation_seed": int(record["evaluation_seed"]),
            "decision_step": int(record["decision_step"]),
            "simulation_time_seconds": float(record["simulation_time_seconds"]),
            "controller_reward_mean": _controller_reward(
                record["controller_reward_agents"]
            ),
            "queue_network_mean": float(record["queue_network_mean"]),
            "delay_network_weighted_mean": float(
                record["delay_network_weighted_mean"]
            ),
            "throughput_interval": int(record["throughput_interval"]),
            "throughput_cumulative": int(record["throughput_cumulative"]),
        })
    frame = pd.DataFrame(rows).sort_values(
        ["controller_id", "evaluation_seed", "decision_step"]
    ).reset_index(drop=True)
    if len(frame) != 28800 or frame.controller_id.nunique() != 80:
        raise ValueError(
            "Expected 80 controllers and 28,800 decision records; got "
            f"{frame.controller_id.nunique()} and {len(frame)}"
        )
    if set(frame.evaluation_seed.unique()) != {10000}:
        raise ValueError("Expected the fixed Plan 1 evaluation seed 10000")
    expected_time = frame.decision_step.to_numpy(float) * 10.0
    if not np.allclose(frame.simulation_time_seconds.to_numpy(float), expected_time):
        raise ValueError("Expected decision steps at fixed 10-second intervals")
    counts = frame.groupby("controller_id").decision_step.agg(
        ["count", "min", "max", "nunique"]
    )
    invalid = counts[
        (counts["count"] != 360)
        | (counts["nunique"] != 360)
        | (counts["min"] != 1)
        | (counts["max"] != 360)
    ]
    if not invalid.empty:
        raise ValueError("Every final-checkpoint evaluation must contain steps 1--360")
    if set(frame.source_scene.unique()) != set(SCENES):
        raise ValueError("Source-scene coverage is incomplete")
    if set(frame.target_scene.unique()) != set(SCENES):
        raise ValueError("Target-scene coverage is incomplete")
    seed_counts = frame.groupby(
        ["source_scene", "target_scene"]
    ).training_seed.nunique()
    if not (seed_counts == 5).all():
        raise ValueError("Every source-target cell must contain five training seeds")

    prepared = []
    for _, group in frame.groupby(
        ["controller_id", "evaluation_seed"], sort=False
    ):
        group = group.sort_values("decision_step").copy()
        interval = int(round(group.simulation_time_seconds.diff().dropna().median()))
        if interval <= 0:
            interval = 10
        window = max(1, int(round(smoothing_window_seconds / interval)))
        group["controller_reward_mean_trend"] = (
            group.controller_reward_mean.expanding().mean()
        )
        group["queue_network_mean_trend"] = (
            group.queue_network_mean.rolling(window, min_periods=1).mean()
        )
        group["delay_network_weighted_mean_trend"] = (
            group.delay_network_weighted_mean.expanding().mean()
        )
        group["throughput_interval_trend"] = (
            group.throughput_interval.rolling(window, min_periods=1).sum()
        )
        group["throughput_cumulative_trend"] = group.throughput_cumulative
        prepared.append(group)
    return pd.concat(prepared, ignore_index=True)


def _bootstrap_summary(frame):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    group_fields = ("source_scene", "target_scene", "decision_step")
    for identity, group in frame.groupby(list(group_fields), sort=True):
        source_scene, target_scene, decision_step = identity
        if group.training_seed.nunique() != 5 or len(group) != 5:
            raise ValueError(
                "Expected exactly one record for each of five seeds at every curve point"
            )
        for field, stem, _, _ in METRICS:
            values = group.sort_values("training_seed")[field].to_numpy(float)
            indices = rng.integers(
                0, len(values), size=(BOOTSTRAP_SAMPLES, len(values))
            )
            bootstrap_means = values[indices].mean(axis=1)
            lower, upper = np.percentile(bootstrap_means, (2.5, 97.5))
            rows.append({
                "source_scene": source_scene,
                "target_scene": target_scene,
                "decision_step": int(decision_step),
                "metric": stem,
                "value_field": field,
                "seed_mean": float(values.mean()),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "n_training_seeds": 5,
            })
    return pd.DataFrame(rows)


def _metric_rows(summary, stem):
    return summary[summary.metric == stem].sort_values(
        ["target_scene", "source_scene", "decision_step"]
    )


def _render_within_scene(summary, output_dir, dpi):
    outputs = []
    for _, stem, title, ylabel in METRICS:
        metric = _metric_rows(summary, stem)
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), squeeze=False)
        for axis, scene in zip(axes.flat, SCENES):
            rows = metric[
                (metric.source_scene == scene) & (metric.target_scene == scene)
            ]
            x = rows.decision_step.to_numpy(float)
            mean = rows.seed_mean.to_numpy(float)
            lower = rows.ci_lower.to_numpy(float)
            upper = rows.ci_upper.to_numpy(float)
            color = SCENE_COLORS[scene]
            axis.plot(x, mean, color=color, linewidth=1.9)
            axis.fill_between(x, lower, upper, color=color, alpha=0.16, linewidth=0)
            axis.set(
                title=f"{scene} final checkpoint → {scene}",
                xlabel="Decision step (10 s/step)",
                ylabel=ylabel,
                xlim=(1, 360),
            )
            axis.grid(alpha=0.18)
        fig.suptitle(f"{title}: within-scene final-checkpoint evaluation", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.4, w_pad=1.2)
        outputs.extend(_save(fig, output_dir / "within_scene" / stem, dpi))
    return outputs


def _render_cross_scene(summary, output_dir, dpi):
    outputs = []
    for _, stem, title, ylabel in METRICS:
        metric = _metric_rows(summary, stem)
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), squeeze=False)
        for axis, target_scene in zip(axes.flat, SCENES):
            for source_scene in SCENES:
                rows = metric[
                    (metric.source_scene == source_scene)
                    & (metric.target_scene == target_scene)
                ]
                x = rows.decision_step.to_numpy(float)
                mean = rows.seed_mean.to_numpy(float)
                lower = rows.ci_lower.to_numpy(float)
                upper = rows.ci_upper.to_numpy(float)
                color = SCENE_COLORS[source_scene]
                in_domain = source_scene == target_scene
                axis.plot(
                    x, mean, color=color,
                    linewidth=2.3 if in_domain else 1.55,
                    linestyle="-" if in_domain else "--",
                )
                axis.fill_between(
                    x, lower, upper, color=color,
                    alpha=0.13 if in_domain else 0.07, linewidth=0,
                )
            axis.set(
                title=f"Target evaluation scene: {target_scene}",
                xlabel="Decision step (10 s/step)",
                ylabel=ylabel,
                xlim=(1, 360),
            )
            axis.grid(alpha=0.18)
        handles = [
            Line2D([], [], color=SCENE_COLORS[scene], linewidth=2,
                   label=f"Source model: {scene}")
            for scene in SCENES
        ] + [
            Line2D([], [], color="0.2", linewidth=2.3, linestyle="-",
                   label="In-domain source"),
            Line2D([], [], color="0.2", linewidth=1.55, linestyle="--",
                   label="Cross-scene source"),
        ]
        fig.legend(
            handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
            ncols=3, frameon=False,
        )
        fig.suptitle(f"{title}: final-checkpoint cross-scene evaluation", y=0.995)
        fig.tight_layout(rect=(0, 0.105, 1, 0.97), h_pad=1.4, w_pad=1.2)
        outputs.extend(_save(fig, output_dir / "cross_scene" / stem, dpi))
    return outputs


def _write_prepared_table(path, frame):
    columns = (
        "controller_id", "source_scene", "target_scene", "source_network",
        "target_network", "training_seed", "evaluation_seed", "decision_step",
        "simulation_time_seconds", "controller_reward_mean",
        "controller_reward_mean_trend", "queue_network_mean",
        "queue_network_mean_trend", "delay_network_weighted_mean",
        "delay_network_weighted_mean_trend", "throughput_interval",
        "throughput_interval_trend", "throughput_cumulative",
        "throughput_cumulative_trend",
    )
    frame.to_csv(path, columns=columns, index=False, quoting=csv.QUOTE_MINIMAL)


def run_plan1_final_timeseries(args):
    evaluation_package = Path(args.evaluation_package).expanduser().resolve()
    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    root_manifest_path = analysis_dir / "plotting_manifest.json"
    if not root_manifest_path.is_file():
        raise FileNotFoundError(
            f"Plan 1 plotting manifest is missing: {root_manifest_path}"
        )
    package = validate_evaluation_package(str(evaluation_package))
    manifest = package["manifest"]
    if manifest.get("status") != "completed":
        raise ValueError("Evaluation package is not completed")
    if int(manifest.get("episode_count", -1)) != 80:
        raise ValueError("Expected 80 completed final-checkpoint evaluations")
    if int(manifest.get("decision_record_count", -1)) != 28800:
        raise ValueError("Expected 28,800 decision records")
    collection = package["collection"]
    controllers = collection.get("controllers", [])
    if len(controllers) != 80:
        raise ValueError("Evaluation collection must describe 80 controllers")
    controller_metadata = {
        item["controller_id"]: {
            "source_scene": item["source_scene"],
            "target_scene": item["target_scene"],
            "source_network": item["source_network"],
            "target_network": item["target_network"],
        }
        for item in controllers
    }
    if len(controller_metadata) != 80:
        raise ValueError("Evaluation controller identities are not unique")

    supplement_name = "final_checkpoint_decision_timeseries"
    supplement_manifest_path = analysis_dir / f"{supplement_name}_manifest.json"
    if supplement_manifest_path.exists() and not args.refresh_existing:
        raise FileExistsError(
            f"Supplement already exists: {supplement_manifest_path}; "
            "use --refresh-existing to rebuild derived outputs"
        )
    records = load_evaluation_records(evaluation_package)
    frame = _prepare_records(
        records, controller_metadata, int(args.smoothing_window_seconds)
    )
    summary = _bootstrap_summary(frame)

    tables_dir = analysis_dir / "tables"
    figure_dir = analysis_dir / "figures" / "results" / supplement_name
    tables_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    prepared_table = tables_dir / f"{supplement_name}.csv"
    summary_table = tables_dir / f"{supplement_name}_seed_summary.csv"
    _write_prepared_table(prepared_table, frame)
    summary.to_csv(summary_table, index=False)

    figures = []
    figures.extend(_render_within_scene(summary, figure_dir, int(args.dpi)))
    figures.extend(_render_cross_scene(summary, figure_dir, int(args.dpi)))
    relative_figures = sorted(str(path.relative_to(analysis_dir)) for path in figures)
    relative_tables = sorted(
        str(path.relative_to(analysis_dir))
        for path in (prepared_table, summary_table)
    )
    supplement_manifest = {
        "schema_version": 1,
        "status": "completed",
        "tool": "tools.experiment_plotting",
        "tool_version": __version__,
        "supplement": supplement_name,
        "created_at_utc": datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat().replace("+00:00", "Z"),
        "source_evaluation_package": str(evaluation_package),
        "source_package_id": manifest["package_id"],
        "source_records_sha256": _sha256(evaluation_package / "records.jsonl"),
        "controller_count": 80,
        "decision_record_count": 28800,
        "within_scene_controller_count": 20,
        "within_scene_decision_record_count": 7200,
        "cross_scene_off_diagonal_controller_count": 60,
        "cross_scene_off_diagonal_decision_record_count": 21600,
        "training_seeds": [0, 1, 2, 3, 4],
        "evaluation_seeds": sorted(int(x) for x in frame.evaluation_seed.unique()),
        "decision_steps": [1, 360],
        "action_interval_seconds": 10,
        "smoothing_window_seconds": int(args.smoothing_window_seconds),
        "statistics": "five-training-seed mean with 95% percentile bootstrap CI",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "metric_semantics": {
            "reward": "controller_reward_agents mean; cumulative mean over decision steps",
            "queue": "queue_network_mean; moving mean over the configured window",
            "delay": "delay_network_weighted_mean; cumulative mean over decision steps",
            "throughput_interval": "throughput_interval; rolling sum over the configured window",
            "throughput_cumulative": "throughput_cumulative without smoothing",
            "travel_time": "episode-level only; intentionally excluded from decision-step figures",
        },
        "tables": relative_tables,
        "figures": relative_figures,
    }
    _atomic_json(supplement_manifest_path, supplement_manifest)

    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    supplements = [
        item for item in root_manifest.get("supplements", [])
        if item.get("name") != supplement_name
    ]
    supplements.append({
        "name": supplement_name,
        "manifest": str(supplement_manifest_path.relative_to(analysis_dir)),
        "source_package_id": manifest["package_id"],
        "tables": relative_tables,
        "figures": relative_figures,
    })
    root_manifest["supplements"] = supplements
    _atomic_json(root_manifest_path, root_manifest)
    return supplement_manifest_path
