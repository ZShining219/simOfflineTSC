"""Plan 2 full-dataset final-checkpoint cross-scene evaluation support.

The preparation phase consumes the frozen Plan 2 run list and writes schema-v2
evaluation manifests.  The plotting phase combines newly evaluated off-diagonal
controllers with the existing same-scene decision-record reproductions.  No
training, checkpoint mutation, dataset writes, or replay writes occur here.
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
from .plotting import BOOTSTRAP_SAMPLES, _save
from utils.logger import (
    build_sumo_traffic_identity,
    load_evaluation_collection_manifest,
    validate_evaluation_package,
)


NETWORKS = (
    "sumohz1x1_config2", "sumohz1x1",
    "sumohz1x1_config4", "sumohz1x1_config3",
)
SCENES = ("S1", "S2", "S3", "S4")
SCENE_BY_NETWORK = dict(zip(NETWORKS, SCENES))
NETWORK_BY_SCENE = dict(zip(SCENES, NETWORKS))
ALGORITHMS = ("batch_dqn", "cql_dqn")
OFFLINE_SEEDS = (1000, 1001, 1002, 1003, 1004)
SCENE_COLORS = {
    "S1": "#0072B2", "S2": "#E69F00",
    "S3": "#009E73", "S4": "#CC79A7",
}
BOOTSTRAP_SEED = 20260728

METRICS = (
    (
        "controller_reward_mean_trend", "controller_reward_mean",
        "Cumulative mean DQN reward", "DQN reward (higher is better)",
    ),
    (
        "queue_network_mean_trend", "queue_network_mean",
        "Mean lane queue (60-s moving mean)",
        "Mean queued vehicles per lane (lower is better)",
    ),
    (
        "delay_network_weighted_mean_trend", "delay_weighted_mean",
        "Cumulative vehicle-weighted delay", "Weighted delay (lower is better)",
    ),
    (
        "throughput_interval_trend", "throughput_interval",
        "Completed vehicles per 60-s window",
        "Completed vehicles (higher is better)",
    ),
    (
        "throughput_cumulative_trend", "throughput_cumulative",
        "Cumulative completed vehicles", "Completed vehicles (higher is better)",
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


def _included_full_runs(run_list):
    with Path(run_list).open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["include"].lower() == "true"
            and row["algorithm"] in ALGORITHMS
            and row["dataset_kind"] == "single_scene"
            and row["dataset_stage"] == "full"
        ]
    if len(rows) != 40:
        raise ValueError(f"Expected 40 included Plan 2 full runs, got {len(rows)}")
    indexed = {}
    for row in rows:
        key = (
            row["algorithm"], row["evaluation_network"],
            int(row["offline_training_seed"]),
        )
        if key in indexed:
            raise ValueError(f"Duplicate Plan 2 full run identity: {key}")
        indexed[key] = row
    expected = {
        (algorithm, network, seed)
        for algorithm in ALGORITHMS for network in NETWORKS
        for seed in OFFLINE_SEEDS
    }
    if set(indexed) != expected:
        raise ValueError("Plan 2 full run matrix is incomplete")
    return indexed


def _final_evaluation(run_dir):
    run_dir = Path(run_dir)
    summary = json.loads(
        (run_dir / "evaluation" / "summary.json").read_text(encoding="utf-8")
    )
    final_update = int(summary["final_update"])
    if final_update != 144000:
        raise ValueError(f"Expected final update 144000, got {final_update}")
    record = next(
        item for item in summary["evaluations"]
        if int(item["training_update"]) == final_update
    )
    checkpoint = run_dir / summary["final_checkpoint"]
    if _sha256(checkpoint) != record["checkpoint_sha256"]:
        raise ValueError(f"Final checkpoint hash mismatch: {checkpoint}")
    return summary, record, checkpoint


def prepare_plan2_cross_scene_manifests(args):
    run_list = Path(args.run_list).expanduser().resolve()
    evaluation_root = Path(args.evaluation_root).expanduser().resolve()
    manifest_dir = evaluation_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    runs = _included_full_runs(run_list)
    manifest_paths = {}
    manifest_hashes = {}
    for algorithm in ALGORITHMS:
        controllers = []
        for source_network in NETWORKS:
            for seed in OFFLINE_SEEDS:
                source = runs[(algorithm, source_network, seed)]
                _, source_final, checkpoint = _final_evaluation(source["run_dir"])
                evaluation_seed = int(source_final["evaluation_seed"])
                for target_network in NETWORKS:
                    if target_network == source_network:
                        continue
                    target = runs[(algorithm, target_network, seed)]
                    target_simulator = (
                        Path(target["run_dir"]) / "config" / "simulator_source.cfg"
                    )
                    target_identity = build_sumo_traffic_identity(target_simulator)
                    source_scene = SCENE_BY_NETWORK[source_network]
                    target_scene = SCENE_BY_NETWORK[target_network]
                    controllers.append({
                        "controller_id": (
                            f"{algorithm}_{source_scene}_seed{seed}_to_{target_scene}"
                        ),
                        "agent": algorithm,
                        "network": target_network,
                        "training_seed": seed,
                        "run_dir": str(Path(source["run_dir"]).resolve()),
                        "checkpoint": str(checkpoint.resolve()),
                        "source_scene": source_scene,
                        "source_network": source_network,
                        "target_scene": target_scene,
                        "target_network": target_network,
                        "target_run_dir": str(Path(target["run_dir"]).resolve()),
                        "expected_vehicle_count": int(
                            target_identity["expected_vehicle_count"]
                        ),
                        "checkpoint_role": "final",
                        "source_policy": f"{algorithm}_full_final_greedy",
                        "evaluation_seeds": [evaluation_seed],
                    })
        if len(controllers) != 60:
            raise ValueError(f"Expected 60 off-diagonal {algorithm} controllers")
        manifest = {
            "schema_version": 2,
            "package_id": f"plan2_{algorithm}_full_cross_scene_paired_seed_v1",
            "world": "sumo",
            "evaluation_seeds": [21000, 21001, 21002, 21003, 21004],
            "sampling_interval_seconds": 10,
            "smoothing_window_seconds": 60,
            "metrics": ["reward", "queue", "delay", "throughput", "travel_time"],
            "record_state_diagnostics": False,
            "expected_controller_count": 60,
            "expected_episode_count": 60,
            "reward_definitions": {
                "controller_reward_mean": (
                    "DQN-family agent reward: -12 times mean incoming-lane "
                    "waiting count"
                ),
                "reward_network_mean": (
                    "negative queue_intersections mean; diagnostic only"
                ),
            },
            "controllers": controllers,
        }
        path = manifest_dir / f"{algorithm}_cross_scene_manifest.json"
        if path.exists():
            raise FileExistsError(f"Evaluation manifest already exists: {path}")
        _atomic_json(path, manifest)
        load_evaluation_collection_manifest(path)
        manifest_paths[algorithm] = str(path)
        manifest_hashes[algorithm] = _sha256(path)
    prepare_manifest = {
        "schema_version": 1,
        "status": "prepared",
        "purpose": "Plan 2 full final-checkpoint 4x4 paired-seed frozen evaluation",
        "source_run_list": str(run_list),
        "source_run_list_sha256": _sha256(run_list),
        "algorithms": manifest_paths,
        "manifest_sha256": manifest_hashes,
        "new_off_diagonal_controller_count": 120,
        "expected_new_decision_record_count": 43200,
        "max_workers": 2,
    }
    _atomic_json(evaluation_root / "prepare_manifest.json", prepare_manifest)
    return evaluation_root / "prepare_manifest.json"


def _controller_reward(value):
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("controller_reward_agents must contain finite values")
    return float(values.mean())


def _load_off_diagonal(cross_root, algorithm):
    package_dir = Path(cross_root) / algorithm
    package = validate_evaluation_package(str(package_dir))
    if package["manifest"].get("status") != "completed":
        raise ValueError(f"Incomplete cross-scene package: {package_dir}")
    controllers = package["collection"]["controllers"]
    if len(controllers) != 60:
        raise ValueError(f"Expected 60 off-diagonal controllers for {algorithm}")
    metadata = {item["controller_id"]: item for item in controllers}
    records = []
    with (package_dir / "records.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    if len(records) != 21600:
        raise ValueError(f"Expected 21,600 off-diagonal records for {algorithm}")
    summaries = pd.read_csv(package_dir / "summary.csv")
    return metadata, records, summaries, package_dir


def _load_diagonal(in_domain_root, algorithm):
    root = Path(in_domain_root) / algorithm
    manifest = json.loads(
        (root / "collection_manifest.json").read_text(encoding="utf-8")
    )
    controllers = [item for item in manifest["controllers"] if item["policy"] == "full"]
    if len(controllers) != 20:
        raise ValueError(f"Expected 20 in-domain full controllers for {algorithm}")
    metadata = {}
    records = []
    summary_rows = []
    for item in controllers:
        scene = SCENE_BY_NETWORK[item["network"]]
        metadata[item["controller_id"]] = {
            **item,
            "source_scene": scene,
            "target_scene": scene,
            "source_network": item["network"],
            "target_network": item["network"],
        }
        path = root / item["controller_id"]
        with (path / "records.jsonl").open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle)
        audit = json.loads((path / "audit.json").read_text(encoding="utf-8"))
        summary = audit["reproduced_summary"]
        summary_rows.append({
            **summary,
            "controller_id": item["controller_id"],
            "source_scene": scene,
            "target_scene": scene,
            "source_network": item["network"],
            "target_network": item["network"],
            "training_seed": int(item["offline_training_seed"]),
            "evaluation_seed": int(item["evaluation_seed"]),
        })
    if len(records) != 7200:
        raise ValueError(f"Expected 7,200 in-domain records for {algorithm}")
    return metadata, records, pd.DataFrame(summary_rows), root


def _prepare_records(algorithm, metadata, records, smoothing_window_seconds):
    if smoothing_window_seconds <= 0:
        raise ValueError("smoothing_window_seconds must be positive")
    rows = []
    for record in records:
        controller_id = record["controller_id"]
        identity = metadata[controller_id]
        training_seed = int(
            record.get("training_seed", identity.get("offline_training_seed"))
        )
        rows.append({
            "algorithm": algorithm,
            "controller_id": controller_id,
            "source_scene": identity["source_scene"],
            "target_scene": identity["target_scene"],
            "source_network": identity["source_network"],
            "target_network": identity["target_network"],
            "training_seed": training_seed,
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
        ["algorithm", "controller_id", "decision_step"]
    ).reset_index(drop=True)
    expected_time = frame.decision_step.to_numpy(float) * 10.0
    if not np.allclose(frame.simulation_time_seconds.to_numpy(float), expected_time):
        raise ValueError("Expected fixed 10-second Plan 2 decision intervals")
    counts = frame.groupby("controller_id").decision_step.agg(
        ["count", "min", "max", "nunique"]
    )
    if not (
        counts["count"].eq(360)
        & counts["min"].eq(1)
        & counts["max"].eq(360)
        & counts["nunique"].eq(360)
    ).all():
        raise ValueError("Every Plan 2 controller must contain decision steps 1--360")
    prepared = []
    for _, group in frame.groupby("controller_id", sort=False):
        group = group.sort_values("decision_step").copy()
        window = max(1, int(round(smoothing_window_seconds / 10)))
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
    group_fields = ["algorithm", "source_scene", "target_scene", "decision_step"]
    for identity, group in frame.groupby(group_fields, sort=True):
        algorithm, source_scene, target_scene, decision_step = identity
        if group.training_seed.nunique() != 5 or len(group) != 5:
            raise ValueError("Every Plan 2 curve point requires five paired seeds")
        for field, stem, _, _ in METRICS:
            values = group.sort_values("training_seed")[field].to_numpy(float)
            indices = rng.integers(
                0, len(values), size=(BOOTSTRAP_SAMPLES, len(values))
            )
            means = values[indices].mean(axis=1)
            lower, upper = np.percentile(means, (2.5, 97.5))
            rows.append({
                "algorithm": algorithm,
                "source_scene": source_scene,
                "target_scene": target_scene,
                "decision_step": int(decision_step),
                "metric": stem,
                "value_field": field,
                "seed_mean": float(values.mean()),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "n_paired_seeds": 5,
            })
    return pd.DataFrame(rows)


def _render_within_scene(summary, output_dir, dpi):
    outputs = []
    for algorithm in ALGORITHMS:
        for _, stem, title, ylabel in METRICS:
            metric = summary[
                (summary.algorithm == algorithm) & (summary.metric == stem)
            ]
            fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), squeeze=False)
            for axis, scene in zip(axes.flat, SCENES):
                rows = metric[
                    (metric.source_scene == scene) & (metric.target_scene == scene)
                ].sort_values("decision_step")
                x = rows.decision_step.to_numpy(float)
                mean = rows.seed_mean.to_numpy(float)
                lower = rows.ci_lower.to_numpy(float)
                upper = rows.ci_upper.to_numpy(float)
                color = SCENE_COLORS[scene]
                axis.plot(x, mean, color=color, linewidth=1.9)
                axis.fill_between(x, lower, upper, color=color, alpha=0.16, linewidth=0)
                axis.set(
                    title=f"{scene} full checkpoint → {scene}",
                    xlabel="Decision step (10 s/step)", ylabel=ylabel, xlim=(1, 360),
                )
                axis.grid(alpha=0.18)
            fig.suptitle(
                f"{title}: {algorithm} within-scene full final checkpoint", y=0.995
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.4, w_pad=1.2)
            outputs.extend(_save(
                fig, Path(output_dir) / algorithm / "within_scene" / stem, dpi
            ))
    return outputs


def _render_cross_scene(summary, output_dir, dpi):
    outputs = []
    for algorithm in ALGORITHMS:
        for _, stem, title, ylabel in METRICS:
            metric = summary[
                (summary.algorithm == algorithm) & (summary.metric == stem)
            ]
            fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), squeeze=False)
            for axis, target_scene in zip(axes.flat, SCENES):
                for source_scene in SCENES:
                    rows = metric[
                        (metric.source_scene == source_scene)
                        & (metric.target_scene == target_scene)
                    ].sort_values("decision_step")
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
                    xlabel="Decision step (10 s/step)", ylabel=ylabel, xlim=(1, 360),
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
            fig.suptitle(
                f"{title}: {algorithm} full final-checkpoint cross-scene evaluation",
                y=0.995,
            )
            fig.tight_layout(rect=(0, 0.105, 1, 0.97), h_pad=1.4, w_pad=1.2)
            outputs.extend(_save(
                fig, Path(output_dir) / algorithm / "cross_scene" / stem, dpi
            ))
    return outputs


def _normalize_summaries(algorithm, diagonal, off_diagonal):
    rows = []
    for frame in (diagonal, off_diagonal):
        for item in frame.to_dict("records"):
            source_network = item.get("source_network", item.get("network"))
            target_network = item.get("target_network", item.get("network"))
            rows.append({
                "algorithm": algorithm,
                "controller_id": item["controller_id"],
                "source_scene": item.get(
                    "source_scene", SCENE_BY_NETWORK[source_network]
                ),
                "source_network": source_network,
                "target_scene": item.get(
                    "target_scene", SCENE_BY_NETWORK[target_network]
                ),
                "target_network": target_network,
                "training_seed": int(item["training_seed"]),
                "evaluation_seed": int(item["evaluation_seed"]),
                "travel_time": float(item["travel_time"]),
                "reward_mean": float(item["reward_mean"]),
                "queue": float(item["queue"]),
                "real_delay": float(item["real_delay"]),
                "throughput": float(item["throughput"]),
            })
    frame = pd.DataFrame(rows)
    expected = 80
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} final summaries for {algorithm}")
    return frame


def _aggregate_final_summaries(frame):
    metrics = ("travel_time", "reward_mean", "queue", "real_delay", "throughput")
    rows = []
    for identity, group in frame.groupby(
        ["algorithm", "source_scene", "target_scene"], sort=True
    ):
        algorithm, source_scene, target_scene = identity
        if len(group) != 5 or group.training_seed.nunique() != 5:
            raise ValueError("Every final Plan 2 source-target cell requires five seeds")
        for metric in metrics:
            values = group[metric].to_numpy(float)
            rows.append({
                "algorithm": algorithm,
                "source_scene": source_scene,
                "target_scene": target_scene,
                "metric": metric,
                "paired_seed_count": 5,
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
            })
    return pd.DataFrame(rows)


def _relative_degradation(summary):
    travel = summary[summary.metric == "travel_time"]
    baseline = {
        (row.algorithm, row.target_scene): row.mean
        for row in travel.itertuples()
        if row.source_scene == row.target_scene
    }
    rows = []
    for row in travel.itertuples():
        reference = baseline[(row.algorithm, row.target_scene)]
        rows.append({
            "algorithm": row.algorithm,
            "source_scene": row.source_scene,
            "target_scene": row.target_scene,
            "travel_time_mean": row.mean,
            "in_domain_travel_time_mean": reference,
            "relative_degradation": row.mean / reference - 1.0,
        })
    return pd.DataFrame(rows)


def plot_plan2_cross_scene(args):
    cross_root = Path(args.cross_evaluation_root).expanduser().resolve()
    in_domain_root = Path(args.in_domain_root).expanduser().resolve()
    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    manifest_path = analysis_dir / "plotting_manifest.json"
    if manifest_path.exists() and not args.refresh_existing:
        raise FileExistsError(
            f"Plan 2 cross-scene analysis already exists: {manifest_path}"
        )
    prepared_parts = []
    final_parts = []
    source_packages = {}
    for algorithm in ALGORITHMS:
        off_meta, off_records, off_summary, off_path = _load_off_diagonal(
            cross_root, algorithm
        )
        diag_meta, diag_records, diag_summary, diag_path = _load_diagonal(
            in_domain_root, algorithm
        )
        metadata = {**diag_meta, **off_meta}
        if len(metadata) != 80:
            raise ValueError(f"Expected 80 combined controllers for {algorithm}")
        prepared_parts.append(_prepare_records(
            algorithm, metadata, diag_records + off_records,
            int(args.smoothing_window_seconds),
        ))
        final_parts.append(_normalize_summaries(
            algorithm, diag_summary, off_summary
        ))
        source_packages[algorithm] = {
            "off_diagonal": str(off_path),
            "off_diagonal_records_sha256": _sha256(off_path / "records.jsonl"),
            "in_domain": str(diag_path),
            "in_domain_collection_sha256": _sha256(
                diag_path / "collection_manifest.json"
            ),
        }
    prepared = pd.concat(prepared_parts, ignore_index=True)
    controller_count = len(
        prepared[["algorithm", "controller_id"]].drop_duplicates()
    )
    if len(prepared) != 57600 or controller_count != 160:
        raise ValueError("Expected 160 controllers and 57,600 combined records")
    curve_summary = _bootstrap_summary(prepared)
    final_raw = pd.concat(final_parts, ignore_index=True)
    final_summary = _aggregate_final_summaries(final_raw)
    relative = _relative_degradation(final_summary)

    tables_dir = analysis_dir / "tables"
    figures_dir = analysis_dir / "figures" / "results"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    table_paths = {
        "decision_timeseries": tables_dir / "full_cross_scene_decision_timeseries.csv",
        "curve_summary": tables_dir / "full_cross_scene_curve_seed_summary.csv",
        "final_evaluation": tables_dir / "full_cross_scene_evaluation.csv",
        "final_summary": tables_dir / "full_cross_scene_summary.csv",
        "relative_degradation": tables_dir / "full_cross_scene_relative_degradation.csv",
    }
    prepared.to_csv(table_paths["decision_timeseries"], index=False)
    curve_summary.to_csv(table_paths["curve_summary"], index=False)
    final_raw.to_csv(table_paths["final_evaluation"], index=False)
    final_summary.to_csv(table_paths["final_summary"], index=False)
    relative.to_csv(table_paths["relative_degradation"], index=False)

    figure_paths = []
    figure_paths.extend(_render_within_scene(
        curve_summary, figures_dir, int(args.dpi)
    ))
    figure_paths.extend(_render_cross_scene(
        curve_summary, figures_dir, int(args.dpi)
    ))
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "tool": "tools.experiment_plotting",
        "tool_version": __version__,
        "package_id": "plan2_full_cross_scene_plotting_v1_20260728",
        "created_at_utc": datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat().replace("+00:00", "Z"),
        "scope": (
            "Plan 2 full current-scene final checkpoints only; 4x4 source-target; "
            "Batch-DQN and CQL-DQN"
        ),
        "source_packages": source_packages,
        "algorithms": list(ALGORITHMS),
        "offline_training_seeds": list(OFFLINE_SEEDS),
        "evaluation_seeds": [21000, 21001, 21002, 21003, 21004],
        "seed_pairing": "1000->21000, 1001->21001, ..., 1004->21004",
        "controller_count": 160,
        "reused_in_domain_controller_count": 40,
        "new_off_diagonal_controller_count": 120,
        "decision_record_count": 57600,
        "new_off_diagonal_decision_record_count": 43200,
        "decision_steps": [1, 360],
        "action_interval_seconds": 10,
        "smoothing_window_seconds": int(args.smoothing_window_seconds),
        "statistics": (
            "mean with 95% percentile bootstrap CI over five paired "
            "offline-training/evaluation seeds"
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "metric_semantics": {
            "reward": (
                "controller_reward_agents mean; cumulative mean over decision steps"
            ),
            "queue": "queue_network_mean; moving mean over configured window",
            "delay": (
                "delay_network_weighted_mean; cumulative mean over decision steps"
            ),
            "throughput_interval": (
                "throughput_interval; rolling sum over configured window"
            ),
            "throughput_cumulative": "throughput_cumulative without smoothing",
            "travel_time": (
                "episode-level final summary only; excluded from decision-step figures"
            ),
        },
        "tables": sorted(
            str(path.relative_to(analysis_dir)) for path in table_paths.values()
        ),
        "figures": sorted(
            str(path.relative_to(analysis_dir)) for path in figure_paths
        ),
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path
