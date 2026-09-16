"""Build the Guozhong Project mid-term-report queue figures for 2026-08-04.

The script only consumes archived evaluation records.  It does not train a
controller and it does not rerun SUMO.  The independent S2 online-DQN curve is
the frozen-checkpoint revisit from the archived CONT-FIFO forgetting suite.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output_data" / "国重项目中期汇报_260804"
RESOURCE_CSV = (
    ROOT
    / "output_data"
    / "resource_efficiency_ha_vs_continuous"
    / "resource_efficiency_episode_level.csv"
)
BASELINE_RECORDS = (
    ROOT
    / "output_data"
    / "evaluations"
    / "plan1"
    / "plan1_best_checkpoint_reevaluation_v1_20260723"
    / "records.jsonl"
)
FORGETTING_CSV = (
    ROOT
    / "output_data"
    / "resource_efficiency_ha_vs_continuous"
    / "forgetting_episode_level.csv"
)

SCENES = ("S1", "S2", "S3", "S4")
NETWORK_TO_SCENE = {
    "sumohz1x1_config2": "S1",
    "sumohz1x1": "S2",
    "sumohz1x1_config4": "S3",
    "sumohz1x1_config3": "S4",
}
ORDER2_METHODS = (
    "FixedTime",
    "Online DQN",
    "SemiOffline",
)
ONLINE_METHOD = "Online DQN"
SEMIOFFLINE_METHOD = "SemiOffline"
S2_METHODS = (
    "FixedTime",
    ONLINE_METHOD,
    SEMIOFFLINE_METHOD,
)
METHOD_COLORS = {
    "FixedTime": "#D55E00",
    ONLINE_METHOD: "#009E73",
    SEMIOFFLINE_METHOD: "#CC79A7",
}
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_TRAFFIC_SEEDS = (10000, 10001, 10002, 10003, 10004)
DECISIONS = 360
INTERVAL_SECONDS = 10
DURATION_SECONDS = DECISIONS * INTERVAL_SECONDS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_source_path(value: str | Path) -> Path:
    """Resolve archived paths after the repository's data/ prefix migration."""

    path = Path(value)
    if path.is_file():
        return path.resolve()
    text = str(path)
    candidates = []
    if "/data/output_data/" in text:
        candidates.append(Path(text.replace("/data/output_data/", "/output_data/")))
    if text.startswith("data/output_data/"):
        candidates.append(ROOT / text[len("data/") :])
    if not path.is_absolute():
        candidates.append(ROOT / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve archived source path: {value}")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stats(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Statistics received empty or non-finite values")
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if array.size > 1 else 0.0
    half = 1.96 * sd / math.sqrt(array.size) if array.size > 1 else 0.0
    return {
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "min": float(array.min()),
        "max": float(array.max()),
        "n": int(array.size),
    }


def load_baseline_decisions() -> tuple[list[dict], list[dict]]:
    records = read_jsonl(BASELINE_RECORDS)
    rows = []
    for record in records:
        if record.get("record_type") != "DECISION_METRICS":
            continue
        network = record.get("network")
        scene = NETWORK_TO_SCENE.get(network)
        controller_id = str(record.get("controller_id", ""))
        if scene is None:
            continue
        if controller_id.startswith("maxpressure_"):
            method = "MaxPressure"
        elif controller_id.startswith("fixedtime_"):
            method = "FixedTime"
        else:
            continue
        evaluation_seed = record.get("evaluation_seed")
        if evaluation_seed is None:
            continue
        rows.append(
            {
                "method": method,
                "scene": scene,
                "seed": int(evaluation_seed),
                "seed_kind": "evaluation_traffic_seed",
                "decision_step": int(record["decision_step"]),
                "simulation_time_seconds": float(record["simulation_time_seconds"]),
                "queue": float(record["queue_network_sum"]),
                "source_records": str(BASELINE_RECORDS.resolve()),
            }
        )
    if not rows:
        raise ValueError("No MaxPressure/FixedTime decision records were found")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["scene"], row["seed"])].append(row)
    expected_methods = {"MaxPressure", "FixedTime"}
    if {key[0] for key in grouped} != expected_methods:
        raise ValueError("Baseline records do not contain both requested controllers")
    for key, group in grouped.items():
        group.sort(key=lambda item: item["decision_step"])
        if len(group) != DECISIONS or [
            item["decision_step"] for item in group
        ] != list(range(1, DECISIONS + 1)):
            raise ValueError(f"Baseline decision grid is incomplete: {key}")
        expected_time = np.arange(1, DECISIONS + 1, dtype=float) * INTERVAL_SECONDS
        if not np.allclose(
            [item["simulation_time_seconds"] for item in group], expected_time
        ):
            raise ValueError(f"Baseline time grid is invalid: {key}")
    scalar_rows = []
    for (method, scene, seed), group in sorted(grouped.items()):
        value = float(np.mean([item["queue"] for item in group]))
        scalar_rows.append(
            {
                "method": method,
                "scene": scene,
                "seed": seed,
                "seed_kind": "evaluation_traffic_seed",
                "queue": value,
                "queue_auc_vehicle_seconds": value * DURATION_SECONDS,
                "source_metric": "mean(queue_network_sum over 360 decisions)",
                "source_records": str(BASELINE_RECORDS.resolve()),
            }
        )
    return rows, scalar_rows


def load_resource_rows() -> list[dict]:
    with RESOURCE_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def training_order(resource_rows: list[dict], order_id: str, seed: int) -> list[str]:
    stage_networks = {}
    for row in resource_rows:
        if (
            row["method"] == "P1C-DHOA-R25"
            and row["order_id"] == order_id
            and int(row["training_seed"]) == seed
        ):
            stage = int(row["stage_index"])
            stage_networks.setdefault(stage, set()).add(row["training_network"])
    if set(stage_networks) != {1, 2, 3, 4}:
        raise ValueError(f"Missing P1C {order_id} stage network identities")
    sequence = []
    for stage in range(1, 5):
        networks = stage_networks[stage]
        if len(networks) != 1:
            raise ValueError(f"Ambiguous P1C training network at stage {stage}")
        sequence.append(NETWORK_TO_SCENE[next(iter(networks))])
    return sequence


def build_order2_rows(
    resource_rows: list[dict], baseline_scalar: list[dict]
) -> tuple[list[dict], list[dict]]:
    rows = []
    for item in baseline_scalar:
        if item["method"] != "FixedTime":
            continue
        rows.append(
            {
                "scope": "O2 final stage",
                "method": item["method"],
                "source_method": item["method"],
                "scene": item["scene"],
                "seed": item["seed"],
                "seed_kind": item["seed_kind"],
                "queue": item["queue"],
                "queue_auc_vehicle_seconds": item["queue_auc_vehicle_seconds"],
                "source_metric": item["source_metric"],
                "source_records": item["source_records"],
                "order_id": "O2",
                "stage_index": 4,
                "global_episode": 400,
                "training_network": "baseline controller",
                "evaluation_network": next(
                    network
                    for network, mapped_scene in NETWORK_TO_SCENE.items()
                    if mapped_scene == item["scene"]
                ),
            }
        )
    selected = [
        row
        for row in resource_rows
        if row["method"] == "P1C-DHOA-R25"
        and row["order_id"] == "O2"
        and int(row["stage_index"]) == 4
    ]
    if len(selected) != 20:
        raise ValueError(f"Expected 20 O2 stage-4 P1C rows, found {len(selected)}")
    for row in selected:
        seed = int(row["training_seed"])
        scene = row["scene"]
        baseline_source = resolve_source_path(row["baseline_decisions_path"])
        project_source = resolve_source_path(row["project_decisions_path"])
        common = {
            "scope": "O2 final stage",
            "scene": scene,
            "seed": seed,
            "seed_kind": "training_seed",
            "order_id": row["order_id"],
            "stage_index": int(row["stage_index"]),
            "global_episode": int(row["global_episode"]),
            "training_network": row["training_network"],
            "evaluation_network": row["evaluation_network"],
        }
        rows.append(
            {
                **common,
                "method": ONLINE_METHOD,
                "source_method": ONLINE_METHOD,
                "queue": float(row["continuous_queue_mean"]),
                "queue_auc_vehicle_seconds": float(row["continuous_queue_auc"]),
                "source_metric": "continuous_queue_mean",
                "source_records": str(baseline_source),
                "source_summary": str(resolve_source_path(row["baseline_summary_path"])),
            }
        )
        rows.append(
            {
                **common,
                "method": SEMIOFFLINE_METHOD,
                "source_method": SEMIOFFLINE_METHOD,
                "queue": float(row["ha_queue_mean"]),
                "queue_auc_vehicle_seconds": float(row["ha_queue_auc"]),
                "source_metric": "ha_queue_mean",
                "source_records": str(project_source),
                "source_summary": str(resolve_source_path(row["project_summary_path"])),
            }
        )
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["scene"])].append(row)
    if any(len(group) != 5 for group in grouped.values()):
        raise ValueError("Order-2 comparison does not have five seeds per method/scene")
    summary = []
    for method in ORDER2_METHODS:
        for scene in SCENES:
            group = grouped[(method, scene)]
            values = [float(item["queue"]) for item in group]
            result = stats(values)
            summary.append(
                {
                    "scope": "O2 final stage",
                    "method": method,
                    "scene": scene,
                    "n_seeds": result["n"],
                    "seed_kind": group[0]["seed_kind"],
                    "queue_mean": result["mean"],
                    "queue_sd": result["sd"],
                    "queue_ci95_low": result["ci95_low"],
                    "queue_ci95_high": result["ci95_high"],
                    "queue_min": result["min"],
                    "queue_max": result["max"],
                    "queue_auc_mean_vehicle_seconds": result["mean"] * DURATION_SECONDS,
                }
            )
    return rows, summary


def load_cont_s2() -> tuple[list[dict], dict]:
    """Load the final-stage S2 revisit from the CONT forgetting suite.

    O2 trains S2 first and then trains S3, S1 and S4.  Therefore the O2
    final-stage S2 evaluation is a genuine frozen-checkpoint forgetting
    revisit, rather than an online training trajectory.
    """

    with FORGETTING_CSV.open(newline="", encoding="utf-8") as handle:
        forgetting_rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in forgetting_rows
        if row["method"] == "CONT-FIFO"
        and row["order_id"] == "O2"
        and row["scene"] == "S2"
        and int(row["evaluation_stage"]) == 4
        and int(row["additional_scenes_trained"]) == 3
        and row["stage_label"] == "final_stage"
    ]
    if len(selected) != 5:
        raise ValueError(
            "Expected five CONT-FIFO O2 final-stage S2 forgetting rows, "
            f"found {len(selected)}"
        )
    if {int(row["training_seed"]) for row in selected} != set(EXPECTED_SEEDS):
        raise ValueError("CONT-FIFO O2 S2 forgetting seeds are incomplete")

    rows = []
    checkpoint_paths = []
    decision_paths = []
    summary_paths = []
    checkpoint_episodes = set()
    for forgetting_row in sorted(selected, key=lambda row: int(row["training_seed"])):
        seed = int(forgetting_row["training_seed"])
        summary_path = resolve_source_path(forgetting_row["evaluation_summary_path"])
        decision_path = summary_path.with_name("decisions.jsonl")
        if not decision_path.is_file():
            raise FileNotFoundError(f"CONT forgetting decisions are missing: {decision_path}")
        records = [
            row
            for row in read_jsonl(decision_path)
            if row.get("record_type") == "DECISION_METRICS"
        ]
        records.sort(key=lambda item: int(item["decision_step"]))
        if len(records) != DECISIONS:
            raise ValueError(f"CONT O2 S2 seed {seed} has {len(records)} decisions")
        if [int(item["decision_step"]) for item in records] != list(
            range(1, DECISIONS + 1)
        ):
            raise ValueError(f"CONT O2 S2 seed {seed} has a non-contiguous decision grid")
        expected_time = np.arange(1, DECISIONS + 1, dtype=float) * INTERVAL_SECONDS
        if not np.allclose(
            [float(item["simulation_time_seconds"]) for item in records], expected_time
        ):
            raise ValueError(f"CONT O2 S2 seed {seed} has an invalid time grid")
        if {item.get("network") for item in records} != {"sumohz1x1"}:
            raise ValueError(f"CONT O2 S2 seed {seed} is not evaluated on network sumohz1x1")

        checkpoint_values = {
            resolve_source_path(item["checkpoint_path"]) for item in records
        }
        if len(checkpoint_values) != 1:
            raise ValueError(f"CONT O2 S2 seed {seed} uses multiple checkpoints")
        checkpoint_path = next(iter(checkpoint_values))
        checkpoint_paths.append(checkpoint_path)
        decision_paths.append(decision_path.resolve())
        summary_paths.append(summary_path.resolve())
        checkpoint_episodes.update(int(item["checkpoint_episode"]) for item in records)
        values = [float(item["queue_network_sum"]) for item in records]
        expected_auc = float(forgetting_row["evaluation_queue_auc"])
        actual_auc = float(sum(values) * INTERVAL_SECONDS)
        if not math.isclose(actual_auc, expected_auc, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"CONT queue AUC mismatch for O2 S2 seed {seed}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not math.isclose(
            float(np.mean(values)), float(summary["queue"]), rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError(f"CONT queue mean mismatch for O2 S2 seed {seed}")
        for record in records:
            rows.append(
                {
                    "method": ONLINE_METHOD,
                    "seed": seed,
                    "seed_kind": "training_seed",
                    "decision_step": int(record["decision_step"]),
                    "simulation_time_seconds": float(record["simulation_time_seconds"]),
                    "queue": float(record["queue_network_sum"]),
                    "source_records": str(decision_path.resolve()),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_record_episode": int(record["checkpoint_episode"]),
                    "forgetting_stage": forgetting_row["stage_label"],
                    "additional_scenes_trained": int(
                        forgetting_row["additional_scenes_trained"]
                    ),
                }
            )

    if checkpoint_episodes != {400}:
        raise ValueError(f"Unexpected CONT checkpoint episodes: {checkpoint_episodes}")
    metadata = {
        "series": "CONT-FIFO",
        "order_id": "O2",
        "training_order": "S2 -> S3 -> S1 -> S4",
        "evaluated_scene": "S2",
        "evaluation_stage": 4,
        "additional_scenes_trained": 3,
        "stage_label": "final_stage",
        "stage_local_checkpoint": "stage_04_episode_0100.pt",
        "checkpoint_record_episode": 400,
        "training_seeds": list(EXPECTED_SEEDS),
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "checkpoint_sha256": [sha256_file(path) for path in checkpoint_paths],
        "decision_paths": [str(path) for path in decision_paths],
        "summary_paths": [str(path) for path in summary_paths],
        "source_forgetting_csv": str(FORGETTING_CSV.resolve()),
        "checkpoint_role": "frozen final-stage CONT checkpoint revisited on S2 after three additional scenes",
    }
    return rows, metadata


def load_p1c_s2(resource_rows: list[dict]) -> tuple[list[dict], dict]:
    selected = [
        row
        for row in resource_rows
        if row["method"] == "P1C-DHOA-R25"
        and row["order_id"] == "O1"
        and int(row["stage_index"]) == 4
        and row["scene"] == "S2"
    ]
    if len(selected) != 5:
        raise ValueError(f"Expected 5 O1-last-S2 P1C rows, found {len(selected)}")
    sequence = training_order(resource_rows, "O1", 0)
    if sequence != ["S4", "S1", "S3", "S2"]:
        raise ValueError(f"O1 P1C training order is not S2-last: {sequence}")
    rows = []
    checkpoint_sources = []
    for item in sorted(selected, key=lambda row: int(row["training_seed"])):
        seed = int(item["training_seed"])
        decision_path = resolve_source_path(item["project_decisions_path"])
        decision_records = [
            record
            for record in read_jsonl(decision_path)
            if record.get("record_type") == "DECISION_METRICS"
        ]
        decision_records.sort(key=lambda record: int(record["decision_step"]))
        if len(decision_records) != DECISIONS:
            raise ValueError(f"P1C O1 seed {seed} has {len(decision_records)} decisions")
        values = [float(record["queue_network_sum"]) for record in decision_records]
        if not math.isclose(float(np.mean(values)), float(item["ha_queue_mean"]), rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"P1C queue mean mismatch for O1 seed {seed}")
        checkpoint_path = resolve_source_path(decision_records[0]["checkpoint_path"])
        checkpoint_sources.append(checkpoint_path)
        for record in decision_records:
            rows.append(
                {
                    "method": SEMIOFFLINE_METHOD,
                    "seed": seed,
                    "seed_kind": "training_seed",
                    "decision_step": int(record["decision_step"]),
                    "simulation_time_seconds": float(record["simulation_time_seconds"]),
                    "queue": float(record["queue_network_sum"]),
                    "source_records": str(decision_path),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_record_episode": int(record["checkpoint_episode"]),
                    "stage_local_checkpoint": "stage_04_episode_0100.pt",
                }
            )
    metadata = {
        "order_id": "O1",
        "training_order": "S4 -> S1 -> S3 -> S2",
        "last_training_scene": "S2",
        "stage_index": 4,
        "global_episode": 400,
        "stage_local_checkpoint": "stage_04_episode_0100.pt",
        "checkpoint_paths": [str(path) for path in checkpoint_sources],
        "checkpoint_sha256": [sha256_file(path) for path in checkpoint_sources],
        "checkpoint_record_episode": 400,
    }
    return rows, metadata


def validate_dynamic_rows(rows: list[dict], methods: tuple[str, ...]) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["seed"])].append(row)
    for method in methods:
        method_groups = [key for key in groups if key[0] == method]
        if len(method_groups) != 5:
            raise ValueError(f"{method} has {len(method_groups)} seeds, expected 5")
        for key in method_groups:
            group = sorted(groups[key], key=lambda item: item["decision_step"])
            if [item["decision_step"] for item in group] != list(
                range(1, DECISIONS + 1)
            ):
                raise ValueError(f"Non-contiguous S2 time series: {key}")
            expected_time = np.arange(1, DECISIONS + 1, dtype=float) * INTERVAL_SECONDS
            if not np.allclose(
                [item["simulation_time_seconds"] for item in group], expected_time
            ):
                raise ValueError(f"Invalid S2 time grid: {key}")


def aggregate_timeseries(rows: list[dict], methods: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["decision_step"])].append(float(row["queue"]))
    output = []
    for method in methods:
        for step in range(1, DECISIONS + 1):
            result = stats(grouped[(method, step)])
            output.append(
                {
                    "method": method,
                    "decision_step": step,
                    "simulation_time_seconds": step * INTERVAL_SECONDS,
                    "n_seeds": result["n"],
                    "queue_mean": result["mean"],
                    "queue_sd": result["sd"],
                    "queue_ci95_low": result["ci95_low"],
                    "queue_ci95_high": result["ci95_high"],
                    "queue_min": result["min"],
                    "queue_max": result["max"],
                }
            )
    return output


def seed_summaries(rows: list[dict], methods: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["seed"], row["seed_kind"])].append(row)
    output = []
    for method in methods:
        for (group_method, seed, seed_kind), group in sorted(grouped.items()):
            if group_method != method:
                continue
            group = sorted(group, key=lambda item: item["decision_step"])
            values = np.asarray([item["queue"] for item in group], dtype=float)
            output.append(
                {
                    "method": method,
                    "seed": seed,
                    "seed_kind": seed_kind,
                    "n_decisions": len(values),
                    "queue_mean": float(values.mean()),
                    "queue_auc_vehicle_seconds": float(values.sum() * INTERVAL_SECONDS),
                    "queue_final": float(values[-1]),
                    "queue_min": float(values.min()),
                    "queue_max": float(values.max()),
                    "queue_time_sd": float(values.std(ddof=1)),
                }
            )
    return output


def summary_over_seeds(seed_rows: list[dict], methods: tuple[str, ...]) -> list[dict]:
    output = []
    for method in methods:
        group = [row for row in seed_rows if row["method"] == method]
        if len(group) != 5:
            raise ValueError(f"Expected five seed summaries for {method}")
        result = {"method": method, "n_seeds": 5, "seed_kind": group[0]["seed_kind"]}
        for field in (
            "queue_mean",
            "queue_auc_vehicle_seconds",
            "queue_final",
            "queue_min",
            "queue_max",
        ):
            values = [float(row[field]) for row in group]
            field_stats = stats(values)
            for suffix in ("mean", "sd", "ci95_low", "ci95_high", "min", "max"):
                result[f"{field}_{suffix}"] = field_stats[suffix]
        output.append(result)
    return output


def plot_order2(summary: list[dict], raw: list[dict], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(11.5, 6.8), constrained_layout=True)
    x = np.arange(len(SCENES), dtype=float)
    for method in ORDER2_METHODS:
        method_rows = [row for row in raw if row["method"] == method]
        by_seed = defaultdict(dict)
        for row in method_rows:
            by_seed[row["seed"]][row["scene"]] = float(row["queue"])
        for seed in sorted(by_seed):
            axis.plot(
                x,
                [by_seed[seed][scene] for scene in SCENES],
                color=METHOD_COLORS[method],
                alpha=0.18,
                linewidth=0.9,
            )
        current = [row for row in summary if row["method"] == method]
        current.sort(key=lambda row: SCENES.index(row["scene"]))
        means = np.asarray([row["queue_mean"] for row in current])
        lower = np.asarray([row["queue_ci95_low"] for row in current])
        upper = np.asarray([row["queue_ci95_high"] for row in current])
        axis.errorbar(
            x,
            means,
            yerr=np.vstack((means - lower, upper - means)),
            color=METHOD_COLORS[method],
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            capsize=4,
            label=method,
        )
    axis.set_xticks(x, SCENES)
    axis.set_xlabel("Scenario")
    axis.set_ylabel("Queue (vehicles)")
    axis.set_title("Illustrative Scenario-Order Revisit Performance")
    axis.grid(axis="y", alpha=0.25)
    axis.set_ylim(bottom=0)
    axis.legend(loc="upper left", frameon=True)
    outputs = [
        output_dir / "order2_queue_comparison.png",
        output_dir / "order2_queue_comparison.pdf",
    ]
    for path in outputs:
        fig.savefig(path, dpi=240 if path.suffix == ".png" else None)
    plt.close(fig)
    return outputs


def _plot_timeseries(
    aggregate: list[dict],
    raw: list[dict],
    methods: tuple[str, ...],
    title: str,
    output_dir: Path,
    stem: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(12.5, 7.0), constrained_layout=True)
    time = np.arange(1, DECISIONS + 1, dtype=float) * INTERVAL_SECONDS
    for method in methods:
        method_rows = [row for row in raw if row["method"] == method]
        by_seed = defaultdict(dict)
        for row in method_rows:
            by_seed[row["seed"]][row["decision_step"]] = float(row["queue"])
        for seed in sorted(by_seed):
            axis.plot(
                time,
                [by_seed[seed][step] for step in range(1, DECISIONS + 1)],
                color=METHOD_COLORS[method],
                alpha=0.15,
                linewidth=0.75,
            )
        current = [row for row in aggregate if row["method"] == method]
        current.sort(key=lambda row: row["decision_step"])
        mean = np.asarray([row["queue_mean"] for row in current])
        sd = np.asarray([row["queue_sd"] for row in current])
        axis.fill_between(
            time,
            np.maximum(0.0, mean - sd),
            mean + sd,
            color=METHOD_COLORS[method],
            alpha=0.13,
            linewidth=0,
        )
        axis.plot(
            time,
            mean,
            color=METHOD_COLORS[method],
            linewidth=2.0,
            label=method,
        )
    axis.set_xlabel("Simulation time (s; 10 s per decision)")
    axis.set_ylabel("Queue (vehicles)")
    axis.set_title(title)
    axis.grid(alpha=0.22)
    axis.set_xlim(0, DURATION_SECONDS)
    axis.set_ylim(bottom=0)
    axis.legend(loc="upper left", frameon=True)
    outputs = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in outputs:
        fig.savefig(path, dpi=240 if path.suffix == ".png" else None)
    plt.close(fig)
    return outputs


def source_inventory(paths: list[tuple[str, Path]]) -> list[dict]:
    output = []
    seen = set()
    for role, path in paths:
        path = path.resolve()
        key = (role, str(path))
        if key in seen:
            continue
        seen.add(key)
        if not path.is_file():
            raise FileNotFoundError(f"Source inventory path does not exist: {path}")
        output.append(
            {
                "role": role,
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return output


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_readme(
    order_summary: list[dict],
    s2_summary: list[dict],
    online_metadata: dict,
    p1c_metadata: dict,
) -> None:
    seed_labels = {
        "evaluation_traffic_seed": "traffic seed",
        "training_seed": "training seed",
    }
    lines = [
        "# 国重项目中期汇报 260804",
        "",
        "本图包统一展示 FixedTime、Online DQN、SemiOffline 三种方法的 queue 回顾结果。所有 Queue 均采用 `queue_network_sum`（网络排队车辆数）；每个 episode 包含 360 个决策，每个决策间隔 10 s。",
        "",
        "## 图表",
        "",
        "- `figures/order2_queue_comparison.png` / `.pdf`：例证场景顺序回顾性能；横轴为四个场景，纵轴为 Queue。",
        "- `figures/s2_queue_timeseries_comparison.png` / `.pdf`：回顾性能下样例 Queue 场景的场景内 Queue 情况；横轴为仿真时间，纵轴为 Queue。",
        "",
        "两张图均只保留 FixedTime、Online DQN、SemiOffline；误差条为五个 seed 的正态近似 95% CI，时间序列阴影为五个 seed 的样本 ±1 SD。",
        "",
        "## 例证场景顺序回顾性能：量化数据",
        "",
        "| 方法 | 场景 | 均值 | 样本 SD | 95% CI | AUC (vehicle·s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in order_summary:
        lines.append(
            f"| {row['method']} | {row['scene']} | {fmt(row['queue_mean'])} | "
            f"{fmt(row['queue_sd'])} | [{fmt(row['queue_ci95_low'])}, {fmt(row['queue_ci95_high'])}] | "
            f"{fmt(row['queue_auc_mean_vehicle_seconds'])} |"
        )
    lines.extend(
        [
            "",
            "## 回顾性能下样例 Queue 场景：量化数据",
            "",
            "| 方法 | seed 类型 | Queue mean | Queue mean SD | AUC mean (vehicle·s) | Final Queue mean |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in s2_summary:
        lines.append(
            f"| {row['method']} | {seed_labels.get(row['seed_kind'], row['seed_kind'])} | {fmt(row['queue_mean_mean'])} | "
            f"{fmt(row['queue_mean_sd'])} | {fmt(row['queue_auc_vehicle_seconds_mean'])} | "
            f"{fmt(row['queue_final_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## 数据说明",
            "",
            "- 展示名称统一为 FixedTime、Online DQN、SemiOffline；不在图表和量化表中展开内部实验代号。",
            f"- Online DQN 的回顾顺序为 `{online_metadata['training_order']}`，S2 训练完成后又经历 {online_metadata['additional_scenes_trained']} 个场景，曲线来自冻结 checkpoint 回顾。",
            f"- SemiOffline 的回顾顺序为 `{p1c_metadata['training_order']}`，最后训练场景为 S2，曲线来自该阶段冻结 checkpoint 回顾。",
            "- FixedTime 使用固定时序控制回顾数据；Queue 均值为 360 个决策点的平均值，AUC 为 Queue 随时间的积分量。",
            "",
            "## 可复现入口",
            "",
            "派生图表由仓库脚本 `tools/plot_guozhong_midterm_260804.py` 生成；来源文件、SHA256、checkpoint 身份和验证条件记录在 `report_manifest.json` 与 `metadata/source_inventory.csv`。",
            "",
            "本目录只代表数据整理与图表生成完成；图中的 seed 波动不额外外推为完整交通随机性结论。",
        ]
    )
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if (
        not RESOURCE_CSV.is_file()
        or not FORGETTING_CSV.is_file()
        or not BASELINE_RECORDS.is_file()
    ):
        raise FileNotFoundError("Required archived source file is missing")

    baseline_dynamic, baseline_scalar = load_baseline_decisions()
    resource_rows = load_resource_rows()
    order2_raw, order2_summary = build_order2_rows(resource_rows, baseline_scalar)
    online_rows, online_metadata = load_cont_s2()
    p1c_rows, p1c_metadata = load_p1c_s2(resource_rows)

    s2_rows = [
        {
            "method": method,
            "seed": row["seed"],
            "seed_kind": row["seed_kind"],
            "decision_step": row["decision_step"],
            "simulation_time_seconds": row["simulation_time_seconds"],
            "queue": row["queue"],
            "source_records": row["source_records"],
        }
        for row in baseline_dynamic
        if row["scene"] == "S2" and row["method"] == "FixedTime"
        for method in (row["method"],)
    ]
    s2_rows.extend(online_rows)
    s2_rows.extend(
        {
            "method": row["method"],
            "seed": row["seed"],
            "seed_kind": row["seed_kind"],
            "decision_step": row["decision_step"],
            "simulation_time_seconds": row["simulation_time_seconds"],
            "queue": row["queue"],
            "source_records": row["source_records"],
            "checkpoint_path": row["checkpoint_path"],
        }
        for row in p1c_rows
    )
    validate_dynamic_rows(s2_rows, S2_METHODS)
    s2_aggregate = aggregate_timeseries(s2_rows, S2_METHODS)
    s2_seed_summary = seed_summaries(s2_rows, S2_METHODS)
    s2_summary = summary_over_seeds(s2_seed_summary, S2_METHODS)

    tables = OUTPUT / "tables"
    figures = OUTPUT / "figures"
    metadata = OUTPUT / "metadata"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    for stale_path in (
        figures / "s2_maxpressure_queue_timeseries.png",
        figures / "s2_maxpressure_queue_timeseries.pdf",
    ):
        if stale_path.exists():
            stale_path.unlink()
    write_csv(tables / "order2_queue_seed_values.csv", order2_raw)
    write_csv(tables / "order2_queue_summary.csv", order2_summary)
    write_csv(tables / "s2_queue_timeseries_seed_values.csv", s2_rows)
    write_csv(tables / "s2_queue_timeseries_aggregate.csv", s2_aggregate)
    write_csv(tables / "s2_queue_seed_summary.csv", s2_seed_summary)
    write_csv(tables / "s2_queue_summary.csv", s2_summary)

    figure_paths = []
    figure_paths.extend(plot_order2(order2_summary, order2_raw, figures))
    figure_paths.extend(
        _plot_timeseries(
            s2_aggregate,
            s2_rows,
            S2_METHODS,
            "Within-Scenario Queue Dynamics for a Representative Revisit Scenario",
            figures,
            "s2_queue_timeseries_comparison",
        )
    )
    inventory_paths = [
        ("resource_efficiency_episode_level", RESOURCE_CSV),
        ("cont_forgetting_episode_level", FORGETTING_CSV),
        ("plan1_baseline_decision_records", BASELINE_RECORDS),
    ]
    for path in online_metadata["summary_paths"]:
        inventory_paths.append(("online_dqn_forgetting_summary", Path(path)))
    for path in online_metadata["decision_paths"]:
        inventory_paths.append(("online_dqn_forgetting_decisions", Path(path)))
    for path in online_metadata["checkpoint_paths"]:
        inventory_paths.append(("online_dqn_frozen_checkpoint", Path(path)))
    for row in p1c_rows:
        inventory_paths.append(("semi_offline_s2_decisions", Path(row["source_records"])))
        inventory_paths.append(("semi_offline_frozen_checkpoint", Path(row["checkpoint_path"])))
    inventory = source_inventory(inventory_paths)
    write_csv(metadata / "source_inventory.csv", inventory)

    write_readme(order2_summary, s2_summary, online_metadata, p1c_metadata)
    (OUTPUT / "reproduction_command.txt").write_text(
        "cd /projects/simOfflineTSC\n"
        "/home/dev/miniforge3/envs/colight/bin/python "
        "tools/plot_guozhong_midterm_260804.py\n",
        encoding="utf-8",
    )

    derived_files = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "report_manifest.json":
            derived_files.append(
                {
                    "path": str(path.relative_to(OUTPUT)),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    report_manifest = {
        "schema_version": 1,
        "status": "completed",
        "report": "国重项目中期汇报 260804",
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "queue_definition": "queue_network_sum, measured at each 10-second decision; episode mean is the mean of 360 decision values",
        "presentation": {
            "display_methods": list(ORDER2_METHODS),
            "order2_title_zh": "例证场景顺序回顾性能",
            "order2_title_en": "Illustrative Scenario-Order Revisit Performance",
            "order2_description_zh": "四个场景上的 Queue 回顾性能比较；横轴为场景，纵轴为 Queue。",
            "s2_title_zh": "回顾性能下样例 Queue 场景的场景内 Queue 情况",
            "s2_title_en": "Within-Scenario Queue Dynamics for a Representative Revisit Scenario",
            "s2_description_zh": "样例 S2 场景内的 Queue 回顾变化；横轴为仿真时间，纵轴为 Queue。",
        },
        "statistics": {
            "seed_summary": "mean, sample SD, min/max and normal-approximate 95% CI over five source seeds",
            "order2_plot": "individual seed lines plus mean and 95% CI error bars",
            "s2_timeseries_plot": "individual seed lines plus mean and sample +/-1 SD ribbon",
        },
        "order2_scope": {
            "order_id": "O2",
            "training_order": "S2 -> S3 -> S1 -> S4",
            "stage_index": 4,
            "global_episode": 400,
            "scenes": list(SCENES),
            "methods": list(ORDER2_METHODS),
            "seed_axes": {
                "FixedTime": "evaluation_traffic_seed 10000-10004",
                "Online DQN": "training_seed 0-4 from frozen-checkpoint revisit records",
                "SemiOffline": "training_seed 0-4 from frozen-checkpoint revisit records",
            },
        },
        "s2_scope": {
            "scene": "S2",
            "duration_seconds": DURATION_SECONDS,
            "decisions": DECISIONS,
            "methods": list(S2_METHODS),
            "online_dqn": online_metadata,
            "semi_offline": p1c_metadata,
        },
        "figures": [str(path.relative_to(OUTPUT)) for path in figure_paths],
        "tables": [
            "tables/order2_queue_seed_values.csv",
            "tables/order2_queue_summary.csv",
            "tables/s2_queue_timeseries_seed_values.csv",
            "tables/s2_queue_timeseries_aggregate.csv",
            "tables/s2_queue_seed_summary.csv",
            "tables/s2_queue_summary.csv",
        ],
        "source_inventory": inventory,
        "files": derived_files,
        "limitations": [
            "FixedTime uses explicit traffic evaluation seeds; Online DQN and SemiOffline use training_seed variability from five frozen-checkpoint revisit records.",
            "Both figures and all exported queue tables contain only FixedTime, Online DQN and SemiOffline.",
            "The independent S2 Online DQN curve is a final-stage frozen-checkpoint revisit after three additional scenes; it is not a training trajectory or a new training run.",
            "The SemiOffline S2 curve is the final stage of its listed training order (S4 -> S1 -> S3 -> S2); the archived checkpoint record is episode 400 with stage-local filename stage_04_episode_0100.pt.",
        ],
    }
    write_json(OUTPUT / "report_manifest.json", report_manifest)
    print(f"Completed report package: {OUTPUT}")
    print(f"Order-2 rows: {len(order2_raw)}; S2 decision rows: {len(s2_rows)}")


if __name__ == "__main__":
    main()
