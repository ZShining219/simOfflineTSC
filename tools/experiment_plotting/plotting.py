import math
import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(
        Path(tempfile.gettempdir()) / "simofflinetsc-matplotlib"
    )

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "dqn": "#4C78A8", "fixedtime": "#F58518", "maxpressure": "#54A24B",
}


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _save(fig, output_base, dpi):
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = output_base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=dpi if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _label(row):
    return f"{row['agent']} | {row['network']} | seed {row['training_seed']}"


def _baseline_levels(rows, metric):
    levels = {}
    for row in rows:
        if row["role"] == "baseline" and _finite(row.get(metric)):
            key = (row["network"], row["agent"])
            if row["record_type"] == "FINAL_EVALUATION" or key not in levels:
                levels[key] = row[metric]
    return levels


def render_learning_metrics(rows, output_base, metrics, title, dpi=160):
    fig, axes = plt.subplots(
        len(metrics), 1, figsize=(10, max(3.8, 3.4 * len(metrics))),
        squeeze=False, constrained_layout=True,
    )
    run_keys = list(dict.fromkeys(row["run_key"] for row in rows))
    for axis, (metric, ylabel) in zip(axes[:, 0], metrics):
        plotted = False
        for run_index, run_key in enumerate(run_keys):
            run_rows = [row for row in rows if row["run_key"] == run_key]
            if run_rows and run_rows[0]["role"] == "baseline":
                continue
            for series_name, record_types, linestyle in (
                ("train", {"TRAIN"}, "-"),
                ("evaluation", {"EVALUATION", "FINAL_EVALUATION"}, "--"),
            ):
                selected = [
                    row for row in run_rows
                    if row["record_type"] in record_types and _finite(row.get(metric))
                ]
                if not selected:
                    continue
                selected.sort(key=lambda row: row["episode"])
                sample = selected[0]
                marker = "o" if series_name == "evaluation" else "."
                color = (
                    plt.get_cmap("tab10")(run_index % 10)
                    if sample["agent"] == "dqn"
                    else COLORS.get(sample["agent"])
                )
                axis.plot(
                    [row["episode"] for row in selected],
                    [row[metric] for row in selected],
                    marker=marker, markersize=3.5, linewidth=1.4,
                    linestyle=linestyle, color=color,
                    label=f"{_label(sample)} | {series_name}",
                )
                plotted = True
        for (network, agent), value in _baseline_levels(rows, metric).items():
            axis.axhline(
                value, linestyle="--", linewidth=0.9, alpha=0.45,
                color=COLORS.get(agent), label=f"{agent} {network} reference",
            )
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Completed training episode")
        axis.grid(alpha=0.2)
        if plotted:
            axis.legend(fontsize=7, ncols=2)
        else:
            axis.text(0.5, 0.5, "No compatible data", ha="center", va="center")
    fig.suptitle(title)
    return _save(fig, output_base, dpi)


def render_action_distribution(action_rows, output_base, dpi=160):
    fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    latest = {}
    for row in action_rows:
        key = (row["run_key"], row["action"])
        current = latest.get(key)
        if current is None or row["episode"] >= current["episode"]:
            latest[key] = row
    if latest:
        run_keys = list(dict.fromkeys(key[0] for key in latest))
        actions = sorted({key[1] for key in latest})
        width = 0.8 / max(1, len(run_keys))
        x = np.arange(len(actions))
        for index, run_key in enumerate(run_keys):
            values = [latest.get((run_key, action), {}).get("fraction", 0.0) for action in actions]
            sample = next(value for key, value in latest.items() if key[0] == run_key)
            color = (
                plt.get_cmap("tab10")(index % 10)
                if sample["agent"] == "dqn"
                else COLORS.get(sample["agent"])
            )
            axis.bar(x + (index - (len(run_keys) - 1) / 2) * width, values,
                     width=width, label=_label(sample), color=color)
        axis.set_xticks(x, [str(action) for action in actions])
        axis.legend(fontsize=7, ncols=2)
    else:
        axis.text(0.5, 0.5, "No schema v2 action-distribution data", ha="center", va="center")
    axis.set_title("Latest action distribution")
    axis.set_xlabel("Action")
    axis.set_ylabel("Fraction")
    axis.grid(axis="y", alpha=0.2)
    return _save(fig, output_base, dpi)


def render_final_best(comparison_rows, output_base, dpi=160):
    fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    grouped = {}
    for row in comparison_rows:
        grouped.setdefault(row["run_key"], {})[row["selection"]] = row
    if grouped:
        labels = []
        final_values = []
        best_values = []
        for selections in grouped.values():
            sample = next(iter(selections.values()))
            labels.append(_label(sample))
            final_values.append(selections.get("final", {}).get("travel_time", np.nan))
            best_values.append(selections.get("best", {}).get("travel_time", np.nan))
        x = np.arange(len(labels))
        axis.bar(x - 0.2, final_values, width=0.4, label="Final", color="#9ECAE1")
        axis.bar(x + 0.2, best_values, width=0.4, label="Best", color="#3182BD")
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No final/best data", ha="center", va="center")
    axis.set_title("Final and best evaluation travel time")
    axis.set_ylabel("Average travel time")
    axis.grid(axis="y", alpha=0.2)
    return _save(fig, output_base, dpi)


def render_auc(auc_rows, output_base, dpi=160):
    fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    selected = [row for row in auc_rows if row["metric"] == "travel_time"]
    if selected:
        labels = [f"{_label(row)} | {row.get('curve_source', 'TRAIN').lower()}" for row in selected]
        axis.bar(np.arange(len(selected)), [row["auc"] for row in selected], color="#72B7B2")
        axis.set_xticks(np.arange(len(selected)), labels, rotation=20, ha="right")
    else:
        axis.text(0.5, 0.5, "No first-100 AUC data", ha="center", va="center")
    axis.set_title("First-100 travel-time AUC by curve source")
    axis.set_ylabel("AUC")
    axis.grid(axis="y", alpha=0.2)
    return _save(fig, output_base, dpi)


def render_all(metric_rows, action_rows, comparison_rows, auc_rows, figure_dir, dpi=160):
    figure_dir = Path(figure_dir)
    outputs = []
    specifications = (
        ("travel_time", (("travel_time", "Average travel time"),), "Travel time"),
        ("delays", (("delay", "Approximate delay"), ("real_delay", "Real delay (s)")), "Delay diagnostics"),
        ("queue_throughput", (("queue", "Queue"), ("throughput", "Throughput")), "Traffic performance"),
        ("reward_loss", (("reward_mean", "Mean reward"), ("loss_mean", "Mean loss")), "Training signal"),
        ("epsilon_replay", (("epsilon", "Epsilon"), ("replay_size", "Replay size")), "Exploration and replay"),
        ("phase_switching", (("phase_switch_frequency", "Switch frequency"), ("phase_switches", "Switch count")), "Phase switching"),
        ("interaction_costs", (("wall_time_seconds", "Wall time (s)"), ("gradient_updates", "Gradient updates")), "Interaction costs"),
    )
    for filename, metrics, title in specifications:
        outputs.extend(render_learning_metrics(
            metric_rows, figure_dir / filename, metrics, title, dpi=dpi,
        ))
    outputs.extend(render_action_distribution(action_rows, figure_dir / "action_distribution", dpi=dpi))
    outputs.extend(render_final_best(comparison_rows, figure_dir / "final_best_comparison", dpi=dpi))
    outputs.extend(render_auc(auc_rows, figure_dir / "first_100_auc", dpi=dpi))
    return outputs
