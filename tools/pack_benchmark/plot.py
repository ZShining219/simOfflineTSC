"""Plot a canonical benchmark dataset.  Plot code never recomputes metrics."""
import json
import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(
        Path(tempfile.gettempdir()) / "simofflinetsc-pack-benchmark-matplotlib"
    )

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import METRICS, get_metric


PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")


def _save(fig, base_path, dpi=160):
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = base_path.with_suffix(f".{suffix}")
        metadata = (
            {"Software": "tools.pack_benchmark"}
            if suffix == "png"
            else {"Creator": "tools.pack_benchmark", "CreationDate": None, "ModDate": None}
        )
        fig.savefig(
            path,
            dpi=dpi if suffix == "png" else None,
            bbox_inches="tight",
            metadata=metadata,
        )
        paths.append(path)
    plt.close(fig)
    return paths


def load_dataset(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _cell_map(dataset):
    return {(cell["pack_id"], cell["arm_id"]): cell for cell in dataset["cells"]}


def _value(cells, pack_id, arm_id, metric_id):
    cell = cells.get((pack_id, arm_id))
    if not cell:
        return np.nan
    value = (cell.get("metrics") or {}).get(metric_id)
    return np.nan if value is None else value


def _pack_order(dataset):
    arms = [arm["arm_id"] for arm in dataset["arms"]]
    direct = arms[0] if arms else None
    packs = [pack["pack_id"] for pack in dataset["packs"]]
    if direct:
        return sorted(
            packs,
            key=lambda pack: (
                -( _value(_cell_map(dataset), pack, direct, "att_arrived_s")
                    if np.isfinite(_value(_cell_map(dataset), pack, direct, "att_arrived_s"))
                    else -1e9 )
            ),
        )
    return sorted(packs)


def _arm_order(dataset):
    return [arm["arm_id"] for arm in dataset["arms"]]


def _arm_label(dataset, arm_id):
    for arm in dataset["arms"]:
        if arm["arm_id"] == arm_id:
            return arm.get("label") or arm_id
    return arm_id


def _colors(arms):
    return {arm: PALETTE[index % len(PALETTE)] for index, arm in enumerate(arms)}


def _grouped_bars(ax, dataset, metric_id, title, packs, arms):
    values = [
        [_value(_cell_map(dataset), pack, arm, metric_id) for pack in packs]
        for arm in arms
    ]
    x = np.arange(len(packs))
    width = 0.8 / max(1, len(arms))
    colors = _colors(arms)
    for index, arm in enumerate(arms):
        ax.bar(
            x + (index - (len(arms) - 1) / 2) * width,
            values[index],
            width,
            label=_arm_label(dataset, arm),
            color=colors[arm],
        )
    metric = get_metric(metric_id)
    ax.set_title(title)
    ax.set_ylabel(f"{metric.label} ({metric.unit})")
    ax.set_xticks(x)
    ax.set_xticklabels(packs, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=7)


def _stacked_backlog(ax, dataset, packs, arms):
    cells = _cell_map(dataset)
    x = np.arange(len(packs))
    width = 0.8 / max(1, len(arms))
    colors = _colors(arms)
    for index, arm in enumerate(arms):
        uninserted = [
            _value(cells, pack, arm, "uninserted_loaded_veh") for pack in packs
        ]
        unfinished = [
            _value(cells, pack, arm, "unfinished_departed_veh") for pack in packs
        ]
        offset = (index - (len(arms) - 1) / 2) * width
        ax.bar(
            x + offset, uninserted, width, color=colors[arm],
            label=_arm_label(dataset, arm) + " uninserted",
        )
        ax.bar(
            x + offset, unfinished, width, bottom=uninserted, color=colors[arm],
            alpha=0.45, label=_arm_label(dataset, arm) + " in-net",
        )
    ax.set_title("End-of-run backlog: uninserted + in-net unfinished")
    ax.set_ylabel("vehicles")
    ax.set_xticks(x)
    ax.set_xticklabels(packs, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=6, ncol=2)


def _heatmap(ax, dataset, metric_id, packs, arms):
    cells = _cell_map(dataset)
    values = np.array([
        [_value(cells, pack, arm, metric_id) for arm in arms] for pack in packs
    ], dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = "YlOrRd" if not get_metric(metric_id).higher_is_better else "YlGn"
    image = ax.imshow(masked, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([_arm_label(dataset, arm) for arm in arms], fontsize=8,
                       rotation=20, ha="right")
    ax.set_yticks(range(len(packs)))
    ax.set_yticklabels(packs, fontsize=8)
    for row in range(len(packs)):
        for col in range(len(arms)):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.1f}", ha="center", va="center",
                        fontsize=7)
    ax.set_title(get_metric(metric_id).label)
    plt.colorbar(image, ax=ax, shrink=0.8, label=get_metric(metric_id).unit)


def _scatter(ax, dataset, packs, arms):
    cells = _cell_map(dataset)
    colors = _colors(arms)
    for arm in arms:
        points = []
        for pack in packs:
            x = _value(cells, pack, arm, "throughput_arrived_veh")
            y = _value(cells, pack, arm, "att_arrived_s")
            backlog = _value(cells, pack, arm, "backlog_total_veh")
            if np.isfinite(x) and np.isfinite(y):
                size = 20 + (0 if not np.isfinite(backlog) else max(0, backlog)) * 0.6
                points.append((pack, x, y, size))
        ax.scatter(
            [point[1] for point in points],
            [point[2] for point in points],
            s=[point[3] for point in points],
            color=colors[arm], alpha=0.8,
            label=_arm_label(dataset, arm), edgecolors="black", linewidths=0.4,
        )
        for pack, x, y, _ in points:
            ax.annotate(pack, (x, y), fontsize=5, alpha=0.75,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Throughput (arrived vehicles)")
    ax.set_ylabel("Average travel time (s)")
    ax.set_title("ATT vs throughput (bubble size = backlog)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)


def _caption(dataset, figure_id, title, metric_ids, extra=()):
    lines = [f"## {figure_id}: {title}", ""]
    lines.append(
        f"Benchmark `{dataset['benchmark_id']}` / analysis `{dataset['analysis_id']}`; "
        f"{len(dataset['packs'])} packs x {len(dataset['arms'])} arms; "
        f"horizon {dataset['horizon_s']} s, decision interval "
        f"{dataset['decision_interval_s']} s, seed {dataset['seed']}, "
        f"episode {dataset['episode']}."
    )
    lines.append("")
    lines.append("Metric definitions used in this figure:")
    lines.append("")
    for metric_id in metric_ids:
        metric = get_metric(metric_id)
        lines.append(
            f"- `{metric_id}`: {metric.label}; unit `{metric.unit}`; "
            f"population: {metric.population}; time basis: {metric.time_basis}; "
            f"cross-arm comparable: {metric.cross_arm_comparable}. "
            f"{metric.definition}"
        )
    for note in extra:
        lines.append("")
        lines.append(note)
    lines.append("")
    return "\n".join(lines)


def plot_dataset(dataset, output_dir, dpi=160):
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    packs = _pack_order(dataset)
    arms = _arm_order(dataset)
    figures = []

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    _grouped_bars(axes[0], dataset, "att_arrived_s",
                  "Average travel time (arrived vehicles)", packs, arms)
    _grouped_bars(axes[1], dataset, "throughput_arrived_veh",
                  "Throughput (arrived vehicles)", packs, arms)
    _grouped_bars(axes[2], dataset, "backlog_total_veh",
                  "Total backlog (uninserted + unfinished)", packs, arms)
    fig.suptitle("xiasha1 demand packs x controllers: mandatory three-metric overview")
    paths = _save(fig, figures_dir / "fig01_overview_three_metrics", dpi)
    figures.append({
        "figure_id": "fig01_overview_three_metrics",
        "title": "Mandatory three-metric overview",
        "paths": [str(path.relative_to(output_dir)) for path in paths],
        "metric_ids": ["att_arrived_s", "throughput_arrived_veh", "backlog_total_veh"],
        "caption": _caption(
            dataset, "fig01_overview_three_metrics",
            "Mandatory three-metric overview",
            ["att_arrived_s", "throughput_arrived_veh", "backlog_total_veh"],
            ["ATT is the mean over arrived vehicles only; it must be read together "
             "with throughput and backlog."],
        ),
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    _grouped_bars(axes[0], dataset, "queue_decision_mean_veh",
                  "Mean queue (in-lane waiting vehicles)", packs, arms)
    _grouped_bars(axes[1], dataset, "waiting_end_in_net_s",
                  "Mean accumulated waiting at end (in-net vehicles)", packs, arms)
    fig.suptitle("Queue and end-of-run waiting: read the definitions before comparing")
    paths = _save(fig, figures_dir / "fig02_queue_waiting", dpi)
    figures.append({
        "figure_id": "fig02_queue_waiting",
        "title": "Queue and end-of-run waiting",
        "paths": [str(path.relative_to(output_dir)) for path in paths],
        "metric_ids": ["queue_decision_mean_veh", "waiting_end_in_net_s"],
        "caption": _caption(
            dataset, "fig02_queue_waiting", "Queue and end-of-run waiting",
            ["queue_decision_mean_veh", "waiting_end_in_net_s"],
            ["Queue is a decision-step time mean; waiting is an end-of-episode "
             "snapshot over remaining vehicles. They are not the same time basis.",
             "waiting_end_in_net_s is a mean over a small remaining population in "
             "some saturated packs; see waiting_n in tables/cells.csv.",
             "The demand-verification chain's SUMO summary halting count is a "
             "different metric and is not plotted here."],
        ),
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    _heatmap(axes[0], dataset, "att_arrived_s", packs, arms)
    _heatmap(axes[1], dataset, "throughput_arrived_veh", packs, arms)
    _heatmap(axes[2], dataset, "backlog_total_veh", packs, arms)
    fig.suptitle("Metric heatmaps: pack x controller")
    paths = _save(fig, figures_dir / "fig03_metric_heatmaps", dpi)
    figures.append({
        "figure_id": "fig03_metric_heatmaps",
        "title": "Metric heatmaps",
        "paths": [str(path.relative_to(output_dir)) for path in paths],
        "metric_ids": ["att_arrived_s", "throughput_arrived_veh", "backlog_total_veh"],
        "caption": _caption(
            dataset, "fig03_metric_heatmaps", "Metric heatmaps",
            ["att_arrived_s", "throughput_arrived_veh", "backlog_total_veh"],
            ["For ATT and backlog, lower is better; for throughput, higher is better. "
             "The color scale is per metric."],
        ),
    })

    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    _stacked_backlog(ax, dataset, packs, arms)
    paths = _save(fig, figures_dir / "fig04_backlog_decomposition", dpi)
    figures.append({
        "figure_id": "fig04_backlog_decomposition",
        "title": "Backlog decomposition",
        "paths": [str(path.relative_to(output_dir)) for path in paths],
        "metric_ids": ["uninserted_loaded_veh", "unfinished_departed_veh",
                       "backlog_total_veh"],
        "caption": _caption(
            dataset, "fig04_backlog_decomposition", "Backlog decomposition",
            ["uninserted_loaded_veh", "unfinished_departed_veh",
             "backlog_total_veh"],
            ["Uninserted = loaded but not departed; in-net unfinished = departed "
             "but not arrived. The sum is the total backlog."],
        ),
    })

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    _scatter(ax, dataset, packs, arms)
    paths = _save(fig, figures_dir / "fig05_tradeoff_scatter", dpi)
    figures.append({
        "figure_id": "fig05_tradeoff_scatter",
        "title": "ATT vs throughput trade-off",
        "paths": [str(path.relative_to(output_dir)) for path in paths],
        "metric_ids": ["att_arrived_s", "throughput_arrived_veh", "backlog_total_veh"],
        "caption": _caption(
            dataset, "fig05_tradeoff_scatter", "ATT vs throughput trade-off",
            ["att_arrived_s", "throughput_arrived_veh", "backlog_total_veh"],
            ["Do not rank controllers by throughput alone: a controller can complete "
             "more trips while leaving a larger backlog and higher ATT."],
        ),
    })

    captions = "\n".join(figure["caption"] for figure in figures)
    (output_dir / "captions.md").write_text(captions, encoding="utf-8")
    figure_manifest = {
        "schema_version": 1,
        "benchmark_id": dataset["benchmark_id"],
        "analysis_id": dataset["analysis_id"],
        "dpi": dpi,
        "figures": figures,
    }
    (output_dir / "figures_manifest.json").write_text(
        json.dumps(figure_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return figure_manifest
