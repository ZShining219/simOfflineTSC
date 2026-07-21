import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    matplotlib_cache = Path(tempfile.gettempdir()) / "simofflinetsc-matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .models import APPROACHES, MOVEMENTS


COLORS = {"W": "#4C78A8", "S": "#F58518", "E": "#54A24B", "N": "#E45756"}
MOVEMENT_LABELS = {"left": "Left", "through": "Through", "right": "Right"}


def _annotate_bars(axis, bars, labels):
    for bar, label in zip(bars, labels):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def render_profile(metrics, output_path: Path, dpi: int = 160):
    standard = metrics["standard_metrics"]
    profile = metrics["profile_metrics"]
    scenario = metrics["scenario"]
    five_total = profile["temporal_total_5min"]
    five_approach = profile["temporal_approach_5min"]
    fifteen_total = profile["temporal_total_15min"]
    x5 = np.arange(len(five_total))
    x15 = np.arange(len(fifteen_total))
    relative_peak_start = standard["peak_15min_start_seconds"] - scenario["begin_seconds"]
    relative_peak_end = standard["peak_15min_end_seconds"] - scenario["begin_seconds"]
    peak_start = int(relative_peak_start // 60)
    peak_end = int(relative_peak_end // 60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(f"SUMO Traffic Demand Profile: {scenario['id']}", fontsize=16)

    axis = axes[0, 0]
    bars = axis.bar(x5, five_total, color="#72B7B2")
    axis.plot(x5, five_total, color="#1F4E5F", marker="o", linewidth=1.5)
    peak_index = int(relative_peak_start // 300)
    axis.axvspan(peak_index - 0.5, peak_index + 2.5, color="#E45756", alpha=0.15)
    axis.set_title("(a) Total demand by 5-minute interval")
    axis.set_xlabel("Time interval (min)")
    axis.set_ylabel("Vehicles / 5 min")
    axis.set_xticks(x5, [f"{index * 5}-{(index + 1) * 5}" for index in x5], rotation=45)

    axis = axes[0, 1]
    bottom = np.zeros(len(x5))
    for approach in APPROACHES:
        values = np.asarray(five_approach[approach])
        axis.bar(x5, values, bottom=bottom, label=approach, color=COLORS[approach])
        bottom += values
    axis.set_title("(b) Approach demand by 5-minute interval")
    axis.set_xlabel("Time interval (min)")
    axis.set_ylabel("Vehicles / 5 min")
    axis.set_xticks(x5, [f"{index * 5}-{(index + 1) * 5}" for index in x5], rotation=45)
    axis.legend(title="Approach", ncols=4, fontsize=8)

    axis = axes[0, 2]
    approach_counts = standard["approach_vehicle_count"]
    approach_shares = standard["approach_share"]
    approach_values = [approach_counts[item] for item in APPROACHES]
    bars = axis.bar(APPROACHES, approach_values, color=[COLORS[item] for item in APPROACHES])
    _annotate_bars(
        axis,
        bars,
        [f"{value}\n({approach_shares[item]:.1%})" for item, value in zip(APPROACHES, approach_values)],
    )
    axis.set_title("(c) Hourly demand by approach")
    axis.set_xlabel("Approach")
    axis.set_ylabel("Vehicles / hour")
    axis.set_ylim(0, max(approach_values) * 1.2)

    axis = axes[1, 0]
    matrix = np.asarray(
        [[standard["movement_matrix"][approach][movement] for movement in MOVEMENTS] for approach in APPROACHES]
    )
    image = axis.imshow(matrix, cmap="YlGnBu", aspect="auto")
    axis.set_title("(d) Turning-movement matrix")
    axis.set_xticks(range(len(MOVEMENTS)), [MOVEMENT_LABELS[item] for item in MOVEMENTS])
    axis.set_yticks(range(len(APPROACHES)), APPROACHES)
    axis.set_xlabel("Movement")
    axis.set_ylabel("Approach")
    for row in range(matrix.shape[0]):
        row_total = matrix[row].sum()
        for column in range(matrix.shape[1]):
            share = matrix[row, column] / row_total if row_total else 0.0
            axis.text(column, row, f"{matrix[row, column]}\n{share:.1%}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axis, shrink=0.8, label="Vehicles / hour")

    axis = axes[1, 1]
    bars = axis.bar(x15, fifteen_total, color="#B279A2")
    peak_15_index = int(relative_peak_start // 900)
    bars[peak_15_index].set_color("#E45756")
    _annotate_bars(axis, bars, [str(value) for value in fifteen_total])
    axis.set_title(f"(e) 15-minute demand | PHF={standard['peak_hour_factor']:.3f}")
    axis.set_xlabel("Time interval (min)")
    axis.set_ylabel("Vehicles / 15 min")
    axis.set_xticks(x15, [f"{index * 15}-{(index + 1) * 15}" for index in x15])
    axis.set_ylim(0, max(fifteen_total) * 1.18)

    axis = axes[1, 2]
    axis.axis("off")
    movement_share = standard["movement_share"]
    summary_lines = [
        "(f) Standard metric summary",
        "",
        f"Analysis window: {scenario['duration_seconds']:.0f} s",
        f"Hourly volume: {standard['hourly_volume_vph']:.0f} veh/h",
        f"Peak 15-min count: {standard['peak_15min_vehicle_count']} veh",
        f"Peak 15-min rate: {standard['peak_15min_equivalent_vph']:.0f} veh/h",
        f"Peak window: {peak_start}-{peak_end} min",
        f"PHF: {standard['peak_hour_factor']:.3f}",
        "",
        f"Dominant approach: {standard['dominant_approach']} "
        f"({standard['dominant_approach_share']:.1%})",
        f"EW / NS share: {standard['east_west_share']:.1%} / {standard['north_south_share']:.1%}",
        f"Left / through / right: {movement_share['left']:.1%} / "
        f"{movement_share['through']:.1%} / {movement_share['right']:.1%}",
        f"Second-half change: {profile['half_hour_change_pct']:+.1f}%",
    ]
    axis.text(0.02, 0.98, "\n".join(summary_lines), va="top", ha="left", fontsize=11, family="monospace")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
