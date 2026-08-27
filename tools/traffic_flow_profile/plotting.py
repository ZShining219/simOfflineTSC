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
from matplotlib.ticker import MaxNLocator

COLORS = {"W": "#4C78A8", "S": "#F58518", "E": "#54A24B", "N": "#E45756"}
MOVEMENT_LABELS = {
    "left": "Left",
    "through": "Through",
    "right": "Right",
    "u_turn": "U-turn",
}


def _label_colors(labels):
    fallback = plt.get_cmap("tab20")
    return {
        label: COLORS.get(label, fallback(index % 20))
        for index, label in enumerate(labels)
    }


def _shared_upper_limit(values):
    maximum = max(values, default=0)
    if maximum <= 0:
        return 1.0
    locator = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10])
    ticks = locator.tick_values(0, maximum * 1.08)
    return float(ticks[-1])


def calculate_shared_plot_limits(metrics_collection):
    """Return common metric limits for a comparable group of profile figures."""
    five_minute_values = []
    approach_values = []
    movement_values = []
    fifteen_minute_values = []
    crossing_values = []

    for metrics in metrics_collection:
        standard = metrics["standard_metrics"]
        profile = metrics["profile_metrics"]
        five_minute_values.extend(profile["temporal_total_5min"])
        fifteen_minute_values.extend(profile["temporal_total_15min"])
        if metrics["scenario"].get("scope", "signal") == "signal":
            approach_values.extend(standard["approach_vehicle_count"].values())
            for approach in standard["approach_order"]:
                movement_values.extend(standard["movement_matrix"][approach].values())
        else:
            approach_values.extend(
                standard.get("route_origin_approach_count", {}).values()
            )
            movement_values.extend(
                standard.get("signal_crossing_movement_count", {}).values()
            )
            crossing_values.extend(
                standard.get("signal_crossing_count_by_junction", {}).values()
            )

    return {
        "five_minute": _shared_upper_limit(five_minute_values),
        "approach": _shared_upper_limit(approach_values),
        "movement": _shared_upper_limit(movement_values),
        "fifteen_minute": _shared_upper_limit(fifteen_minute_values),
        "crossing": _shared_upper_limit(crossing_values),
    }


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


def _render_network_profile(metrics, output_path: Path, dpi: int, plot_limits):
    standard = metrics["standard_metrics"]
    profile = metrics["profile_metrics"]
    scenario = metrics["scenario"]
    five_total = profile["temporal_total_5min"]
    fifteen_total = profile["temporal_total_15min"]
    x5 = np.arange(len(five_total))
    x15 = np.arange(len(fifteen_total))
    relative_peak_start = standard["peak_15min_start_seconds"] - scenario["begin_seconds"]
    peak_index = int(relative_peak_start // 900)

    origin_order = tuple(standard["route_origin_approach_order"])
    origin_counts = standard["route_origin_approach_count"]
    origin_shares = standard["route_origin_approach_share"]
    signal_order = tuple(standard["signal_order"])
    signal_crossings = standard["signal_crossing_count_by_junction"]
    movement_order = tuple(standard["signal_crossing_movement_order"])
    movement_counts = standard["signal_crossing_movement_count"]
    movement_shares = standard["signal_crossing_movement_share"]
    origin_colors = _label_colors(origin_order)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(f"SUMO Network Demand Profile: {scenario['id']}", fontsize=16)

    axis = axes[0, 0]
    axis.bar(x5, five_total, color="#72B7B2")
    axis.plot(x5, five_total, color="#1F4E5F", marker="o", linewidth=1.5)
    axis.set_title("(a) Unique departures by 5-minute interval")
    axis.set_xlabel("Time interval (min)")
    axis.set_ylabel("Vehicles / 5 min")
    axis.set_xticks(x5, [f"{index * 5}-{(index + 1) * 5}" for index in x5], rotation=45)
    axis.set_ylim(0, plot_limits["five_minute"])

    axis = axes[0, 1]
    bars = axis.bar(x15, fifteen_total, color="#B279A2")
    if bars:
        bars[min(peak_index, len(bars) - 1)].set_color("#E45756")
    _annotate_bars(axis, bars, [str(value) for value in fifteen_total])
    axis.set_title(f"(b) 15-minute departures | PHF={standard['peak_hour_factor']:.3f}")
    axis.set_xlabel("Time interval (min)")
    axis.set_ylabel("Vehicles / 15 min")
    axis.set_xticks(x15, [f"{index * 15}-{(index + 1) * 15}" for index in x15])
    axis.set_ylim(0, plot_limits["fifteen_minute"])

    axis = axes[0, 2]
    origin_values = [origin_counts[item] for item in origin_order]
    bars = axis.bar(
        origin_order,
        origin_values,
        color=[origin_colors[item] for item in origin_order],
    )
    _annotate_bars(
        axis,
        bars,
        [
            f"{value}\n({origin_shares[item]:.1%})"
            for item, value in zip(origin_order, origin_values)
        ],
    )
    axis.set_title("(c) Unique demand by route-origin direction")
    axis.set_xlabel("Origin direction")
    axis.set_ylabel("Unique vehicles")
    axis.set_ylim(0, plot_limits["approach"])

    axis = axes[1, 0]
    signal_values = [signal_crossings[item] for item in signal_order]
    bars = axis.bar(signal_order, signal_values, color="#59A14F")
    _annotate_bars(axis, bars, [str(value) for value in signal_values])
    axis.set_title("(d) Route crossings by signal junction")
    axis.set_xlabel("Signal junction")
    axis.set_ylabel("Crossing events")
    axis.tick_params(axis="x", labelrotation=35)
    axis.set_ylim(0, plot_limits["crossing"])

    axis = axes[1, 1]
    movement_values = [movement_counts[item] for item in movement_order]
    bars = axis.bar(
        [MOVEMENT_LABELS.get(item, item) for item in movement_order],
        movement_values,
        color="#F28E2B",
    )
    _annotate_bars(
        axis,
        bars,
        [
            f"{value}\n({movement_shares[item]:.1%})"
            for item, value in zip(movement_order, movement_values)
        ],
    )
    axis.set_title("(e) Aggregate movements across signal crossings")
    axis.set_xlabel("Movement")
    axis.set_ylabel("Crossing events")
    axis.set_ylim(0, plot_limits["movement"])

    axis = axes[1, 2]
    axis.axis("off")
    summary_lines = [
        "(f) Network-level metric summary",
        "",
        f"Analysis window: {scenario['duration_seconds']:.0f} s",
        f"Unique departures: {standard['vehicle_count']}",
        f"Equivalent hourly volume: {standard['hourly_volume_vph']:.0f} veh/h",
        f"Peak 15-min count: {standard['peak_15min_vehicle_count']} veh",
        f"Peak 15-min rate: {standard['peak_15min_equivalent_vph']:.0f} veh/h",
        f"PHF: {standard['peak_hour_factor']:.3f}",
        f"Second-half change: {profile['half_hour_change_pct']:+.1f}%",
        "",
        f"Signal junctions: {scenario['signal_junction_count']}",
        f"Signal crossings: {standard['signal_crossing_count']}",
        f"Crossings / vehicle: {standard['signal_crossings_per_vehicle']:.2f}",
        f"Unclassified route origins: {standard['route_origin_unclassified_count']}",
        "",
        "Unique totals count each demand element once.",
        "Crossing totals count every traversed signal junction.",
    ]
    axis.text(0.02, 0.98, "\n".join(summary_lines), va="top", ha="left", fontsize=11, family="monospace")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_profile(metrics, output_path: Path, dpi: int = 160, plot_limits=None):
    if plot_limits is None:
        plot_limits = calculate_shared_plot_limits([metrics])
    if metrics["scenario"].get("scope", "signal") == "network":
        _render_network_profile(metrics, output_path, dpi, plot_limits)
        return

    standard = metrics["standard_metrics"]
    profile = metrics["profile_metrics"]
    scenario = metrics["scenario"]
    approaches = tuple(standard["approach_order"])
    movements = tuple(standard["movement_order"])
    colors = _label_colors(approaches)
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
    axis.set_ylim(0, plot_limits["five_minute"])

    axis = axes[0, 1]
    bottom = np.zeros(len(x5))
    for approach in approaches:
        values = np.asarray(five_approach[approach])
        axis.bar(x5, values, bottom=bottom, label=approach, color=colors[approach])
        bottom += values
    axis.set_title("(b) Approach demand by 5-minute interval")
    axis.set_xlabel("Time interval (min)")
    axis.set_ylabel("Vehicles / 5 min")
    axis.set_xticks(x5, [f"{index * 5}-{(index + 1) * 5}" for index in x5], rotation=45)
    axis.set_ylim(0, plot_limits["five_minute"])
    axis.legend(title="Approach", ncols=4, fontsize=8)

    axis = axes[0, 2]
    approach_counts = standard["approach_vehicle_count"]
    approach_shares = standard["approach_share"]
    approach_values = [approach_counts[item] for item in approaches]
    bars = axis.bar(approaches, approach_values, color=[colors[item] for item in approaches])
    _annotate_bars(
        axis,
        bars,
        [f"{value}\n({approach_shares[item]:.1%})" for item, value in zip(approaches, approach_values)],
    )
    axis.set_title("(c) Hourly demand by approach")
    axis.set_xlabel("Approach")
    axis.set_ylabel("Vehicles / hour")
    axis.set_ylim(0, plot_limits["approach"])

    axis = axes[1, 0]
    matrix = np.asarray(
        [[standard["movement_matrix"][approach][movement] for movement in movements] for approach in approaches]
    )
    image = axis.imshow(
        matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=plot_limits["movement"]
    )
    axis.set_title("(d) Turning-movement matrix")
    axis.set_xticks(range(len(movements)), [MOVEMENT_LABELS.get(item, item) for item in movements])
    axis.set_yticks(range(len(approaches)), approaches)
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
    axis.set_ylim(0, plot_limits["fifteen_minute"])

    axis = axes[1, 2]
    axis.axis("off")
    movement_share = standard["movement_share"]
    movement_summary = " / ".join(
        f"{MOVEMENT_LABELS.get(item, item)} {movement_share[item]:.1%}"
        for item in movements
    )
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
        f"Dominant approach: {standard['dominant_approach'] or 'n/a'} "
        f"({standard['dominant_approach_share']:.1%})",
        f"EW / NS share: {standard['east_west_share']:.1%} / {standard['north_south_share']:.1%}",
        movement_summary,
        f"Second-half change: {profile['half_hour_change_pct']:+.1f}%",
    ]
    axis.text(0.02, 0.98, "\n".join(summary_lines), va="top", ha="left", fontsize=11, family="monospace")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
