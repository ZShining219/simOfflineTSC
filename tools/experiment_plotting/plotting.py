"""Semantic, publication-oriented plots for validated experiment records."""

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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from .profiles import get_profile


COLORS = {
    "dqn": "#0072B2",
    "batch_dqn": "#009E73",
    "cql_dqn": "#D55E00",
    "fixedtime": "#7F7F7F",
    "maxpressure": "#E69F00",
}
SOURCE_STYLES = {"train": ":", "evaluation": "-"}
BOOTSTRAP_SEED = 20260722
BOOTSTRAP_SAMPLES = 1000


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _save(fig, output_base, dpi):
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = output_base.with_suffix(f".{suffix}")
        metadata = (
            {"Software": "tools.experiment_plotting"}
            if suffix == "png"
            else {
                "Creator": "tools.experiment_plotting",
                "CreationDate": None,
                "ModDate": None,
            }
        )
        fig.savefig(
            path, dpi=dpi if suffix == "png" else None,
            bbox_inches="tight", metadata=metadata,
        )
        paths.append(path)
    plt.close(fig)
    return paths


def _frame(rows):
    return pd.DataFrame.from_records(rows)


def _ordered(values, preferred=()):
    encountered = list(dict.fromkeys(value for value in values if pd.notna(value)))
    return [value for value in preferred if value in encountered] + [
        value for value in encountered if value not in preferred
    ]


def _grid(count, width=6.0, height=4.0):
    columns = 2 if count > 1 else 1
    rows = max(1, math.ceil(count / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(width * columns, height * rows),
        squeeze=False, constrained_layout=True,
    )
    flat = list(axes.flat)
    for axis in flat[count:]:
        axis.set_visible(False)
    return fig, flat[:count]


def _agent_color(agent):
    if agent in COLORS:
        return COLORS[agent]
    stable_index = sum(
        (index + 1) * ord(character) for index, character in enumerate(str(agent))
    ) % 10
    return sns.color_palette("colorblind", 10)[stable_index]


def _evaluation_rows(frame):
    return frame[frame["record_type"].isin(("EVALUATION", "FINAL_EVALUATION"))]


def _non_baseline_rows(frame):
    if "role" not in frame:
        return frame
    return frame[frame["role"] != "baseline"]


def _baseline_levels(frame, network, metric):
    if "role" not in frame or metric not in frame:
        return []
    selected = frame[
        (frame["role"] == "baseline")
        & (frame["network"] == network)
        & frame[metric].map(_finite)
    ]
    levels = []
    for agent, group in selected.groupby("agent", sort=False):
        final = group[group["record_type"] == "FINAL_EVALUATION"]
        row = final.iloc[-1] if not final.empty else group.iloc[-1]
        levels.append((agent, float(row[metric])))
    return levels


def _plot_seed_mean_ci(
    axis, rows, x_field, y_field, seed_field, color, linestyle="-",
    linewidth=1.6,
):
    """Plot a seed mean and a vectorized percentile-bootstrap confidence band."""

    pivot = rows.pivot_table(
        index=seed_field, columns=x_field, values=y_field, aggfunc="mean",
    ).sort_index(axis=1)
    if pivot.empty:
        return False
    x_values = pivot.columns.to_numpy()
    values = pivot.to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0)
    axis.plot(
        x_values, mean, color=color, linestyle=linestyle, linewidth=linewidth,
    )
    if values.shape[0] > 1:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        indices = rng.integers(
            0, values.shape[0], size=(BOOTSTRAP_SAMPLES, values.shape[0]),
        )
        bootstrap_means = np.nanmean(values[indices, :], axis=1)
        lower, upper = np.nanpercentile(bootstrap_means, (2.5, 97.5), axis=0)
        axis.fill_between(x_values, lower, upper, color=color, alpha=0.18, linewidth=0)
    return True


def _seed_summary_facets(
    rows, output_base, value_field, title, ylabel, profile, dpi=160,
    include_baselines=True, zero_line=False,
):
    frame = _frame(rows)
    if frame.empty or value_field not in frame:
        fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        axis.text(0.5, 0.5, "No compatible data", ha="center", va="center")
        axis.set_axis_off()
        fig.suptitle(title)
        return _save(fig, output_base, dpi)

    network_field = profile.network_field
    algorithm_field = profile.algorithm_field
    seed_field = profile.seed_field
    frame = frame[frame[value_field].map(_finite)].copy()
    if not include_baselines and "role" in frame:
        frame = frame[frame["role"] != "baseline"]
    if frame.empty:
        fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        axis.text(0.5, 0.5, "No compatible data", ha="center", va="center")
        axis.set_axis_off()
        fig.suptitle(title)
        return _save(fig, output_base, dpi)
    networks = _ordered(frame[network_field], profile.network_order)
    fig, axes = _grid(len(networks))
    for axis, network in zip(axes, networks):
        group = frame[frame[network_field] == network]
        algorithms = _ordered(group[algorithm_field], profile.algorithm_order)
        palette = {agent: _agent_color(agent) for agent in algorithms}
        np.random.seed(BOOTSTRAP_SEED)
        sns.stripplot(
            data=group, x=algorithm_field, y=value_field, order=algorithms,
            hue=algorithm_field, hue_order=algorithms, palette=palette,
            jitter=0.12, alpha=0.58, size=5, ax=axis, legend=False,
        )
        for index, algorithm in enumerate(algorithms):
            values = group.loc[
                group[algorithm_field] == algorithm, value_field
            ].astype(float)
            if values.empty:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            axis.errorbar(
                index, mean, yerr=std, fmt="D", markersize=5.5,
                color=palette[algorithm], markeredgecolor="black",
                markeredgewidth=0.5, capsize=4, linewidth=1.4, zorder=5,
            )
        if zero_line:
            axis.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.65)
        axis.set_title(str(network))
        axis.set_xlabel("")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.2)
        if seed_field in group:
            axis.text(
                0.99, 0.98, f"seeds: {group[seed_field].nunique()}",
                transform=axis.transAxes, ha="right", va="top",
                fontsize=8, color="#555555",
            )
    fig.suptitle(title)
    fig.legend(
        handles=(
            Line2D([], [], marker="o", linestyle="", color="#666666", alpha=0.6,
                   label="Individual seed"),
            Line2D([], [], marker="D", linestyle="", color="black",
                   label="Mean ± sample SD"),
        ),
        loc="outside upper right", frameon=False,
    )
    return _save(fig, output_base, dpi)


def render_final_performance(comparison_rows, output_base, profile, dpi=160):
    selected = [row for row in comparison_rows if row.get("selection") == "final"]
    return _seed_summary_facets(
        selected, output_base, "travel_time",
        "Final evaluation travel time", "Average travel time",
        profile, dpi=dpi,
    )


def render_evaluation_learning_curve(metric_rows, output_base, profile, dpi=160):
    frame = _frame(metric_rows)
    selected = _evaluation_rows(_non_baseline_rows(frame))
    selected = selected[selected["travel_time"].map(_finite)].copy()
    networks = _ordered(frame[profile.network_field], profile.network_order)
    fig, axes = _grid(len(networks), width=6.2, height=4.1)
    legend_agents = []
    legend_baselines = []
    for axis, network in zip(axes, networks):
        group = selected[selected[profile.network_field] == network]
        algorithms = _ordered(group[profile.algorithm_field], profile.algorithm_order)
        for algorithm in algorithms:
            algorithm_rows = group[group[profile.algorithm_field] == algorithm]
            if algorithm_rows.empty:
                continue
            _plot_seed_mean_ci(
                axis, algorithm_rows, profile.progress_field, "travel_time",
                profile.seed_field, _agent_color(algorithm), linewidth=1.6,
            )
            if algorithm not in legend_agents:
                legend_agents.append(algorithm)
        for baseline, value in _baseline_levels(frame, network, "travel_time"):
            axis.axhline(
                value, color=_agent_color(baseline), linestyle="--",
                linewidth=1.1, alpha=0.9,
            )
            if baseline not in legend_baselines:
                legend_baselines.append(baseline)
        axis.set_title(str(network))
        axis.set_xlabel(profile.progress_label)
        axis.set_ylabel("Average travel time")
        axis.grid(alpha=0.2)
    handles = [
        Line2D([], [], color=_agent_color(agent), linewidth=2, label=f"{agent} mean (95% CI)")
        for agent in legend_agents
    ] + [
        Line2D([], [], color=_agent_color(agent), linestyle="--", linewidth=1.3,
               label=f"{agent} baseline")
        for agent in legend_baselines
    ]
    if handles:
        fig.legend(handles=handles, loc="outside upper right", frameon=False)
    fig.suptitle("Evaluation learning curve across seeds")
    return _save(fig, output_base, dpi)


def render_learning_speed(auc_rows, output_base, profile, dpi=160):
    selected = [
        row for row in auc_rows
        if row.get("metric") == "travel_time"
        and row.get("curve_source") == "EVALUATION"
    ]
    return _seed_summary_facets(
        selected, output_base, "auc",
        "Early learning: first-100 evaluation travel-time AUC",
        "AUC (lower is better)", profile, dpi=dpi,
        include_baselines=False,
    )


def _latest_action_concentration(action_rows):
    frame = _frame(action_rows)
    if frame.empty:
        return []
    frame = _evaluation_rows(_non_baseline_rows(frame))
    if frame.empty:
        return []
    latest_episode = frame.groupby("run_key")["episode"].transform("max")
    latest = frame[frame["episode"] == latest_episode]
    identity = [
        "run_key", "role", "agent", "network", "training_seed", "run_dir",
    ]
    return (
        latest.groupby(identity, as_index=False)["fraction"]
        .max()
        .rename(columns={"fraction": "maximum_action_fraction"})
        .to_dict("records")
    )


def render_action_concentration(action_rows, output_base, profile, dpi=160):
    return _seed_summary_facets(
        _latest_action_concentration(action_rows), output_base,
        "maximum_action_fraction", "Final evaluation action concentration",
        "Largest action fraction", profile, dpi=dpi, include_baselines=False,
    )


def _training_cost_rows(metric_rows):
    frame = _frame(metric_rows)
    if frame.empty:
        return []
    frame = _non_baseline_rows(frame)
    frame = frame[
        (frame["record_type"] == "TRAIN")
        & frame["wall_time_seconds"].map(_finite)
    ]
    identity = [
        "run_key", "role", "agent", "network", "training_seed", "run_dir",
    ]
    costs = frame.groupby(identity, as_index=False)["wall_time_seconds"].sum()
    costs["wall_time_hours"] = costs["wall_time_seconds"] / 3600.0
    return costs.to_dict("records")


def render_training_cost(metric_rows, output_base, profile, dpi=160):
    return _seed_summary_facets(
        _training_cost_rows(metric_rows), output_base, "wall_time_hours",
        "Measured training wall-clock cost", "Wall-clock time (hours)",
        profile, dpi=dpi, include_baselines=False,
    )


def render_individual_travel_time(metric_rows, output_base, profile, dpi=160):
    frame = _non_baseline_rows(_frame(metric_rows))
    frame = frame[frame["travel_time"].map(_finite)].copy()
    networks = _ordered(frame[profile.network_field], profile.network_order)
    seeds = _ordered(frame[profile.seed_field])
    seed_palette = dict(zip(seeds, sns.color_palette("colorblind", len(seeds))))
    fig, axes = _grid(len(networks), width=6.2, height=4.1)
    for axis, network in zip(axes, networks):
        group = frame[frame[profile.network_field] == network]
        for (_, seed), run in group.groupby(["run_key", profile.seed_field], sort=False):
            for source, record_types in (
                ("train", ("TRAIN",)),
                ("evaluation", ("EVALUATION", "FINAL_EVALUATION")),
            ):
                series = run[run["record_type"].isin(record_types)].sort_values(
                    profile.progress_field
                )
                if series.empty:
                    continue
                axis.plot(
                    series[profile.progress_field], series["travel_time"],
                    color=seed_palette[seed], linestyle=SOURCE_STYLES[source],
                    linewidth=1.0 if source == "train" else 1.4,
                    alpha=0.45 if source == "train" else 0.9,
                )
        for baseline, value in _baseline_levels(frame=_frame(metric_rows), network=network,
                                                metric="travel_time"):
            axis.axhline(value, color=_agent_color(baseline), linestyle="--", linewidth=1.0)
        axis.set_title(str(network))
        axis.set_xlabel(profile.progress_label)
        axis.set_ylabel("Average travel time")
        axis.grid(alpha=0.2)
    handles = [
        Line2D([], [], color=seed_palette[seed], linewidth=2, label=f"seed {seed}")
        for seed in seeds
    ] + [
        Line2D([], [], color="#333333", linestyle=SOURCE_STYLES[source],
               label=source)
        for source in ("train", "evaluation")
    ]
    if handles:
        fig.legend(handles=handles, loc="outside upper right", frameon=False, ncols=2)
    fig.suptitle("Individual-seed travel-time diagnostics")
    return _save(fig, output_base, dpi)


def render_metric_diagnostic(
    metric_rows, output_base, metric, title, ylabel, profile,
    sources=("evaluation",), include_baselines=False, dpi=160,
):
    frame = _frame(metric_rows)
    selected = _non_baseline_rows(frame)
    selected = selected[selected[metric].map(_finite)].copy()
    networks = _ordered(selected[profile.network_field], profile.network_order)
    fig, axes = _grid(len(networks), width=6.2, height=4.0)
    algorithms_seen = []
    for axis, network in zip(axes, networks):
        group = selected[selected[profile.network_field] == network]
        algorithms = _ordered(group[profile.algorithm_field], profile.algorithm_order)
        for algorithm in algorithms:
            for source in sources:
                record_types = (
                    ("TRAIN",) if source == "train"
                    else ("EVALUATION", "FINAL_EVALUATION")
                )
                series = group[
                    (group[profile.algorithm_field] == algorithm)
                    & group["record_type"].isin(record_types)
                ]
                if series.empty:
                    continue
                _plot_seed_mean_ci(
                    axis, series, profile.progress_field, metric,
                    profile.seed_field, _agent_color(algorithm),
                    linestyle=SOURCE_STYLES[source], linewidth=1.5,
                )
                if algorithm not in algorithms_seen:
                    algorithms_seen.append(algorithm)
        if include_baselines:
            for baseline, value in _baseline_levels(frame, network, metric):
                axis.axhline(
                    value, color=_agent_color(baseline), linestyle="--",
                    linewidth=1.0, alpha=0.9,
                )
        axis.set_title(str(network))
        axis.set_xlabel(profile.progress_label)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    handles = [
        Line2D([], [], color=_agent_color(agent), linewidth=2, label=agent)
        for agent in algorithms_seen
    ]
    if len(sources) > 1:
        handles.extend(
            Line2D([], [], color="#333333", linestyle=SOURCE_STYLES[source], label=source)
            for source in sources
        )
    if handles:
        fig.legend(handles=handles, loc="outside upper right", frameon=False)
    fig.suptitle(title)
    return _save(fig, output_base, dpi)


def render_action_distribution(action_rows, output_base, profile, dpi=160):
    frame = _frame(action_rows)
    frame = _evaluation_rows(_non_baseline_rows(frame))
    if frame.empty:
        fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        axis.text(0.5, 0.5, "No action-distribution data", ha="center", va="center")
        axis.set_axis_off()
        return _save(fig, output_base, dpi)
    latest_episode = frame.groupby("run_key")["episode"].transform("max")
    latest = frame[frame["episode"] == latest_episode].copy()
    networks = _ordered(latest[profile.network_field], profile.network_order)
    fig, axes = _grid(len(networks), width=6.2, height=4.2)
    for index, (axis, network) in enumerate(zip(axes, networks)):
        group = latest[latest[profile.network_field] == network].copy()
        group["run_label"] = group.apply(
            lambda row: f"{row[profile.algorithm_field]} seed {row[profile.seed_field]}",
            axis=1,
        )
        matrix = group.pivot_table(
            index="run_label", columns="action", values="fraction", aggfunc="first",
        ).fillna(0.0)
        matrix = matrix.reindex(sorted(matrix.columns), axis=1)
        sns.heatmap(
            matrix, vmin=0.0, vmax=1.0, cmap="viridis", annot=True, fmt=".2f",
            linewidths=0.35, cbar=index == len(networks) - 1, ax=axis,
        )
        axis.set_title(str(network))
        axis.set_xlabel("Action")
        axis.set_ylabel("")
    fig.suptitle("Latest evaluation action distribution")
    return _save(fig, output_base, dpi)


def render_final_best_gap(comparison_rows, output_base, profile, dpi=160):
    frame = _frame(comparison_rows)
    if frame.empty:
        return _seed_summary_facets(
            [], output_base, "gap", "Final-to-best travel-time gap", "Gap",
            profile, dpi=dpi,
        )
    frame = _non_baseline_rows(frame)
    identity = [
        "run_key", "role", "agent", "network", "training_seed", "run_dir",
    ]
    wide = frame.pivot_table(
        index=identity, columns="selection", values="travel_time", aggfunc="first",
    ).reset_index()
    if "final" not in wide or "best" not in wide:
        rows = []
    else:
        wide["final_minus_best"] = wide["final"] - wide["best"]
        rows = wide.to_dict("records")
    return _seed_summary_facets(
        rows, output_base, "final_minus_best",
        "Final-to-best evaluation travel-time gap",
        "Final minus best travel time", profile, dpi=dpi,
        include_baselines=False, zero_line=True,
    )


def render_efficiency_curve(
    efficiency_rows, output_base, x_field, y_field, title, xlabel, ylabel,
    profile, dpi=160,
):
    frame = _frame(efficiency_rows)
    frame = frame[
        frame[x_field].map(_finite) & frame[y_field].map(_finite)
    ].copy()
    networks = _ordered(frame[profile.network_field], profile.network_order)
    fig, axes = _grid(len(networks), width=6.2, height=4.0)
    for axis, network in zip(axes, networks):
        group = frame[frame[profile.network_field] == network]
        y_pivot = group.pivot_table(
            index=profile.seed_field, columns="episode", values=y_field,
            aggfunc="mean",
        ).sort_index(axis=1)
        x_by_episode = group.groupby("episode")[x_field].mean().reindex(
            y_pivot.columns
        )
        if not y_pivot.empty:
            values = y_pivot.to_numpy(dtype=float)
            x_values = x_by_episode.to_numpy(dtype=float)
            mean = np.nanmean(values, axis=0)
            axis.plot(
                x_values, mean, color=_agent_color("dqn"), linewidth=1.6,
            )
            if values.shape[0] > 1:
                rng = np.random.default_rng(BOOTSTRAP_SEED)
                indices = rng.integers(
                    0, values.shape[0],
                    size=(BOOTSTRAP_SAMPLES, values.shape[0]),
                )
                bootstrap_means = np.mean(values[indices, :], axis=1)
                lower, upper = np.percentile(
                    bootstrap_means, (2.5, 97.5), axis=0
                )
                axis.fill_between(
                    x_values, lower, upper, color=_agent_color("dqn"),
                    alpha=0.18, linewidth=0,
                )
        axis.set_title(str(network))
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    if networks:
        fig.legend(
            handles=[Line2D([], [], color=_agent_color("dqn"), linewidth=2,
                            label="DQN mean (95% bootstrap CI)")],
            loc="outside upper right", frameon=False,
        )
    fig.suptitle(title)
    return _save(fig, output_base, dpi)


def render_aulc(aulc_rows, output_base, curve, title, ylabel, profile, dpi=160):
    selected = [row for row in aulc_rows if row.get("curve") == curve]
    return _seed_summary_facets(
        selected, output_base, "aulc", title, ylabel, profile,
        dpi=dpi, include_baselines=False,
    )


def _prepare_evaluation_timeseries(records, smoothing_window_seconds):
    frame = _frame(records)
    if frame.empty:
        return frame
    identity = ["controller_id", "evaluation_seed"]
    prepared = []
    for _, group in frame.groupby(identity, sort=False, dropna=False):
        group = group.sort_values("simulation_time_seconds").copy()
        interval = max(1, int(group["action_interval_seconds"].iloc[0]))
        window = max(1, int(round(smoothing_window_seconds / interval)))
        group["reward_network_mean_trend"] = (
            group["reward_network_mean"].expanding().mean()
        )
        group["reward_network_sum_trend"] = (
            group["reward_network_sum"].expanding().mean()
        )
        group["delay_network_weighted_mean_trend"] = (
            group["delay_network_weighted_mean"].expanding().mean()
        )
        group["queue_network_mean_trend"] = (
            group["queue_network_mean"].rolling(window, min_periods=1).mean()
        )
        group["queue_network_sum_trend"] = (
            group["queue_network_sum"].rolling(window, min_periods=1).mean()
        )
        group["throughput_interval_trend"] = (
            group["throughput_interval"].rolling(window, min_periods=1).sum()
        )
        group["throughput_cumulative_trend"] = group["throughput_cumulative"]
        prepared.append(group)
    return pd.concat(prepared, ignore_index=True)


def _render_timeseries_main(
    frame, output_base, value_field, title, ylabel, profile, dpi,
):
    networks = _ordered(frame["network"], profile.network_order)
    fig, axes = _grid(len(networks), width=6.3, height=4.1)
    agents_seen = []
    for axis, network in zip(axes, networks):
        network_rows = frame[frame["network"] == network]
        for agent in _ordered(network_rows["agent"], profile.algorithm_order):
            agent_rows = network_rows[network_rows["agent"] == agent]
            if agent == "dqn":
                series = agent_rows.groupby(
                    ["training_seed", "simulation_time_seconds"],
                    as_index=False, dropna=False,
                )[value_field].mean()
                seed_field = "training_seed"
            else:
                series = agent_rows
                seed_field = "evaluation_seed"
            _plot_seed_mean_ci(
                axis, series, "simulation_time_seconds", value_field,
                seed_field, _agent_color(agent), linewidth=1.7,
            )
            if agent not in agents_seen:
                agents_seen.append(agent)
        axis.set_title(str(network))
        axis.set_xlabel("Simulation time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    if agents_seen:
        fig.legend(
            handles=[
                Line2D([], [], color=_agent_color(agent), linewidth=2,
                       label=f"{agent} mean (95% bootstrap CI)")
                for agent in agents_seen
            ],
            loc="outside upper right", frameon=False,
        )
    fig.suptitle(title)
    return _save(fig, output_base, dpi)


def _render_timeseries_training_seed_diagnostic(
    frame, output_base, value_field, title, ylabel, profile, dpi,
):
    networks = _ordered(frame["network"], profile.network_order)
    dqn_seeds = _ordered(frame.loc[frame["agent"] == "dqn", "training_seed"])
    seed_palette = dict(zip(
        dqn_seeds, sns.color_palette("colorblind", max(1, len(dqn_seeds)))
    ))
    fig, axes = _grid(len(networks), width=6.3, height=4.1)
    for axis, network in zip(axes, networks):
        network_rows = frame[frame["network"] == network]
        dqn = network_rows[network_rows["agent"] == "dqn"].groupby(
            ["training_seed", "simulation_time_seconds"], as_index=False
        )[value_field].mean()
        for seed, series in dqn.groupby("training_seed", sort=False):
            axis.plot(
                series["simulation_time_seconds"], series[value_field],
                color=seed_palette[seed], linewidth=1.25, alpha=0.9,
            )
        for agent in ("fixedtime", "maxpressure"):
            baseline = network_rows[network_rows["agent"] == agent].groupby(
                "simulation_time_seconds", as_index=False
            )[value_field].mean()
            if not baseline.empty:
                axis.plot(
                    baseline["simulation_time_seconds"], baseline[value_field],
                    color=_agent_color(agent), linewidth=1.5, linestyle="--",
                )
        axis.set_title(str(network))
        axis.set_xlabel("Simulation time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    handles = [
        Line2D([], [], color=seed_palette[seed], linewidth=2, label=f"DQN seed {int(seed)}")
        for seed in dqn_seeds
    ] + [
        Line2D([], [], color=_agent_color(agent), linestyle="--", linewidth=2,
               label=agent)
        for agent in ("fixedtime", "maxpressure")
    ]
    if handles:
        fig.legend(handles=handles, loc="outside upper right", frameon=False, ncols=2)
    fig.suptitle(title)
    return _save(fig, output_base, dpi)


def _render_timeseries_raw_diagnostic(
    frame, output_base, value_field, title, ylabel, profile, dpi,
):
    networks = _ordered(frame["network"], profile.network_order)
    fig, axes = _grid(len(networks), width=6.3, height=4.1)
    for axis, network in zip(axes, networks):
        network_rows = frame[frame["network"] == network]
        for (_, _), series in network_rows.groupby(
            ["controller_id", "evaluation_seed"], sort=False
        ):
            agent = series["agent"].iloc[0]
            axis.plot(
                series["simulation_time_seconds"], series[value_field],
                color=_agent_color(agent), linewidth=0.75,
                alpha=0.32 if agent == "dqn" else 0.5,
            )
        axis.set_title(str(network))
        axis.set_xlabel("Simulation time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    fig.legend(
        handles=[
            Line2D([], [], color=_agent_color(agent), linewidth=2, label=agent)
            for agent in ("dqn", "fixedtime", "maxpressure")
        ],
        loc="outside upper right", frameon=False,
    )
    fig.suptitle(title)
    return _save(fig, output_base, dpi)


def render_evaluation_timeseries(
    records, figure_dir, smoothing_window_seconds, profile, dpi=160,
):
    """Render two-level seed summaries plus five-seed and raw diagnostics."""
    frame = _prepare_evaluation_timeseries(records, smoothing_window_seconds)
    figure_dir = Path(figure_dir)
    results_dir = figure_dir / "results" / "best_checkpoint_timeseries"
    diagnostics_dir = figure_dir / "diagnostics" / "best_checkpoint_timeseries"
    outputs = []
    specs = (
        ("reward_network_mean_trend", "reward_network_mean",
         "Cumulative mean network reward", "Reward (network mean)"),
        ("reward_network_sum_trend", "reward_network_sum",
         "Cumulative mean total network reward", "Reward (network sum)"),
        ("queue_network_mean_trend", "queue_network_mean",
         f"Network queue ({smoothing_window_seconds:g}-s moving mean)",
         "Queued vehicles (lane mean)"),
        ("queue_network_sum_trend", "queue_network_sum",
         f"Total network queue ({smoothing_window_seconds:g}-s moving mean)",
         "Queued vehicles (network sum)"),
        ("delay_network_weighted_mean_trend", "delay_weighted_mean",
         "Cumulative vehicle-weighted network delay", "Weighted delay"),
        ("throughput_interval_trend", "throughput_interval",
         f"Completed vehicles per {smoothing_window_seconds:g}-s window",
         "Completed vehicles"),
        ("throughput_cumulative_trend", "throughput_cumulative",
         "Cumulative completed vehicles", "Completed vehicles"),
    )
    for field, stem, title, ylabel in specs:
        outputs.extend(_render_timeseries_main(
            frame, results_dir / stem, field, title, ylabel, profile, dpi,
        ))
        outputs.extend(_render_timeseries_training_seed_diagnostic(
            frame, diagnostics_dir / f"{stem}_training_seed_means", field,
            f"{title}: DQN training-seed means", ylabel, profile, dpi,
        ))
        outputs.extend(_render_timeseries_raw_diagnostic(
            frame, diagnostics_dir / f"{stem}_raw_episodes", field,
            f"{title}: all evaluation episodes", ylabel, profile, dpi,
        ))
    return outputs


def render_all(
    metric_rows, action_rows, comparison_rows, auc_rows, figure_dir, dpi=160,
    learning_speed_rows=None, profile=None, efficiency_rows=None, aulc_rows=None,
):
    """Render a stable result package and detailed diagnostic package."""

    del learning_speed_rows  # Reserved for profile-specific threshold plots.
    profile = profile or get_profile("plan1")
    figure_dir = Path(figure_dir)
    results_dir = figure_dir / "results"
    diagnostics_dir = figure_dir / "diagnostics"
    outputs = []

    outputs.extend(render_final_performance(
        comparison_rows, results_dir / "final_travel_time", profile, dpi=dpi,
    ))
    outputs.extend(render_evaluation_learning_curve(
        metric_rows, results_dir / "evaluation_learning_curve", profile, dpi=dpi,
    ))
    outputs.extend(render_learning_speed(
        auc_rows, results_dir / "learning_speed_auc", profile, dpi=dpi,
    ))
    outputs.extend(render_action_concentration(
        action_rows, results_dir / "action_concentration", profile, dpi=dpi,
    ))
    outputs.extend(render_training_cost(
        metric_rows, results_dir / "training_cost", profile, dpi=dpi,
    ))
    efficiency_rows = efficiency_rows or []
    aulc_rows = aulc_rows or []
    for x_field, stem, title, xlabel in (
        ("environment_transitions", "travel_time_vs_transitions",
         "Travel time vs cumulative environment transitions",
         "Cumulative environment transitions"),
        ("gradient_updates", "travel_time_vs_gradient_updates",
         "Travel time vs cumulative gradient updates",
         "Cumulative gradient updates"),
        ("cumulative_wall_time_seconds", "travel_time_vs_wall_time",
         "Travel time vs cumulative measured training time",
         "Cumulative training wall time (s)"),
    ):
        outputs.extend(render_efficiency_curve(
            efficiency_rows, results_dir / stem, x_field, "travel_time",
            title, xlabel, "Average travel time", profile, dpi=dpi,
        ))
    outputs.extend(render_efficiency_curve(
        efficiency_rows, results_dir / "replay_fill_fraction", "episode",
        "replay_fill_fraction", "Replay-buffer fill fraction",
        profile.progress_label, "Replay fill fraction", profile, dpi=dpi,
    ))
    outputs.extend(render_efficiency_curve(
        efficiency_rows, results_dir / "update_to_data_ratio", "episode",
        "update_to_data_ratio", "Cumulative update-to-data ratio",
        profile.progress_label, "Gradient updates / collected transitions",
        profile, dpi=dpi,
    ))
    outputs.extend(render_aulc(
        aulc_rows, results_dir / "raw_travel_time_aulc",
        "raw_travel_time", "Travel-time AULC by scenario",
        "Mean travel time over learning (lower is better)", profile, dpi=dpi,
    ))
    outputs.extend(render_aulc(
        aulc_rows, results_dir / "normalized_control_aulc",
        "cross_scene_normalized_control_score",
        "Cross-scenario normalized control-score AULC",
        "Normalized AULC (higher is better)", profile, dpi=dpi,
    ))

    outputs.extend(render_individual_travel_time(
        metric_rows, diagnostics_dir / "individual_travel_time", profile, dpi=dpi,
    ))
    diagnostic_specs = (
        ("reward_mean", "Reward diagnostics", "Mean reward", ("train", "evaluation"), True),
        ("loss_mean", "Training loss diagnostics", "Mean loss", ("train",), False),
        ("queue", "Evaluation queue diagnostics", "Queue", ("evaluation",), True),
        ("delay", "Evaluation approximate-delay diagnostics", "Approximate delay", ("evaluation",), True),
        ("real_delay", "Evaluation real-delay diagnostics", "Real delay (s)", ("evaluation",), True),
        ("throughput", "Evaluation throughput diagnostics", "Throughput", ("evaluation",), True),
        ("epsilon", "Exploration schedule", "Epsilon", ("train",), False),
        ("replay_size", "Replay-buffer growth", "Replay size", ("train",), False),
        ("phase_switch_frequency", "Evaluation phase-switching diagnostics", "Switch frequency", ("evaluation",), True),
    )
    for metric, title, ylabel, sources, baselines in diagnostic_specs:
        outputs.extend(render_metric_diagnostic(
            metric_rows, diagnostics_dir / metric, metric, title, ylabel,
            profile, sources=sources, include_baselines=baselines, dpi=dpi,
        ))
    outputs.extend(render_action_distribution(
        action_rows, diagnostics_dir / "action_distribution", profile, dpi=dpi,
    ))
    outputs.extend(render_final_best_gap(
        comparison_rows, diagnostics_dir / "final_best_gap", profile, dpi=dpi,
    ))
    return outputs
