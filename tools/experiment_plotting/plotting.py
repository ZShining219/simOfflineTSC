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

from .profiles import get_profile, S1_S4_SCENE_LABELS


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


def _scene_tick(scene):
    for network, label in S1_S4_SCENE_LABELS.items():
        if label.startswith(f'{scene} '):
            return f'{scene}\n{network}'
    return scene


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


def render_convergence_curves(curve_rows, convergence_rows, output_base, dpi=160):
    frame = _frame(curve_rows)
    selected = frame[frame['metric'] == 'travel_time'].copy()
    scenes = _ordered(selected['scene'], ('S1','S2','S3','S4'))
    fig, axes = _grid(len(scenes), width=6.4, height=4.2)
    seed_palette = dict(zip(
        sorted(selected['training_seed'].unique()),
        sns.color_palette('colorblind', selected['training_seed'].nunique()),
    ))
    for axis, scene in zip(axes, scenes):
        group = selected[selected['scene'] == scene]
        for seed, run in group.groupby('training_seed'):
            series = run.dropna(subset=['moving_average_10']).sort_values('episode')
            axis.plot(series['episode'], series['moving_average_10'],
                      color=seed_palette[seed], linewidth=1.1, alpha=.78)
            match = [row for row in convergence_rows if row['scene'] == scene
                     and row['metric'] == 'travel_time' and row['tolerance'] == .05
                     and row['training_seed'] == seed]
            if match and match[0]['convergence_episode'] is not None:
                episode = match[0]['convergence_episode']
                value = series.loc[series['episode'] == episode, 'moving_average_10']
                if not value.empty:
                    axis.scatter([episode], [value.iloc[0]], color=seed_palette[seed],
                                 marker='o', edgecolor='black', linewidth=.4, s=28, zorder=4)
        for reference, style in ((50,'--'),(75,':'),(100,'-.'),(400,'--')):
            axis.axvline(reference, color='#555555', linestyle=style, linewidth=.8, alpha=.55)
        network = group['network'].iloc[0]
        axis.set_title(f'{scene} ({network})')
        axis.set_xlabel('Completed training episode')
        axis.set_ylabel('Travel time (10-episode moving average)')
        axis.grid(alpha=.2)
    fig.legend(handles=[
        Line2D([], [], color=seed_palette[seed], label=f'training seed {int(seed)}')
        for seed in sorted(seed_palette)
    ], loc='outside upper right', frameon=False)
    fig.suptitle('Convergence curves and ±5% stable-entry episodes')
    return _save(fig, output_base, dpi)


def render_convergence_confirmation_metrics(curve_rows, convergence_rows,
                                            output_base, dpi=160):
    frame = _frame(curve_rows)
    metrics = ('travel_time', 'queue', 'reward_mean')
    scenes = ['S1', 'S2', 'S3', 'S4']
    seeds = sorted(frame['training_seed'].unique())
    palette = dict(zip(seeds, sns.color_palette('colorblind', len(seeds))))
    fig, axes = plt.subplots(4, 3, figsize=(17, 15), constrained_layout=True)
    for row_index, scene in enumerate(scenes):
        for column_index, metric in enumerate(metrics):
            axis = axes[row_index, column_index]
            group = frame[(frame['scene'] == scene) & (frame['metric'] == metric)]
            for seed, run in group.groupby('training_seed'):
                series = run.dropna(subset=['moving_average_10']).sort_values('episode')
                axis.plot(series['episode'], series['moving_average_10'],
                          color=palette[seed], linewidth=.9, alpha=.72)
                match = [item for item in convergence_rows
                         if item['scene'] == scene and item['metric'] == metric
                         and item['tolerance'] == .05
                         and item['training_seed'] == seed]
                if match and match[0]['convergence_episode'] is not None:
                    episode = match[0]['convergence_episode']
                    point = series.loc[series['episode'] == episode, 'moving_average_10']
                    if not point.empty:
                        axis.scatter(episode, point.iloc[0], color=palette[seed],
                                     edgecolor='black', linewidth=.25, s=15, zorder=4)
            for reference in (50, 75, 100):
                axis.axvline(reference, color='#555555', linestyle='--',
                             linewidth=.65, alpha=.45)
            axis.set_title(f"{scene} / {group['network'].iloc[0]} — {metric.replace('_', ' ')}")
            axis.set_xlabel('Completed training episode')
            axis.set_ylabel('10-episode moving average')
            axis.grid(alpha=.18)
    fig.legend(handles=[Line2D([], [], color=palette[seed],
                               label=f'training seed {int(seed)}')
                        for seed in seeds], loc='outside upper right', frameon=False)
    fig.suptitle('Primary and confirmation convergence metrics (±5% entry points)')
    return _save(fig, output_base, dpi)


def render_convergence_distribution(convergence_rows, output_base, dpi=160):
    frame = _frame(convergence_rows)
    frame = frame[(frame['metric'] == 'travel_time') & (frame['tolerance'] == .05)].copy()
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    order = _ordered(frame['scene'], ('S1','S2','S3','S4'))
    sns.boxplot(data=frame, x='scene', y='convergence_episode', order=order,
                color='#B8D9EA', width=.55, showfliers=False, ax=axis)
    sns.stripplot(data=frame, x='scene', y='convergence_episode', order=order,
                  hue='training_seed', palette='colorblind', jitter=.12, size=6, ax=axis)
    axis.set_xticks(range(len(order)), [_scene_tick(value) for value in order],
                    rotation=15, ha='right')
    for reference in (50,75,100):
        axis.axhline(reference, color='#555555', linestyle='--', linewidth=.8, alpha=.55)
    axis.set_xlabel('Scene (mapping shown in source table)')
    axis.set_ylabel('Convergence episode')
    axis.set_title('Travel-time convergence episode by training seed (±5%)')
    axis.grid(axis='y', alpha=.2)
    axis.legend(title='Training seed', frameon=False, ncols=5)
    return _save(fig, output_base, dpi)


def render_budget_sufficiency(budget_rows, output_base, dpi=160):
    frame = _frame(budget_rows)
    frame = frame[frame['metric'] == 'travel_time'].copy()
    scenes = _ordered(frame['scene'], ('S1','S2','S3','S4'))
    fig, axes = _grid(len(scenes), width=6.2, height=4.1)
    for axis, scene in zip(axes, scenes):
        group = frame[frame['scene'] == scene]
        _plot_seed_mean_ci(axis, group, 'episode_budget', 'absolute_relative_gap',
                           'training_seed', _agent_color('dqn'))
        for seed, run in group.groupby('training_seed'):
            axis.scatter(run['episode_budget'], run['absolute_relative_gap'],
                         s=13, alpha=.36, color=_agent_color('dqn'))
        axis.axhline(.05, color='#D55E00', linestyle='--', linewidth=1, label='5% gap')
        axis.set_title(f"{scene} ({group['network'].iloc[0]})")
        axis.set_xlabel('Episode budget')
        axis.set_ylabel('Absolute relative gap to final-20 reference')
        axis.grid(alpha=.2)
    fig.suptitle('Budget sufficiency relative to final stable performance')
    return _save(fig, output_base, dpi)


def render_mean_action_distribution(summary_rows, output_base, dpi=160):
    frame = _frame(summary_rows)
    scenes = _ordered(frame['scene'], ('S1','S2','S3','S4'))
    fig, axes = _grid(len(scenes), width=6.3, height=4.0)
    for axis, scene in zip(axes, scenes):
        group = frame[frame['scene'] == scene].sort_values('action')
        axis.bar(group['action'], group['mean'], yerr=group['sample_sd'], capsize=3,
                 color=sns.color_palette('colorblind', 8), edgecolor='black', linewidth=.35)
        axis.set_title(f"{scene} ({group['network'].iloc[0]})")
        axis.set_xlabel('Action ID')
        axis.set_ylabel('Final action fraction (mean ± sample SD)')
        axis.set_xticks(range(8))
        axis.grid(axis='y', alpha=.2)
    fig.suptitle('Final greedy action distributions across training seeds')
    return _save(fig, output_base, dpi)


def render_distance_heatmap(distance_rows, metric, output_base, dpi=160):
    frame = _frame(distance_rows)
    frame = frame[(frame['aggregation'] == 'scene_mean') & (frame['metric'] == metric)]
    order = ('S1','S2','S3','S4')
    matrix = frame.pivot(index='source_scene', columns='target_scene', values='distance').reindex(index=order, columns=order)
    fig, axis = plt.subplots(figsize=(6.5, 5.3), constrained_layout=True)
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='mako', square=True,
                vmin=0, linewidths=.4, ax=axis)
    axis.set_xticklabels([_scene_tick(value) for value in order], rotation=25, ha='right')
    axis.set_yticklabels([_scene_tick(value) for value in order], rotation=0)
    axis.set_xlabel('Target scene')
    axis.set_ylabel('Source scene')
    axis.set_title(('Total Variation' if metric == 'total_variation' else 'Jensen–Shannon') + ' distance')
    return _save(fig, output_base, dpi)


def render_within_between_distance(distance_rows, output_base, dpi=160):
    frame = _frame(distance_rows)
    frame = frame[frame['aggregation'] == 'model_pair'].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for axis, metric in zip(axes, ('total_variation','jensen_shannon')):
        group = frame[frame['metric'] == metric]
        sns.boxplot(data=group, x='pair_kind', y='distance', color='#B8D9EA', ax=axis, showfliers=False)
        sns.stripplot(data=group, x='pair_kind', y='distance', color='#333333', alpha=.35, size=3, ax=axis)
        axis.set_xlabel('Model-pair category')
        axis.set_ylabel('Distance')
        axis.set_title(metric.replace('_',' ').title())
        axis.grid(axis='y', alpha=.2)
    fig.suptitle('Within-scene seed variation versus between-scene action distance')
    return _save(fig, output_base, dpi)


def render_cross_scene_heatmaps(summary_rows, output_base, dpi=160):
    frame = _frame(summary_rows)
    metrics = ('travel_time','queue','real_delay','approximate_delay','throughput',
               'phase_switch_frequency','reward_mean')
    fig, axes = plt.subplots(3, 3, figsize=(17, 14), constrained_layout=True)
    order = ('S1','S2','S3','S4')
    for axis, metric in zip(axes.flat, metrics):
        group = frame[frame['metric'] == metric]
        matrix = group.pivot(index='source_scene', columns='target_scene', values='mean').reindex(index=order, columns=order)
        sns.heatmap(matrix, annot=True, fmt='.2f' if metric != 'phase_switch_frequency' else '.3f',
                    cmap='viridis_r' if metric != 'throughput' else 'viridis',
                    linewidths=.4, ax=axis)
        axis.set_xticks(range(len(order)), [_scene_tick(value) for value in order],
                        rotation=25, ha='right')
        axis.set_yticklabels([_scene_tick(value) for value in order], rotation=0)
        axis.set_title(metric.replace('_',' ').title())
        axis.set_xlabel('Target evaluation scene')
        axis.set_ylabel('Source training scene')
    for axis in axes.flat[len(metrics):]:
        axis.set_visible(False)
    fig.suptitle('Final-checkpoint frozen cross-scene evaluation (training-seed means)')
    return _save(fig, output_base, dpi)


def render_relative_degradation_heatmaps(relative_rows, output_base, dpi=160):
    frame = _frame(relative_rows)
    metrics = ('travel_time','queue','real_delay','approximate_delay','throughput',
               'reward_mean')
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    order = ('S1','S2','S3','S4')
    limit = max(.01, float(frame[frame['metric'].isin(metrics)]['relative_degradation'].abs().max()))
    for axis, metric in zip(axes.flat, metrics):
        group = frame[frame['metric'] == metric]
        matrix = group.pivot(index='source_scene', columns='target_scene', values='relative_degradation').reindex(index=order, columns=order)
        sns.heatmap(matrix, annot=True, fmt='.1%', cmap='vlag', center=0,
                    vmin=-limit, vmax=limit, linewidths=.4, ax=axis)
        axis.set_xticks(range(len(order)), [_scene_tick(value) for value in order],
                        rotation=25, ha='right')
        axis.set_yticklabels([_scene_tick(value) for value in order], rotation=0)
        axis.set_title(metric.replace('_',' ').title())
        axis.set_xlabel('Target evaluation scene')
        axis.set_ylabel('Source training scene')
    fig.suptitle('Relative degradation versus target-scene in-domain final models')
    return _save(fig, output_base, dpi)


def render_cross_scene_seed_scatter(raw_rows, output_base, dpi=160):
    frame = _frame(raw_rows)
    order = ('S1','S2','S3','S4')
    fig, axes = _grid(4, width=6.2, height=4.2)
    for axis, target in zip(axes, order):
        group = frame[frame['target_scene'] == target]
        sns.stripplot(data=group, x='source_scene', y='travel_time', order=order,
                      hue='training_seed', palette='colorblind', jitter=.11, size=6, ax=axis)
        means = group.groupby('source_scene')['travel_time'].mean().reindex(order)
        sds = group.groupby('source_scene')['travel_time'].std(ddof=1).reindex(order)
        axis.errorbar(range(4), means, yerr=sds, fmt='D', color='black',
                      markersize=5, capsize=4, linewidth=1.2, zorder=5)
        axis.set_title(f"Target {_scene_tick(target).replace(chr(10), ' / ')}")
        axis.set_xticks(range(len(order)), [_scene_tick(value) for value in order],
                        rotation=25, ha='right')
        axis.set_xlabel('Source training scene / network')
        axis.set_ylabel('Average travel time')
        axis.grid(axis='y', alpha=.2)
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    fig.legend(handles=[Line2D([],[],marker='o',linestyle='',color=_agent_color('dqn'),label='Individual training seed'),
                        Line2D([],[],marker='D',linestyle='',color='black',label='Mean ± sample SD')],
               loc='outside upper right', frameon=False)
    fig.suptitle('Individual training-seed zero-shot transfer results')
    return _save(fig, output_base, dpi)


def render_episode100_pair_gate(raw_rows, relative_rows, output_base, dpi=160):
    """Render the targeted G0 individual-seed gate and paired degradation."""
    raw = _frame(raw_rows)
    relative = _frame([row for row in relative_rows if row['metric'] == 'travel_time'])
    directions = (('S3', 'S4'), ('S4', 'S3'), ('S2', 'S3'), ('S3', 'S2'))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    palette = dict(zip(range(5), sns.color_palette('colorblind', 5)))
    for axis, (source, target) in zip(axes.flat, directions):
        transferred = raw[(raw['source_scene'] == source) & (raw['target_scene'] == target)]
        reference = raw[(raw['source_scene'] == target) & (raw['target_scene'] == target)]
        paired = relative[(relative['source_scene'] == source)
                          & (relative['target_scene'] == target)]
        for seed in range(5):
            left = transferred[transferred['training_seed'] == seed]['travel_time']
            right = reference[reference['training_seed'] == seed]['travel_time']
            if len(left) != 1 or len(right) != 1:
                raise ValueError(f'Incomplete G0 plot pair: {source}->{target} seed {seed}')
            axis.plot([0, 1], [left.iloc[0], right.iloc[0]], color=palette[seed],
                      marker='o', linewidth=1.0, alpha=.85)
        means = [transferred['travel_time'].mean(), reference['travel_time'].mean()]
        sds = [transferred['travel_time'].std(ddof=1), reference['travel_time'].std(ddof=1)]
        axis.errorbar([0, 1], means, yerr=sds, fmt='D', color='black',
                      capsize=4, linewidth=1.3, markersize=5, zorder=6)
        degradation = paired['relative_degradation']
        axis.set_title(
            f'{source} → {target}: mean degradation {degradation.mean():.1%} '
            f'± {degradation.std(ddof=1):.1%}'
        )
        axis.set_xticks([0, 1], [f'{source} episode-100 model',
                                  f'{target} same-seed in-domain'])
        axis.set_ylabel('Average travel time')
        axis.grid(axis='y', alpha=.2)
    handles = [
        Line2D([], [], color=palette[seed], marker='o', label=f'seed {seed}')
        for seed in range(5)
    ] + [Line2D([], [], color='black', marker='D', label='mean ± sample SD')]
    fig.legend(handles=handles, loc='outside upper right', frameon=False, ncols=2)
    fig.suptitle('Figure G0: episode-100 targeted pair sensitivity gate')
    paths = _save(fig, output_base, dpi)

    degradation_fig, axis = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
    labels = [f'{source}→{target}' for source, target in directions]
    for position, (source, target) in enumerate(directions):
        group = relative[(relative['source_scene'] == source)
                         & (relative['target_scene'] == target)].sort_values(
                             'training_seed')
        for _, row in group.iterrows():
            seed = int(row['training_seed'])
            axis.scatter(position, row['relative_degradation'],
                         color=palette[seed], s=44, zorder=4)
        axis.errorbar(
            position, group['relative_degradation'].mean(),
            yerr=group['relative_degradation'].std(ddof=1), fmt='D',
            color='black', capsize=4, linewidth=1.3, markersize=6, zorder=5,
        )
    axis.axhline(0, color='#555555', linewidth=.9)
    axis.axhline(.05, color='#999999', linestyle='--', linewidth=.9,
                 label='5% descriptive reference')
    axis.set_xticks(range(len(labels)), labels)
    axis.set_ylabel('Same-seed travel-time relative degradation')
    axis.set_title('Episode-100 four-direction travel-time degradation')
    axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axis.grid(axis='y', alpha=.2)
    axis.legend(frameon=False)
    paths += _save(
        degradation_fig,
        Path(output_base).with_name(Path(output_base).name + '_degradation'), dpi,
    )
    return paths


def render_policy_scene_heatmap(scene_rows, metric, output_base, dpi=160):
    frame = _frame([row for row in scene_rows
                    if row['probe_source_scene'] == 'ALL' and row['metric'] == metric])
    scenes = ['S1', 'S2', 'S3', 'S4']
    matrix = frame.pivot(index='source_scene', columns='target_scene', values='mean')
    matrix = matrix.reindex(index=scenes, columns=scenes)
    fig, axis = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='mako', square=True,
                vmin=0, ax=axis, cbar_kws={'label': metric.replace('_', ' ')})
    axis.set_xticklabels([_scene_tick(value) for value in scenes], rotation=25, ha='right')
    axis.set_yticklabels([_scene_tick(value) for value in scenes], rotation=0)
    axis.set_xlabel('Compared model scene')
    axis.set_ylabel('Model scene')
    axis.set_title(f'Common FixedTime probe: {metric.replace("_", " ")}')
    return _save(fig, output_base, dpi)


def render_policy_source_stratified(scene_rows, output_base, dpi=160):
    frame = _frame([row for row in scene_rows
                    if row['probe_source_scene'] != 'ALL'
                    and row['metric'] == 'action_disagreement'])
    scenes = ['S1', 'S2', 'S3', 'S4']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    limit = max(.01, float(frame['mean'].max()))
    for axis, probe_scene in zip(axes.flat, scenes):
        group = frame[frame['probe_source_scene'] == probe_scene]
        matrix = group.pivot(index='source_scene', columns='target_scene', values='mean')
        matrix = matrix.reindex(index=scenes, columns=scenes)
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='mako', square=True,
                    vmin=0, vmax=limit, ax=axis,
                    cbar=probe_scene == scenes[-1],
                    cbar_kws={'label': 'Top-1 disagreement'})
        axis.set_xticklabels([_scene_tick(value) for value in scenes],
                             rotation=25, ha='right')
        axis.set_yticklabels([_scene_tick(value) for value in scenes], rotation=0)
        axis.set_xlabel('Compared model scene')
        axis.set_ylabel('Model scene')
        axis.set_title(f"Probe source: {_scene_tick(probe_scene).replace(chr(10), ' / ')}")
    fig.suptitle('Policy disagreement stratified by FixedTime probe source scene')
    return _save(fig, output_base, dpi)


def render_policy_model_heatmap(model_rows, output_base, dpi=160):
    selected = [row for row in model_rows if row['probe_source_scene'] == 'ALL']
    labels = sorted({row['left_model'] for row in selected} |
                    {row['right_model'] for row in selected},
                    key=lambda value: (int(value.split('_')[0][1:]),
                                       int(value.rsplit('seed', 1)[1])))
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    index = {label: position for position, label in enumerate(labels)}
    for row in selected:
        left, right = index[row['left_model']], index[row['right_model']]
        matrix[left, right] = matrix[right, left] = row['action_disagreement']
    fig, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    sns.heatmap(matrix, cmap='mako', vmin=0, vmax=max(.01, float(matrix.max())),
                xticklabels=labels, yticklabels=labels, square=True, ax=axis,
                cbar_kws={'label': 'Top-1 disagreement'})
    axis.tick_params(axis='x', rotation=90, labelsize=7)
    axis.tick_params(axis='y', rotation=0, labelsize=7)
    axis.set_title('20 final models on identical FixedTime probe states')
    return _save(fig, output_base, dpi)


def render_within_between_policy(model_rows, output_base, dpi=160):
    frame = _frame([row for row in model_rows if row['probe_source_scene'] == 'ALL'])
    melted = frame.melt(id_vars=['pair_kind'],
                        value_vars=['action_disagreement',
                                    'high_confidence_disagreement',
                                    'q_cosine_distance'],
                        var_name='metric', value_name='value').dropna()
    fig, axis = plt.subplots(figsize=(9, 5.3), constrained_layout=True)
    sns.boxplot(data=melted, x='metric', y='value', hue='pair_kind',
                palette='colorblind', ax=axis)
    sns.stripplot(data=melted, x='metric', y='value', hue='pair_kind',
                  dodge=True,
                  palette={'within_scene': '#333333', 'between_scene': '#333333'},
                  alpha=.25, size=2, legend=False, ax=axis)
    axis.set_xlabel('')
    axis.set_ylabel('Pairwise difference')
    axis.tick_params(axis='x', rotation=12)
    axis.set_title('Within-scene seed variation vs between-scene variation')
    axis.legend(title='Model pair')
    return _save(fig, output_base, dpi)


def render_state_distance_heatmap(distance_rows, output_base, dpi=160):
    frame = _frame(distance_rows)
    scenes = ['S1', 'S2', 'S3', 'S4']
    matrix = frame.pivot(index='source_scene', columns='target_scene', values='distance')
    matrix = matrix.reindex(index=scenes, columns=scenes)
    fig, axis = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='crest', square=True,
                ax=axis, cbar_kws={'label': 'Mean standardized Wasserstein distance'})
    axis.set_xticklabels([_scene_tick(value) for value in scenes], rotation=25, ha='right')
    axis.set_yticklabels([_scene_tick(value) for value in scenes], rotation=0)
    axis.set_xlabel('Target scene')
    axis.set_ylabel('Source scene')
    axis.set_title('FixedTime state-distribution distance')
    return _save(fig, output_base, dpi)


def render_state_pca(pca_rows, output_base, dpi=160):
    frame = _frame(pca_rows)
    frame['scene_display'] = frame['scene'].map(_scene_tick)
    fig, axis = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)
    sns.scatterplot(data=frame, x='pc1', y='pc2', hue='scene_display',
                    hue_order=[_scene_tick(value) for value in ['S1','S2','S3','S4']],
                    alpha=.35, s=12, ax=axis)
    axis.set_xlabel(f"PC1 ({frame['pc1_explained_variance'].iloc[0]:.1%})")
    axis.set_ylabel(f"PC2 ({frame['pc2_explained_variance'].iloc[0]:.1%})")
    axis.set_title('Common FixedTime state distribution (PCA)')
    return _save(fig, output_base, dpi)


def _representative_points(frame, maximum):
    if len(frame) <= maximum:
        return frame
    indices = np.linspace(0, len(frame) - 1, maximum, dtype=int)
    return frame.iloc[indices]


def _density_mass_contours(frame, x_limits, y_limits, bins=72, sigma=1.15):
    from scipy.ndimage import gaussian_filter

    counts, x_edges, y_edges = np.histogram2d(
        frame['pc1'], frame['pc2'], bins=bins,
        range=(x_limits, y_limits),
    )
    density = gaussian_filter(counts.T.astype(float), sigma=sigma)
    positive = density[density > 0]
    if positive.size < 4:
        return None
    ordered = np.sort(positive)[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    thresholds = []
    for mass in (.95, .80, .50):
        threshold = ordered[min(np.searchsorted(cumulative, mass), len(ordered) - 1)]
        thresholds.append(float(threshold))
    levels = sorted(set(thresholds))
    if len(levels) < 3:
        levels = np.linspace(float(positive.min()), float(positive.max()), 5)[1:4].tolist()
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    return x_centers, y_centers, density, levels


def render_dqn_training_state_coverage(
        fixedtime_rows, dqn_rows, episode_start, episode_end, full_axis_limits,
        reference_axis_limits, explained_variance, output_base, dpi=160):
    """Compare one DQN training window with FixedTime using density small multiples."""
    fixedtime = _frame(fixedtime_rows)
    dqn = _frame(dqn_rows)
    scenes = ('S1', 'S2', 'S3', 'S4')
    palette = dict(zip(scenes, sns.color_palette('colorblind', len(scenes))))

    fig = plt.figure(figsize=(13.6, 11.4))
    grid = fig.add_gridspec(
        3, 2, height_ratios=(1, 1, .64),
        left=.075, right=.975, bottom=.13, top=.85,
        wspace=.14, hspace=.25,
    )
    scene_axes = [fig.add_subplot(grid[index // 2, index % 2])
                  for index in range(4)]
    overview = fig.add_subplot(grid[2, :])

    for axis, scene in zip(scene_axes, scenes):
        reference = fixedtime[fixedtime['scene'] == scene]
        stage = dqn[dqn['scene'] == scene]
        inside_mask = (
            stage['pc1'].between(*reference_axis_limits['x'])
            & stage['pc2'].between(*reference_axis_limits['y'])
        )
        stage_in_view = stage[inside_mask]
        reference_density = _density_mass_contours(
            reference, reference_axis_limits['x'], reference_axis_limits['y']
        )
        dqn_density = _density_mass_contours(
            stage_in_view, reference_axis_limits['x'], reference_axis_limits['y']
        )
        axis.scatter(
            reference['pc1'], reference['pc2'], color='#777777',
            s=5, alpha=.075, linewidths=0, rasterized=True, zorder=1,
        )
        if reference_density is not None:
            x_grid, y_grid, density, levels = reference_density
            upper = float(density.max()) * 1.001
            axis.contourf(
                x_grid, y_grid, density,
                levels=levels + [upper], cmap='Greys', alpha=.22,
                antialiased=True, zorder=1,
            )
            axis.contour(
                x_grid, y_grid, density, levels=levels,
                colors=['#9A9A9A'], linestyles='--', linewidths=.9,
                alpha=.85, zorder=2,
            )
        stage_sample = _representative_points(stage_in_view, 1800)
        axis.scatter(
            stage_sample['pc1'], stage_sample['pc2'], color=palette[scene],
            s=5, alpha=.15, linewidths=0, rasterized=True, zorder=3,
        )
        if dqn_density is not None:
            x_grid, y_grid, density, levels = dqn_density
            axis.contour(
                x_grid, y_grid, density, levels=levels,
                colors=[palette[scene]], linestyles='-',
                linewidths=(1.0, 1.5, 2.1), alpha=.98, zorder=4,
            )
        inside = inside_mask.mean()
        axis.text(
            .018, .975, f'{inside:.1%} of DQN states inside reference-scale view',
            transform=axis.transAxes, ha='left', va='top', fontsize=8.4,
            color='#333333',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .78, 'pad': 2},
            zorder=6,
        )
        axis.set_xlim(*reference_axis_limits['x'])
        axis.set_ylim(*reference_axis_limits['y'])
        axis.set_title(
            f'{scene} · {len(reference):,} FixedTime / {len(stage):,} DQN states',
            color=palette[scene], fontsize=11, fontweight='semibold',
        )
        axis.grid(alpha=.14, linewidth=.6)

    for index, axis in enumerate(scene_axes):
        if index % 2 == 0:
            axis.set_ylabel(f'PC2 ({explained_variance[1]:.1%})')
        else:
            axis.set_ylabel('')
        if index >= 2:
            axis.set_xlabel(f'PC1 ({explained_variance[0]:.1%})')
        else:
            axis.set_xlabel('')

    overview.scatter(
        fixedtime['pc1'], fixedtime['pc2'], color='#888888',
        s=4, alpha=.08, linewidths=0, rasterized=True, zorder=1,
    )
    for scene in scenes:
        stage = dqn[dqn['scene'] == scene]
        sample = _representative_points(stage, 3000)
        overview.scatter(
            sample['pc1'], sample['pc2'], color=palette[scene],
            s=4, alpha=.16, linewidths=0, rasterized=True, zorder=2,
        )
    x0, x1 = reference_axis_limits['x']
    y0, y1 = reference_axis_limits['y']
    overview.plot(
        [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
        color='#333333', linestyle='--', linewidth=1.0, zorder=3,
    )
    overview.set_xlim(*full_axis_limits['x'])
    overview.set_ylim(*full_axis_limits['y'])
    overview.set_xlabel(f'PC1 ({explained_variance[0]:.1%})')
    overview.set_ylabel(f'PC2 ({explained_variance[1]:.1%})')
    overview.set_title(
        'Full-range context · identical limits in all five stage figures · '
        'dashed box marks the detailed reference-scale view',
        fontsize=10,
    )
    overview.grid(alpha=.12, linewidth=.6)

    fig.suptitle(
        'Plan 1 In-domain DQN Training-State Coverage vs FixedTime Reference\n'
        f'Episodes {episode_start}–{episode_end} · four scenes · five training seeds per scene',
        fontsize=16,
    )
    legend_handles = [
        Line2D([], [], color='#888888', linestyle='--', linewidth=1.2,
               label='FixedTime reference density (all reference states)'),
        Line2D([], [], color='#222222', linestyle='-', linewidth=1.8,
               label='DQN density (all states inside reference-scale view)'),
        Line2D([], [], linestyle='none', marker='o', markersize=5,
               markerfacecolor='#555555', markeredgecolor='none', alpha=.55,
               label='Representative raw points (not used for density)'),
    ]
    fig.legend(
        handles=legend_handles, loc='lower center', ncols=3,
        bbox_to_anchor=(.5, .068), frameon=False, fontsize=9,
    )
    fig.text(
        .5, .028,
        'DQN trajectories record random warm-up / epsilon-greedy training behavior. '
        'Main-panel contours summarize 50%, 80%, and 95% highest-density regions '
        'within the reference-scale view; '
        'they do not represent greedy-policy evaluation.',
        fontsize=8.5, color='#444444', ha='center',
    )
    return _save(fig, output_base, dpi)


def render_classifier_confusion(confusion_rows, output_base, dpi=160):
    frame = _frame(confusion_rows)
    scenes = ['S1', 'S2', 'S3', 'S4']
    matrix = frame.pivot(index='true_scene', columns='predicted_scene',
                         values='normalized_count').reindex(index=scenes, columns=scenes)
    fig, axis = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1,
                square=True, ax=axis, cbar_kws={'label': 'Row-normalized fraction'})
    axis.set_xticklabels([_scene_tick(value) for value in scenes], rotation=25, ha='right')
    axis.set_yticklabels([_scene_tick(value) for value in scenes], rotation=0)
    axis.set_xlabel('Predicted scene')
    axis.set_ylabel('True scene')
    axis.set_title('Traffic-seed-grouped logistic regression')
    return _save(fig, output_base, dpi)


def render_state_distance_vs_policy(relation_rows, output_base, dpi=160):
    frame = _frame(relation_rows)
    fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    sns.scatterplot(data=frame, x='state_distribution_distance',
                    y='policy_disagreement', s=85, ax=axis)
    for row in relation_rows:
        axis.annotate(f"{row['scene_a']}-{row['scene_b']}",
                      (row['state_distribution_distance'], row['policy_disagreement']),
                      xytext=(4, 4), textcoords='offset points')
    axis.set_xlabel('Standardized Wasserstein state distance')
    axis.set_ylabel('Common-state policy disagreement')
    axis.set_title('State-distribution difference is not policy conflict')
    axis.grid(alpha=.2)
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
            if agent not in {"fixedtime", "maxpressure"}:
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
    seeded = frame[~frame["agent"].isin(("fixedtime", "maxpressure"))]
    dqn_seeds = _ordered(seeded["training_seed"])
    seed_palette = dict(zip(
        dqn_seeds, sns.color_palette("colorblind", max(1, len(dqn_seeds)))
    ))
    fig, axes = _grid(len(networks), width=6.3, height=4.1)
    for axis, network in zip(axes, networks):
        network_rows = frame[frame["network"] == network]
        dqn = network_rows[~network_rows["agent"].isin(
            ("fixedtime", "maxpressure")
        )].groupby(
            ["agent", "training_seed", "simulation_time_seconds"],
            as_index=False,
        )[value_field].mean()
        for (_, seed), series in dqn.groupby(
            ["agent", "training_seed"], sort=False
        ):
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
        if profile.name == 'sequential_frozen':
            continue
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
