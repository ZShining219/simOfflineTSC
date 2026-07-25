"""Plots for validated Plan 3/4 sequential-analysis reports."""

import csv
import json
import shutil
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from . import __version__
from .plotting import _save
from utils.logger import validate_evaluation_package

POLICIES = ('clear', 'fifo', 'fifo_matched_wait')
COLORS = dict(zip(POLICIES, sns.color_palette('colorblind', 3)))
LABELS = {
    'clear': 'Clear replay', 'fifo': 'FIFO replay',
    'fifo_matched_wait': 'FIFO + matched wait',
}
SCENE_LABELS = {
    'sumohz1x1_config2': 'S1', 'sumohz1x1': 'S2',
    'sumohz1x1_config4': 'S3', 'sumohz1x1_config3': 'S4',
}
METRIC_SPECS = (
    ('normalized_travel_time', 'Travel time / Plan 1 reference'),
    ('real_delay', 'Real delay (s)'),
    ('delay', 'Approximate delay'),
    ('queue', 'Mean queued vehicles'),
    ('throughput', 'Throughput (vehicles)'),
    ('reward_mean', 'Mean reward'),
)


def _write_csv(path, rows):
    path = Path(path)
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean_band(axis, frame, x, y, policy):
    pivot = frame.pivot_table(index='training_seed', columns=x, values=y)
    values = pivot.to_numpy(float)
    xs = pivot.columns.to_numpy()
    mean = np.nanmean(values, axis=0)
    axis.plot(xs, mean, color=COLORS[policy], label=LABELS[policy], linewidth=1.8)
    if len(values) > 1:
        rng = np.random.default_rng(20260724)
        idx = rng.integers(0, len(values), size=(1000, len(values)))
        means = np.nanmean(values[idx], axis=1)
        lo, hi = np.nanpercentile(means, (2.5, 97.5), axis=0)
        axis.fill_between(xs, lo, hi, color=COLORS[policy], alpha=.16)


def _flatten(report):
    runs, curves, stages, retention, forgetting = [], [], [], [], []
    replay, replay_sources, replacement, stage_scene = [], [], [], []
    for run in report['runs']:
        base = {key: run[key] for key in ('logical_run_id','order_id','training_seed','policy')}
        network_by_stage = {row['stage_index']: row['network'] for row in run['stages']}
        previous_samples = {}
        runs.append({**base, 'primary_normalized_auc': run['primary_normalized_auc'],
                     **run['secondary_metrics']})
        for row in run['adaptation_curves']:
            curves.append({**base, **row})
        for row in run['stages']:
            stages.append({**base, **{k:v for k,v in row.items() if k != 'metrics'}})
        for network, value in run['final_retention_normalized'].items():
            retention.append({**base, 'network':network, 'final_retention_normalized':value})
        for network, value in run['stage_end_forgetting_travel_time'].items():
            forgetting.append({**base, 'network':network, 'forgetting_travel_time':value})
        for row in run.get('stage_scene_evaluations', []):
            stage_scene.append({
                **base, **row,
                'scene': SCENE_LABELS.get(row['evaluation_network'], row['evaluation_network']),
            })
        for row in run['replay_diagnostics']:
            current_network = row.get('current_network') or network_by_stage.get(
                row.get('stage_index')
            )
            cumulative_samples = row.get('samples_drawn_by_scene', {})
            episode_samples = {
                network: int(count) - int(previous_samples.get(network, 0))
                for network, count in cumulative_samples.items()
            }
            episode_sample_count = sum(episode_samples.values())
            replay.append({
                **base, **{k:v for k,v in row.items() if not isinstance(v, dict)},
                'cumulative_current_sample_fraction': row.get(
                    'current_sample_fraction'
                ),
                'episode_current_sample_fraction': (
                    np.nan if not episode_sample_count else
                    episode_samples.get(current_network, 0) / episode_sample_count
                ),
                'current_buffer_fraction': row.get('replay_ratio_by_scene', {}).get(
                    current_network
                ),
            })
            for source, fraction in row.get('replay_ratio_by_scene', {}).items():
                replay_sources.append({
                    **base, 'stage_index': row.get('stage_index'),
                    'local_episode': row.get('local_episode'),
                    'source_network': source,
                    'source_scene': SCENE_LABELS.get(source, source),
                    'source_kind': ('current' if source == current_network
                                    else 'historical'),
                    'provenance': 'buffer', 'fraction': fraction,
                })
            if episode_sample_count:
                for source, count in episode_samples.items():
                    replay_sources.append({
                        **base, 'stage_index': row.get('stage_index'),
                        'local_episode': row.get('local_episode'),
                        'source_network': source,
                        'source_scene': SCENE_LABELS.get(source, source),
                        'source_kind': ('current' if source == current_network
                                        else 'historical'),
                        'provenance': 'sample',
                        'fraction': count / episode_sample_count,
                    })
            for threshold, value in (row.get('replacement_thresholds') or {}).items():
                if value:
                    replacement.append({
                        **base, 'stage_index': row.get('stage_index'),
                        'threshold': float(threshold),
                        'reached_local_episode': (
                            int(value['global_episode']) - int(row['global_episode'])
                            + int(row['local_episode'])
                        ),
                        'global_episode': value['global_episode'],
                        'global_step': value['global_step'],
                        'historical_ratio': value['historical_ratio'],
                    })
            previous_samples = dict(cumulative_samples)
    # Keep replay as the final element for compatibility with the original
    # plotting helper contract used by downstream checks.
    return (runs, curves, stages, retention, forgetting, replay_sources,
            replacement, stage_scene, replay)


def _adaptation_metric(curves, output, dpi, metric, ylabel):
    frame = pd.DataFrame(curves)
    orders = sorted(frame.order_id.unique())
    fig, axes = plt.subplots(len(orders), 3, figsize=(15, 4.2*len(orders)), squeeze=False)
    for i, order in enumerate(orders):
        for j, stage in enumerate((2,3,4)):
            axis = axes[i,j]
            selected = frame[(frame.order_id==order)&(frame.stage_index==stage)]
            for policy in POLICIES:
                _mean_band(axis, selected[selected.policy==policy], 'local_episode',
                           metric, policy)
            if metric == 'normalized_travel_time':
                axis.axhline(1, color='0.25', linestyle='--', linewidth=.9)
            scene = SCENE_LABELS.get(selected.network.iloc[0], selected.network.iloc[0])
            axis.set(title=f'{order} · Stage {stage} · {scene}',
                     xlabel='Local training episode', ylabel=ylabel)
            axis.grid(alpha=.18)
    handles = [Line2D([], [], color=COLORS[p], linewidth=2, label=LABELS[p])
               for p in POLICIES]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .008),
               ncols=3, frameon=False)
    fig.suptitle(
        f'Sequential adaptation · {ylabel} · mean and 95% seed bootstrap CI',
        y=.995,
    )
    fig.tight_layout(rect=(0, .045, 1, .965), h_pad=1.5, w_pad=1.0)
    return _save(fig, output/f'adaptation_{metric}', dpi)


def _adaptation(curves, output, dpi):
    paths = []
    for metric, ylabel in METRIC_SPECS:
        paths += _adaptation_metric(curves, output, dpi, metric, ylabel)
    return paths


def _summary_plot(rows, value, ylabel, title, output, dpi):
    frame = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(8,5), constrained_layout=True)
    sns.pointplot(data=frame, x='order_id', y=value, hue='policy', hue_order=POLICIES,
                  palette=COLORS, errorbar=('ci',95), n_boot=1000, seed=20260724,
                  dodge=.25, ax=axis)
    axis.set(xlabel='Order', ylabel=ylabel, title=title)
    axis.legend(title=None, labels=[LABELS[p] for p in POLICIES], frameon=False)
    return _save(fig, output, dpi)


def _stage_end_plot(rows, output, dpi):
    frame = pd.DataFrame(rows)
    melted = frame.melt(
        id_vars=['order_id','training_seed','policy','stage_index'],
        value_vars=['zero_shot_normalized','final_normalized'],
        var_name='checkpoint', value_name='normalized_travel_time',
    )
    melted['checkpoint'] = melted.checkpoint.map({
        'zero_shot_normalized':'Zero-shot', 'final_normalized':'Stage final',
    })
    grid = sns.catplot(
        data=melted, x='stage_index', y='normalized_travel_time', hue='policy',
        col='order_id', row='checkpoint', kind='point', hue_order=POLICIES,
        palette=COLORS, errorbar=('ci',95), n_boot=1000, seed=20260724,
        dodge=.25, height=3.5, aspect=1.25,
    )
    grid.set_axis_labels('Stage', 'Travel time / Plan 1 reference')
    grid.set_titles('{row_name} · {col_name}')
    grid.figure.subplots_adjust(top=.90)
    grid.figure.suptitle('Zero-shot transfer and stage-final performance', y=.985)
    if grid.legend:
        for text, policy in zip(grid.legend.texts, POLICIES): text.set_text(LABELS[policy])
    return _save(grid.figure, output/'zero_shot_and_stage_final', dpi)


def _replay_plot(rows, output, dpi, value, ylabel, stem, title):
    frame = pd.DataFrame(rows)
    frame = frame[frame.policy.isin(('fifo','fifo_matched_wait'))]
    orders = sorted(frame.order_id.unique())
    fig, axes = plt.subplots(len(orders), 3, figsize=(15,4*len(orders)), squeeze=False)
    for i, order in enumerate(orders):
        for j, stage in enumerate((2,3,4)):
            axis=axes[i,j]
            selected=frame[(frame.order_id==order)&(frame.stage_index==stage)]
            for policy in ('fifo','fifo_matched_wait'):
                _mean_band(axis, selected[selected.policy==policy], 'local_episode',
                           value, policy)
            axis.set(title=f'{order} · stage {stage}', xlabel='Local training episode',
                     ylabel=ylabel, ylim=(0,1.03))
    handles = [Line2D([], [], color=COLORS[p], linewidth=2, label=LABELS[p])
               for p in ('fifo', 'fifo_matched_wait')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .008),
               ncols=2, frameon=False)
    fig.suptitle(title+' · mean and 95% seed bootstrap CI', y=.995)
    fig.tight_layout(rect=(0, .045, 1, .965), h_pad=1.5, w_pad=1.0)
    return _save(fig, output/stem, dpi)


def _retention_matrix(rows, output, dpi):
    frame = pd.DataFrame(rows)
    frame['scene'] = frame.network.map(SCENE_LABELS).fillna(frame.network)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    for axis, policy in zip(axes, POLICIES):
        table = (frame[frame.policy == policy].pivot_table(
            index='order_id', columns='scene', values='final_retention_normalized',
            aggfunc='mean'
        ).reindex(index=('O1', 'O2', 'O3', 'O4'),
                  columns=('S1', 'S2', 'S3', 'S4')))
        sns.heatmap(table, annot=True, fmt='.3f', cmap='vlag', center=1,
                    cbar=policy == POLICIES[-1], ax=axis)
        axis.set(title=LABELS[policy], xlabel='Evaluation scene', ylabel='Order')
    fig.suptitle('Final frozen retention matrix · travel time / Plan 1 reference')
    return _save(fig, output/'final_retention_matrix', dpi)


def _stage_scene_matrix(rows, output, dpi, value, ylabel, stem, title, center=None):
    frame = pd.DataFrame(rows)
    frame = frame[frame.evaluation_scene_introduced_stage <= frame.stage_index]
    orders = sorted(frame.order_id.unique())
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), squeeze=False)
    for axis, order in zip(axes.flat, orders):
        selected = frame[frame.order_id == order]
        table = selected.pivot_table(
            index='scene', columns='stage_index', values=value, aggfunc='mean'
        ).reindex(index=('S1', 'S2', 'S3', 'S4'), columns=(1, 2, 3, 4))
        annot = table.map(lambda x: '' if pd.isna(x) else f'{x:.3f}')
        sns.heatmap(table, annot=annot, fmt='', cmap='vlag', center=center,
                    cbar=axis is axes.flat[-1], ax=axis, mask=table.isna())
        axis.set(title=order, xlabel='Training stage completed', ylabel='Evaluated scene')
    for axis in axes.flat[len(orders):]:
        axis.set_visible(False)
    fig.suptitle(title, y=.99)
    fig.tight_layout(rect=(0, .02, 1, .95), h_pad=1.5, w_pad=1.2)
    return _save(fig, output/stem, dpi)


def _stage_scene_policy_plot(rows, output, dpi, value, ylabel, stem, title, center=None):
    frame = pd.DataFrame(rows)
    frame = frame[frame.evaluation_scene_introduced_stage <= frame.stage_index]
    orders = sorted(frame.order_id.unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
    for axis, order in zip(axes.flat, orders):
        selected = frame[frame.order_id == order]
        policy_frame = selected.pivot_table(
            index=['scene', 'stage_index'], columns='policy', values=value,
            aggfunc='mean'
        )
        policy_frame = policy_frame.reindex(
            index=pd.MultiIndex.from_product(
                [('S1', 'S2', 'S3', 'S4'), (1, 2, 3, 4)],
                names=['scene', 'stage_index'],
            ), columns=POLICIES
        )
        annot = policy_frame.map(lambda x: '' if pd.isna(x) else f'{x:.3f}')
        sns.heatmap(policy_frame, annot=annot, fmt='', cmap='vlag', center=center,
                    cbar=axis is axes.flat[-1], ax=axis, mask=policy_frame.isna())
        axis.set(title=order, xlabel='Policy', ylabel='Scene · completed stage')
    for axis in axes.flat[len(orders):]:
        axis.set_visible(False)
    fig.suptitle(title, y=.99)
    fig.tight_layout(rect=(0, .02, 1, .95), h_pad=1.5, w_pad=1.2)
    return _save(fig, output/stem, dpi)


def _replay_source_plot(rows, output, dpi, provenance, policy):
    frame = pd.DataFrame(rows)
    frame = frame[(frame.provenance == provenance) & (frame.policy == policy)]
    orders = sorted(frame.order_id.unique())
    palette = dict(zip(('S1', 'S2', 'S3', 'S4'), sns.color_palette('deep', 4)))
    fig, axes = plt.subplots(len(orders), 3, figsize=(15, 3.8 * len(orders)),
                             squeeze=False)
    for i, order in enumerate(orders):
        for j, stage in enumerate((2, 3, 4)):
            axis = axes[i, j]
            selected = frame[(frame.order_id == order) & (frame.stage_index == stage)]
            table = selected.pivot_table(
                index='local_episode', columns='source_scene', values='fraction',
                aggfunc='mean', fill_value=0
            ).reindex(columns=('S1', 'S2', 'S3', 'S4'), fill_value=0)
            if not table.empty:
                axis.stackplot(table.index, *(table[c] for c in table.columns),
                               labels=table.columns, colors=[palette[c] for c in table.columns],
                               alpha=.82)
            axis.set(title=f'{order} · Stage {stage}', xlabel='Local training episode',
                     ylabel='Transition fraction', ylim=(0, 1))
            axis.grid(alpha=.15)
    handles = [Line2D([], [], color=palette[s], linewidth=7, label=s)
               for s in ('S1', 'S2', 'S3', 'S4')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .008),
               ncols=4, frameon=False)
    label = 'Buffer composition' if provenance == 'buffer' else 'Actual sample provenance'
    fig.suptitle(f'{label} · {LABELS[policy]} · five-seed mean', y=.995)
    fig.tight_layout(rect=(0, .05, 1, .965), h_pad=1.5, w_pad=1.0)
    return _save(fig, output/f'replay_{provenance}_composition_{policy}', dpi)


def _replacement_plot(rows, output, dpi):
    frame = pd.DataFrame(rows)
    frame = frame[frame.policy.isin(('fifo', 'fifo_matched_wait'))].drop_duplicates(
        ['logical_run_id', 'stage_index', 'threshold']
    )
    frame['threshold_label'] = frame.threshold.map({.5: '≤50%', .1: '≤10%', 0.: '0%'})
    grid = sns.catplot(
        data=frame, x='stage_index', y='reached_local_episode',
        hue='threshold_label', hue_order=('≤50%', '≤10%', '0%'),
        col='order_id', row='policy', kind='point', errorbar=('ci', 95),
        n_boot=1000, seed=20260724, dodge=.25, height=3.4, aspect=1.05,
    )
    grid.set_axis_labels('Stage', 'Local episode first reached')
    grid.set_titles('{row_name} · {col_name}')
    grid.figure.subplots_adjust(top=.90)
    grid.figure.suptitle('Historical replay replacement speed', y=.985)
    return _save(grid.figure, output/'replay_replacement_thresholds', dpi)


def _replay_performance_alignment(curves, replay, output, dpi, metric, ylabel):
    performance = pd.DataFrame(curves)
    composition = pd.DataFrame(replay)
    keys = ['logical_run_id', 'order_id', 'training_seed', 'policy',
            'stage_index', 'local_episode']
    merged = performance.merge(
        composition[keys + ['current_buffer_fraction']], on=keys, how='inner'
    )
    merged = merged[merged.policy.isin(('fifo', 'fifo_matched_wait'))]
    orders = sorted(merged.order_id.unique())
    fig, axes = plt.subplots(len(orders), 3, figsize=(15, 4 * len(orders)), squeeze=False)
    for i, order in enumerate(orders):
        for j, stage in enumerate((2, 3, 4)):
            axis = axes[i, j]
            twin = axis.twinx()
            selected = merged[(merged.order_id == order) & (merged.stage_index == stage)]
            for policy in ('fifo', 'fifo_matched_wait'):
                part = selected[selected.policy == policy]
                mean = part.groupby('local_episode')[[metric, 'current_buffer_fraction']].mean()
                axis.plot(mean.index, mean[metric], color=COLORS[policy], linewidth=1.8)
                twin.plot(mean.index, mean.current_buffer_fraction, color=COLORS[policy],
                          linestyle=':', linewidth=1.4, alpha=.8)
            axis.set(title=f'{order} · Stage {stage}', xlabel='Local training episode',
                     ylabel=ylabel)
            twin.set(ylabel='Current-scene replay fraction', ylim=(0, 1.03))
            axis.grid(alpha=.16)
    handles = []
    for policy in ('fifo', 'fifo_matched_wait'):
        handles.extend((
            Line2D([], [], color=COLORS[policy], linewidth=2, label=f'{LABELS[policy]} · metric'),
            Line2D([], [], color=COLORS[policy], linestyle=':', linewidth=2,
                   label=f'{LABELS[policy]} · replay fraction'),
        ))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .005),
               ncols=2, frameon=False)
    fig.suptitle(f'Replay/performance alignment · {ylabel} · five-seed mean', y=.995)
    fig.tight_layout(rect=(0, .065, 1, .965), h_pad=1.6, w_pad=1.3)
    return _save(fig, output/f'replay_alignment_{metric}', dpi)


def _prepare_frozen_timeseries(records, smoothing_window_seconds):
    frame = pd.DataFrame(records).sort_values(
        ['controller_id', 'evaluation_seed', 'simulation_time_seconds']
    )
    prepared = []
    for _, group in frame.groupby(['controller_id', 'evaluation_seed'], sort=False):
        group = group.copy()
        interval = max(1, int(group.action_interval_seconds.iloc[0]))
        window = max(1, int(round(smoothing_window_seconds / interval)))
        group['reward_network_mean_trend'] = group.reward_network_mean.expanding().mean()
        group['reward_network_sum_trend'] = group.reward_network_sum.expanding().mean()
        group['delay_network_weighted_mean_trend'] = (
            group.delay_network_weighted_mean.expanding().mean()
        )
        group['queue_network_mean_trend'] = (
            group.queue_network_mean.rolling(window, min_periods=1).mean()
        )
        group['queue_network_sum_trend'] = (
            group.queue_network_sum.rolling(window, min_periods=1).mean()
        )
        group['throughput_interval_trend'] = (
            group.throughput_interval.rolling(window, min_periods=1).sum()
        )
        group['throughput_cumulative_trend'] = group.throughput_cumulative
        prepared.append(group)
    return pd.concat(prepared, ignore_index=True)


def _plot_frozen_mean_ci(axis, frame, field, color, label, linestyle='-'):
    if frame.empty:
        return
    grouped = frame.groupby(
        ['training_seed', 'simulation_time_seconds'], as_index=False, dropna=False
    )[field].mean()
    pivot = grouped.pivot_table(
        index='training_seed', columns='simulation_time_seconds', values=field
    )
    xs = pivot.columns.to_numpy(float)
    values = pivot.to_numpy(float)
    axis.plot(xs, np.nanmean(values, axis=0), color=color, linewidth=1.8,
              linestyle=linestyle, label=label)
    if len(values) > 1:
        rng = np.random.default_rng(20260724)
        idx = rng.integers(0, len(values), size=(1000, len(values)))
        means = np.nanmean(values[idx], axis=1)
        lo, hi = np.nanpercentile(means, (2.5, 97.5), axis=0)
        axis.fill_between(xs, lo, hi, color=color, alpha=.12, linewidth=0)


def _render_frozen_order_facets(records, output, orders, smoothing_window_seconds, dpi):
    frame = _prepare_frozen_timeseries(records, smoothing_window_seconds)
    specs = (
        ('reward_network_mean_trend', 'reward_network_mean',
         'Cumulative mean network reward', 'Reward (network mean)'),
        ('reward_network_sum_trend', 'reward_network_sum',
         'Cumulative mean total network reward', 'Reward (network sum)'),
        ('queue_network_mean_trend', 'queue_network_mean',
         f'Network queue ({smoothing_window_seconds:g}-s moving mean)',
         'Queued vehicles (lane mean)'),
        ('queue_network_sum_trend', 'queue_network_sum',
         f'Total network queue ({smoothing_window_seconds:g}-s moving mean)',
         'Queued vehicles (network sum)'),
        ('delay_network_weighted_mean_trend', 'delay_weighted_mean',
         'Cumulative vehicle-weighted network delay', 'Weighted delay'),
        ('throughput_interval_trend', 'throughput_interval',
         f'Completed vehicles per {smoothing_window_seconds:g}-s window',
         'Completed vehicles'),
        ('throughput_cumulative_trend', 'throughput_cumulative',
         'Cumulative completed vehicles', 'Completed vehicles'),
    )
    baseline_colors = {'dqn': '0.15', 'maxpressure': '#CC3311',
                       'fixedtime': '#777777'}
    outputs = []
    for field, stem, title, ylabel in specs:
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.9), squeeze=False)
        for axis, order in zip(axes.flat, orders):
            for policy in POLICIES:
                agent = f'{order.lower()}_{policy}'
                _plot_frozen_mean_ci(
                    axis, frame[frame.agent == agent], field, COLORS[policy],
                    LABELS[policy]
                )
            for agent, linestyle in (('dqn', '--'), ('maxpressure', '-.'),
                                     ('fixedtime', ':')):
                _plot_frozen_mean_ci(
                    axis, frame[frame.agent == agent], field,
                    baseline_colors[agent],
                    {'dqn': 'Plan 1 DQN', 'maxpressure': 'MaxPressure',
                     'fixedtime': 'FixedTime'}[agent], linestyle
                )
            axis.set(title=order, xlabel='Simulation time (s)', ylabel=ylabel)
            axis.grid(alpha=.18)
        for axis in axes.flat[len(orders):]:
            axis.set_visible(False)
        handles = [
            Line2D([], [], color=COLORS[p], linewidth=2, label=LABELS[p])
            for p in POLICIES
        ] + [
            Line2D([], [], color=baseline_colors[a], linestyle=s, linewidth=2, label=l)
            for a, s, l in (('dqn', '--', 'Plan 1 DQN'),
                            ('maxpressure', '-.', 'MaxPressure'),
                            ('fixedtime', ':', 'FixedTime'))
        ]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .012),
                   ncols=6, frameon=False)
        fig.suptitle(f'{title} · frozen one-scene validation', y=.99)
        fig.tight_layout(rect=(0, .075, 1, .945), h_pad=1.5, w_pad=1.2)
        outputs += _save(fig, output/'results'/'frozen_timeseries'/stem, dpi)
    return outputs


def run_sequential_plotting(args):
    source = Path(args.analysis_report).expanduser().resolve()
    report = json.loads(source.read_text(encoding='utf-8'))
    scope = report.get('analysis_scope', {})
    if report.get('mode') not in ('formal','formal_partial'):
        raise ValueError('Sequential plotting requires a validated formal analysis report')
    if not scope.get('complete_selected_order_matrix'):
        raise ValueError('Selected-order matrix is not complete')
    output = Path(args.output_root).expanduser().resolve()/args.analysis_id
    if output.exists(): raise FileExistsError(f'Analysis output already exists: {output}')
    tables, figures, inputs = output/'tables', output/'figures', output/'inputs'
    tables.mkdir(parents=True); figures.mkdir(); inputs.mkdir()
    try:
        shutil.copy2(source, inputs/'sequential_analysis.json')
        (runs, curves, stages, retention, forgetting, replay_sources,
         replacement, stage_scene, replay) = _flatten(report)
        for name, rows in [('run_summary',runs),('adaptation_curves',curves),
                           ('stage_summary',stages),('final_retention',retention),
                           ('forgetting',forgetting),('replay_diagnostics',replay),
                           ('replay_sources', replay_sources),
                           ('replay_replacement', replacement),
                           ('stage_scene_evaluations', stage_scene)]:
            _write_csv(tables/f'{name}.csv', rows)
        paths=[]
        paths += _adaptation(curves, figures, args.dpi)
        paths += _summary_plot(runs, 'primary_normalized_auc', 'Normalized travel-time AUC',
                               'Adaptation efficiency across completed orders', figures/'primary_auc', args.dpi)
        paths += _stage_end_plot(stages, figures, args.dpi)
        paths += _stage_scene_matrix(
            stage_scene, figures, args.dpi, 'forgetting_travel_time',
            'Travel-time change (s)', 'forgetting_stage_scene_matrix',
            'Stage-wise scene forgetting · evaluated after each completed stage', center=0,
        )
        paths += _stage_scene_policy_plot(
            stage_scene, figures, args.dpi, 'forgetting_travel_time',
            'Travel-time change (s)', 'forgetting_stage_scene_policy',
            'Stage-wise scene forgetting by policy', center=0,
        )
        paths += _stage_scene_matrix(
            stage_scene, figures, args.dpi, 'final_retention_normalized',
            'Travel time / same-scene Plan 1 reference', 'retention_stage_scene_matrix',
            'Stage-wise scene retention · same-scene Plan 1 reference', center=1,
        )
        paths += _stage_scene_policy_plot(
            stage_scene, figures, args.dpi, 'final_retention_normalized',
            'Travel time / same-scene Plan 1 reference', 'retention_stage_scene_policy',
            'Stage-wise scene retention by policy', center=1,
        )
        paths += _replay_plot(
            replay, figures, args.dpi, 'current_buffer_fraction',
            'Current-scene fraction in replay buffer', 'replay_buffer_composition',
            'FIFO replay composition',
        )
        paths += _replay_plot(
            replay, figures, args.dpi, 'episode_current_sample_fraction',
            'Current-scene fraction in this episode samples',
            'replay_current_sample_fraction', 'FIFO replay sampling provenance',
        )
        for provenance in ('buffer', 'sample'):
            for policy in ('fifo', 'fifo_matched_wait'):
                paths += _replay_source_plot(
                    replay_sources, figures, args.dpi, provenance, policy
                )
        paths += _replacement_plot(replacement, figures, args.dpi)
        for metric, ylabel in (
            ('travel_time', 'Travel time (s)'), ('real_delay', 'Real delay (s)'),
            ('delay', 'Approximate delay'), ('queue', 'Mean queued vehicles'),
            ('throughput', 'Throughput (vehicles)'),
        ):
            paths += _replay_performance_alignment(
                curves, replay, figures, args.dpi, metric, ylabel
            )
        manifest = {
            'schema_version':1, 'tool':'tools.experiment_plotting', 'tool_version':__version__,
            'analysis_type':'sequential_partial' if report['mode']=='formal_partial' else 'sequential_formal',
            'analysis_id':args.analysis_id, 'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'source_report':str(source), 'analysis_scope':scope,
            'formal_inference_performed':report['formal_inference_performed'],
            'notice':report.get('pilot_notice'), 'included_run_count':len(runs),
            'tables':sorted(str(p.relative_to(output)) for p in tables.iterdir()),
            'figures':sorted(str(p.relative_to(output)) for p in paths),
            'plot_semantics':{'curve':'five-seed mean with 95% percentile bootstrap CI',
                              'lower_travel_time_and_auc_is_better':True,
                              'replay_sample_fraction':'per-episode deltas of cumulative sample counts',
                              'partial_scope_is_descriptive_only':not report['formal_inference_performed']},
            'library_versions':{p:metadata.version(p) for p in ('matplotlib','numpy','pandas','seaborn')},
        }
        (output/'plotting_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return output


def run_sequential_frozen_plotting(args):
    evaluation_roots = [
        Path(path).expanduser().resolve()
        for path in args.sequential_evaluation_root
    ]
    baseline_root = Path(args.baseline_package).expanduser().resolve()
    output = Path(args.output_root).expanduser().resolve() / args.analysis_id
    if output.exists():
        raise FileExistsError(f'Analysis output already exists: {output}')
    baseline = validate_evaluation_package(str(baseline_root))
    sequential_records = []
    sequential_summaries = []
    committed_paths = sorted(
        path for root in evaluation_roots
        for path in root.glob('physical/*/committed.json')
    )
    for committed_path in committed_paths:
        committed = json.loads(committed_path.read_text(encoding='utf-8'))
        decisions_path = Path(committed['decisions_path'])
        request = json.loads((decisions_path.parent / 'request.json').read_text(
            encoding='utf-8'
        ))
        records = [
            json.loads(line) for line in decisions_path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        if len(records) != 360:
            raise ValueError(f'Expected 360 sequential decisions: {decisions_path}')
        sequential_records.extend(records)
        summary = json.loads(Path(committed['summary_path']).read_text(encoding='utf-8'))
        sequential_summaries.append({
            'controller_id': request['controller_id'],
            'agent': request['agent'], 'training_seed': request['training_seed'],
            'evaluation_seed': request['evaluation_seed'],
            'network': request['evaluation_network'],
            **{key: summary[key] for key in (
                'travel_time', 'reward_mean', 'queue', 'delay', 'real_delay',
                'throughput', 'unfinished_vehicles',
            )},
        })
    orders = tuple(dict.fromkeys(order.upper() for order in args.orders))
    expected_agents = {
        f'{order.lower()}_{policy}'
        for order in orders
        for policy in ('clear', 'fifo', 'fifo_matched_wait')
    }
    expected_units = {
        (agent, seed) for agent in expected_agents for seed in range(5)
    }
    actual_units = {
        (row['agent'], int(row['training_seed'])) for row in sequential_summaries
    }
    if actual_units != expected_units:
        raise ValueError(
            f'Sequential frozen matrix mismatch: missing={sorted(expected_units-actual_units)}, '
            f'extra={sorted(actual_units-expected_units)}'
        )
    with (baseline_root / 'records.jsonl').open(encoding='utf-8') as handle:
        baseline_records = [
            json.loads(line) for line in handle if line.strip()
        ]
    baseline_records = [
        row for row in baseline_records
        if row['network'] == args.network
        and int(row['evaluation_seed']) == int(args.evaluation_seed)
        and row['agent'] in {'dqn', 'fixedtime', 'maxpressure'}
    ]
    if len(baseline_records) != 7 * 360:
        raise ValueError(
            f'Expected 2520 Plan 1/baseline decision records, got {len(baseline_records)}'
        )
    with (baseline_root / 'summary.csv').open(newline='', encoding='utf-8') as handle:
        baseline_summaries = [
            row for row in csv.DictReader(handle)
            if row['network'] == args.network
            and int(row['evaluation_seed']) == int(args.evaluation_seed)
            and row['agent'] in {'dqn', 'fixedtime', 'maxpressure'}
        ]
    if len(baseline_summaries) != 7:
        raise ValueError(
            f'Expected seven Plan 1/baseline episode summaries, got '
            f'{len(baseline_summaries)}'
        )
    all_records = sequential_records + baseline_records
    if any(
        row['network'] != args.network
        or int(row['evaluation_seed']) != int(args.evaluation_seed)
        or float(row['simulation_time_seconds']) not in range(10, 3601, 10)
        for row in all_records
    ):
        raise ValueError('Frozen comparison records do not share one scene/seed/time grid')
    figures = output / 'figures'
    tables = output / 'tables'
    inputs = output / 'inputs'
    figures.mkdir(parents=True); tables.mkdir(); inputs.mkdir()
    try:
        figure_paths = _render_frozen_order_facets(
            all_records, figures, orders, args.smoothing_window_seconds, args.dpi,
        )
        _write_csv(
            tables / 'episode_summary.csv',
            sequential_summaries + baseline_summaries,
        )
        with (inputs / 'combined_decision_records.jsonl').open(
            'w', encoding='utf-8'
        ) as handle:
            for row in all_records:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        shutil.copy2(baseline_root / 'manifest.json', inputs / 'baseline_manifest.json')
        manifest = {
            'schema_version': 1, 'tool': 'tools.experiment_plotting',
            'tool_version': __version__, 'analysis_type': 'sequential_frozen_timeseries',
            'analysis_id': args.analysis_id,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'sequential_evaluation_roots': [str(path) for path in evaluation_roots],
            'network': args.network, 'scene': args.scene,
            'evaluation_seed': int(args.evaluation_seed),
            'simulation_duration_seconds': 3600,
            'sampling_interval_seconds': 10,
            'smoothing_window_seconds': args.smoothing_window_seconds,
            'orders': list(orders),
            'sequential_controller_count': len(sequential_summaries),
            'baseline_controller_count': 7,
            'decision_record_count': len(all_records),
            'controllers': sorted({row['agent'] for row in all_records}),
            'figures': sorted(str(path.relative_to(output)) for path in figure_paths),
            'tables': ['tables/episode_summary.csv'],
            'inputs': [
                'inputs/combined_decision_records.jsonl',
                'inputs/baseline_manifest.json',
            ],
            'semantics': {
                'comparison': 'one frozen target scene and one common evaluation traffic seed',
                'sequential_checkpoint': 'stage_4 local_episode_100 final online snapshot',
                'plan1_checkpoint': 'existing selected best checkpoint',
                'curve': 'order-faceted mean with 95% percentile bootstrap CI over five training seeds; baselines share the same evaluation seed',
                'renderer': '_render_frozen_order_facets',
            },
        }
        (output / 'plotting_manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return output
