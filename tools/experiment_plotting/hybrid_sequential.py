"""Publication-oriented plots for validated CS-HR M0--M3 reports."""

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
from .sequential import SCENE_LABELS, _write_csv


CONDITIONS = ('M0', 'M1', 'M2', 'M3')
LABELS = {
    'M0': 'M0 · Online-only', 'M1': 'M1 · Seq-FIFO',
    'M2': 'M2 · Naive hybrid', 'M3': 'M3 · CS-HR',
}
COLORS = dict(zip(CONDITIONS, sns.color_palette('colorblind', 4)))


def _flatten_hybrid(report):
    runs, curves, retention, forgetting = [], [], [], []
    stage_scene, replay, replay_sources = [], [], []
    for run in report['runs']:
        base = {
            key: run[key] for key in (
                'logical_run_id', 'order_id', 'training_seed', 'condition'
            )
        }
        runs.append({
            **base, 'primary_normalized_auc': run['primary_normalized_auc'],
            **run['secondary_metrics'],
        })
        curves.extend({**base, **row} for row in run['adaptation_curves'])
        final_stage = max(row['stage_index'] for row in run['stage_scene_evaluations'])
        network_positions = {
            row['evaluation_network']: int(row['evaluation_scene_introduced_stage'])
            for row in run['stage_scene_evaluations']
        }
        for row in run['stage_scene_evaluations']:
            flattened = {
                **base, **row,
                'scene': SCENE_LABELS.get(
                    row['evaluation_network'], row['evaluation_network']
                ),
            }
            stage_scene.append(flattened)
            if int(row['stage_index']) == final_stage:
                retention.append({
                    **base,
                    'scene_position': int(row['evaluation_scene_introduced_stage']),
                    'scene': flattened['scene'],
                    'value': row['final_retention_normalized'],
                })
                if int(row['evaluation_scene_introduced_stage']) < final_stage:
                    forgetting.append({
                        **base,
                        'scene_position': int(row['evaluation_scene_introduced_stage']),
                        'scene': flattened['scene'],
                        'value': row['forgetting_travel_time'],
                    })
        previous_kind, previous_scene = {}, {}
        for row in sorted(
                run['replay_diagnostics'],
                key=lambda item: (int(item['stage_index']), int(item['local_episode']))):
            cumulative_kind = row.get('samples_drawn_by_kind') or {}
            kind_delta = {
                kind: int(count) - int(previous_kind.get(kind, 0))
                for kind, count in cumulative_kind.items()
            }
            previous_kind = dict(cumulative_kind)
            total = sum(kind_delta.values())
            replay.append({
                **base, 'stage_index': int(row['stage_index']),
                'local_episode': int(row['local_episode']),
                'actual_historical_fraction': (
                    np.nan if total <= 0 else kind_delta.get('historical', 0) / total
                ),
                'sample_age_mean': row.get('sample_age_mean'),
                'sample_age_p95': row.get('sample_age_p95'),
            })
            cumulative_scene = row.get('samples_drawn_by_scene') or {}
            scene_delta = {
                network: int(count) - int(previous_scene.get(network, 0))
                for network, count in cumulative_scene.items()
            }
            previous_scene = dict(cumulative_scene)
            scene_total = sum(scene_delta.values())
            if scene_total > 0:
                for network, count in scene_delta.items():
                    replay_sources.append({
                        **base, 'stage_index': int(row['stage_index']),
                        'local_episode': int(row['local_episode']),
                        'source_network': network,
                        'source_scene': SCENE_LABELS.get(network, network),
                        'source_position': network_positions.get(network),
                        'fraction': count / scene_total,
                    })
    return {
        'runs': runs, 'adaptation_curves': curves,
        'final_retention': retention, 'forgetting': forgetting,
        'stage_scene_evaluations': stage_scene,
        'replay_diagnostics': replay, 'replay_sources': replay_sources,
    }


def _mean_band(axis, frame, x, y, condition):
    pivot = frame.pivot_table(
        index=['order_id', 'training_seed'], columns=x, values=y,
    )
    if pivot.empty:
        return
    values = pivot.to_numpy(float)
    xs = pivot.columns.to_numpy()
    axis.plot(xs, np.nanmean(values, axis=0), color=COLORS[condition],
              linewidth=1.8, label=LABELS[condition])
    if len(values) > 1:
        rng = np.random.default_rng(20260726)
        indices = rng.integers(0, len(values), size=(1000, len(values)))
        means = np.nanmean(values[indices], axis=1)
        lo, hi = np.nanpercentile(means, (2.5, 97.5), axis=0)
        axis.fill_between(xs, lo, hi, color=COLORS[condition], alpha=.15)


def _adaptation_plot(rows, output, dpi):
    frame = pd.DataFrame(rows)
    orders = sorted(frame.order_id.unique())
    fig, axes = plt.subplots(
        len(orders), 3, figsize=(15, 4 * len(orders)), squeeze=False,
    )
    for i, order in enumerate(orders):
        for j, stage in enumerate((2, 3, 4)):
            axis = axes[i, j]
            selected = frame[(frame.order_id == order) & (frame.stage_index == stage)]
            for condition in CONDITIONS:
                _mean_band(
                    axis, selected[selected.condition == condition],
                    'local_episode', 'normalized_travel_time', condition,
                )
            axis.axhline(1, color='0.3', linestyle='--', linewidth=.8)
            axis.set(title=f'{order} · Stage {stage}', xlabel='Local episode',
                     ylabel='Travel time / Plan 1 reference')
            axis.grid(alpha=.18)
    handles = [Line2D([], [], color=COLORS[c], linewidth=2, label=LABELS[c])
               for c in CONDITIONS]
    fig.legend(handles=handles, loc='lower center', ncols=4, frameon=False)
    fig.suptitle('CS-HR adaptation curves · paired-seed mean and 95% bootstrap CI')
    fig.tight_layout(rect=(0, .055, 1, .96))
    return _save(fig, output / 'adaptation_normalized_travel_time', dpi)


def _position_plot(rows, ylabel, title, output, dpi, center=None):
    frame = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    sns.pointplot(
        data=frame, x='scene_position', y='value', hue='condition',
        hue_order=CONDITIONS, palette=COLORS, errorbar=('ci', 95),
        n_boot=1000, seed=20260726, dodge=.25, ax=axis,
    )
    if center is not None:
        axis.axhline(center, color='0.3', linestyle='--', linewidth=.8)
    axis.set(xlabel='Scene position in order', ylabel=ylabel, title=title)
    axis.legend(title=None, frameon=False)
    return _save(fig, output, dpi)


def _lower_triangle_plot(rows, output, dpi):
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    for axis, condition in zip(axes.flat, CONDITIONS):
        selected = frame[frame.condition == condition]
        table = selected.pivot_table(
            index='evaluation_scene_introduced_stage', columns='stage_index',
            values='final_retention_normalized', aggfunc='mean',
        ).reindex(index=(1, 2, 3, 4), columns=(1, 2, 3, 4))
        sns.heatmap(table, mask=table.isna(), annot=True, fmt='.3f', cmap='vlag',
                    center=1, cbar=condition == 'M3', ax=axis)
        axis.set(title=LABELS[condition], xlabel='Completed training stage',
                 ylabel='Evaluated scene position')
    fig.suptitle('Strict lower-triangular retention matrix')
    fig.tight_layout(rect=(0, 0, 1, .96))
    return _save(fig, output / 'lower_triangular_retention_matrix', dpi)


def _replay_quota_plot(rows, output, dpi):
    frame = pd.DataFrame(rows)
    frame = frame[(frame.condition.isin(('M2', 'M3'))) & (frame.stage_index > 1)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), squeeze=False)
    for axis, stage in zip(axes.flat, (2, 3, 4)):
        selected = frame[frame.stage_index == stage]
        for condition in ('M2', 'M3'):
            _mean_band(
                axis, selected[selected.condition == condition],
                'local_episode', 'actual_historical_fraction', condition,
            )
        axis.axhline(.5, color='0.25', linestyle='--', linewidth=.9)
        axis.set(title=f'Stage {stage}', xlabel='Local episode',
                 ylabel='Actual historical sample fraction', ylim=(0, 1.02))
        axis.grid(alpha=.18)
    fig.legend(handles=[
        Line2D([], [], color=COLORS[c], linewidth=2, label=LABELS[c])
        for c in ('M2', 'M3')
    ], loc='lower center', ncols=2, frameon=False)
    fig.suptitle('Hybrid batch quota · per-episode count deltas')
    fig.tight_layout(rect=(0, .07, 1, .94))
    return _save(fig, output / 'replay_actual_historical_fraction', dpi)


def _source_plot(rows, output, dpi):
    frame = pd.DataFrame(rows)
    frame = frame[(frame.condition == 'M3') & (frame.stage_index > 1)]
    stages = sorted(frame.stage_index.unique())
    fig, axes = plt.subplots(1, len(stages), figsize=(5 * len(stages), 4.2), squeeze=False)
    palette = dict(zip((1, 2, 3, 4), sns.color_palette('deep', 4)))
    for axis, stage in zip(axes.flat, stages):
        table = frame[frame.stage_index == stage].pivot_table(
            index='local_episode', columns='source_position', values='fraction',
            aggfunc='mean', fill_value=0,
        ).reindex(columns=(1, 2, 3, 4), fill_value=0)
        if not table.empty:
            axis.stackplot(
                table.index, *(table[column] for column in table.columns),
                colors=[palette[column] for column in table.columns], alpha=.82,
            )
        axis.set(title=f'Stage {stage}', xlabel='Local episode',
                 ylabel='Sample fraction', ylim=(0, 1))
    fig.legend(handles=[
        Line2D([], [], color=palette[p], linewidth=7, label=f'Position {p}')
        for p in (1, 2, 3, 4)
    ], loc='lower center', ncols=4, frameon=False)
    fig.suptitle('M3 sample-source composition by scene position')
    fig.tight_layout(rect=(0, .07, 1, .94))
    return _save(fig, output / 'm3_sample_source_composition', dpi)


def _effect_plot(report, output, dpi):
    rows = []
    for contrast, metrics in report['paired_effects'].items():
        for metric, effect in metrics.items():
            interval = (effect.get('bootstrap') or {}).get('confidence_interval_95')
            rows.append({
                'contrast': contrast, 'metric': metric,
                'effect': effect['effect_mean'],
                'low': interval[0] if interval else np.nan,
                'high': interval[1] if interval else np.nan,
            })
    frame = pd.DataFrame(rows)
    metrics = ('adaptation_auc', 'final_retention', 'forgetting_travel_time')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
    for axis, metric in zip(axes.flat, metrics):
        selected = frame[frame.metric == metric].reset_index(drop=True)
        y = np.arange(len(selected))
        axis.scatter(selected.effect, y, color='#0072B2')
        valid = selected[['low', 'high']].notna().all(axis=1)
        if valid.any():
            subset = selected[valid]
            axis.errorbar(
                subset.effect, y[valid],
                xerr=np.vstack((subset.effect - subset.low,
                                subset.high - subset.effect)),
                fmt='none', color='#0072B2', capsize=3,
            )
        axis.axvline(0, color='0.3', linestyle='--', linewidth=.8)
        axis.set(yticks=y, yticklabels=selected.contrast, title=metric,
                 xlabel='Baseline − candidate (positive favors candidate)')
        axis.grid(axis='x', alpha=.18)
    fig.suptitle('Pre-specified paired CS-HR effects')
    fig.tight_layout(rect=(0, 0, 1, .95))
    return _save(fig, output / 'paired_effects', dpi)


def run_hybrid_sequential_plotting(args):
    source = Path(args.analysis_report).expanduser().resolve()
    report = json.loads(source.read_text(encoding='utf-8'))
    if report.get('experiment') != 'cs_hr' or report.get('mode') not in (
            'formal', 'formal_partial'):
        raise ValueError('CS-HR plotting requires a validated formal analysis report')
    output = Path(args.output_root).expanduser().resolve() / args.analysis_id
    if output.exists():
        raise FileExistsError(f'Analysis output already exists: {output}')
    tables, figures, inputs = output / 'tables', output / 'figures', output / 'inputs'
    tables.mkdir(parents=True); figures.mkdir(); inputs.mkdir()
    try:
        shutil.copy2(source, inputs / 'cs_hr_analysis.json')
        flattened = _flatten_hybrid(report)
        for name, rows in flattened.items():
            _write_csv(tables / f'{name}.csv', rows)
        paths = []
        paths += _adaptation_plot(flattened['adaptation_curves'], figures, args.dpi)
        paths += _position_plot(
            flattened['final_retention'], 'Travel time / Plan 1 reference',
            'Final retention by scene position', figures / 'final_retention',
            args.dpi, center=1,
        )
        paths += _position_plot(
            flattened['forgetting'], 'Travel-time change (s)',
            'Final forgetting by scene position', figures / 'forgetting',
            args.dpi, center=0,
        )
        paths += _lower_triangle_plot(
            flattened['stage_scene_evaluations'], figures, args.dpi,
        )
        paths += _replay_quota_plot(flattened['replay_diagnostics'], figures, args.dpi)
        if flattened['replay_sources']:
            paths += _source_plot(flattened['replay_sources'], figures, args.dpi)
        paths += _effect_plot(report, figures, args.dpi)
        manifest = {
            'schema_version': 1, 'tool': 'tools.experiment_plotting',
            'tool_version': __version__, 'analysis_type': 'cs_hr_formal',
            'analysis_id': args.analysis_id,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_report': str(source),
            'formal_inference_performed': report['formal_inference_performed'],
            'included_run_count': len(flattened['runs']),
            'tables': sorted(str(path.relative_to(output)) for path in tables.iterdir()),
            'figures': sorted(str(path.relative_to(output)) for path in paths),
            'plot_semantics': {
                'lower_cost_is_better': True,
                'replay_fraction': 'per-episode deltas of cumulative kind/source counts',
                'future_scene_cells': 'absent_and_masked_not_zero_filled',
                'partial_scope_is_descriptive_only': not report[
                    'formal_inference_performed'
                ],
            },
            'library_versions': {
                package: metadata.version(package)
                for package in ('matplotlib', 'numpy', 'pandas', 'seaborn')
            },
        }
        (output / 'plotting_manifest.json').write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return output
