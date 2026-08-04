#!/usr/bin/env python3
"""Read-only Plan 1 observation-conditioned outcome analysis.

This program consumes formal trajectory shards and final checkpoints.  It never
launches SUMO/training and writes only analysis/plan1_observation_conditioned_outcomes.
Stages are idempotent: each command overwrites only its own derived products.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
RUNLIST = ROOT / 'data/output_data/analysis/plan1/p1_formal_20_runlist_20260722.csv'
OUT = ROOT / 'analysis/plan1_observation_conditioned_outcomes'
SCENES = ('sumohz1x1_config2', 'sumohz1x1', 'sumohz1x1_config4', 'sumohz1x1_config3')
PAIRS = tuple(combinations(SCENES, 2))
METRICS = ('queue', 'waiting_time', 'approximate_delay', 'real_delay', 'reward',
           'next_l1', 'next_l2', 'next_max_lane', 'next_total', 'state_to_next_total')


def out_file(name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / name


def pair_name(a: str, b: str) -> str:
    return a + '__vs__' + b


def write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(out_file(name), index=False)


def runs() -> pd.DataFrame:
    x = pd.read_csv(RUNLIST)
    return x[(x.role == 'formal') & (x.agent == 'dqn') &
             (x.include.astype(str).str.lower() == 'true')].copy()


def scopes():
    return {'P1_ALL_400': 400, 'HOA_USED_100': 100}


def state_columns(prefix='state'):
    return [f'{prefix}_{i}' for i in range(8)]


MATCH_COLS = state_columns() + ['current_phase', 'action']


def analysis_config() -> None:
    cfg = {
        'scenes': list(SCENES),
        'data_scopes': {'P1_ALL_400': 'episodes 1-400', 'HOA_USED_100': 'episodes 1-100'},
        'strict_matching': 'state_0..7 exactly equal + current_phase + action',
        'near_strict_matching': {'same_phase_action': True, 'per_lane_abs_difference_max': 1,
                                 'state_raw_l1_max': 4, 'mutual': True, 'one_to_one': True},
        'mnn_distances': ['raw_l1', 'pooled_iqr_normalized_l1', 'log1p_l1'],
        'gamma': 0.95, 'formal_target_has_terminal_mask': False,
        'cluster_bootstrap': {'seed': 20260731, 'resamples': 500,
                              'cluster': 'training seed, then matching group after episode aggregation'},
        'provenance': {'runlist': str(RUNLIST), 'sumo_or_training_started': False},
    }
    out_file('analysis_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')


def semantics() -> None:
    text = '''# Reward and diagnostic semantics

All four formal scenes use the same SUMO, DQN, trainer, generator, and `Metrics` implementation. A trajectory record is appended after ten one-second `env.step` calls: `state`, `current_phase`, and `action` are pre-interval; `next_state`, `next_phase`, reward, and diagnostics are post-interval.

## Reward

`agent/dqn.py:48-52,119-130` uses incoming-lane `lane_waiting_count`, averages across the eight incoming lanes, negates it, then multiplies by 12. Therefore one simulator-step reward is `-12 * mean_8(waiting vehicle count)`, or `-1.5 * sum_8(waiting vehicle count)`. `trainer/tsc_trainer.py:481-487` saves the mean of the ten simulator-step rewards in the action interval. It has no direct calculation dependence on saved `queue`, `waiting_time`, `approximate_delay`, or `real_delay`, although all reflect traffic state.

## Diagnostics

* **queue** — `agent/dqn.py:99-102`, `common/metrics.py:39-48,92-104`, saved at `trainer/tsc_trainer.py:517`: cumulative mean, over decisions elapsed in the episode, of the intersection-level incoming `lane_waiting_count` aggregate. Unit: vehicles. It is an episode-running average, not a current instantaneous queue.
* **approximate_delay** — `agent/dqn.py:102-105`, `world/world_sumo.py:890-915`, `common/metrics.py:79-90`, saved at `trainer/tsc_trainer.py:518-521`: lane delay is `1 - mean_vehicle_speed / lane_speed_limit` (empty lane zero), averaged across incoming lanes and then cumulatively across decisions elapsed in the episode. Unit: dimensionless normalized speed deficit; not seconds or an instantaneous field.
* **waiting_time** — `common/metrics.py:72-73`, `world/world_sumo.py:944-955`, saved at `trainer/tsc_trainer.py:522-523`: at interval end, mean SUMO accumulated waiting time among currently active network vehicles. Unit: seconds. It is a current snapshot of vehicle history, not a lane total or interval value.
* **real_delay** — `common/metrics.py:69-70`, `world/world_sumo.py:1003-1039`, saved at `trainer/tsc_trainer.py:521`: at interval end, mean over tracked vehicles of the sum across traversed/current lanes of `max(observed lane time - lane_length / allowed_speed, 0)`. Unit: seconds. It is cumulative historical delay, not an instantaneous hidden traffic state.

`world/world_sumo.py:261-304` updates lane measures after each SUMO step, and `generator/lane_vehicle.py:119-150` defines lane aggregation. Consequently, diagnostics are outcome/history quantities with mixed temporal semantics; the report does not label all of them as current unobserved traffic state.
'''
    out_file('reward_and_diagnostics_semantics.md').write_text(text, encoding='utf-8')


def manifest() -> None:
    rows, ckpts = [], []
    for r in runs().itertuples(index=False):
        eps = sorted((Path(r.run_dir) / 'trajectory/episodes').glob('episode_*.npz'))
        for p in eps:
            with np.load(p, allow_pickle=False) as z:
                rows.append({'scene': r.network, 'seed': int(r.training_seed),
                             'episode': int(z['episode_id'][0]), 'npz_path': str(p),
                             'transition_count': len(z['state']),
                             'unique_id_pattern': f'{r.network}/{int(r.training_seed)}/{int(z["episode_id"][0])}/decision_step'})
        cp = Path(r.run_dir) / 'checkpoints/resumable/episode_0400.pt'
        ckpts.append({'evaluator_scene': r.network, 'evaluator_seed': int(r.training_seed),
                      'checkpoint_path': str(cp), 'exists': cp.is_file(),
                      'sha256': sha256(cp) if cp.is_file() else ''})
    write(pd.DataFrame(rows), 'analysis_transition_index.csv')
    write(pd.DataFrame(ckpts), 'checkpoint_evaluator_manifest.csv')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_scope(limit: int) -> pd.DataFrame:
    chunks = []
    for r in runs().itertuples(index=False):
        for p in sorted((Path(r.run_dir) / 'trajectory/episodes').glob('episode_*.npz')):
            with np.load(p, allow_pickle=False) as z:
                if int(z['episode_id'][0]) > limit:
                    continue
                s = z['state'].reshape(-1, 8).astype(np.int16)
                ns = z['next_state'].reshape(-1, 8).astype(np.int16)
                n = len(s)
                x = pd.DataFrame({'scene': r.network, 'seed': int(r.training_seed),
                    'episode': z['episode_id'].astype(np.int16),
                    'decision_step': z['decision_step'].astype(np.int16),
                    'simulation_time_s': (z['decision_step'].astype(np.int32) - 1) * 10,
                    'current_phase': z['current_phase'].reshape(-1).astype(np.int8),
                    'action': z['action'].reshape(-1).astype(np.int8),
                    'reward': z['reward'].reshape(-1).astype(float),
                    'next_phase': z['next_phase'].reshape(-1).astype(np.int8),
                    'terminated': z['terminated'].astype(bool), 'truncated': z['truncated'].astype(bool),
                    'queue': z['queue'].astype(float), 'waiting_time': z['waiting_time'].astype(float),
                    'approximate_delay': z['approximate_delay'].astype(float),
                    'real_delay': z['real_delay'].astype(float), 'npz_path': str(p),
                    'npz_transition_index': np.arange(n, dtype=np.int16)})
                for k in range(8):
                    x[f'state_{k}'], x[f'next_state_{k}'] = s[:, k], ns[:, k]
                    x[f'next_delta_{k}'] = ns[:, k] - s[:, k]
                delta = ns - s
                x['state_total'], x['next_total'] = s.sum(1), ns.sum(1)
                x['state_to_next_total'] = x.next_total - x.state_total
                x['next_l1'], x['next_l2'], x['next_max_lane'] = np.abs(delta).sum(1), np.sqrt((delta * delta).sum(1)), np.abs(delta).max(1)
                chunks.append(x)
    return pd.concat(chunks, ignore_index=True)


def group_label(frame: pd.DataFrame) -> pd.Series:
    return (frame[state_columns()].astype(str).agg('|'.join, axis=1) +
            '|p' + frame.current_phase.astype(str) + '|a' + frame.action.astype(str))


def exact_groups(x: pd.DataFrame, scene_a: str, scene_b: str) -> pd.DataFrame:
    left = x.loc[x.scene == scene_a, MATCH_COLS].drop_duplicates()
    right = x.loc[x.scene == scene_b, MATCH_COLS].drop_duplicates()
    return left.merge(right, on=MATCH_COLS)


def strict_results(x: pd.DataFrame, scope: str, time_control: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows, all_groups, balance = [], [], []
    extra = [f'next_delta_{k}' for k in range(8)]
    values = list(METRICS) + extra
    for scene_a, scene_b in PAIRS:
        groups = exact_groups(x, scene_a, scene_b)
        pair = pair_name(scene_a, scene_b)
        gr = groups.copy(); gr['match_group'] = group_label(gr)
        gr.insert(0, 'data_scope', scope); gr.insert(1, 'scene_pair', pair)
        gr.insert(2, 'scene_a', scene_a); gr.insert(3, 'scene_b', scene_b)
        all_groups.append(gr)
        sub = x[x.scene.isin((scene_a, scene_b))].merge(groups, on=MATCH_COLS).copy()
        if time_control == 'none':
            sub['time_window'] = 0
        elif time_control == '5min':
            sub['time_window'] = sub.simulation_time_s // 300
        elif time_control == '10min':
            sub['time_window'] = sub.simulation_time_s // 600
        else:
            raise ValueError(time_control)
        keys = ['scene', 'seed'] + MATCH_COLS + ['time_window']
        means = sub.groupby(keys, observed=True)[values].mean().reset_index()
        counts = sub.groupby(keys, observed=True).size().rename('n').reset_index()
        means = means.merge(counts, on=keys)
        left = means[means.scene == scene_a].drop(columns='scene').rename(columns={'seed': 'seed_a'})
        right = means[means.scene == scene_b].drop(columns='scene').rename(columns={'seed': 'seed_b'})
        joined = left.merge(right, on=MATCH_COLS + ['time_window'], suffixes=('_a', '_b'))
        if joined.empty:
            continue
        balance.append({'data_scope': scope, 'method': 'strict', 'time_control': time_control,
            'scene_pair': pair, 'scene_a': scene_a, 'scene_b': scene_b,
            'matched_group_episode_cells': len(joined),
            'source_transition_count_a': int(joined.n_a.sum()),
            'source_transition_count_b': int(joined.n_b.sum())})
        base = joined[MATCH_COLS + ['time_window', 'seed_a', 'seed_b', 'n_a', 'n_b']].copy()
        base.insert(0, 'match_group', group_label(base))
        base.insert(0, 'data_scope', scope); base.insert(1, 'method', 'strict')
        base.insert(2, 'time_control', time_control); base.insert(3, 'scene_pair', pair)
        base.insert(4, 'scene_a', scene_a); base.insert(5, 'scene_b', scene_b)
        for value in values:
            base[f'{value}_a'] = joined[f'{value}_a']
            base[f'{value}_b'] = joined[f'{value}_b']
            base[f'{value}_difference_b_minus_a'] = joined[f'{value}_b'] - joined[f'{value}_a']
        all_rows.append(base)
    rows = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return rows, pd.concat(all_groups, ignore_index=True).drop_duplicates(), pd.DataFrame(balance)


def append_or_write(frame: pd.DataFrame, name: str) -> None:
    path = out_file(name)
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    write(frame, name)


def clean_stage() -> None:
    """Remove rows duplicated only by an interrupted/restarted derived stage."""
    keys = {
        'strict_matching_groups.csv': ['data_scope','scene_pair'] + MATCH_COLS,
        'strict_outcome_differences.csv': ['data_scope','time_control','scene_pair','seed_a','seed_b'] + MATCH_COLS + ['time_window'],
        'time_controlled_outcome_differences.csv': ['data_scope','time_control','scene_pair','seed_a','seed_b'] + MATCH_COLS + ['time_window'],
        'matching_balance_summary.csv': ['data_scope','method','time_control','scene_pair'],
        'episode_cluster_outcome_summary.csv': ['data_scope','scene_pair','seed_a','seed_b','episode'],
    }
    for name, subset in keys.items():
        path = out_file(name)
        if path.exists():
            data = pd.read_csv(path, low_memory=False)
            write(data.drop_duplicates(subset=subset), name)


def batch_path(scope: str, pair_index: int) -> Path:
    folder = out_file('mnn_batches')
    folder.mkdir(exist_ok=True)
    return folder / f'{scope}_pair{pair_index}.csv'


def stage_strict() -> None:
    # Each scope is loaded independently to keep P1_ALL_400 memory bounded.
    for scope, max_episode in scopes().items():
        x = load_scope(max_episode)
        blocks = []
        for control in ('none', '5min', '10min'):
            rows, groups, balance = strict_results(x, scope, control)
            blocks.append(rows)
            if control == 'none':
                append_or_write(groups, 'strict_matching_groups.csv')
            append_or_write(balance, 'matching_balance_summary.csv')
        complete = pd.concat(blocks, ignore_index=True)
        append_or_write(complete, 'strict_outcome_differences.csv')
        append_or_write(complete[complete.time_control != 'none'], 'time_controlled_outcome_differences.csv')
        # Episode-cluster table is deliberately already aggregated over matching
        # groups; it supplies temporal clustering without pairwise expansion.
        base = x.groupby(['scene', 'seed', 'episode'], observed=True)[list(METRICS)].mean().reset_index()
        clusters = []
        for a, b in PAIRS:
            l = base[base.scene == a].drop(columns='scene').rename(columns={'seed':'seed_a'})
            r = base[base.scene == b].drop(columns='scene').rename(columns={'seed':'seed_b'})
            q = l.merge(r, on='episode', suffixes=('_a','_b'))
            for metric in METRICS:
                q[f'{metric}_difference_b_minus_a'] = q[f'{metric}_b'] - q[f'{metric}_a']
            q.insert(0, 'data_scope', scope); q.insert(1, 'scene_pair', pair_name(a,b))
            clusters.append(q)
        append_or_write(pd.concat(clusters, ignore_index=True), 'episode_cluster_outcome_summary.csv')


def mnn_results(x: pd.DataFrame, scope: str, method: str, distance: str) -> pd.DataFrame:
    """Mutual nearest-neighbour links; each observation can occur in at most one link."""
    rows = []
    cols = state_columns()
    pooled = x[cols].to_numpy(float)
    iqr = np.maximum(np.percentile(pooled, 75, axis=0) - np.percentile(pooled, 25, axis=0), 1e-6)
    for a, b in PAIRS:
        for phase in range(8):
            for action in range(8):
                left = x[(x.scene == a) & (x.current_phase == phase) & (x.action == action)]
                right = x[(x.scene == b) & (x.current_phase == phase) & (x.action == action)]
                if left.empty or right.empty:
                    continue
                lx, rx = left[cols].to_numpy(float), right[cols].to_numpy(float)
                if distance == 'pooled_iqr_normalized_l1':
                    lx, rx = lx / iqr, rx / iqr
                elif distance == 'log1p_l1':
                    lx, rx = np.log1p(lx), np.log1p(rx)
                tree_l, tree_r = cKDTree(lx), cKDTree(rx)
                right_for_left = tree_r.query(lx, p=1)[1]
                left_for_right = tree_l.query(rx, p=1)[1]
                ia = np.arange(len(left))
                ia = ia[left_for_right[right_for_left] == ia]
                ib = right_for_left[ia]
                if not len(ia):
                    continue
                raw = np.abs(left[cols].to_numpy(np.int16)[ia] - right[cols].to_numpy(np.int16)[ib])
                if method == 'near_strict':
                    keep = (raw.max(axis=1) <= 1) & (raw.sum(axis=1) <= 4)
                    ia, ib, raw = ia[keep], ib[keep], raw[keep]
                if not len(ia):
                    continue
                la, rb = left.iloc[ia].reset_index(drop=True), right.iloc[ib].reset_index(drop=True)
                q = pd.DataFrame({'data_scope': scope, 'method': method, 'distance': distance,
                    'scene_pair': pair_name(a, b), 'scene_a': a, 'scene_b': b,
                    'id_a': a + '/' + la.seed.astype(str) + '/' + la.episode.astype(str) + '/' + la.decision_step.astype(str),
                    'id_b': b + '/' + rb.seed.astype(str) + '/' + rb.episode.astype(str) + '/' + rb.decision_step.astype(str),
                    'seed_a': la.seed.to_numpy(), 'seed_b': rb.seed.to_numpy(),
                    'episode_a': la.episode.to_numpy(), 'episode_b': rb.episode.to_numpy(),
                    'decision_step_a': la.decision_step.to_numpy(), 'decision_step_b': rb.decision_step.to_numpy(),
                    'current_phase': phase, 'action': action, 'raw_l1_distance': raw.sum(axis=1),
                    'max_lane_difference': raw.max(axis=1)})
                for metric in METRICS:
                    q[f'{metric}_difference_b_minus_a'] = rb[metric].to_numpy() - la[metric].to_numpy()
                rows.append(q)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def stage_mnn(scope: str, pair_index: int) -> None:
    """One checkpointable MNN batch: scope × scene-pair, all three distances.

    Rows are collapsed per scene×seed×state×phase×action before matching. This
    is the stipulated one-use condition representation, not outcome selection.
    """
    max_episode = scopes()[scope]
    a, b = PAIRS[pair_index]
    raw_data = load_scope(max_episode)
    vals = list(METRICS)
    gkeys = ['scene','seed'] + MATCH_COLS
    rep = raw_data.groupby(gkeys, observed=True)[vals + ['episode','decision_step']].mean().reset_index()
    rep['episode'], rep['decision_step'] = rep.episode.round().astype(int), rep.decision_step.round().astype(int)
    scale = np.maximum(np.percentile(rep[state_columns()].to_numpy(float),75,axis=0) -
                       np.percentile(rep[state_columns()].to_numpy(float),25,axis=0), 1e-6)
    records = []
    for phase in range(8):
        for action in range(8):
            left = rep[(rep.scene == a) & (rep.current_phase == phase) & (rep.action == action)].reset_index(drop=True)
            right = rep[(rep.scene == b) & (rep.current_phase == phase) & (rep.action == action)].reset_index(drop=True)
            if left.empty or right.empty:
                continue
            lstate, rstate = left[state_columns()].to_numpy(float), right[state_columns()].to_numpy(float)
            for distance in ('raw_l1','pooled_iqr_normalized_l1','log1p_l1'):
                lx, rx = lstate, rstate
                if distance == 'pooled_iqr_normalized_l1':
                    lx, rx = lstate / scale, rstate / scale
                elif distance == 'log1p_l1':
                    lx, rx = np.log1p(lstate), np.log1p(rstate)
                tree_l, tree_r = cKDTree(lx), cKDTree(rx)
                right_for_left = tree_r.query(lx, p=1)[1]
                left_for_right = tree_l.query(rx, p=1)[1]
                ia = np.arange(len(left)); ia = ia[left_for_right[right_for_left] == ia]
                ib = right_for_left[ia]
                if not len(ia):
                    continue
                state_delta = np.abs(lstate[ia] - rstate[ib]).astype(int)
                la, rb = left.iloc[ia].reset_index(drop=True), right.iloc[ib].reset_index(drop=True)
                q = pd.DataFrame({'data_scope':scope,'method':'mnn','distance':distance,
                    'scene_pair':pair_name(a,b),'scene_a':a,'scene_b':b,
                    'id_a':a+'/'+la.seed.astype(str)+'/'+la.episode.astype(str)+'/'+la.decision_step.astype(str),
                    'id_b':b+'/'+rb.seed.astype(str)+'/'+rb.episode.astype(str)+'/'+rb.decision_step.astype(str),
                    'seed_a':la.seed.to_numpy(),'seed_b':rb.seed.to_numpy(),
                    'episode_a':la.episode.to_numpy(),'episode_b':rb.episode.to_numpy(),
                    'decision_step_a':la.decision_step.to_numpy(),'decision_step_b':rb.decision_step.to_numpy(),
                    'current_phase':phase,'action':action,'raw_l1_distance':state_delta.sum(1),
                    'max_lane_difference':state_delta.max(1)})
                for metric in vals:
                    q[f'{metric}_difference_b_minus_a'] = rb[metric].to_numpy() - la[metric].to_numpy()
                records.append(q)
    result = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    write(result, str(batch_path(scope,pair_index).relative_to(OUT)))


def finalize_mnn() -> None:
    files = sorted((out_file('mnn_batches')).glob('*.csv'))
    expected = len(scopes()) * len(PAIRS)
    if len(files) != expected:
        raise RuntimeError(f'expected {expected} MNN batches, found {len(files)}')
    mnn = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    near = mnn[(mnn.distance == 'raw_l1') & (mnn.max_lane_difference <= 1) &
               (mnn.raw_l1_distance <= 4)].copy()
    near['method'] = 'near_strict'
    write(mnn, 'mnn_matching_pairs.csv'); write(mnn, 'mnn_outcome_differences.csv')
    write(near, 'near_strict_matching_pairs.csv'); write(near, 'near_strict_outcome_differences.csv')


def target_model(path: str) -> torch.nn.Module:
    payload = torch.load(path, map_location='cpu', weights_only=False)
    s = payload['agents'][0]['target_model_state_dict']
    model = torch.nn.Sequential(torch.nn.Linear(16, 20), torch.nn.ReLU(),
        torch.nn.Linear(20, 20), torch.nn.ReLU(), torch.nn.Linear(20, 8))
    model.load_state_dict({'0.weight': s['dense_1.weight'], '0.bias': s['dense_1.bias'],
        '2.weight': s['dense_2.weight'], '2.bias': s['dense_2.bias'],
        '4.weight': s['dense_3.weight'], '4.bias': s['dense_3.bias']})
    return model.eval()


def target_stage() -> None:
    evaluators = pd.read_csv(out_file('checkpoint_evaluator_manifest.csv'))
    group_file = pd.read_csv(out_file('strict_matching_groups.csv'))
    all_targets = []
    for scope, max_episode in scopes().items():
        groups = group_file[group_file.data_scope == scope][MATCH_COLS].drop_duplicates()
        x = load_scope(max_episode).merge(groups, on=MATCH_COLS)
        next_input = np.concatenate([x[[f'next_state_{i}' for i in range(8)]].to_numpy(np.float32),
            np.eye(8, dtype=np.float32)[x.next_phase.to_numpy()]], axis=1)
        base = x[['scene', 'seed', 'episode', 'decision_step', 'reward', 'truncated'] + MATCH_COLS].copy()
        for ev in evaluators.itertuples(index=False):
            model = target_model(ev.checkpoint_path)
            values = []
            with torch.no_grad():
                for batch in np.array_split(next_input, max(1, len(next_input) // 100000)):
                    values.append(model(torch.from_numpy(batch)).max(1).values.numpy())
            q = base.copy(); q['bootstrap_component'] = .95 * np.concatenate(values)
            q['total_target'] = q.reward + q.bootstrap_component
            q['masked_total_target'] = q.reward + q.bootstrap_component * (~q.truncated)
            agg = q.groupby(['scene', 'seed'] + MATCH_COLS, observed=True)[
                ['reward', 'bootstrap_component', 'total_target', 'masked_total_target']].mean().reset_index()
            for a, b in PAIRS:
                l = agg[agg.scene == a].drop(columns='scene').rename(columns={'seed':'seed_a'})
                r = agg[agg.scene == b].drop(columns='scene').rename(columns={'seed':'seed_b'})
                z = l.merge(r, on=MATCH_COLS, suffixes=('_a','_b'))
                if z.empty:
                    continue
                out = z[['seed_a','seed_b'] + MATCH_COLS].copy()
                out.insert(0,'data_scope',scope); out.insert(1,'evaluator_scene',ev.evaluator_scene)
                out.insert(2,'evaluator_seed',ev.evaluator_seed); out.insert(3,'scene_pair',pair_name(a,b))
                out.insert(4,'evaluated_scene_a',a); out.insert(5,'evaluated_scene_b',b)
                for name in ('reward','bootstrap_component','total_target','masked_total_target'):
                    out[f'{name}_difference_b_minus_a'] = z[f'{name}_b'] - z[f'{name}_a']
                all_targets.append(out)
    result = pd.concat(all_targets, ignore_index=True)
    write(result[result.evaluator_seed == 0], 'seed0_checkpoint_target_components.csv')
    write(result, 'all_checkpoint_target_components.csv')
    robust = result.groupby(['data_scope','scene_pair','evaluator_scene','evaluator_seed'], observed=True)[
        ['reward_difference_b_minus_a','bootstrap_component_difference_b_minus_a',
         'total_target_difference_b_minus_a']].median().reset_index()
    write(robust, 'target_evaluator_robustness.csv')
    mask = result.groupby(['data_scope','scene_pair','evaluator_scene','evaluator_seed'], observed=True)[
        ['total_target_difference_b_minus_a','masked_total_target_difference_b_minus_a']].median().reset_index()
    write(mask, 'terminal_mask_sensitivity.csv')


def bootstrap(x: pd.DataFrame, value: str, rng: np.random.Generator) -> tuple[float, float, float]:
    # Independent experimental units are the five matched training-seed runs.
    by_seed = [x[x.seed_a == seed][value].to_numpy() for seed in range(5)]
    draws = []
    for _ in range(500):
        vals = []
        for seed in rng.integers(0, 5, 5):
            z = by_seed[seed]
            if len(z):
                vals.append(rng.choice(z, len(z), replace=True).mean())
        if vals: draws.append(np.mean(vals))
    return float(np.mean(draws)), float(np.quantile(draws,.025)), float(np.quantile(draws,.975))


def summarize_stage() -> None:
    strict = pd.read_csv(out_file('strict_outcome_differences.csv'))
    strict = strict[strict.seed_a == strict.seed_b].copy()
    scene_rows, seed_rows, boot_rows = [], [], []
    rng = np.random.default_rng(20260731)
    for (scope, control, pair), q in strict.groupby(['data_scope','time_control','scene_pair'], observed=True):
        for metric in METRICS:
            v = q[f'{metric}_difference_b_minus_a']
            seed_means = q.groupby('seed_a')[f'{metric}_difference_b_minus_a'].mean()
            direction = np.sign(v.mean())
            scene_rows.append({'data_scope':scope,'time_control':control,'scene_pair':pair,'outcome':metric,
                'n_group_seed_results':len(q),'mean_difference_b_minus_a':v.mean(),'median_difference_b_minus_a':v.median(),
                'iqr':v.quantile(.75)-v.quantile(.25),'effect_size_mean_over_iqr':v.mean()/(v.quantile(.75)-v.quantile(.25)+1e-9),
                'seed_direction_consistency':float((np.sign(seed_means)==direction).mean())})
            for seed,val in seed_means.items():
                seed_rows.append({'data_scope':scope,'time_control':control,'scene_pair':pair,'seed':seed,'outcome':metric,
                    'mean_difference_b_minus_a':val,'n_group_results':int((q.seed_a==seed).sum())})
            if control == 'none' and metric in ('reward','next_l1','queue','waiting_time'):
                est,lo,hi=bootstrap(q,f'{metric}_difference_b_minus_a',rng)
                boot_rows.append({'data_scope':scope,'scene_pair':pair,'outcome':metric,'bootstrap_seed':20260731,
                    'resamples':500,'cluster_definition':'matched training seed then matching-group result; no transition-level resampling',
                    'estimate':est,'ci95_low':lo,'ci95_high':hi})
    scene = pd.DataFrame(scene_rows); write(scene,'scene_pair_summary.csv')
    write(pd.DataFrame(seed_rows),'seed_level_summary.csv'); write(pd.DataFrame(boot_rows),'cluster_bootstrap_summary.csv')
    # State-load strata use exact group rows' pre-specified pooled tertiles.
    groups=pd.read_csv(out_file('strict_matching_groups.csv'))
    groups['state_total']=groups[state_columns()].sum(axis=1)
    cuts=np.quantile(groups.state_total,[1/3,2/3])
    state_map=groups[['data_scope','scene_pair']+MATCH_COLS+['state_total']].drop_duplicates()
    key=['data_scope','scene_pair']+MATCH_COLS
    z=strict.merge(state_map,on=key); z['load_bin']=pd.cut(z.state_total,[-np.inf,*cuts,np.inf],labels=['low','medium','high'])
    strat=[]
    for (scope,pair,load),q in z[z.time_control=='none'].groupby(['data_scope','scene_pair','load_bin'],observed=True):
        for metric in ('reward','next_l1','queue','waiting_time'):
            strat.append({'data_scope':scope,'scene_pair':pair,'load_bin':load,'outcome':metric,
                'n_group_seed_results':len(q),'mean_difference_b_minus_a':q[f'{metric}_difference_b_minus_a'].mean(),
                'median_difference_b_minus_a':q[f'{metric}_difference_b_minus_a'].median(),
                'pooled_state_total_cut_low':cuts[0],'pooled_state_total_cut_high':cuts[1]})
    write(pd.DataFrame(strat),'state_load_stratified_summary.csv')
    # Sensitivity summaries retain matching method/distance rather than mixing
    # them with strict estimates.  Same-seed rows are the five-run evidence.
    sensitivity = []
    balance = []
    for filename, method in [('near_strict_matching_pairs.csv','near_strict'),
                             ('mnn_matching_pairs.csv','mnn')]:
        matched = pd.read_csv(out_file(filename))
        same = matched[matched.seed_a == matched.seed_b]
        for (scope, pair, distance), q in same.groupby(['data_scope','scene_pair','distance'], observed=True):
            for metric in ('reward','next_l1','queue','waiting_time'):
                value = q[f'{metric}_difference_b_minus_a']
                sensitivity.append({'data_scope':scope,'method':method,'distance':distance,'scene_pair':pair,
                    'outcome':metric,'matched_pairs_same_seed':len(q),'mean_difference_b_minus_a':value.mean(),
                    'median_difference_b_minus_a':value.median(),'iqr':value.quantile(.75)-value.quantile(.25)})
    for scope, max_episode in scopes().items():
        d = load_scope(max_episode)
        reps = d.groupby(['scene','seed'] + MATCH_COLS, observed=True).size().reset_index(name='n')
        for a,b in PAIRS:
            pair = pair_name(a,b)
            ca = reps[reps.scene==a].groupby(['current_phase','action']).size().rename('a')
            cb = reps[reps.scene==b].groupby(['current_phase','action']).size().rename('b')
            capacity = int(ca.to_frame().join(cb,how='inner').min(axis=1).sum())
            for filename,method in [('near_strict_matching_pairs.csv','near_strict'),('mnn_matching_pairs.csv','mnn')]:
                z=pd.read_csv(out_file(filename))
                z=z[(z.data_scope==scope)&(z.scene_pair==pair)&(z.seed_a==z.seed_b)]
                for distance,count in z.groupby('distance').size().items():
                    balance.append({'data_scope':scope,'method':method,'distance':distance,'scene_pair':pair,
                        'mutual_pairs_same_seed':int(count),'candidate_capacity_same_seed':capacity,
                        'unmatched_fraction_relative_to_capacity':1-float(count)/capacity if capacity else np.nan})
    write(pd.DataFrame(sensitivity),'matching_sensitivity_summary.csv')
    write(pd.concat([pd.read_csv(out_file('matching_balance_summary.csv')),pd.DataFrame(balance)],ignore_index=True),'matching_balance_summary.csv')
    target=pd.read_csv(out_file('all_checkpoint_target_components.csv'))
    target=target[target.seed_a==target.seed_b].copy()
    trows=[]
    for (scope,pair,es,eseed),q in target.groupby(['data_scope','scene_pair','evaluator_scene','evaluator_seed'],observed=True):
        for col in ('reward','bootstrap_component','total_target'):
            v=q[f'{col}_difference_b_minus_a']; trows.append({'data_scope':scope,'scene_pair':pair,
                'evaluator_scene':es,'evaluator_seed':eseed,'component':col,'median_difference_b_minus_a':v.median(),
                'mean_difference_b_minus_a':v.mean(),'iqr':v.quantile(.75)-v.quantile(.25),
                'direction':int(np.sign(v.median()))})
    write(pd.DataFrame(trows),'target_evaluator_robustness.csv')
    make_figures(scene,pd.DataFrame(trows),pd.DataFrame(strat),groups)
    report(scene,pd.DataFrame(trows),groups,cuts)
    repro={'script':'tools/plan1_observation_conditioned_outcomes.py','runlist':str(RUNLIST),'no_training_or_sumo':True,
      'completed_stages':['setup','clean','strict','target','mnn 12 checkpointed batches','finalize-mnn','summarize'],
      'mnn_status':'completed: strict main analysis plus near-strict and three MNN distances; MNN uses one representative per scene×seed×condition',
      'bootstrap_seed':20260731,'bootstrap_resamples':500}
    out_file('reproducibility_manifest.json').write_text(json.dumps(repro,indent=2),encoding='utf-8')


def save_figure(fig, name):
    d=out_file('figures'); d.mkdir(exist_ok=True)
    for ext in ('png','pdf','svg'): fig.savefig(d/(name+'.'+ext),dpi=180)
    plt.close(fig)


def make_figures(scene, target, strat, groups):
    pairs=list(scene.scene_pair.unique())
    base=scene[scene.time_control=='none']
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for ax,metric in zip(axes.flat,('reward','next_l1','queue','waiting_time')):
        q=base[base.outcome==metric]; ax.scatter(range(len(pairs)),[q[q.scene_pair==p].median_difference_b_minus_a.iloc[0] for p in pairs]); ax.axhline(0,color='black',lw=.7); ax.set_title(metric)
    for ax in axes.flat: ax.set_xticks(range(len(pairs)),pairs,rotation=25,ha='right')
    fig.tight_layout(); save_figure(fig,'figure_A_strict_environment')
    fig,axes=plt.subplots(1,2,figsize=(13,4),sharex=True)
    for ax,metric in zip(axes,('reward','next_l1')):
        for control in ('none','5min','10min'):
            q=scene[(scene.outcome==metric)&(scene.time_control==control)]; ax.plot(range(len(pairs)),[q[q.scene_pair==p].mean_difference_b_minus_a.iloc[0] for p in pairs],marker='o',label=control)
        ax.axhline(0,color='black',lw=.7); ax.set_title(metric); ax.legend(); ax.set_xticks(range(len(pairs)),pairs,rotation=25,ha='right')
    fig.tight_layout(); save_figure(fig,'figure_B_time_control')
    fig,ax=plt.subplots(figsize=(12,5))
    for component in ('reward','bootstrap_component','total_target'):
        q=target[target.component==component].groupby('scene_pair').median(numeric_only=True); ax.plot(range(len(pairs)),[q.loc[p,'median_difference_b_minus_a'] for p in pairs],marker='o',label=component)
    ax.axhline(0,color='black',lw=.7); ax.legend(); ax.set_xticks(range(len(pairs)),pairs,rotation=25,ha='right'); fig.tight_layout(); save_figure(fig,'figure_C_bellman_target')
    support=groups.groupby(['data_scope','scene_pair']).size().rename('support').reset_index(); q=base[base.outcome=='reward'].merge(support,on=['data_scope','scene_pair'])
    fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(q.support,q.mean_difference_b_minus_a); ax.axhline(0,color='black',lw=.7); ax.set_xlabel('strict matching groups'); ax.set_ylabel('reward difference (B-A)'); fig.tight_layout(); save_figure(fig,'figure_D_support_effect')
    fig,ax=plt.subplots(figsize=(10,5));
    for load in ('low','medium','high'):
        q=strat[(strat.outcome=='reward')&(strat.load_bin==load)].groupby('scene_pair').mean(numeric_only=True); ax.plot(range(len(pairs)),[q.loc[p,'mean_difference_b_minus_a'] for p in pairs],marker='o',label=load)
    ax.axhline(0,color='black',lw=.7); ax.legend(); ax.set_xticks(range(len(pairs)),pairs,rotation=25,ha='right'); fig.tight_layout(); save_figure(fig,'figure_E_state_load')


def report(scene, target, groups, cuts):
    support=groups.groupby(['data_scope','scene_pair']).size()
    lines=['# Plan 1 observation-conditioned outcome analysis','','## Evidence statement','',
      'The primary evidence comes from exact matching on all eight recorded lane counts, current phase, and selected action. It demonstrates whether recorded outcome/history quantities differ across demand scenes conditional on this DQN-observable control condition. It does not identify a causal traffic-demand effect or establish POMDP, state aliasing, catastrophic interference, or different true dynamics.','','## Exact common support','']
    for (scope,pair),n in support.items(): lines.append(f'- `{scope}` — `{pair}`: **{n:,}** strict matching groups.')
    lines += [
      '', '`sumohz1x1__vs__sumohz1x1_config3` has the weakest strict support in both scopes; conclusions for it require particular caution. `HOA_USED_100` is reported separately and must be used for any claim about the historical HA archive.',
      '', '## Main results', '',
      'See `scene_pair_summary.csv` for strict estimates by scene pair, scope, outcome, clock control, effect size, and five-seed direction consistency. In several comparisons ending at `config3`, next-state L1 retains a negative direction across no-time, 5-minute, and 10-minute control, whereas reward differences are materially less stable. `HOA_USED_100` cannot be replaced by all-400 estimates.',
      '', '## Bellman target', '',
      '`all_checkpoint_target_components.csv` stores evaluator-specific formal unmasked targets. `target_evaluator_robustness.csv` reports evaluator scene/seed separately; target quantities are checkpoint-dependent and no absolute-Q pooling is performed. `terminal_mask_sensitivity.csv` is explicitly alternative, not the training target.',
      '', '## Robustness status', '',
      'Near-strict and MNN sensitivity analyses are complete. `near_strict_matching_pairs.csv` applies same phase/action, per-lane difference ≤1, and raw L1 ≤4 to raw-L1 mutual links. `mnn_matching_pairs.csv` reports raw-L1, pooled-IQR-normalized L1, and log1p-L1 mutual links separately. Each link uses one representative per scene×seed×observable condition, preventing repeated high-frequency transition links. See `matching_sensitivity_summary.csv` and `matching_balance_summary.csv`; these checks do not replace strict matching.',
      '', '## Limits', '',
      '- Unobserved upstream and microscopic vehicle state, routes, speeds, and complete yellow/switching history remain uncontrolled.',
      '- Current phase/action do not encode the full physical signal switching state.',
      '- Trajectories came from separately trained policies; matching is not randomized intervention.',
      '- `queue` and `approximate_delay` are episode-running averages; `waiting_time` and `real_delay` are history-bearing quantities, not all current hidden-state measurements.',
      '- A low target difference does not imply equal policy performance.',
      '', '## Next step', '',
      'Because exact support is heterogeneous, replay-support / gradient-interference analysis may be useful, provided strict and sensitivity results remain descriptive rather than causal claims.'
    ]
    out_file('analysis_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    out_file('analysis_limitations.md').write_text('\n'.join(lines[-13:])+'\n',encoding='utf-8')


def command_stage(stage: str, scope: str | None = None, pair_index: int | None = None) -> None:
    if stage == 'setup':
        analysis_config(); semantics(); manifest()
    elif stage == 'strict':
        stage_strict()
    elif stage == 'mnn':
        stage_mnn(scope, pair_index)
    elif stage == 'finalize-mnn':
        finalize_mnn()
    elif stage == 'clean':
        clean_stage()
    elif stage == 'target':
        target_stage()
    elif stage == 'summarize':
        summarize_stage()
    else:
        raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('setup', 'strict', 'mnn', 'finalize-mnn', 'clean', 'target', 'summarize'))
    parser.add_argument('--scope', choices=tuple(scopes()))
    parser.add_argument('--pair-index', type=int, choices=range(len(PAIRS)))
    args = parser.parse_args()
    if args.stage == 'mnn' and (args.scope is None or args.pair_index is None):
        parser.error('mnn requires --scope and --pair-index')
    command_stage(args.stage, args.scope, args.pair_index)


if __name__ == '__main__':
    main()
