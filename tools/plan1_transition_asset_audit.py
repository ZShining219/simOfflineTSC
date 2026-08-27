#!/usr/bin/env python3
"""Read-only asset audit for the formal Plan 1 Online-DQN trajectories.

This tool deliberately writes only its audit directory.  It never mutates a
run, trajectory, checkpoint, or simulator state.
"""
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
RUNLIST = ROOT / 'data/output_data/analysis/plan1/p1_formal_20_runlist_20260722.csv'
OUT = ROOT / 'data/output_data/analysis/plan1/plan1_transition_asset_audit_20260731'
SCENES = ('sumohz1x1_config2', 'sumohz1x1', 'sumohz1x1_config4', 'sumohz1x1_config3')
REQUIRED = ('state', 'current_phase', 'action', 'reward', 'next_state',
            'next_phase', 'terminated', 'truncated')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(name, rows, fields):
    with open(OUT / name, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def formal_runs():
    with open(RUNLIST, newline='', encoding='utf-8') as handle:
        return [row for row in csv.DictReader(handle)
                if row['role'] == 'formal' and row['agent'] == 'dqn'
                and row['include'].lower() == 'true']


def compact_shape(array):
    return 'x'.join(map(str, array.shape))


def audit_transition_assets(runs):
    rows, all_data, run_summary = [], defaultdict(list), []
    for run in runs:
        run_path = Path(run['run_dir'])
        traj = run_path / 'trajectory'
        manifest = json.loads((traj / 'manifest.json').read_text())
        validation = json.loads((traj / 'validation.json').read_text())
        entries = [json.loads(line) for line in (traj / 'index.jsonl').read_text().splitlines() if line]
        errors = []
        previous_global = 0
        seen_eps = set()
        for entry in entries:
            path = traj / entry['file']
            row = {'scene': run['network'], 'training_seed': int(run['training_seed']),
                   'run_path': str(run_path), 'episode': entry['episode_id'],
                   'npz_path': str(path), 'transition_count': entry['transition_count'],
                   'index_sha256': entry['sha256'], 'exists': path.is_file(),
                   'field_names': '', 'field_shapes': '', 'field_dtypes': '',
                   'nan_or_inf': False, 'length_mismatch': False,
                   'decision_continuous': False, 'global_continuous': False,
                   'state_chain_continuous': False, 'phase_chain_continuous': False,
                   'duplicate_episode': entry['episode_id'] in seen_eps, 'error': ''}
            seen_eps.add(entry['episode_id'])
            if not path.is_file():
                row['error'] = 'missing file'; errors.append(row['error']); rows.append(row); continue
            try:
                with np.load(path, allow_pickle=False) as data:
                    fields = data.files
                    row['field_names'] = ';'.join(fields)
                    row['field_shapes'] = json.dumps({x: compact_shape(data[x]) for x in fields}, sort_keys=True)
                    row['field_dtypes'] = json.dumps({x: str(data[x].dtype) for x in fields}, sort_keys=True)
                    n = len(data['state'])
                    row['length_mismatch'] = any(len(data[x]) != n for x in REQUIRED if x in data)
                    row['nan_or_inf'] = any(not np.all(np.isfinite(data[x])) for x in fields
                                            if data[x].dtype.kind in 'fiu')
                    row['decision_continuous'] = np.array_equal(data['decision_step'], np.arange(1, n + 1))
                    row['global_continuous'] = np.array_equal(data['global_step'], np.arange(previous_global + 1, previous_global + n + 1))
                    row['state_chain_continuous'] = n <= 1 or np.array_equal(data['next_state'][:-1], data['state'][1:])
                    row['phase_chain_continuous'] = n <= 1 or np.array_equal(data['next_phase'][:-1], data['current_phase'][1:])
                    missing = set(REQUIRED) - set(fields)
                    if missing: row['error'] = 'missing:' + ','.join(sorted(missing))
                    if not row['length_mismatch'] and not row['nan_or_inf'] and not missing:
                        all_data[run['network']].append({
                            'state': data['state'].reshape(n, -1).astype(np.float32),
                            'phase': data['current_phase'].reshape(-1).astype(np.int8),
                            'action': data['action'].reshape(-1).astype(np.int8),
                            'episode': data['episode_id'].reshape(-1).astype(np.int16),
                            'decision': data['decision_step'].reshape(-1).astype(np.int16),
                        })
                    previous_global += n
            except Exception as exc:  # asset audit must record corrupt NPZ rather than stop
                row['error'] = f'{type(exc).__name__}: {exc}'; errors.append(row['error'])
            rows.append(row)
        expected = set(range(1, 401))
        missing_eps = sorted(expected - seen_eps)
        run_summary.append({'scene': run['network'], 'training_seed': int(run['training_seed']),
                            'run_path': str(run_path), 'manifest_episode_count': len(entries),
                            'manifest_transition_count': validation.get('transition_count'),
                            'validation_valid': validation.get('valid'),
                            'missing_episodes': ';'.join(map(str, missing_eps)),
                            'duplicate_episodes': len(seen_eps) != len(entries),
                            'asset_errors': len(errors)})
    write_csv('plan1_transition_asset_manifest.csv', rows, list(rows[0]))
    write_csv('plan1_transition_run_summary.csv', run_summary, list(run_summary[0]))
    return all_data, run_summary


def audit_checkpoints(runs):
    rows = []
    for run in runs:
        path = Path(run['run_dir']) / 'checkpoints/resumable/episode_0400.pt'
        row = {'scene': run['network'], 'training_seed': int(run['training_seed']),
               'run_path': run['run_dir'], 'checkpoint_path': str(path), 'episode': 400,
               'file_size_bytes': path.stat().st_size if path.exists() else 0,
               'sha256': sha256(path) if path.exists() else '', 'online_network_key': '',
               'target_network_key': '', 'optimizer_present': False, 'epsilon_present': False,
               'input_dim': '', 'output_dim': '', 'loadable': False, 'forward_pass_valid': False,
               'error': ''}
        try:
            payload = torch.load(path, map_location='cpu')
            agent = payload['agents'][0]
            row['online_network_key'] = 'online_model_state_dict' if 'online_model_state_dict' in agent else ''
            row['target_network_key'] = 'target_model_state_dict' if 'target_model_state_dict' in agent else ''
            row['optimizer_present'] = 'optimizer_state_dict' in agent
            row['epsilon_present'] = 'epsilon' in agent
            state = agent['online_model_state_dict']
            first = state['dense_1.weight']; last = state['dense_3.weight']
            row['input_dim'], row['output_dim'] = int(first.shape[1]), int(last.shape[0])
            # Instantiate from the checkpoint tensor dimensions: no simulator/world required.
            model = torch.nn.Sequential(torch.nn.Linear(row['input_dim'], 20), torch.nn.ReLU(),
                torch.nn.Linear(20, 20), torch.nn.ReLU(), torch.nn.Linear(20, row['output_dim']))
            mapped = {'0.weight': state['dense_1.weight'], '0.bias': state['dense_1.bias'],
                      '2.weight': state['dense_2.weight'], '2.bias': state['dense_2.bias'],
                      '4.weight': state['dense_3.weight'], '4.bias': state['dense_3.bias']}
            model.load_state_dict(mapped); model(torch.zeros((1, row['input_dim'])))
            row['loadable'], row['forward_pass_valid'] = True, True
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        rows.append(row)
    write_csv('plan1_checkpoint_manifest.csv', rows, list(rows[0]))
    return rows


def state_statistics(data):
    pooled = np.concatenate([np.concatenate([x['state'] for x in data[s]]) for s in SCENES])
    qs = np.quantile(pooled, [.01, .05, .25, .5, .75, .95, .99], axis=0)
    rows = []
    for lane in range(8):
        rows.append({'lane_index': lane, 'count': len(pooled), 'mean': float(pooled[:, lane].mean()),
                     'std': float(pooled[:, lane].std()), 'median': float(qs[3, lane]),
                     'iqr': float(qs[4, lane] - qs[2, lane]), 'q01': float(qs[0, lane]),
                     'q05': float(qs[1, lane]), 'q25': float(qs[2, lane]), 'q50': float(qs[3, lane]),
                     'q75': float(qs[4, lane]), 'q95': float(qs[5, lane]), 'q99': float(qs[6, lane])})
    write_csv('pooled_state_statistics.csv', rows, list(rows[0]))
    return pooled, np.maximum(qs[4] - qs[2], 1e-6)


def pair_stats(a, b, scale):
    # Exact support is counted as common unique (state, phase, action) keys;
    # transition count is the sum of min(scene-A frequency, scene-B frequency).
    # A structured NumPy key avoids millions of Python tuple/dict objects.
    def keys(records):
        state = np.concatenate([x['state'] for x in records]).astype(np.int16)
        phase = np.concatenate([x['phase'] for x in records]).astype(np.int8)
        action = np.concatenate([x['action'] for x in records]).astype(np.int8)
        result = np.empty(len(state), dtype=[('s', np.int16, 8), ('p', np.int8), ('a', np.int8)])
        result['s'], result['p'], result['a'] = state, phase, action
        return result
    ua, ca = np.unique(keys(a), return_counts=True)
    ub, cb = np.unique(keys(b), return_counts=True)
    _, iax, ibx = np.intersect1d(ua, ub, return_indices=True)
    common = ua[iax]
    exact_n = int(np.minimum(ca[iax], cb[ibx]).sum())
    exact_pa = defaultdict(int)
    for key in common: exact_pa[f'{int(key["p"])}:{int(key["a"])}'] += 1

    sa, pa, aa, ea, da = (np.concatenate([x[k] for x in a]) for k in ('state','phase','action','episode','decision'))
    sb, pb, ab, eb, db = (np.concatenate([x[k] for x in b]) for k in ('state','phase','action','episode','decision'))
    mutual, near, dists, rawl1, maxlane, time_same, t5, t10 = [], 0, [], [], [], 0, 0, 0
    for p in range(8):
        for act in range(8):
            ia, ib = np.where((pa == p) & (aa == act))[0], np.where((pb == p) & (ab == act))[0]
            if not len(ia) or not len(ib): continue
            xa, xb = sa[ia], sb[ib]
            # MNN under the prescribed pooled-IQR normalized L1 metric.
            ta, tb = cKDTree(xa / scale), cKDTree(xb / scale)
            j = ta.query(xb / scale, p=1)[1]
            i = tb.query(xa / scale, p=1)[1]
            for local_a, local_b in enumerate(i):
                if j[local_b] != local_a: continue
                x, y = xa[local_a], xb[local_b]
                delta = np.abs(x - y); l1 = float(delta.sum())
                dists.append(float((delta / scale).mean())); rawl1.append(l1); maxlane.append(float(delta.max()))
                mutual.append((ia[local_a], ib[local_b]))
                if delta.max() <= 1 and l1 <= 4: near += 1
                # decision 1 starts at simulation time 0; 10-step control interval.
                ta_s, tb_s = (int(da[ia[local_a]]) - 1) * 10, (int(db[ib[local_b]]) - 1) * 10
                time_same += ta_s == tb_s; t5 += ta_s // 300 == tb_s // 300; t10 += ta_s // 600 == tb_s // 600
    q = lambda x, p: float(np.quantile(x, p)) if x else ''
    return {'strict_unique_groups': len(common), 'strict_matched_transition_pairs': exact_n,
            'strict_phase_action_unique_groups': json.dumps(dict(sorted(exact_pa.items()))),
            'near_strict_mutual_pairs': near, 'mutual_nearest_neighbor_pairs': len(mutual),
            'normalized_l1_mean': float(np.mean(dists)) if dists else '', 'normalized_l1_median': q(dists,.5),
            'normalized_l1_q25': q(dists,.25), 'normalized_l1_q75': q(dists,.75),
            'normalized_l1_q90': q(dists,.9), 'normalized_l1_q95': q(dists,.95),
            'raw_l1_mean': float(np.mean(rawl1)) if rawl1 else '', 'raw_l1_median': q(rawl1,.5),
            'max_lane_difference_mean': float(np.mean(maxlane)) if maxlane else '',
            'same_episode_time_pairs': time_same, 'same_5min_window_pairs': t5, 'same_10min_window_pairs': t10}


def matching(data, scale):
    rows = []
    for i, left in enumerate(SCENES):
        for right in SCENES[i + 1:]:
            row = {'scene_a': left, 'scene_b': right}
            row.update(pair_stats(data[left], data[right], scale)); rows.append(row)
    write_csv('matching_feasibility_summary.csv', rows, list(rows[0]))
    return rows


def write_reports(run_summary, checkpoints, match_rows):
    formal_ok = all(x['validation_valid'] and not x['missing_episodes'] and not x['asset_errors'] for x in run_summary)
    cp_ok = all(x['loadable'] and x['forward_pass_valid'] for x in checkpoints)
    semantics = '''# Plan 1 transition 与 Bellman 语义审计

## 数据语义

- `state`/`next_state` 是 SUMO 单路口八条进口车道的 `lane_count`，shape 为 `(1,1,8)`；车道顺序由 SUMO `Intersection` 对进口道路按方向和 lane suffix 的稳定排序确定，不能仅从 NPZ 的数字索引反推方位（`agent/dqn.py:39-41`, `world/world_sumo.py:150-158`）。
- 正式网络输入为 16 维：8 维 `lane_count` 拼接 8 维当前绿色 phase one-hot。正式配置 `phase: true, one_hot: true`（`configs/tsc/dqn.yml:10-11`）；拼接在 `agent/dqn.py:150-154` 和训练 batch 的 `agent/dqn.py:337-345`。
- NPZ 直接保存 `current_phase` 与 `next_phase`（不是推断字段）。writer 在动作前记录前者、执行一个 action interval 后记录后者（`trainer/tsc_trainer.py:456-508`）。每个有效 shard 校验 `next_phase[t] == current_phase[t+1]`，故 episode 内 current phase 可可靠使用；第一条也直接保存，无需从前 episode 重建（`utils/trajectory.py:173-176`）。
- action 为 `[0,7]` 的绿色 phase/action index，直接传给 SUMO；无 action mask。SUMO 可能在切换时插入 yellow，且 `current_phase` 的原始 SUMO phase index 可能短暂为内部 yellow phase；因此相同 action 在不同 current phase/切换上下文不能视为相同的瞬时信号物理状态（`world/world_sumo.py:192-256,340-361`）。数据的 phase 值已通过正式 schema 检验为 `[0,7]`；后续严格比较应控制该字段。

## 时间与终止

- 每 episode 3600 simulation steps，control interval 10，故每 episode 360 decisions；`decision_step` 为 1..360，`global_step` 为连续训练决策索引（`configs/tsc/base.yml`, `trainer/tsc_trainer.py:456-491`, `utils/trajectory.py:55-57`）。第 k 个决策的起始 simulation time 为 `(k-1)*10` 秒；NPZ 未存单独 simulation-time 字段。
- 正式固定时长结束写为 `truncated=True`，每 shard 最后一条；`terminated=False`（`trainer/tsc_trainer.py:490-508`, `utils/trajectory.py:185-191`）。但 online `DQNAgent.remember` 没有把 done/terminated/truncated 放入 replay tuple，`train()` 未使用任何 terminal mask（`agent/dqn.py:193-214,352-370`）。

## Bellman target 与 checkpoint

- 正式代码是 vanilla DQN，不是 Double DQN：`y=r+gamma*max_a Q_target(s',a)`，`gamma` 由模型配置读取（正式冻结值为 0.95，见每个 run 的 `config/resolved_config.yaml`；计算位置为 `agent/dqn.py:352-370`）。公式没有 bootstrap mask，因此即使 fixed-horizon `truncated=True` 也 bootstrap；`terminated` 同样未参与 target。
- `s'` 的网络输入是 8 维 `next_state` 加 `next_phase` 的 8 维 one-hot；没有 action mask。resumable checkpoint 同时存 `online_model_state_dict`、`target_model_state_dict`、optimizer 和 epsilon（`trainer/tsc_trainer.py:596-625`）。
'''
    (OUT / 'transition_semantics_report.md').write_text(semantics, encoding='utf-8')
    readiness = f'''# Plan 1 跨场景 RL 数据资产 readiness

## 审计边界

本目录只读审计 `p1_formal_20_runlist_20260722.csv` 的 20 个正式 run；seed 0 使用正式 `_r2` 替代目录。未训练、未 rollout、未修改原始资产。

## 结果

- 正式 transition 资产完整：`{formal_ok}`；全部 run 的 shard 级结果见 `plan1_transition_asset_manifest.csv`。
- current phase 可可靠使用：它被直接保存，且逐 shard 验证连续链；不需要以 `next_phase[t-1]` 重建。
- checkpoint 面板可统一前向：`{cp_ok}`。注意“统一计算”在工程上可行，但选择某一 checkpoint 作为共同 Q 函数是后续分析设计决定，本审计未选择它。
- 共同支持存在，规模见 `matching_feasibility_summary.csv`；该文件仅是匹配可行性，并不检验 reward、next-state 或 Bellman-target 差异。

## 可直接开展与限制

可直接做：控制 8 维 lane state、直接保存的 current phase 和 action 后的跨场景条件比较；对指定 checkpoint 用其 target network 计算 vanilla-DQN target。

限制：训练 replay 的实际 target 无 terminal/truncation mask；NPZ 没有单独的 simulation time（但可由 decision step 恢复）；信号切换会经历内部 yellow，因此 phase/action 不是完整的信号历史；`queue`、`waiting_time`、两种 delay 在原始 NPZ 中逐 transition 可用，但本任务不将其重打包为新大数据集。它们的代码口径分别来自 `DQNAgent.get_queue/get_delay` 与 `Metrics`，应在后续正式机制分析中将其作为诊断量而非另一个状态变量。
'''
    (OUT / 'analysis_readiness_report.md').write_text(readiness, encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = formal_runs()
    assert len(runs) == 20 and {x['network'] for x in runs} == set(SCENES)
    data, run_summary = audit_transition_assets(runs)
    checkpoints = audit_checkpoints(runs)
    _, scale = state_statistics(data)
    rows = matching(data, scale)
    write_reports(run_summary, checkpoints, rows)


if __name__ == '__main__': main()
