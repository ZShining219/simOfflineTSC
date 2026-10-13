"""HITL session packager: recorder JSONL -> offline trajectory archive.

探索件（归 dev/hitl_live/，登记 dev/INDEX.md）。把
``hitl/live`` ``--record`` 录出的会话目录（transitions/
proposals/interventions + manifest）打包成离线档案：

    <out>/
        trajectory/               Plan 1 兼容分片（EpisodeTrajectoryWriter）
            episodes/episode_XXXX.npz
            manifest.json index.jsonl validation.json
        source_shards.jsonl       Plan 2 风格 source index
        datasets/<network>/<stage>/manifest.json
            ↑ 可直接喂 ``dataset.offline_trajectory_dataset`` 的
              ``OfflineTrajectoryDataset`` / ``validate_offline_dataset``
        config/simulator_resolved.cfg   语义哈希用（dir 已解析为绝对路径）
        config/source_sim.cfg           录制时的原始仿真配置副本
        hitl/                     审计副产物（不进标准分片）
            session_manifest.json proposals.jsonl interventions.jsonl
            labels.jsonl          逐转移归属标签（transition_unique_key 对齐）
        manifest.json             archive 根清单
        packaging_report.json     本打包操作的报告（丢弃/隔离/自检结果）

与 Plan 2 正式构建的差异（有意为之）：
- 变长 episode：直播会话按 reset/关闭边界切分，决策数不固定；
  ``expected_decisions_per_episode`` 记录为最大单集决策数，逐集校验由本件
  自己的 ``_validate_trajectory_dir`` 完成（语义检查与 Plan 1 writer
  ``validate()`` 同级，去掉"等长+连续 episode id"假设）。
- 不伪造 run_manifest/run_status/config 归档——本档案不是"一次正式 Plan 1
  run"，只是一份可被 ``OfflineTrajectoryDataset`` 消费的转移集合；
  语义哈希（roadnet/tl_program/action_mapping/feature/reward 五元组）仍按
  同一 ``_scene_semantics`` 计算，保证与正式数据集同口径可比对。
- HITL 溯源（谁建议、谁拍板、是否接管、查询原因）不落标准数组字段——
  标准字段保持训练管线契约干净；``behavior_mode`` 数组打粗粒度归属标签
  （hitl_autonomous/hitl_expert_agreed/hitl_expert_took_over/hitl_operator），
  逐转移明细在 ``hitl/labels.jsonl`` 供混合池加权采样用。

用法（repo 根目录，conda env: colight）：

    python dev/hitl_live/package.py \
        --session output_data/hitl_live/<session_id> \
        --out output_data/hitl_archives/hitl_<tag> \
        --network sumohz1x1 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset.offline_trajectory_dataset import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    OFFLINE_DATASET_BUILDER_VERSION,
    OFFLINE_DATASET_SCHEMA_VERSION,
    OfflineTrajectoryDataset,
    TRAJECTORY_REQUIRED_FIELDS,
    _atomic_json,
    _git_commit,
    _scene_semantics,
    _sha256,
    _stable_hash,
    validate_offline_dataset,
)
from utils.trajectory import (  # noqa: E402
    TRAJECTORY_ARRAY_FIELDS,
    TRAJECTORY_SCHEMA_VERSION,
    EpisodeTrajectoryWriter,
)

SEMANTIC_COMPATIBILITY_FIELDS = (
    'roadnet_control_topology_sha256', 'tl_program_sha256',
    'action_mapping_sha256', 'feature_schema_sha256',
    'reward_definition_sha256',
)
SOURCE_ROOT_ID = 'hitl_live_root'
TRANSITION_UNIQUE_KEY = ['run_id', 'shard_sha256', 'episode_id', 'decision_step']

# behavior_mode 归属标签优先级（多路口合并时取最高）
_AUTHORSHIP_ORDER = (
    'hitl_operator',
    'hitl_expert_took_over',
    'hitl_expert_agreed',
    'hitl_autonomous',
)

REQUIRED_ROW_FIELDS = (
    'episode_id', 'decision_step', 'global_step', 'rank',
    'state', 'current_phase', 'action', 'reward',
    'next_state', 'next_phase', 'terminated', 'truncated',
)


class PackagingError(RuntimeError):
    """Fatal packaging problem (corrupt files / invalid inputs)."""


# --------------------------------------------------------------------- load


def _load_session(session_dir):
    session_dir = Path(session_dir)
    manifest_path = session_dir / 'manifest.json'
    if not manifest_path.is_file():
        raise PackagingError(f'session manifest missing: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows = []
    transitions_path = session_dir / 'transitions.jsonl'
    if transitions_path.is_file():
        with open(transitions_path, encoding='utf-8') as handle:
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise PackagingError(
                        f'transitions.jsonl:{lineno} malformed JSON: {error}'
                    )
    if not rows:
        raise PackagingError(
            f'no transitions recorded in {session_dir} — nothing to package')
    return manifest, rows


# ------------------------------------------------------------- merge episodes


def _authorship(row):
    """Per-rank authorship tag -> behavior_mode vocabulary."""
    hitl = row.get('hitl') or {}
    if row.get('mid_interval_override') or hitl.get('expert_id') == 'operator':
        return 'hitl_operator'
    if hitl.get('intervened') and hitl.get('expert_id'):
        return 'hitl_expert_took_over'
    if hitl.get('queried'):
        return 'hitl_expert_agreed'
    return 'hitl_autonomous'


def _merge_group(rows, action_dim):
    """Merge same-decision per-rank rows into one Plan-1 transition dict.

    Returns (record, problems); problems is a list of strings — non-empty
    means the group is unusable and its episode must be quarantined.
    """
    problems = []
    rows = sorted(rows, key=lambda r: r['rank'])
    ranks = [r['rank'] for r in rows]
    if len(set(ranks)) != len(ranks):
        problems.append(f'duplicate rank rows at decision {rows[0]["decision_step"]}')
    null_fields = ('state', 'next_state', 'current_phase', 'next_phase',
                   'action', 'reward')
    for row in rows:
        for field in null_fields:
            if row.get(field) is None:
                problems.append(
                    f"rank {row['rank']} decision {row['decision_step']}: "
                    f'{field} is null')
        if row.get('reward_error'):
            problems.append(
                f"rank {row['rank']} decision {row['decision_step']}: "
                'reward sampling error')
        if row.get('reward_steps') is not None and row['reward_steps'] < 1:
            problems.append(
                f"rank {row['rank']} decision {row['decision_step']}: "
                'zero reward samples')
        if problems:
            return None, problems
    try:
        state = np.stack(
            [np.asarray(r['state'], dtype=np.float32) for r in rows])
        next_state = np.stack(
            [np.asarray(r['next_state'], dtype=np.float32) for r in rows])
        phase = np.asarray([int(r['current_phase']) for r in rows],
                           dtype=np.int64)
        next_phase = np.asarray([int(r['next_phase']) for r in rows],
                                dtype=np.int64)
        action = np.asarray([int(r['action']) for r in rows], dtype=np.int64)
        reward = np.asarray([float(r['reward']) for r in rows],
                            dtype=np.float32)
    except (TypeError, ValueError) as error:
        return None, [f'row values not convertible: {error}']
    for name, array, hi in (
            ('action', action, action_dim),
            ('current_phase', phase, action_dim),
            ('next_phase', next_phase, action_dim)):
        if np.any(array < 0) or np.any(array >= hi):
            problems.append(
                f'{name} out of range [0,{hi}): {array.tolist()} '
                '(yellow program index or bad action)')
    for name, array in (('state', state), ('next_state', next_state),
                        ('reward', reward)):
        if not np.all(np.isfinite(array)):
            problems.append(f'{name} contains NaN/Inf')
    if problems:
        return None, problems
    metrics_rows = [r.get('metrics') or {} for r in rows]
    junctions = [m.get('junctions') or {} for m in metrics_rows]

    def _junction_metric(field):
        values = []
        for row, jm in zip(rows, junctions):
            value = (jm.get(row['junction']) or {}).get(field)
            if value is not None:
                values.append(float(value))
        return float(np.mean(values)) if values else 0.0

    def _network_metric(field):
        for m in metrics_rows:
            if m.get(field) is not None:
                return float(m[field])
        return 0.0

    tags = sorted(
        {_authorship(r) for r in rows},
        key=_AUTHORSHIP_ORDER.index)
    record = {
        'state': state, 'current_phase': phase, 'action': action,
        'reward': reward, 'next_state': next_state, 'next_phase': next_phase,
        'terminated': bool(all(bool(r['terminated']) for r in rows)),
        'truncated': bool(all(bool(r['truncated']) for r in rows)),
        'epsilon': 0.0,
        'behavior_mode': tags[0],
        'queue': _junction_metric('queue'),
        'approximate_delay': _junction_metric('approximate_delay'),
        'real_delay': _network_metric('real_delay'),
        'throughput': _network_metric('throughput'),
        'waiting_time': _network_metric('waiting_time'),
        'metrics_missing': sum(
            1 for m in metrics_rows if not m.get('junctions')),
        '_rows': rows,
    }
    return record, problems


def _group_episodes(rows):
    """Rows -> {episode_id: [(decision_step, merged_record)]}; problems
    bucketed per episode for quarantine decisions."""
    episodes = {}
    for index, row in enumerate(rows):
        missing = [f for f in REQUIRED_ROW_FIELDS if f not in row]
        ep = row.get('episode_id')
        episodes.setdefault(ep, {'problems': [], 'groups': {}})
        if missing:
            episodes[ep]['problems'].append(
                f'row {index}: missing fields {missing}')
            continue
        key = (row['decision_step'], row['global_step'])
        episodes[ep]['groups'].setdefault(key, []).append(row)
    return episodes


# ------------------------------------------------------------ validation


def _validate_trajectory_dir(root, action_dim, index_entries):
    """Plan-1-grade shard validation without uniform-length assumptions."""
    errors = []
    total = 0
    previous_global_step = 0
    per_episode = {}
    for entry in index_entries:
        path = os.path.join(root, entry['file'])
        label = entry['file']
        if not os.path.isfile(path):
            errors.append(f'missing trajectory shard: {label}')
            continue
        if _sha256(path) != entry['sha256']:
            errors.append(f'trajectory hash mismatch: {label}')
            continue
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(set(TRAJECTORY_ARRAY_FIELDS) - set(data.files))
            if missing:
                errors.append(f'{label} missing fields {missing}')
                continue
            count = len(data['global_step'])
            per_episode[str(entry['episode_id'])] = int(count)
            total += count
            if count != entry['transition_count']:
                errors.append(f'{label} index count mismatch')
            if int(np.asarray(data['episode_id']).reshape(-1)[0]) != \
                    entry['episode_id']:
                errors.append(f'{label} episode id mismatch')
            expected_decisions = np.arange(1, count + 1)
            if not np.array_equal(data['decision_step'], expected_decisions):
                errors.append(f'{label} decision steps are not contiguous')
            expected_globals = np.arange(
                previous_global_step + 1, previous_global_step + count + 1)
            if not np.array_equal(data['global_step'], expected_globals):
                errors.append(f'{label} global steps are not contiguous')
            previous_global_step += count
            if count > 1 and not np.array_equal(
                    data['next_state'][:-1], data['state'][1:]):
                errors.append(f'{label} state chain is broken')
            if count > 1 and not np.array_equal(
                    data['next_phase'][:-1], data['current_phase'][1:]):
                errors.append(f'{label} phase chain is broken')
            for field in ('state', 'reward', 'next_state', 'epsilon', 'queue',
                          'approximate_delay', 'real_delay', 'throughput',
                          'waiting_time'):
                if not np.all(np.isfinite(data[field])):
                    errors.append(f'{label} {field} contains NaN/Inf')
            if np.any(data['action'] < 0) or np.any(
                    data['action'] >= action_dim):
                errors.append(f'{label} contains illegal actions')
            phases = np.asarray(data['current_phase']).reshape(count, -1)
            next_phases = np.asarray(data['next_phase']).reshape(count, -1)
            if np.any(phases < 0) or np.any(phases >= action_dim) or np.any(
                    next_phases < 0) or np.any(next_phases >= action_dim):
                errors.append(f'{label} phase index outside action domain')
            if np.any(data['terminated']):
                errors.append(f'{label} has unexpected terminated=true')
            if count and (np.any(data['truncated'][:-1])
                          or not bool(data['truncated'][-1])):
                errors.append(f'{label} truncated flags are invalid')
    result = {
        'schema_version': TRAJECTORY_SCHEMA_VERSION,
        'validator': 'hitl_packager',
        'variable_episode_length': True,
        'valid': not errors,
        'errors': errors,
        'episode_count': len(index_entries),
        'transition_count': total,
        'transitions_per_episode': per_episode,
        'evaluation_transition_count': 0,
        'expected_decisions_per_episode': max(per_episode.values(), default=0),
    }
    _atomic_json(os.path.join(root, 'validation.json'), result)
    return result


# --------------------------------------------------------------- packaging


def package_session(session_dir, output_dir, *, network=None,
                    behavior_seed=0, dataset_id=None, stage='hitl',
                    config_path=None, repo_root=None, strict=False,
                    verify_load=True):
    """Package one or more recorded HITL sessions into one offline archive.

    ``session_dir`` may be a single session directory or a list of them.
    With multiple sessions the episodes are merged into one archive:
    ``episode_id`` is renumbered globally (1..N in session order),
    ``global_step`` is rewritten to a single contiguous counter across the
    packaged episodes, and every episode/label keeps its ``run_id`` session
    provenance.

    Returns the packaging report dict.  With ``strict=True`` any quarantined
    episode aborts packaging with ``PackagingError``; otherwise bad episodes
    are dropped and listed in the report.
    """
    raw_dirs = ([session_dir] if isinstance(session_dir, (str, Path))
                else list(session_dir))
    output_dir = Path(output_dir).resolve()
    repo_root = Path(repo_root or _REPO_ROOT).resolve()
    sessions = []
    for raw in raw_dirs:
        path = Path(raw).resolve()
        meta_i, rows_i = _load_session(path)
        sid = str(meta_i.get('session_id') or path.name)
        sessions.append({
            'dir': path, 'meta': meta_i, 'rows': rows_i,
            'session_id': sid,
        })
    session_ids = [s['session_id'] for s in sessions]
    if len(set(session_ids)) != len(session_ids):
        raise PackagingError(f'duplicate session ids: {session_ids}')
    meta = sessions[0]['meta']
    session_id = session_ids[0] if len(sessions) == 1 else (
        '+'.join(session_ids))
    session_dir = sessions[0]['dir']
    session_dir_by_id = {s['session_id']: s['dir'] for s in sessions}
    dataset_id = dataset_id or f'hitl_{session_id}'
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PackagingError(
            f'output dir already exists and is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- source config / semantics inputs
    config_rel = config_path or meta.get('source_config')
    if not config_rel:
        raise PackagingError(
            'session manifest lacks source_config; pass --config')
    cfg_path = Path(config_rel)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    network = network or cfg_path.stem
    config_hash = _stable_hash(cfg)

    junctions = meta.get('junctions') or {}
    action_dims = meta.get('action_dims') or {}
    # cross-session compatibility: identical junction/action geometry and
    # identical simulator config, otherwise the merged archive would mix
    # semantically different transitions
    for other in sessions[1:]:
        other_meta = other['meta']
        if (other_meta.get('action_dims') or {}) != action_dims:
            raise PackagingError(
                f'session {other["session_id"]}: action_dims differ '
                'from the first session')
        if (other_meta.get('junctions') or {}) != junctions:
            raise PackagingError(
                f'session {other["session_id"]}: junctions differ '
                'from the first session')
        other_cfg_rel = config_path or other_meta.get('source_config')
        if not other_cfg_rel:
            raise PackagingError(
                f'session {other["session_id"]} lacks source_config')
        other_cfg_path = Path(other_cfg_rel)
        if not other_cfg_path.is_absolute():
            other_cfg_path = repo_root / other_cfg_path
        if _stable_hash(json.loads(
                other_cfg_path.read_text(encoding='utf-8'))) != config_hash:
            raise PackagingError(
                f'session {other["session_id"]}: simulator config differs')
    dims = {int(v) for v in action_dims.values()}
    if len(dims) != 1:
        raise PackagingError(
            f'non-uniform action dims across junctions: {action_dims}')
    action_dim = dims.pop()
    action_interval = float(meta.get('action_interval') or 0) or None

    # --- renumber episodes globally; combined row stream
    rows = []
    episode_owner = {}
    ep_counter = 0
    for session in sessions:
        local_eps = sorted({
            row.get('episode_id') for row in session['rows']
            if isinstance(row.get('episode_id'), int)})
        id_map = {}
        for local_ep in local_eps:
            ep_counter += 1
            id_map[local_ep] = ep_counter
            episode_owner[ep_counter] = session['session_id']
        for row in session['rows']:
            if row.get('episode_id') in id_map:
                row = dict(row)
                row['episode_id'] = id_map[row['episode_id']]
                row['_run_id'] = session['session_id']
            rows.append(row)

    # --- group + merge
    episodes = _group_episodes(rows)
    merged = {}
    quarantined = {}
    for ep_id, info in episodes.items():
        if not isinstance(ep_id, int) and info['problems']:
            quarantined.setdefault('unparseable', []).extend(info['problems'])
    for ep_id in sorted(k for k in episodes if isinstance(k, int)):
        info = episodes[ep_id]
        groups = info['groups']
        if not groups:
            continue
        steps = sorted(groups)
        episode_problems = list(info['problems'])
        # global_step must be strictly contiguous per decision boundary
        expected_gs = None
        records = []
        expected_n_ranks = None
        for ds, gs in steps:
            if expected_gs is not None and gs != expected_gs:
                episode_problems.append(
                    f'non-contiguous global_step at decision {ds}')
                break
            expected_gs = gs + 1
            record, problems = _merge_group(groups[(ds, gs)], action_dim)
            if problems:
                episode_problems.extend(problems)
            else:
                n_ranks = record['state'].shape[0]
                if expected_n_ranks is None:
                    expected_n_ranks = n_ranks
                elif n_ranks != expected_n_ranks:
                    episode_problems.append(
                        f'rank set changed at decision {ds}: '
                        f'{expected_n_ranks} -> {n_ranks}')
                    break
                records.append((ds, gs, record))
        if records:
            # chain + flag sanity across the merged episode — the dataset
            # loader does not check chains, so quarantine before writing
            for i in range(1, len(records)):
                prev, cur = records[i - 1][2], records[i][2]
                if not np.array_equal(prev['next_state'], cur['state']):
                    episode_problems.append(
                        f'state chain broken at decision {records[i][0]}')
                if not np.array_equal(
                        prev['next_phase'], cur['current_phase']):
                    episode_problems.append(
                        f'phase chain broken at decision {records[i][0]}')
            for ds, _, record in records:
                if record['terminated']:
                    episode_problems.append(
                        f'terminated=true at decision {ds} '
                        '(unsupported in offline sources)')
                if record['truncated'] and record is not records[-1][2]:
                    episode_problems.append(
                        f'truncated=true mid-episode at decision {ds}')
            if not records[-1][2]['truncated']:
                episode_problems.append(
                    'last transition not truncated (episode close missing)')
        if episode_problems:
            quarantined[str(ep_id)] = episode_problems
            continue
        merged[ep_id] = records
        # decision_step sanity: recorded steps may skip (rows dropped only if
        # invalid — those episodes are already quarantined above)
        ds_seq = [ds for ds, _, _ in records]
        if ds_seq != list(range(ds_seq[0], ds_seq[-1] + 1)):
            quarantined[str(ep_id)] = [
                f'decision steps not contiguous: {ds_seq[:5]}...'
            ]
            del merged[ep_id]

    if strict and quarantined:
        raise PackagingError(
            f'episodes quarantined under --strict: {sorted(quarantined)}')
    if not merged:
        raise PackagingError(
            f'no packageable episodes in {session_dir}: {quarantined}')

    metrics_missing = 0

    # --- write Plan-1-style shards
    max_ep_len = max(len(recs) for recs in merged.values())
    writer = EpisodeTrajectoryWriter(
        output_path=str(output_dir),
        network=network,
        behavior_training_seed=int(behavior_seed),
        config_hash=config_hash,
        simulation_steps=int((action_interval or 1) * max_ep_len),
        action_interval=int(action_interval or 1),
        action_dim=action_dim,
    )
    labels_rows = []
    gs_out = 0
    for ep_id in sorted(merged):
        records = merged[ep_id]
        owner = episode_owner.get(ep_id, session_id)
        writer.start_episode(ep_id)
        for ds, gs, record in records:
            metrics_missing += record['metrics_missing']
            gs_out += 1
            writer.append(
                episode_id=ep_id,
                decision_step=ds + 1,
                global_step=gs_out,
                state=record['state'],
                current_phase=record['current_phase'],
                action=record['action'],
                reward=record['reward'],
                next_state=record['next_state'],
                next_phase=record['next_phase'],
                terminated=record['terminated'],
                truncated=record['truncated'],
                epsilon=record['epsilon'],
                behavior_mode=record['behavior_mode'],
                queue=record['queue'],
                approximate_delay=record['approximate_delay'],
                real_delay=record['real_delay'],
                throughput=record['throughput'],
                waiting_time=record['waiting_time'],
            )
            for row in record['_rows']:
                hitl = dict(row.get('hitl') or {})
                hitl.update({
                    'mid_interval_override': bool(
                        row.get('mid_interval_override')),
                    'reward_steps': row.get('reward_steps'),
                    'reward_error': bool(row.get('reward_error')),
                })
                labels_rows.append({
                    'run_id': row.get('_run_id') or owner,
                    'episode_id': ep_id,
                    'decision_step': ds + 1,
                    'global_step': gs_out,
                    'rank': row['rank'],
                    'junction': row.get('junction'),
                    't0': row.get('t0'),
                    't1': row.get('t1'),
                    'hitl': hitl,
                    'behavior_mode': record['behavior_mode'],
                    'packaged': True,
                })
        writer.finish_episode()
    index_entries = list(writer.index_entries)
    traj_validation = _validate_trajectory_dir(
        writer.root, action_dim, index_entries)

    # non-packaged rows still land in labels for audit
    for row in rows:
        ep = row.get('episode_id')
        if ep in merged:
            continue
        missing = [f for f in REQUIRED_ROW_FIELDS if f not in row]
        labels_rows.append({
            'run_id': row.get('_run_id')
            or episode_owner.get(ep, session_id),
            'episode_id': ep,
            'decision_step': (row.get('decision_step') or -1) + 1,
            'global_step': (row.get('global_step') or -1) + 1,
            'rank': row.get('rank'),
            'junction': row.get('junction'),
            't0': row.get('t0'), 't1': row.get('t1'),
            'hitl': dict(row.get('hitl') or {}),
            'behavior_mode': None,
            'packaged': False,
            'quarantine_reasons': (
                [f'missing fields {missing}'] if missing
                else quarantined.get(str(ep), ['episode not packaged'])),
        })

    # --- config archive for semantic hashing (same recipe as Plan 1 runs)
    config_dir = output_dir / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)
    resolved_cfg = dict(cfg)
    cfg_dir = Path(cfg.get('dir', 'data/'))
    if not cfg_dir.is_absolute():
        cfg_dir = repo_root / cfg_dir
    resolved_cfg['dir'] = str(cfg_dir.resolve()) + '/'
    _atomic_json(str(config_dir / 'simulator_resolved.cfg'), resolved_cfg)
    _atomic_json(str(config_dir / 'source_sim.cfg'), cfg)
    semantics = _scene_semantics(str(output_dir))
    if semantics['green_action_count'] != action_dim:
        raise PackagingError(
            f'roadnet green_action_count={semantics["green_action_count"]} '
            f'!= recorded action_dim={action_dim} — network mismatch?')

    # --- Plan-2-style source index + dataset manifest
    source_root = os.path.realpath(writer.root)
    source_index_path = output_dir / 'source_shards.jsonl'
    with open(source_index_path, 'x', encoding='utf-8') as handle:
        for entry in index_entries:
            shard_path = os.path.join(source_root, entry['file'])
            owner_id = episode_owner.get(entry['episode_id'], session_id)
            owner_dir = session_dir_by_id.get(owner_id, session_dir)
            indexed = {
                'schema_version': OFFLINE_DATASET_SCHEMA_VERSION,
                'run_path': str(owner_dir),
                'run_id': owner_id,
                'network': network,
                'behavior_training_seed': int(behavior_seed),
                'episode_id': entry['episode_id'],
                'transition_count': entry['transition_count'],
                'source_root_id': SOURCE_ROOT_ID,
                'relative_path': entry['file'],
                'original_absolute_path': os.path.realpath(shard_path),
                'sha256': entry['sha256'],
                'semantic_validation': {
                    'episode_id': entry['episode_id'],
                    'transition_count': entry['transition_count'],
                    'first_global_step': entry['first_global_step'],
                    'last_global_step': entry['last_global_step'],
                },
            }
            handle.write(json.dumps(indexed, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())

    episode_ids = [entry['episode_id'] for entry in index_entries]
    episode_range = [min(episode_ids), max(episode_ids)]
    expected_total = sum(e['transition_count'] for e in index_entries)
    stage_manifest = (
        output_dir / 'datasets' / network / stage / 'manifest.json')
    stage_payload = {
        'schema_version': OFFLINE_DATASET_SCHEMA_VERSION,
        'dataset_builder_version': OFFLINE_DATASET_BUILDER_VERSION,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'trajectory_schema_version': TRAJECTORY_SCHEMA_VERSION,
        'created_by_commit': _git_commit(),
        'dataset_id': dataset_id,
        'dataset_kind': 'single_scene',
        'dataset_stage': stage,
        'evaluation_network': network,
        'source_networks': [network],
        'behavior_training_seeds': [int(behavior_seed)],
        'episode_range': episode_range,
        'sampling_strategy': 'uniform_transition',
        'source_index': os.path.relpath(
            str(source_index_path), str(stage_manifest.parent)),
        'source_index_sha256': _sha256(str(source_index_path)),
        'source_root_id': SOURCE_ROOT_ID,
        'original_source_root': source_root,
        'action_dim': action_dim,
        'expected_decisions_per_episode': max_ep_len,
        'expected_transition_count': expected_total,
        'transition_unique_key': TRANSITION_UNIQUE_KEY,
        'scene_semantics': {network: semantics},
        'semantic_compatibility_fields': list(
            SEMANTIC_COMPATIBILITY_FIELDS),
    }
    _atomic_json(str(stage_manifest), stage_payload)

    # --- HITL sidecars
    hitl_dir = output_dir / 'hitl'
    hitl_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(str(hitl_dir / 'session_manifest.json'), meta)
    sessions_dir = hitl_dir / 'sessions'
    if len(sessions) > 1:
        sessions_dir.mkdir(exist_ok=True)
        for session in sessions:
            sdir = sessions_dir / session['session_id']
            sdir.mkdir(exist_ok=True)
            _atomic_json(str(sdir / 'session_manifest.json'),
                         session['meta'])
            for name in ('proposals.jsonl', 'interventions.jsonl'):
                src = session['dir'] / name
                if src.is_file():
                    (sdir / name).write_bytes(src.read_bytes())
    for name in ('proposals.jsonl', 'interventions.jsonl'):
        src = session_dir / name
        if len(sessions) == 1 and src.is_file():
            dst = hitl_dir / name
            dst.write_bytes(src.read_bytes())
    labels_path = hitl_dir / 'labels.jsonl'
    with open(labels_path, 'x', encoding='utf-8') as handle:
        for row in labels_rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    hitl_index = {
        'labels_sha256': _sha256(str(labels_path)),
        'label_rows': len(labels_rows),
        'authorship_counts': {
            tag: sum(
                1 for r in labels_rows
                if r.get('packaged') and r.get('behavior_mode') == tag)
            for tag in _AUTHORSHIP_ORDER
        },
    }
    _atomic_json(str(hitl_dir / 'index.json'), hitl_index)

    # --- root manifest
    root_manifest = {
        'schema_version': OFFLINE_DATASET_SCHEMA_VERSION,
        'dataset_builder_version': OFFLINE_DATASET_BUILDER_VERSION,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'trajectory_schema_version': TRAJECTORY_SCHEMA_VERSION,
        'created_by_commit': _git_commit(),
        'archive_kind': 'hitl_live_recording',
        'dataset_id': dataset_id,
        'session_id': session_id,
        'session_ids': session_ids,
        'source_session_dir': str(session_dir),
        'source_session_dirs': [str(s['dir']) for s in sessions],
        'networks': [network],
        'behavior_training_seeds': [int(behavior_seed)],
        'episode_ids': episode_ids,
        'quarantined_episodes': {str(k): v for k, v in
                                 quarantined.items()},
        'transition_count': expected_total,
        'action_dim': action_dim,
        'action_interval': action_interval,
        'variable_episode_length': True,
        'expert_id': meta.get('expert_id'),
        'hitl_gate': meta.get('gate'),
        'student_controller': meta.get('controller')
        or meta.get('default_controller'),
        'sumo_seed_mode': 'fixed_default',
        'config_hash': config_hash,
        'scene_semantics': {network: semantics},
        'dataset_manifests': [os.path.relpath(
            str(stage_manifest), str(output_dir))],
        'source_index': os.path.relpath(
            str(source_index_path), str(output_dir)),
        'source_index_sha256': _sha256(str(source_index_path)),
        'source_root_id': SOURCE_ROOT_ID,
        'original_source_root': source_root,
        'transition_unique_key': TRANSITION_UNIQUE_KEY,
        'trajectory_validation': traj_validation,
    }
    _atomic_json(str(output_dir / 'manifest.json'), root_manifest)

    # --- expert pool manifest (consumed by sequential.expert_pool /
    #     HASODQNAgent's third sampling leg).  observation_dim is the
    #     dataset-layer dim: raw state + one-hot phase.
    with np.load(os.path.join(source_root, index_entries[0]['file']),
                 allow_pickle=False) as first_shard:
        state_dim = int(np.asarray(first_shard['state']).reshape(
            len(first_shard['state']), -1).shape[1])
    expert_pool = {
        'schema_version': 1,
        'kind': 'hitl_expert_pool',
        'dataset_id': dataset_id,
        'session_id': session_id,
        'session_ids': session_ids,
        'network': network,
        'dataset_manifest': {
            'path': os.path.relpath(str(stage_manifest), str(output_dir)),
            'sha256': _sha256(str(stage_manifest)),
        },
        'observation_dim': state_dim + action_dim,
        'action_dim': action_dim,
        'transition_count': expected_total,
        'behavior_training_seed': int(behavior_seed),
        'episode_ids': episode_ids,
        'behavior_mode_counts': hitl_index['authorship_counts'],
        'expert_id': meta.get('expert_id'),
        'student_controller': meta.get('controller')
        or meta.get('default_controller'),
        'hitl_gate': meta.get('gate'),
    }
    expert_pool['pool_digest'] = _stable_hash(expert_pool)
    _atomic_json(str(output_dir / 'expert_pool_manifest.json'), expert_pool)

    # --- self check: dataset-level validation + real load
    report = {
        'dataset_id': dataset_id,
        'session_id': session_id,
        'network': network,
        'output_dir': str(output_dir),
        'episodes_packaged': len(index_entries),
        'episodes_quarantined': sorted(str(k) for k in quarantined),
        'quarantine_reasons': {str(k): v for k, v in quarantined.items()},
        'rows_missing_metrics': metrics_missing,
        'transition_count': expected_total,
        'trajectory_validation': traj_validation,
        'stage_manifest': str(stage_manifest),
    }
    dataset_validation = validate_offline_dataset(str(stage_manifest))
    report['dataset_validation'] = dataset_validation
    if verify_load:
        dataset = OfflineTrajectoryDataset(str(stage_manifest), seed=0)
        report['dataset_load'] = {
            'transition_count': int(dataset.transition_count),
            'observation_dim': int(dataset.observation_dim),
            'statistics': dataset.statistics(),
        }
    report['valid'] = bool(
        traj_validation['valid'] and dataset_validation['valid'])
    _atomic_json(str(output_dir / 'packaging_report.json'), report)
    return report


def _main(argv=None):
    parser = argparse.ArgumentParser(
        description='Package a recorded HITL broadcast session into an '
                    'offline trajectory archive consumable by '
                    'dataset.OfflineTrajectoryDataset.')
    parser.add_argument('--session', required=True, action='append',
                        help='Recorder session dir (with manifest.json + '
                             'transitions.jsonl); repeat to merge multiple '
                             'sessions into one archive')
    parser.add_argument('--out', required=True,
                        help='Output archive dir (must not exist)')
    parser.add_argument('--network', default=None,
                        help='Network id (default: source_config basename)')
    parser.add_argument('--seed', type=int, default=0,
                        help='behavior_training_seed recorded on shards')
    parser.add_argument('--dataset-id', default=None,
                        help='Dataset id (default: hitl_<session_id>)')
    parser.add_argument('--stage', default='hitl',
                        help='Dataset stage name (default: hitl)')
    parser.add_argument('--config', default=None,
                        help='Sim cfg path (default: session manifest '
                             'source_config)')
    parser.add_argument('--repo-root', default=None,
                        help='Repo root for relative config paths')
    parser.add_argument('--strict', action='store_true',
                        help='Abort if any episode is quarantined')
    parser.add_argument('--no-verify-load', action='store_true',
                        help='Skip the OfflineTrajectoryDataset load check')
    args = parser.parse_args(argv)
    report = package_session(
        args.session, args.out, network=args.network,
        behavior_seed=args.seed, dataset_id=args.dataset_id,
        stage=args.stage, config_path=args.config,
        repo_root=args.repo_root, strict=args.strict,
        verify_load=not args.no_verify_load)
    print(json.dumps({
        'dataset_id': report['dataset_id'],
        'network': report['network'],
        'episodes_packaged': report['episodes_packaged'],
        'episodes_quarantined': report['episodes_quarantined'],
        'transition_count': report['transition_count'],
        'valid': report['valid'],
    }, ensure_ascii=False, indent=2))
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    sys.exit(_main())
