"""Pre-registered Plan5 metrics and paired cluster bootstrap."""
import csv
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np

from sequential.core import canonical_digest
from sequential.io import atomic_json, sha256_file

HISTORICAL_SCHEDULE = (0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100)


def historical_metrics(travel_times, schedule=HISTORICAL_SCHEDULE):
    values={int(k): float(travel_times[k]) for k in schedule}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError('Historical travel times must be finite')
    baseline=values[0]
    if baseline == 0: raise ValueError("Episode-0 travel time must be non-zero")
    degradation={e:(v-baseline)/baseline for e,v in values.items()}
    worst=max(degradation.values())
    return {"degradation":degradation,"worst_historical_degradation":worst,
            "episode100_degradation":degradation[100],"recovery_gap":worst-degradation[100]}


def current_adaptation(travel_times, specialist_tt):
    if specialist_tt <= 0: raise ValueError("Specialist travel time must be positive")
    points=sorted((int(e),float(v)/specialist_tt) for e,v in travel_times.items())
    if [episode for episode, _ in points] != list(range(101)):
        raise ValueError("Current evaluations must contain every episode 0..100")
    if not all(math.isfinite(value) for _, value in points):
        raise ValueError('Current normalized travel times must be finite')
    area=sum((b-a)*(y0+y1)/2 for (a,y0),(b,y1) in zip(points,points[1:]))
    time_to=None
    for e in range(0,97):
        if all(travel_times.get(i, math.inf)/specialist_tt <= 1.10 for i in range(e,e+5)):
            time_to=e; break
    return {"normalized":dict(points),"current_adaptation_aulc":area/100.0,
            "episode100_normalized_performance":dict(points)[100],
            "time_to_reference":time_to if time_to is not None else ">100",
            "time_to_reference_censored": time_to is None}


def policy_disagreement(policy0, policy, heldout=None):
    a=np.asarray(policy0); b=np.asarray(policy)
    if a.shape != b.shape: raise ValueError("Policy arrays must have identical shape")
    result=float(np.mean(a != b))
    output={"disagreement":result}
    if heldout is not None:
        c=np.asarray(heldout[0]); d=np.asarray(heldout[1])
        if c.shape != d.shape: raise ValueError("Held-out policy arrays must match")
        output["heldout_disagreement"]=float(np.mean(c != d))
    return output


def _action_distribution(actions, action_dim=8):
    actions = np.asarray(actions, dtype=int)
    if actions.ndim != 1 or actions.size == 0 \
            or np.any(actions < 0) or np.any(actions >= action_dim):
        raise ValueError('Policy actions are outside the frozen action space')
    counts = np.bincount(actions, minlength=action_dim).astype(float)
    frequencies = counts / counts.sum()
    positive = frequencies[frequencies > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return frequencies, entropy


def probe_internal_metrics(baseline, current):
    """Compute frozen within-algorithm probe metrics relative to episode0."""
    if baseline.get('algorithm_id') != current.get('algorithm_id'):
        raise ValueError('Probe comparison algorithm identity changed')
    algorithm_id = current['algorithm_id']
    before = np.asarray(baseline['policy_vectors'], dtype=float)
    after = np.asarray(current['policy_vectors'], dtype=float)
    if before.shape != after.shape or before.ndim != 2 \
            or before.shape[1] != 8 or before.shape[0] == 0 \
            or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('Probe policy vectors are invalid')
    baseline_actions = np.asarray(baseline['actions'], dtype=int)
    actions = np.asarray(current['actions'], dtype=int)
    if baseline_actions.shape != (before.shape[0],) \
            or actions.shape != baseline_actions.shape:
        raise ValueError('Probe action vectors are invalid')
    frequencies, action_entropy = _action_distribution(actions)
    result = {
        'greedy_disagreement': float(np.mean(actions != baseline_actions)),
        'action_entropy': action_entropy,
        **{
            f'action_frequency_{index}': float(value)
            for index, value in enumerate(frequencies)
        },
    }
    if algorithm_id in {'DDQN', 'CTXDDQN'}:
        differences = after - before
        ordered = np.sort(after, axis=1)
        residual = current.get('bellman_residual_mean')
        if residual is None or not math.isfinite(float(residual)):
            raise ValueError('DQN probe Bellman residual is missing')
        result.update({
            'q_drift_l1': float(np.mean(np.sum(np.abs(differences), axis=1))),
            'q_drift_l2': float(np.mean(np.linalg.norm(differences, axis=1))),
            'top1_top2_q_margin': float(np.mean(
                ordered[:, -1] - ordered[:, -2]
            )),
            'bellman_residual': float(residual),
        })
    elif algorithm_id == 'PPO':
        if np.any(before < 0) or np.any(after < 0) \
                or not np.allclose(before.sum(axis=1), 1.0, atol=1e-6) \
                or not np.allclose(after.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError('PPO probe probabilities are invalid')
        midpoint = 0.5 * (before + after)
        epsilon = np.finfo(float).tiny
        js = 0.5 * np.sum(
            before * np.log((before + epsilon) / (midpoint + epsilon))
            + after * np.log((after + epsilon) / (midpoint + epsilon)),
            axis=1,
        )
        entropy = -np.sum(
            after * np.log(after + epsilon), axis=1,
        )
        baseline_values = np.asarray(baseline['critic_values'], dtype=float)
        current_values = np.asarray(current['critic_values'], dtype=float)
        if baseline_values.shape != (before.shape[0],) \
                or current_values.shape != baseline_values.shape \
                or not np.isfinite(baseline_values).all() \
                or not np.isfinite(current_values).all():
            raise ValueError('PPO critic values are invalid')
        result.update({
            'policy_entropy': float(np.mean(entropy)),
            'policy_tv_from_episode0': float(np.mean(
                0.5 * np.sum(np.abs(after - before), axis=1)
            )),
            'policy_js_from_episode0': float(np.mean(js)),
            'critic_value_drift': float(np.mean(np.abs(
                current_values - baseline_values
            ))),
        })
    else:
        raise ValueError('Unsupported Plan5 probe algorithm')
    return result


def context_distinction(rows):
    """Pair exact raw states across scenes; compare integer count vectors."""
    groups={}
    for row in rows:
        if row.get('scene') not in {'S1', 'S2', 'S3', 'S4'}:
            raise ValueError('Context distinction rows require a frozen scene')
        raw=np.asarray(row["raw16"],dtype=np.float32).tobytes()
        groups.setdefault(raw,[]).append(row)
    pair_count=distinguished=0; unique_distinguished=0
    matched_groups = 0
    for entries in groups.values():
        if len(entries)<2: continue
        vectors=[tuple(int(x) for x in item["arrival_counts"]) for item in entries]
        pairs = [
            (left, right)
            for left in range(len(vectors))
            for right in range(left + 1, len(vectors))
            if entries[left]['scene'] != entries[right]['scene']
        ]
        if not pairs:
            continue
        matched_groups += 1
        pair_count += len(pairs)
        group_distinguished = sum(
            vectors[left] != vectors[right] for left, right in pairs
        )
        distinguished += group_distinguished
        if group_distinguished:
            unique_distinguished += 1
    return {"pair_weighted_distinguished_fraction": distinguished/pair_count if pair_count else None,
            "unique_raw_state_weighted_distinguished_fraction": unique_distinguished/matched_groups if matched_groups else None,
            "matched_raw_state_groups": matched_groups,"matched_pairs":pair_count}


def paired_effects(h34, l32):
    if len(h34)!=len(l32) or len(h34)!=5: raise ValueError("Primary inference requires five paired seeds")
    effects=np.asarray(h34,dtype=float)-np.asarray(l32,dtype=float)
    return {"effects":effects.tolist(),"mean":float(np.mean(effects)),"median":float(np.median(effects)),
            "direction_consistency":float(np.mean(effects>0))}


def paired_cluster_bootstrap(h34, l32, resamples=10000, seed=20260827):
    effects=np.asarray(h34,dtype=float)-np.asarray(l32,dtype=float)
    if effects.size != 5: raise ValueError("Bootstrap cluster count must be five seeds")
    rng=np.random.default_rng(seed); draws=rng.integers(0,5,size=(int(resamples),5)); means=effects[draws].mean(axis=1)
    lo,hi=np.percentile(means,[2.5,97.5])
    result=paired_effects(h34,l32); result.update({"bootstrap_resamples":int(resamples),"analysis_seed":int(seed),"ci95":[float(lo),float(hi)]})
    return result


def select_ppo_calibration(rows):
    """Apply the frozen candidate score/IQR/AULC/LR/entropy ordering."""
    grouped = {}
    for row in rows:
        key = (float(row['learning_rate']), float(row['entropy_coefficient']))
        grouped.setdefault(key, []).append(row)
    expected = {
        (lr, entropy) for lr in (1e-4, 2.5e-4, 5e-4)
        for entropy in (0.001, 0.01)
    }
    if set(grouped) != expected or any(
        len(values) != 3
        or {int(row['training_seed']) for row in values} != {100, 101, 102}
        for values in grouped.values()
    ):
        raise ValueError('PPO calibration matrix is incomplete')
    for values in grouped.values():
        for row in values:
            numeric = [
                float(row[field]) for field in (
                    'tt_late_median', 'current_adaptation_aulc',
                    'tt_fixedtime', 'tt_episode0',
                )
            ]
            if not all(math.isfinite(value) for value in numeric) \
                    or numeric[2] <= 0 or numeric[3] <= 0:
                raise ValueError('PPO calibration metrics are non-finite/invalid')
    candidates = []
    for (learning_rate, entropy), values in grouped.items():
        medians = np.asarray([
            float(row['tt_late_median']) for row in values
        ])
        aulc = np.asarray([
            float(row['current_adaptation_aulc']) for row in values
        ])
        score = float(np.median(medians))
        iqr = float(np.percentile(medians, 75) - np.percentile(medians, 25))
        candidates.append({
            'learning_rate': learning_rate,
            'entropy_coefficient': entropy,
            'score': score, 'iqr': iqr,
            'aulc_median': float(np.median(aulc)),
            'runs': sorted(values, key=lambda item: int(item['training_seed'])),
        })
    winner = min(candidates, key=lambda item: (
        item['score'], item['iqr'], item['aulc_median'],
        item['learning_rate'], item['entropy_coefficient'],
    ))
    eligible = sum(
        float(row['tt_late_median']) < float(row['tt_fixedtime'])
        and (
            float(row['tt_episode0']) - float(row['tt_late_median'])
        ) / float(row['tt_episode0']) >= 0.10
        for row in winner['runs']
    )
    winner = dict(winner)
    winner['eligible_seed_count'] = int(eligible)
    winner['formal_eligible'] = eligible >= 2
    return {
        'winner': winner,
        'candidates': sorted(
            candidates,
            key=lambda item: (
                item['learning_rate'], item['entropy_coefficient']
            ),
        ),
        'formal_status': 'READY' if eligible >= 2 else 'HOLD',
    }


def freeze_ppo_calibration(rows, summary_path, config_path, config_sha256):
    """Freeze the complete calibration table and its selected PPO config."""
    from .schema import TABLE_SCHEMAS, validate_json_fields

    rows = [dict(row) for row in rows]
    for row in rows:
        if row.get('status') != 'completed':
            raise ValueError('PPO calibration requires 18 completed runs')
        run_summary = Path(row['run_summary_path'])
        if not run_summary.is_file() \
                or sha256_file(run_summary) != row['run_summary_sha256']:
            raise ValueError('PPO calibration run summary hash/path mismatch')
    selected = select_ppo_calibration(rows)
    candidates = sorted(selected['candidates'], key=lambda item: (
        item['score'], item['iqr'], item['aulc_median'],
        item['learning_rate'], item['entropy_coefficient'],
    ))
    rank = {
        (item['learning_rate'], item['entropy_coefficient']): index + 1
        for index, item in enumerate(candidates)
    }
    candidate_lookup = {
        (item['learning_rate'], item['entropy_coefficient']): item
        for item in selected['candidates']
    }
    winner = selected['winner']
    output_rows = []
    for row in sorted(rows, key=lambda item: (
            float(item['learning_rate']),
            float(item['entropy_coefficient']),
            int(item['training_seed']))):
        key = (float(row['learning_rate']), float(row['entropy_coefficient']))
        candidate = candidate_lookup[key]
        improvement = (
            float(row['tt_episode0']) - float(row['tt_late_median'])
        ) / float(row['tt_episode0'])
        seed_eligible = (
            float(row['tt_late_median']) < float(row['tt_fixedtime'])
            and improvement >= 0.10
        )
        output_rows.append({
            'learning_rate': key[0], 'entropy_coefficient': key[1],
            'training_seed': int(row['training_seed']),
            'logical_run_id': row['logical_run_id'], 'status': row['status'],
            'tt_episode0': float(row['tt_episode0']),
            'tt_late_median': float(row['tt_late_median']),
            'tt_fixedtime': float(row['tt_fixedtime']),
            'improvement_fraction': improvement,
            'current_adaptation_aulc': float(
                row['current_adaptation_aulc']
            ),
            'candidate_score': candidate['score'],
            'candidate_iqr': candidate['iqr'],
            'candidate_aulc_median': candidate['aulc_median'],
            'candidate_rank': rank[key],
            'selected': key == (
                winner['learning_rate'], winner['entropy_coefficient']
            ),
            'seed_eligible': seed_eligible,
            'candidate_eligible_seed_count': (
                winner['eligible_seed_count'] if key == (
                    winner['learning_rate'], winner['entropy_coefficient']
                ) else ''
            ),
            'formal_eligible': (
                winner['formal_eligible'] if key == (
                    winner['learning_rate'], winner['entropy_coefficient']
                ) else ''
            ),
            'run_summary_path': str(Path(row['run_summary_path']).resolve()),
            'run_summary_sha256': row['run_summary_sha256'],
        })

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix='.tmp-plan5-ppo-calibration-summary-',
        dir=summary_path.parent,
    )
    try:
        with os.fdopen(descriptor, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=TABLE_SCHEMAS['ppo_calibration_summary.csv'],
                extrasaction='raise',
            )
            writer.writeheader(); writer.writerows(output_rows)
            handle.flush(); os.fsync(handle.fileno())
        if summary_path.exists():
            if sha256_file(summary_path) != sha256_file(temporary):
                raise FileExistsError(
                    f'PPO calibration summary is immutable: {summary_path}'
                )
            os.unlink(temporary); temporary = None
        else:
            os.replace(temporary, summary_path); temporary = None
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)

    payload = {
        'schema_version': 1, 'algorithm_id': 'PPO',
        'learning_rate': float(winner['learning_rate']),
        'entropy_coefficient': float(winner['entropy_coefficient']),
        'calibration_summary_sha256': sha256_file(summary_path),
        'config_sha256': str(config_sha256),
        'formal_status': selected['formal_status'],
        'eligible_seed_count': int(winner['eligible_seed_count']),
        'selection_order': [
            'score', 'iqr', 'aulc_median',
            'learning_rate', 'entropy_coefficient',
        ],
        'calibration_matrix_digest': canonical_digest(rows),
    }
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, encoding='utf-8') as handle:
            if json.load(handle) != payload:
                raise FileExistsError(
                    f'PPO formal config is immutable: {config_path}'
                )
    else:
        atomic_json(config_path, payload)
    validate_json_fields(config_path, 'ppo_formal_config.json')
    return {
        'selection': selected,
        'summary_path': str(summary_path.resolve()),
        'summary_sha256': sha256_file(summary_path),
        'config_path': str(config_path.resolve()),
        'config_sha256': sha256_file(config_path),
        'formal_status': selected['formal_status'],
    }


def anchor_readiness(rows, algorithm_id):
    selected = [row for row in rows if row['algorithm_id'] == algorithm_id]
    if len(selected) != 20:
        return {'ready': False, 'reason': 'anchor matrix is not 20 runs'}
    common = all(
        bool(row.get(field)) for row in selected
        for field in (
            'completed', 'finite_metrics', 'checkpoint_valid',
            'resume_valid', 'evaluation_isolation_valid',
            'fixed_probe_valid', 'late_metrics_present',
        )
    )
    if not common:
        return {'ready': False, 'reason': 'common anchor gate failed'}
    if algorithm_id == 'PPO':
        counts = {}
        for row in selected:
            passed = (
                float(row['tt_late']) < float(row['tt_fixedtime'])
                and (float(row['tt_episode0']) - float(row['tt_late']))
                / float(row['tt_episode0']) >= 0.10
            )
            counts[row['scene']] = counts.get(row['scene'], 0) + int(passed)
        if set(counts) != {'S1', 'S2', 'S3', 'S4'} \
                or any(value < 4 for value in counts.values()):
            return {
                'ready': False, 'reason': 'PPO per-scene 4/5 gate failed',
                'passing_by_scene': counts,
            }
    return {'ready': True, 'run_count': 20}


def cross_algorithm_cluster_bootstrap(effects_by_algorithm, resamples=10000,
                                      seed=20260827):
    """Resample complete seed clusters and compare two algorithms."""
    if set(effects_by_algorithm) != {'DDQN', 'CTXDDQN'}:
        raise ValueError('Cross-algorithm comparison is DDQN vs CTXDDQN only')
    seeds = set(effects_by_algorithm['DDQN'])
    if seeds != set(range(5)) or set(effects_by_algorithm['CTXDDQN']) != seeds:
        raise ValueError('Cross-algorithm bootstrap requires seeds 0..4')
    per_seed = np.asarray([
        float(effects_by_algorithm['CTXDDQN'][seed])
        - float(effects_by_algorithm['DDQN'][seed])
        for seed in range(5)
    ])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, 5, size=(int(resamples), 5))
    sampled = per_seed[draws].mean(axis=1)
    lo, hi = np.percentile(sampled, [2.5, 97.5])
    return {
        'per_seed_effect_difference': per_seed.tolist(),
        'mean': float(per_seed.mean()), 'median': float(np.median(per_seed)),
        'seed_consistency_positive': float(np.mean(per_seed > 0)),
        'ci95': [float(lo), float(hi)],
        'bootstrap_resamples': int(resamples), 'analysis_seed': int(seed),
    }
