import csv
import ast
import json
import math
from pathlib import Path

import numpy as np
from itertools import combinations, product


PROBE_FEATURE_NAMES = tuple(
    [f'lane_count_{index}' for index in range(8)]
    + [f'phase_{index}_active' for index in range(8)]
)


CORE_FIELDS = (
    "schema_version", "record_type", "episode", "simulation_step",
    "decision_step", "global_decision_step", "gradient_updates",
    "travel_time", "reward_mean", "reward_sum", "queue", "delay",
    "real_delay", "throughput", "loss_mean", "epsilon",
    "wall_time_seconds", "waiting_time", "unfinished_vehicles",
    "phase_switches", "phase_switch_frequency", "replay_size",
    "replay_capacity", "target_updates",
)
AUC_METRICS = (
    "travel_time", "queue", "delay", "real_delay", "reward_mean", "loss_mean",
)
AUC_FIELDS = (
    "run_key", "role", "agent", "network", "training_seed", "run_dir",
    "curve_source", "metric", "episode_start", "episode_end",
    "point_count", "auc",
)
LEARNING_SPEED_FIELDS = (
    "run_key", "role", "agent", "network", "training_seed", "run_dir",
    "curve_source", "reference", "criterion", "threshold", "status",
    "episode",
)
EFFICIENCY_FIELDS = (
    "run_key", "role", "agent", "network", "training_seed", "run_dir",
    "episode", "travel_time", "environment_transitions", "gradient_updates",
    "cumulative_wall_time_seconds", "replay_fill_fraction",
    "update_to_data_ratio",
)
AULC_SUMMARY_FIELDS = (
    "run_key", "role", "agent", "network", "training_seed", "run_dir",
    "curve", "point_count", "x_start", "x_end", "aulc",
)


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _run_columns(spec):
    return {
        "run_key": spec.run_key,
        "role": spec.role,
        "agent": spec.agent,
        "network": spec.network,
        "training_seed": spec.training_seed,
        "run_dir": str(spec.run_dir),
    }


def normalize_records(validated_runs):
    metric_rows = []
    action_rows = []
    for run in validated_runs:
        spec = run["spec"]
        identity = _run_columns(spec)
        for record in run["records"]:
            row = dict(identity)
            row.update({field: record.get(field) for field in CORE_FIELDS})
            row["action_distribution_json"] = json.dumps(
                record.get("action_distribution") or {}, sort_keys=True,
                separators=(",", ":"),
            )
            metric_rows.append(row)
            for action, fraction in sorted(
                (record.get("action_distribution") or {}).items(),
                key=lambda item: int(item[0]),
            ):
                action_rows.append({
                    **identity,
                    "record_type": record["record_type"],
                    "episode": record["episode"],
                    "action": int(action),
                    "fraction": fraction,
                })
    return metric_rows, action_rows


def _candidate_records(rows, run_key):
    records = [row for row in rows if row["run_key"] == run_key]
    evaluations = [
        row for row in records
        if row["record_type"] in {"EVALUATION", "FINAL_EVALUATION"}
    ]
    return evaluations or records


def build_run_summaries(validated_runs, metric_rows):
    summaries = []
    comparisons = []
    for run in validated_runs:
        spec = run["spec"]
        candidates = _candidate_records(metric_rows, spec.run_key)
        final = max(candidates, key=lambda row: (row["episode"], row["record_type"] == "FINAL_EVALUATION"))
        best = min(
            (row for row in candidates if _finite(row.get("travel_time"))),
            key=lambda row: (row["travel_time"], row["episode"]),
        )
        validation = run.get("trajectory_validation") or {}
        summaries.append({
            **_run_columns(spec),
            "record_count": len(run["records"]),
            "metric_schema_versions": ",".join(
                str(value) for value in sorted({r["schema_version"] for r in run["records"]})
            ),
            "config_sha256": run["resolved_config_sha256"],
            "trajectory_valid": validation.get("valid"),
            "trajectory_episodes": validation.get("episode_count"),
            "trajectory_transitions": validation.get("transition_count"),
            "evaluation_transition_count": validation.get("evaluation_transition_count"),
            "final_episode": final["episode"],
            "final_travel_time": final.get("travel_time"),
            "best_episode": best["episode"],
            "best_travel_time": best.get("travel_time"),
        })
        for label, row in (("final", final), ("best", best)):
            comparisons.append({
                **_run_columns(spec),
                "selection": label,
                "episode": row["episode"],
                "travel_time": row.get("travel_time"),
                "queue": row.get("queue"),
                "delay": row.get("delay"),
                "real_delay": row.get("real_delay"),
                "throughput": row.get("throughput"),
            })
    return summaries, comparisons


def calculate_first_100_auc(metric_rows):
    rows = []
    run_keys = list(dict.fromkeys(row["run_key"] for row in metric_rows))
    for run_key in run_keys:
        run_rows = [row for row in metric_rows if row["run_key"] == run_key]
        for curve_source, record_types, minimum_episode in (
            ("TRAIN", {"TRAIN"}, 1),
            ("EVALUATION", {"EVALUATION", "FINAL_EVALUATION"}, 0),
        ):
            selected = [
                row for row in run_rows
                if row["record_type"] in record_types
                and minimum_episode <= row["episode"] <= 100
            ]
            if not selected:
                continue
            identity = {key: selected[0][key] for key in (
                "run_key", "role", "agent", "network",
                "training_seed", "run_dir",
            )}
            for metric in AUC_METRICS:
                points = sorted(
                    (row["episode"], row.get(metric))
                    for row in selected if _finite(row.get(metric))
                )
                if not points:
                    continue
                episodes = np.asarray([point[0] for point in points], dtype=float)
                values = np.asarray([point[1] for point in points], dtype=float)
                auc = (
                    float(np.trapz(values, episodes))
                    if len(points) > 1 else float(values[0])
                )
                rows.append({
                    **identity,
                    "curve_source": curve_source,
                    "metric": metric,
                    "episode_start": int(episodes[0]),
                    "episode_end": int(episodes[-1]),
                    "point_count": len(points),
                    "auc": auc,
                })
    return rows


def _first_attainment(points, threshold, consecutive):
    streak = []
    previous_episode = None
    for episode, value in points:
        if value <= threshold:
            if previous_episode is None or episode == previous_episode + 1:
                streak.append(episode)
            else:
                streak = [episode]
            if len(streak) >= consecutive:
                return streak[0]
        else:
            streak = []
        previous_episode = episode
    return None


def calculate_learning_speed(metric_rows):
    baseline_levels = {}
    for row in metric_rows:
        if (
            row["role"] == "baseline"
            and row["agent"] in {"fixedtime", "maxpressure"}
            and _finite(row.get("travel_time"))
        ):
            baseline_levels[(row["network"], row["agent"])] = row["travel_time"]

    results = []
    run_keys = list(dict.fromkeys(
        row["run_key"] for row in metric_rows
        if row["agent"] == "dqn" and row["role"] != "baseline"
    ))
    for run_key in run_keys:
        run_rows = [row for row in metric_rows if row["run_key"] == run_key]
        evaluations = [
            row for row in run_rows
            if row["record_type"] in {"EVALUATION", "FINAL_EVALUATION"}
            and _finite(row.get("travel_time"))
        ]
        if not evaluations:
            continue
        final = max(evaluations, key=lambda row: row["episode"])
        references = [
            ("final_110_percent", final["travel_time"] * 1.10),
        ]
        for agent in ("fixedtime", "maxpressure"):
            level = baseline_levels.get((final["network"], agent))
            if level is not None:
                references.append((agent, level))
        identity = {key: final[key] for key in (
            "run_key", "role", "agent", "network",
            "training_seed", "run_dir",
        )}
        for curve_source, record_types in (
            ("TRAIN", {"TRAIN"}),
            ("EVALUATION", {"EVALUATION", "FINAL_EVALUATION"}),
        ):
            points = sorted(
                (row["episode"], row["travel_time"])
                for row in run_rows
                if row["record_type"] in record_types
                and _finite(row.get("travel_time"))
            )
            for reference, threshold in references:
                for criterion, consecutive in (("single", 1), ("consecutive_5", 5)):
                    episode = _first_attainment(points, threshold, consecutive)
                    results.append({
                        **identity,
                        "curve_source": curve_source,
                        "reference": reference,
                        "criterion": criterion,
                        "threshold": threshold,
                        "status": "reached" if episode is not None else "not_reached",
                        "episode": episode,
                    })
    return results


DIAGNOSTIC_METRICS = (
    'travel_time', 'queue', 'delay', 'real_delay', 'throughput', 'reward_mean',
)
RESUMABLE_EPISODES = (0, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400)


def _stable_denominator(values, reference):
    finite = np.abs(np.asarray([value for value in values if _finite(value)], dtype=float))
    scale = float(np.median(finite)) if finite.size else 0.0
    return max(abs(float(reference)), 0.01 * scale, 1e-8)


def calculate_convergence(metric_rows, scene_by_network, tolerances=(0.05, 0.03)):
    """Quantify stable convergence on evaluation rows without simulator reruns."""
    per_seed = []
    curves = []
    run_keys = list(dict.fromkeys(row['run_key'] for row in metric_rows))
    for run_key in run_keys:
        rows = sorted((
            row for row in metric_rows
            if row['run_key'] == run_key
            and row['record_type'] in {'EVALUATION', 'FINAL_EVALUATION'}
        ), key=lambda row: row['episode'])
        if not rows or rows[0]['role'] != 'formal':
            continue
        identity = {key: rows[0][key] for key in (
            'run_key', 'network', 'training_seed', 'run_dir',
        )}
        identity['scene'] = scene_by_network[identity['network']]
        for metric in DIAGNOSTIC_METRICS:
            points = [(row['episode'], row.get(metric)) for row in rows if _finite(row.get(metric))]
            if len(points) < 20:
                continue
            episodes = [item[0] for item in points]
            values = [float(item[1]) for item in points]
            final_reference = float(np.mean(values[-20:]))
            denominator = _stable_denominator(values, final_reference)
            moving = []
            for index in range(len(values)):
                moving.append(
                    None if index < 9 else float(np.mean(values[index - 9:index + 1]))
                )
                curves.append({
                    **identity, 'metric': metric, 'episode': episodes[index],
                    'value': values[index], 'moving_average_10': moving[-1],
                    'final_reference_20': final_reference,
                    'stable_denominator': denominator,
                })
            for tolerance in tolerances:
                converged = None
                for index in range(9, len(values) - 19):
                    window = moving[index:index + 20]
                    if all(
                        value is not None
                        and abs(value - final_reference) / denominator <= tolerance
                        for value in window
                    ):
                        converged = episodes[index]
                        break
                per_seed.append({
                    **identity, 'metric': metric, 'tolerance': tolerance,
                    'moving_average_window': 10, 'stable_window': 20,
                    'final_reference_window': 20,
                    'final_reference': final_reference,
                    'stable_denominator': denominator,
                    'status': 'converged' if converged is not None else 'not_converged',
                    'convergence_episode': converged,
                })
    return per_seed, curves


def summarize_convergence(per_seed_rows):
    summaries = []
    groups = {}
    for row in per_seed_rows:
        groups.setdefault((row['scene'], row['network'], row['metric'], row['tolerance']), []).append(row)
    for (scene, network, metric, tolerance), rows in groups.items():
        values = sorted(float(row['convergence_episode']) for row in rows if row['convergence_episode'] is not None)
        summaries.append({
            'scene': scene, 'network': network, 'metric': metric,
            'tolerance': tolerance, 'seed_count': len(rows),
            'converged_seed_count': len(values),
            'median': None if not values else float(np.median(values)),
            'mean': None if not values else float(np.mean(values)),
            'sample_sd': None if len(values) < 2 else float(np.std(values, ddof=1)),
            'q1': None if not values else float(np.percentile(values, 25)),
            'q3': None if not values else float(np.percentile(values, 75)),
            'iqr': None if not values else float(np.percentile(values, 75) - np.percentile(values, 25)),
            'maximum': None if not values else float(max(values)),
        })
    return summaries


def calculate_budget_sufficiency(metric_rows, scene_by_network, budgets=(10,20,30,50,75,100,200,400)):
    rows = []
    evaluations = [row for row in metric_rows if row['role'] == 'formal' and row['record_type'] in {'EVALUATION','FINAL_EVALUATION'}]
    for run_key in dict.fromkeys(row['run_key'] for row in evaluations):
        run = sorted((row for row in evaluations if row['run_key'] == run_key), key=lambda row: row['episode'])
        identity = {key: run[0][key] for key in ('run_key','network','training_seed','run_dir')}
        identity['scene'] = scene_by_network[identity['network']]
        for metric in DIAGNOSTIC_METRICS:
            values = [float(row[metric]) for row in run if _finite(row.get(metric))]
            if len(values) < 20:
                continue
            reference = float(np.mean(values[-20:]))
            denominator = _stable_denominator(values, reference)
            by_episode = {row['episode']: row.get(metric) for row in run}
            for budget in budgets:
                value = by_episode.get(budget)
                if not _finite(value):
                    continue
                rows.append({
                    **identity, 'metric': metric, 'episode_budget': budget,
                    'value': float(value), 'final_reference_20': reference,
                    'absolute_relative_gap': abs(float(value) - reference) / denominator,
                })
    return rows


def recommend_resumable_budget(per_seed_rows):
    primary = [row for row in per_seed_rows if row['metric'] == 'travel_time' and row['tolerance'] == 0.05]
    required = []
    for scene in dict.fromkeys(row['scene'] for row in primary):
        rows = [row for row in primary if row['scene'] == scene]
        values = sorted(row['convergence_episode'] for row in rows if row['convergence_episode'] is not None)
        if len(values) < math.ceil(0.9 * len(rows)):
            required.append(400)
        else:
            required.append(values[math.ceil(0.9 * len(rows)) - 1])
    raw = max(required) if required else 400
    aligned = next((value for value in RESUMABLE_EPISODES if value >= raw), 400)
    return {'raw_required_episode': raw, 'recommended_resumable_episode': aligned}


def _distribution(row, actions=8):
    data = json.loads(row['action_distribution_json'])
    return np.asarray([float(data.get(str(action), 0.0)) for action in range(actions)])


def _js_distance(left, right):
    midpoint = 0.5 * (left + right)
    def kl(first, second):
        mask = first > 0
        return float(np.sum(first[mask] * np.log2(first[mask] / second[mask])))
    return math.sqrt(max(0.0, 0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)))


def calculate_action_distribution_diagnostics(metric_rows, scene_by_network, action_semantics):
    final_rows = []
    for run_key in dict.fromkeys(row['run_key'] for row in metric_rows):
        candidates = [row for row in metric_rows if row['run_key'] == run_key and row['role'] == 'formal' and row['record_type'] == 'FINAL_EVALUATION']
        if candidates:
            final_rows.append(candidates[-1])
    per_seed = []
    vectors = {}
    for row in final_rows:
        vector = _distribution(row)
        key = (scene_by_network[row['network']], int(row['training_seed']))
        vectors[key] = vector
        for action, fraction in enumerate(vector):
            per_seed.append({
                'scene': key[0], 'network': row['network'],
                'training_seed': key[1], 'action': action,
                'action_semantics': action_semantics[action], 'fraction': fraction,
            })
    summary = []
    for scene in dict.fromkeys(row['scene'] for row in per_seed):
        network = next(row['network'] for row in per_seed if row['scene'] == scene)
        for action in range(8):
            values = [row['fraction'] for row in per_seed if row['scene'] == scene and row['action'] == action]
            summary.append({
                'scene': scene, 'network': network, 'action': action,
                'action_semantics': action_semantics[action],
                'mean': float(np.mean(values)),
                'sample_sd': float(np.std(values, ddof=1)),
            })
    distances = []
    keys = list(vectors)
    for left, right in combinations(keys, 2):
        lv, rv = vectors[left], vectors[right]
        pair_kind = 'within_scene' if left[0] == right[0] else 'between_scene'
        for metric, value in (
            ('total_variation', 0.5 * float(np.abs(lv-rv).sum())),
            ('jensen_shannon', _js_distance(lv, rv)),
        ):
            distances.append({
                'aggregation': 'model_pair', 'pair_kind': pair_kind,
                'metric': metric, 'source_scene': left[0], 'source_seed': left[1],
                'target_scene': right[0], 'target_seed': right[1], 'distance': value,
            })
    scenes = list(dict.fromkeys(row['scene'] for row in per_seed))
    means = {scene: np.mean([vector for (name,_),vector in vectors.items() if name == scene], axis=0) for scene in scenes}
    for left, right in product(scenes, scenes):
        lv, rv = means[left], means[right]
        for metric, value in (
            ('total_variation', 0.5 * float(np.abs(lv-rv).sum())),
            ('jensen_shannon', _js_distance(lv, rv)),
        ):
            distances.append({
                'aggregation': 'scene_mean', 'pair_kind': 'same_scene' if left == right else 'between_scene',
                'metric': metric, 'source_scene': left, 'source_seed': None,
                'target_scene': right, 'target_seed': None, 'distance': value,
            })
    return per_seed, summary, distances


CROSS_SCENE_METRIC_DIRECTIONS = {
    'travel_time': 'lower', 'queue': 'lower', 'real_delay': 'lower',
    'delay': 'lower', 'throughput': 'higher', 'reward_mean': 'higher',
}


def normalize_cross_scene_summaries(summary_rows, scene_by_network):
    numeric = (
        'training_seed', 'evaluation_seed', 'evaluation_traffic_seed',
        'checkpoint_episode', 'simulation_steps', 'decision_steps',
        'travel_time', 'reward_mean', 'queue', 'delay', 'real_delay',
        'throughput', 'waiting_time', 'unfinished_vehicles',
        'wall_time_seconds', 'phase_switches', 'phase_switch_frequency',
    )
    rows = []
    for source in summary_rows:
        row = dict(source)
        for field in numeric:
            value = row.get(field)
            if value not in (None, ''):
                row[field] = float(value)
        row['training_seed'] = int(row['training_seed'])
        row['evaluation_traffic_seed'] = int(row['evaluation_traffic_seed'])
        row['source_scene'] = scene_by_network[row['source_network']]
        row['target_scene'] = scene_by_network[row['target_network']]
        row['approximate_delay'] = row.pop('delay')
        row['reward_definition'] = str(row['reward_definition'])
        for field in ('action_distribution', 'isolation_check'):
            if isinstance(row.get(field), str):
                try:
                    row[field] = json.loads(row[field])
                except json.JSONDecodeError:
                    # csv.DictWriter stringifies nested legacy values with
                    # Python literals; parse without executing code.
                    row[field] = ast.literal_eval(row[field])
        rows.append(row)
    return rows


def aggregate_cross_scene(raw_rows):
    metrics = (
        'travel_time', 'queue', 'real_delay', 'approximate_delay',
        'throughput', 'phase_switch_frequency', 'reward_mean',
    )
    grouped = {}
    for row in raw_rows:
        key = (row['source_scene'], row['source_network'],
               row['target_scene'], row['target_network'])
        grouped.setdefault(key, []).append(row)
    summary = []
    for key, rows in grouped.items():
        for metric in metrics:
            by_training_seed = {}
            for row in rows:
                by_training_seed.setdefault(int(row['training_seed']), []).append(
                    float(row[metric])
                )
            values = [float(np.mean(seed_values))
                      for seed_values in by_training_seed.values()]
            traffic_counts = [len(seed_values)
                              for seed_values in by_training_seed.values()]
            if len(set(traffic_counts)) != 1:
                raise ValueError(
                    f'Unequal evaluation traffic-seed counts for {key}, {metric}'
                )
            summary.append({
                'source_scene': key[0], 'source_network': key[1],
                'target_scene': key[2], 'target_network': key[3],
                'metric': metric, 'training_seed_count': len(values),
                'evaluation_traffic_seed_count_per_training_seed': traffic_counts[0],
                'mean': float(np.mean(values)),
                'sample_sd': float(np.std(values, ddof=1)),
                'reward_definition': (
                    rows[0]['reward_definition'] if metric == 'reward_mean' else ''
                ),
            })
    references = {
        (row['target_scene'], row['metric']): row['mean']
        for row in summary if row['source_scene'] == row['target_scene']
    }
    relative = []
    for row in summary:
        metric = row['metric']
        if metric == 'phase_switch_frequency':
            continue
        reference = references[(row['target_scene'], metric)]
        denominator = max(abs(reference), 1e-8)
        direction = CROSS_SCENE_METRIC_DIRECTIONS.get(
            'delay' if metric == 'approximate_delay' else metric
        )
        absolute_difference = (
            row['mean'] - reference
            if direction == 'lower'
            else reference - row['mean']
        )
        degradation = absolute_difference / denominator
        relative.append({
            **{key: row[key] for key in (
                'source_scene','source_network','target_scene','target_network',
                'metric','training_seed_count','reward_definition',
            )},
            'direction': direction, 'target_in_domain_reference': reference,
            'absolute_difference': absolute_difference,
            'relative_denominator': denominator,
            'near_zero_reference': abs(reference) <= 1e-8,
            'relative_degradation': degradation,
        })
    directional = []
    travel = [row for row in relative if row['metric'] == 'travel_time'
              and row['source_scene'] != row['target_scene']]
    by_direction = {(row['source_scene'], row['target_scene']): row for row in travel}
    for row in travel:
        reverse = by_direction[(row['target_scene'], row['source_scene'])]
        directional.append({
            'source_scene': row['source_scene'],
            'source_network': row['source_network'],
            'target_scene': row['target_scene'],
            'target_network': row['target_network'],
            'travel_time_relative_degradation': row['relative_degradation'],
            'reverse_relative_degradation': reverse['relative_degradation'],
            'directional_asymmetry': (
                row['relative_degradation'] - reverse['relative_degradation']
            ),
        })
    directional.sort(key=lambda row: row['travel_time_relative_degradation'])
    return summary, relative, directional


def aggregate_episode100_pair_gate(raw_rows, directions):
    """Aggregate the targeted G0 gate with same-seed target references."""
    required_directions = {tuple(item) for item in directions}
    target_scenes = {target for _, target in required_directions}
    by_identity = {
        (row['source_scene'], row['target_scene'], int(row['training_seed'])): row
        for row in raw_rows
    }
    relative = []
    metric_directions = {
        'travel_time': 'lower', 'queue': 'lower', 'real_delay': 'lower',
        'approximate_delay': 'lower', 'throughput': 'higher',
        'unfinished_vehicles': 'lower',
    }
    for source, target in sorted(required_directions):
        for seed in range(5):
            observed = by_identity.get((source, target, seed))
            reference = by_identity.get((target, target, seed))
            if observed is None or reference is None:
                raise ValueError(
                    f'Missing same-seed G0 comparison: {source}->{target} seed {seed}'
                )
            for metric, direction in metric_directions.items():
                value = float(observed[metric])
                baseline = float(reference[metric])
                degradation = (
                    value - baseline if direction == 'lower' else baseline - value
                ) / max(abs(baseline), 1e-8)
                relative.append({
                    'source_scene': source, 'source_network': observed['source_network'],
                    'target_scene': target, 'target_network': observed['target_network'],
                    'training_seed': seed, 'metric': metric, 'direction': direction,
                    'value': value, 'same_seed_target_reference': baseline,
                    'absolute_difference': (
                        value - baseline if direction == 'lower' else baseline - value
                    ),
                    'relative_degradation': degradation,
                })
    summary = []
    groups = {}
    for row in relative:
        key = (row['source_scene'], row['source_network'],
               row['target_scene'], row['target_network'], row['metric'])
        groups.setdefault(key, []).append(row)
    for key, rows in sorted(groups.items()):
        values = [row['relative_degradation'] for row in rows]
        raw_values = [row['value'] for row in rows]
        references = [row['same_seed_target_reference'] for row in rows]
        summary.append({
            'source_scene': key[0], 'source_network': key[1],
            'target_scene': key[2], 'target_network': key[3],
            'metric': key[4], 'training_seed_count': len(rows),
            'mean_value': float(np.mean(raw_values)),
            'sample_sd_value': float(np.std(raw_values, ddof=1)),
            'mean_same_seed_target_reference': float(np.mean(references)),
            'sample_sd_same_seed_target_reference': float(np.std(references, ddof=1)),
            'mean_relative_degradation': float(np.mean(values)),
            'sample_sd_relative_degradation': float(np.std(values, ddof=1)),
        })
    expected_raw = len(required_directions) * 5 + len(target_scenes) * 5
    if len(raw_rows) != expected_raw:
        raise ValueError(
            f'G0 requires exactly {expected_raw} evaluations, got {len(raw_rows)}'
        )
    travel = {
        (row['source_scene'], row['target_scene']): row
        for row in summary if row['metric'] == 'travel_time'
    }
    low = [travel[('S2', 'S3')], travel[('S3', 'S2')]]
    high = [travel[('S3', 'S4')], travel[('S4', 'S3')]]
    low_worst = max(row['mean_relative_degradation'] for row in low)
    high_worst = max(row['mean_relative_degradation'] for row in high)
    seed_high_worst = []
    for seed in range(5):
        seed_rows = [row for row in relative if row['metric'] == 'travel_time'
                     and (row['source_scene'], row['target_scene'])
                     in {('S3', 'S4'), ('S4', 'S3')}
                     and row['training_seed'] == seed]
        seed_high_worst.append(max(row['relative_degradation'] for row in seed_rows))
    checks = {
        'low_both_mean_degradation_le_5pct': all(
            row['mean_relative_degradation'] <= .05 for row in low
        ),
        'high_direction_exceeds_low_worst': high_worst > low_worst,
        'high_worst_exceeds_low_worst_by_5pp': high_worst - low_worst >= .05,
        'not_driven_by_single_seed': sum(
            value > low_worst for value in seed_high_worst
        ) >= 3,
    }
    audit = {
        'passed': all(checks.values()), 'checks': checks,
        'low_worst_mean_degradation': low_worst,
        'high_worst_mean_degradation': high_worst,
        'high_minus_low_worst_percentage_points': 100 * (high_worst - low_worst),
        'high_worst_per_seed': seed_high_worst,
    }
    return summary, relative, audit


def calculate_policy_disagreement(prediction_rows):
    """Compare final DQN policies on a shared, common-policy probe set."""
    by_model = {}
    model_identity = {}
    for row in prediction_rows:
        model_id = row['model_id']
        by_model.setdefault(model_id, []).append(row)
        model_identity[model_id] = (row['model_scene'], row['model_network'],
                                    int(row['training_seed']))
    for rows in by_model.values():
        rows.sort(key=lambda row: row['probe_id'])
    models = sorted(by_model, key=lambda value: (
        int(model_identity[value][0][1:]), model_identity[value][2]
    ))
    probe_ids = [row['probe_id'] for row in by_model[models[0]]]
    if any([row['probe_id'] for row in by_model[model]] != probe_ids for model in models):
        raise ValueError('Policy predictions are not aligned to identical probe states')
    # The scale-normalized threshold is fixed before pair comparison so a
    # model pair cannot make its own disagreement set easier or harder.
    confidence_threshold = 0.10
    probe_sources = ['ALL'] + sorted({
        row['probe_source_scene'] for row in by_model[models[0]]
    }, key=lambda value: int(value[1:]))
    model_rows = []
    for left_index, left in enumerate(models):
        for right_index in range(left_index + 1, len(models)):
            right = models[right_index]
            left_rows, right_rows = by_model[left], by_model[right]
            for source in probe_sources:
                indices = [
                    index for index, row in enumerate(left_rows)
                    if source == 'ALL' or row['probe_source_scene'] == source
                ]
                left_actions = np.asarray([left_rows[index]['top_action'] for index in indices])
                right_actions = np.asarray([right_rows[index]['top_action'] for index in indices])
                disagreements = left_actions != right_actions
                confident = np.asarray([
                    left_rows[index]['normalized_q_margin'] >= confidence_threshold
                    and right_rows[index]['normalized_q_margin'] >= confidence_threshold
                    for index in indices
                ])
                cosine = []
                for index in indices:
                    lq = np.asarray(left_rows[index]['q_values'], dtype=float)
                    rq = np.asarray(right_rows[index]['q_values'], dtype=float)
                    denominator = np.linalg.norm(lq) * np.linalg.norm(rq)
                    cosine.append(0.0 if denominator <= 1e-12 else
                                  1.0 - float(np.dot(lq, rq) / denominator))
                left_scene, left_network, left_seed = model_identity[left]
                right_scene, right_network, right_seed = model_identity[right]
                model_rows.append({
                    'aggregation': 'model_pair', 'probe_source_scene': source,
                    'left_model': left, 'left_scene': left_scene,
                    'left_network': left_network, 'left_training_seed': left_seed,
                    'right_model': right, 'right_scene': right_scene,
                    'right_network': right_network, 'right_training_seed': right_seed,
                    'pair_kind': ('within_scene' if left_scene == right_scene else 'between_scene'),
                    'probe_state_count': len(indices),
                    'action_agreement': float(np.mean(~disagreements)),
                    'action_disagreement': float(np.mean(disagreements)),
                    'confidence_threshold': confidence_threshold,
                    'high_confidence_state_count': int(confident.sum()),
                    'high_confidence_disagreement': (
                        float(np.mean(disagreements[confident])) if confident.any() else None
                    ),
                    'q_cosine_distance': float(np.mean(cosine)),
                })
    scene_rows = []
    scenes = sorted({value[0] for value in model_identity.values()},
                    key=lambda value: int(value[1:]))
    for source in probe_sources:
        candidates = [row for row in model_rows if row['probe_source_scene'] == source]
        for left_index, left_scene in enumerate(scenes):
            for right_scene in scenes[left_index:]:
                selected = [row for row in candidates if {row['left_scene'], row['right_scene']}
                            == {left_scene, right_scene}]
                if left_scene == right_scene:
                    selected = [row for row in selected if row['left_scene'] == left_scene
                                and row['right_scene'] == right_scene]
                if not selected:
                    continue
                for metric in ('action_agreement', 'action_disagreement',
                               'high_confidence_disagreement', 'q_cosine_distance'):
                    values = [row[metric] for row in selected if row[metric] is not None]
                    value = float(np.mean(values)) if values else None
                    directions = (
                        ((left_scene, right_scene),) if left_scene == right_scene
                        else ((left_scene, right_scene), (right_scene, left_scene))
                    )
                    for source_scene, target_scene in directions:
                        scene_rows.append({
                            'aggregation': 'scene_pair', 'probe_source_scene': source,
                            'source_scene': source_scene, 'target_scene': target_scene,
                            'metric': metric, 'model_pair_count': len(selected),
                            'mean': value, 'sample_sd_across_model_pairs': (
                                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                            ), 'confidence_threshold': confidence_threshold,
                        })
    return model_rows, scene_rows, confidence_threshold


def calculate_state_distribution_diagnostics(probe_rows):
    """Feature summaries and standardized marginal Wasserstein distances."""
    from scipy.stats import wasserstein_distance

    features = np.asarray([row['model_input'] for row in probe_rows], dtype=float)
    scenes = np.asarray([row['scene'] for row in probe_rows])
    names = PROBE_FEATURE_NAMES
    statistics = []
    for scene in sorted(set(scenes), key=lambda value: int(value[1:])):
        values = features[scenes == scene]
        for index, name in enumerate(names):
            column = values[:, index]
            statistics.append({
                'scene': scene, 'feature_index': index, 'feature': name,
                'count': len(column), 'mean': float(np.mean(column)),
                'sample_sd': float(np.std(column, ddof=1)),
                'q05': float(np.quantile(column, .05)),
                'q25': float(np.quantile(column, .25)),
                'median': float(np.quantile(column, .5)),
                'q75': float(np.quantile(column, .75)),
                'q95': float(np.quantile(column, .95)),
            })
    feature_ranges = {}
    for index, name in enumerate(names):
        scene_means = [row['mean'] for row in statistics if row['feature_index'] == index]
        feature_ranges[name] = float(max(scene_means) - min(scene_means))
    ranked = sorted(feature_ranges, key=feature_ranges.get, reverse=True)
    ranks = {name: rank for rank, name in enumerate(ranked, start=1)}
    for row in statistics:
        row['between_scene_mean_range'] = feature_ranges[row['feature']]
        row['difference_rank'] = ranks[row['feature']]
    scale = np.std(features, axis=0, ddof=1)
    scale[scale <= 1e-12] = 1.0
    standardized = features / scale
    distances = []
    ordered = sorted(set(scenes), key=lambda value: int(value[1:]))
    for left in ordered:
        for right in ordered:
            per_feature = [wasserstein_distance(
                standardized[scenes == left, index],
                standardized[scenes == right, index],
            ) for index in range(features.shape[1])]
            distances.append({
                'source_scene': left, 'target_scene': right,
                'distance': float(np.mean(per_feature)),
                'maximum_feature_distance': float(np.max(per_feature)),
                'feature_distance_json': json.dumps(dict(zip(names, per_feature)),
                                                    separators=(',', ':')),
            })
    return statistics, distances


def select_transition_pairs(relative_rows, action_distance_rows,
                            policy_rows, state_distance_rows):
    """Select evidence-consistent pairs without collapsing evidence to a score."""
    travel = {(row['source_scene'], row['target_scene']): row['relative_degradation']
              for row in relative_rows if row['metric'] == 'travel_time'
              and row['source_scene'] != row['target_scene']}
    action = {}
    for row in action_distance_rows:
        if (row['aggregation'] == 'scene_mean'
                and row['metric'] == 'total_variation'
                and row['source_scene'] != row['target_scene']):
            action[frozenset((row['source_scene'], row['target_scene']))] = row['distance']
    within_action_by_scene = {}
    for row in action_distance_rows:
        if (row['aggregation'] == 'model_pair'
                and row['pair_kind'] == 'within_scene'
                and row['metric'] == 'total_variation'):
            within_action_by_scene.setdefault(row['source_scene'], []).append(
                row['distance']
            )
    disagreement_all = {
        frozenset((row['source_scene'], row['target_scene'])): row['mean']
        for row in policy_rows if row['aggregation'] == 'scene_pair'
        and row['probe_source_scene'] == 'ALL' and row['metric'] == 'action_disagreement'
    }
    high_confidence_all = {
        frozenset((row['source_scene'], row['target_scene'])): row['mean']
        for row in policy_rows if row['aggregation'] == 'scene_pair'
        and row['probe_source_scene'] == 'ALL'
        and row['metric'] == 'high_confidence_disagreement'
    }
    state = {frozenset((row['source_scene'], row['target_scene'])): row['distance']
             for row in state_distance_rows if row['source_scene'] != row['target_scene']}
    pairs = sorted(action, key=lambda pair: sorted(pair))
    evidence = []
    for pair in pairs:
        left, right = sorted(pair, key=lambda value: int(value[1:]))
        within_disagreement = np.mean([
            disagreement_all[frozenset((left,))],
            disagreement_all[frozenset((right,))],
        ])
        within_high_confidence = np.mean([
            high_confidence_all[frozenset((left,))],
            high_confidence_all[frozenset((right,))],
        ])
        action_references = [
            float(np.mean(within_action_by_scene[scene]))
            for scene in (left, right) if within_action_by_scene.get(scene)
        ]
        within_action = (
            float(np.mean(action_references)) if len(action_references) == 2
            else None
        )
        evidence.append({
            'scene_a': left, 'scene_b': right,
            'degradation_a_to_b': travel[(left, right)],
            'degradation_b_to_a': travel[(right, left)],
            'mean_directional_degradation': np.mean([travel[(left,right)], travel[(right,left)]]),
            'maximum_directional_degradation': max(travel[(left,right)], travel[(right,left)]),
            'directional_asymmetry': travel[(left,right)] - travel[(right,left)],
            'action_total_variation': action[pair],
            'mean_within_scene_action_total_variation_reference': within_action,
            'action_total_variation_excess_over_within': (
                None if within_action is None else action[pair] - within_action
            ),
            'policy_disagreement': disagreement_all[pair],
            'mean_within_scene_disagreement_reference': within_disagreement,
            'policy_disagreement_excess_over_within': (
                disagreement_all[pair] - within_disagreement
            ),
            'high_confidence_disagreement': high_confidence_all[pair],
            'mean_within_scene_high_confidence_reference': within_high_confidence,
            'high_confidence_excess_over_within': (
                high_confidence_all[pair] - within_high_confidence
            ),
            'state_distribution_distance': state[pair],
        })
    for field in ('maximum_directional_degradation', 'policy_disagreement',
                  'high_confidence_disagreement', 'action_total_variation',
                  'state_distribution_distance'):
        order = sorted(evidence, key=lambda row: row[field])
        for rank, row in enumerate(order, start=1):
            row[f'{field}_rank_low_to_high'] = rank
    high = [row for row in evidence
            if row['maximum_directional_degradation_rank_low_to_high'] >= 4
            and row['high_confidence_disagreement_rank_low_to_high'] >= 4
            and row['action_total_variation_rank_low_to_high'] >= 4
            and row['high_confidence_excess_over_within'] > 0
            and (row['action_total_variation_excess_over_within'] is None
                 or row['action_total_variation_excess_over_within'] > 0)]
    low = [row for row in evidence
           if row['maximum_directional_degradation_rank_low_to_high'] <= 3
           and row['high_confidence_disagreement_rank_low_to_high'] <= 3
           and row['action_total_variation_rank_low_to_high'] <= 3
           and row['high_confidence_excess_over_within'] <= 0
           and (row['action_total_variation_excess_over_within'] is None
                or row['action_total_variation_excess_over_within'] <= 0)]
    high_choice = max(high, key=lambda row: (
        min(row['maximum_directional_degradation_rank_low_to_high'],
            row['high_confidence_disagreement_rank_low_to_high'],
            row['action_total_variation_rank_low_to_high']),
        row['policy_disagreement_rank_low_to_high'],
    )) if high else None
    low_choice = min(low, key=lambda row: (
        max(row['maximum_directional_degradation_rank_low_to_high'],
            row['high_confidence_disagreement_rank_low_to_high'],
            row['action_total_variation_rank_low_to_high']),
        row['policy_disagreement_rank_low_to_high'],
    )) if low else None
    selected = []
    for label, choice in (('high_conflict', high_choice), ('low_conflict', low_choice)):
        if choice is None:
            continue
        for source, target in ((choice['scene_a'], choice['scene_b']),
                               (choice['scene_b'], choice['scene_a'])):
            selected.append({
                'selection': label, 'source_scene': source, 'target_scene': target,
                'travel_time_relative_degradation': travel[(source, target)],
                'reverse_relative_degradation': travel[(target, source)],
                'policy_disagreement': choice['policy_disagreement'],
                'policy_disagreement_excess_over_within': choice['policy_disagreement_excess_over_within'],
                'high_confidence_disagreement': choice['high_confidence_disagreement'],
                'high_confidence_excess_over_within': choice['high_confidence_excess_over_within'],
                'action_total_variation': choice['action_total_variation'],
                'action_total_variation_excess_over_within': choice['action_total_variation_excess_over_within'],
                'state_distribution_distance': choice['state_distribution_distance'],
                'selection_basis': 'consistent half-ranking on worst directed transfer degradation, high-confidence common-state disagreement, and action total variation; raw disagreement supplemental',
            })
    return evidence, selected


def build_efficiency_rows(metric_rows):
    """Align evaluation quality with cumulative interaction and compute cost."""
    rows = []
    run_keys = list(dict.fromkeys(
        row["run_key"] for row in metric_rows
        if row["agent"] == "dqn" and row["role"] != "baseline"
    ))
    for run_key in run_keys:
        run_rows = [row for row in metric_rows if row["run_key"] == run_key]
        training = sorted(
            (row for row in run_rows if row["record_type"] == "TRAIN"),
            key=lambda row: row["episode"],
        )
        cumulative_wall = {}
        elapsed = 0.0
        for row in training:
            if _finite(row.get("wall_time_seconds")):
                elapsed += float(row["wall_time_seconds"])
            cumulative_wall[row["episode"]] = elapsed
        evaluations = sorted(
            (
                row for row in run_rows
                if row["record_type"] in {"EVALUATION", "FINAL_EVALUATION"}
                and _finite(row.get("travel_time"))
            ),
            key=lambda row: row["episode"],
        )
        for row in evaluations:
            transitions = row.get("collected_transitions")
            if not _finite(transitions) or transitions == 0:
                transitions = row.get("global_decision_step")
            capacity = row.get("replay_capacity")
            replay_size = row.get("replay_size")
            replay_fill = (
                float(replay_size) / float(capacity)
                if _finite(replay_size) and _finite(capacity) and capacity > 0
                else None
            )
            utd = row.get("update_to_data_ratio")
            if not _finite(utd):
                updates = row.get("gradient_updates")
                utd = (
                    float(updates) / float(transitions)
                    if _finite(updates) and _finite(transitions) and transitions > 0
                    else 0.0
                )
            wall_time = cumulative_wall.get(row["episode"], 0.0)
            rows.append({
                **{key: row[key] for key in (
                    "run_key", "role", "agent", "network",
                    "training_seed", "run_dir",
                )},
                "episode": row["episode"],
                "travel_time": row["travel_time"],
                "environment_transitions": transitions,
                "gradient_updates": row.get("gradient_updates"),
                "cumulative_wall_time_seconds": wall_time,
                "replay_fill_fraction": replay_fill,
                "update_to_data_ratio": utd,
            })
    return rows


def _mean_curve_area(points):
    if not points:
        return None
    x_values = np.asarray([point[0] for point in points], dtype=float)
    y_values = np.asarray([point[1] for point in points], dtype=float)
    if len(points) == 1 or x_values[-1] == x_values[0]:
        return float(y_values[-1])
    return float(np.trapz(y_values, x_values) / (x_values[-1] - x_values[0]))


def calculate_efficiency_aulc(efficiency_rows, metric_rows):
    """Calculate raw travel-time AULC and baseline-normalized AULC."""
    baselines = {}
    for row in metric_rows:
        if (
            row["role"] == "baseline"
            and row["agent"] in {"fixedtime", "maxpressure"}
            and _finite(row.get("travel_time"))
        ):
            baselines[(row["network"], row["agent"])] = float(row["travel_time"])
    results = []
    by_run = {}
    for row in efficiency_rows:
        by_run.setdefault(row["run_key"], []).append(row)
    normalized_by_seed = {}
    for run_key, rows in by_run.items():
        rows = sorted(rows, key=lambda row: row["environment_transitions"])
        identity = {key: rows[0][key] for key in (
            "run_key", "role", "agent", "network", "training_seed", "run_dir",
        )}
        raw_points = [
            (row["environment_transitions"], row["travel_time"])
            for row in rows
            if _finite(row.get("environment_transitions"))
            and _finite(row.get("travel_time"))
        ]
        raw_aulc = _mean_curve_area(raw_points)
        if raw_aulc is not None:
            results.append({
                **identity, "curve": "raw_travel_time",
                "point_count": len(raw_points), "x_start": raw_points[0][0],
                "x_end": raw_points[-1][0], "aulc": raw_aulc,
            })
        fixed = baselines.get((identity["network"], "fixedtime"))
        pressure = baselines.get((identity["network"], "maxpressure"))
        if fixed is None or pressure is None or fixed == pressure:
            continue
        normalized_points = [
            (x_value, (fixed - travel_time) / (fixed - pressure))
            for x_value, travel_time in raw_points
        ]
        normalized_aulc = _mean_curve_area(normalized_points)
        results.append({
            **identity, "curve": "normalized_control_score",
            "point_count": len(normalized_points),
            "x_start": normalized_points[0][0], "x_end": normalized_points[-1][0],
            "aulc": normalized_aulc,
        })
        normalized_by_seed.setdefault(identity["training_seed"], []).append(
            normalized_aulc
        )
    for training_seed, values in sorted(normalized_by_seed.items()):
        results.append({
            "run_key": f"cross_scene_seed_{training_seed}", "role": "formal",
            "agent": "dqn", "network": "all_scenarios",
            "training_seed": training_seed, "run_dir": "",
            "curve": "cross_scene_normalized_control_score",
            "point_count": len(values), "x_start": None, "x_end": None,
            "aulc": float(np.mean(values)),
        })
    return results


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path
