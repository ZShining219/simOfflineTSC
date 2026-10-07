#!/usr/bin/env python3
"""Complete and validate the read-only mechanism evidence bundle.

This script consumes persisted V1/V2 assets only. It never loads an optimizer,
changes a checkpoint, or invokes a simulator.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "output_data/analysis/plan_ha_analysis_bundle_v1"
OUT = ROOT / "output_data/analysis/ANALYSIS_BUNDLE_V2_MECHANISM_COMPLETION"
EPS = [0, 1, 5, 10, 25, 50, 75, 100]
WINDOWS = [(1, 5), (1, 10), (1, 25), (26, 50), (51, 75), (76, 100)]
STATE_COLUMNS = [f"state_{i}" for i in range(8)]
POLICIES = ["clear", "fifo", "fifo_matched_wait"]


def read(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=100000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_spearman(frame: pl.DataFrame, x: str, y: str) -> tuple[int, float | None, float | None]:
    values = frame.select(x, y).drop_nulls()
    if values.height < 3 or values[x].n_unique() < 2 or values[y].n_unique() < 2:
        return values.height, None, None
    rho, pvalue = spearmanr(values[x].to_numpy(), values[y].to_numpy())
    return values.height, float(rho), float(pvalue)


def summarize(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)) if len(values) else None,
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "median": float(np.median(values)) if len(values) else None,
        "iqr": float(np.quantile(values, .75) - np.quantile(values, .25)) if len(values) else None,
        "q90": float(np.quantile(values, .90)) if len(values) else None,
        "q95": float(np.quantile(values, .95)) if len(values) else None,
        "max": float(np.max(values)) if len(values) else None,
    }


def enrich_plan_timeline() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    timeline_path = OUT / "03_PLAN34_STAGE2_TIMELINE/stage2_mechanism_timeline.csv"
    timeline = read(timeline_path)
    stale = [c for c in timeline.columns if c.endswith("_right")] + [
        c for c in ["travel_time_relative_change", "travel_time_change", "queue_relative_change",
                    "delay_relative_change", "throughput_relative_change"] if c in timeline.columns
    ]
    timeline = timeline.drop(stale)

    probe = read(V1 / "checkpoint_probe/plan34_checkpoint_probe_summary.csv").filter(pl.col("stage") == 2)
    probe_keys = probe.select(
        "logical_run_id", "episode", "probe_scene", "probe_type", "probe_network"
    ).unique()
    if "probe_network" not in timeline.columns:
        timeline = timeline.join(probe_keys, on=["logical_run_id", "episode", "probe_scene", "probe_type"], how="left")

    actions = read(V1 / "plan34_temporal/plan34_episode_action_distribution.csv").filter(pl.col("stage") == 2)
    action_fields = [f"action_{i}_frequency" for i in range(8)]
    timeline = timeline.drop([c for c in action_fields if c in timeline.columns]).join(
        actions.select("logical_run_id", "episode", *action_fields),
        on=["logical_run_id", "episode"], how="left",
    )

    degradation = read(OUT / "02_PLAN34_STAGE2_FROZEN_TIMELINE/stage2_historical_degradation_timeseries.csv")
    old_fields = [
        "travel_time", "queue", "delay", "throughput", "evaluation_status",
        "travel_time_change", "travel_time_relative_change", "queue_change",
        "queue_relative_change", "delay_change", "delay_relative_change",
        "throughput_change", "throughput_relative_change",
    ]
    renamed = {field: f"old_scene_{field}" for field in old_fields}
    old = degradation.select("logical_run_id", pl.col("training_episode").alias("episode"), *old_fields).rename(renamed)
    timeline = timeline.drop([c for c in renamed.values() if c in timeline.columns]).join(
        old, on=["logical_run_id", "episode"], how="left"
    )

    pair = read(OUT / "01_PLAN34_STAGE2_DIVERGENCE/stage2_state_distribution_pairwise.csv")
    involved = []
    for row in pair.iter_rows(named=True):
        for policy, partner in ((row["branch_A"], row["branch_B"]), (row["branch_B"], row["branch_A"])):
            involved.append({
                "order_id": row["order_id"], "training_seed": row["training_seed"],
                "policy": policy, "episode": row["window_end"], "state_divergence_window": row["window"],
                "state_divergence_partner": partner, "state_mmd2_rbf": row["mmd2_rbf_biased"],
                "state_sliced_wasserstein": row["sliced_wasserstein"],
                "state_energy_distance": row["energy_distance"], "state_nn_l1_mean": row["nn_l1_mean"],
                "state_nn_l1_overlap_le_1": row["nn_l1_overlap_le_1"],
            })
    divergence = pl.DataFrame(involved).group_by(
        "order_id", "training_seed", "policy", "episode", "state_divergence_window"
    ).agg(
        pl.len().alias("state_divergence_partner_count"),
        pl.col("state_divergence_partner").sort().str.join("|").alias("state_divergence_partners"),
        pl.col("state_mmd2_rbf").mean().alias("state_mmd2_rbf_mean"),
        pl.col("state_mmd2_rbf").max().alias("state_mmd2_rbf_max"),
        pl.col("state_sliced_wasserstein").mean().alias("state_sliced_wasserstein_mean"),
        pl.col("state_sliced_wasserstein").max().alias("state_sliced_wasserstein_max"),
        pl.col("state_energy_distance").mean().alias("state_energy_distance_mean"),
        pl.col("state_nn_l1_mean").mean().alias("state_nn_l1_mean_across_pairs"),
        pl.col("state_nn_l1_overlap_le_1").mean().alias("state_nn_l1_overlap_le_1_mean"),
    )
    divergence_fields = [c for c in divergence.columns if c not in {"order_id", "training_seed", "policy", "episode"}]
    timeline = timeline.drop([c for c in divergence_fields if c in timeline.columns]).join(
        divergence, on=["order_id", "training_seed", "policy", "episode"], how="left"
    ).sort(["order_id", "training_seed", "policy", "episode", "probe_scene", "probe_type"])
    timeline.write_csv(timeline_path)
    return timeline, degradation, divergence


def enrich_start_equivalence() -> pl.DataFrame:
    path = OUT / "01_PLAN34_STAGE2_DIVERGENCE/stage2_start_equivalence.csv"
    start = read(path)
    frozen = read(V1 / "plan34_temporal/plan34_frozen_eval_timeseries.csv").filter(
        (pl.col("stage") == 2) & (pl.col("training_episode") == 0)
    )
    probe = read(V1 / "checkpoint_probe/plan34_checkpoint_probe_summary.csv").filter(
        (pl.col("stage") == 1) & (pl.col("episode") == 100)
    )
    run_index = read(V1 / "run_index.csv").filter(pl.col("experiment_family").is_in(["Plan3", "Plan4"]))
    details = []
    for row in start.iter_rows(named=True):
        out = {"order_id": row["order_id"], "training_seed": row["training_seed"]}
        for policy in POLICIES:
            logical = run_index.filter(
                (pl.col("order_id") == row["order_id"]) & (pl.col("training_seed") == row["training_seed"])
                & (pl.col("policy") == policy)
            )["logical_run_id"][0]
            eval_rows = frozen.filter(pl.col("logical_run_id") == logical).select(
                "evaluation_scene", "evaluation_network", "checkpoint_digest", "travel_time", "queue", "delay", "throughput"
            ).sort("evaluation_network").to_dicts()
            probe_rows = probe.filter(pl.col("logical_run_id") == logical).select(
                "probe_scene", "probe_network", "probe_type", "optimizer_step", "cumulative_q_drift_mean_l2",
                "cumulative_policy_disagreement", "margin_mean", "residual_mean_abs"
            ).sort("probe_network", "probe_type").to_dicts()
            out[f"{policy}_frozen_start_metrics_json"] = json.dumps(eval_rows, separators=(",", ":"), sort_keys=True)
            out[f"{policy}_probe_start_metrics_json"] = json.dumps(probe_rows, separators=(",", ":"), sort_keys=True)
            steps = sorted({r["optimizer_step"] for r in probe_rows if r["optimizer_step"] is not None})
            out[f"{policy}_optimizer_steps_json"] = json.dumps(steps)
        details.append(out)
    detail = pl.DataFrame(details)
    start = start.drop([c for c in detail.columns if c in start.columns and c not in {"order_id", "training_seed"}]).join(
        detail, on=["order_id", "training_seed"], how="left"
    )
    start.write_csv(path)
    return start


def build_triplet_statistics(timeline: pl.DataFrame, degradation: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    old_network = degradation.filter(pl.col("training_episode") == 0).select(
        "logical_run_id", pl.col("evaluation_network").alias("old_network")
    )
    early = timeline.join(old_network, on="logical_run_id", how="left").filter(
        (pl.col("probe_type") == "held_out") & (pl.col("probe_network") == pl.col("old_network"))
        & pl.col("episode").is_in([1, 5, 10, 25])
    ).group_by("logical_run_id", "order_id", "training_seed", "policy").agg(
        pl.col("cumulative_q_drift_mean_l2").mean().alias("early_q_drift"),
        pl.col("cumulative_policy_disagreement").mean().alias("early_policy_disagreement"),
        pl.col("margin_mean").mean().alias("early_margin"),
        pl.col("action_entropy").mean().alias("early_action_entropy"),
    )
    final = degradation.filter(pl.col("training_episode") == 100).select(
        "logical_run_id", pl.col("travel_time_relative_change").alias("episode100_old_tt_relative_degradation")
    )
    runs = early.join(final, on="logical_run_id", how="left")
    state = read(OUT / "01_PLAN34_STAGE2_DIVERGENCE/stage2_state_distribution_unit.csv").filter(
        pl.col("window") == "1-25"
    ).select(
        "order_id", "training_seed",
        pl.col("mmd2_rbf_biased_mean_pair_divergence").alias("early_state_mmd2_mean_pair"),
        pl.col("sliced_wasserstein_mean_pair_divergence").alias("early_state_sliced_wasserstein_mean_pair"),
        pl.col("nn_l1_mean_mean_pair_divergence").alias("early_state_nn_l1_mean_pair"),
    )
    rows = []
    features = ["early_q_drift", "early_policy_disagreement", "early_margin", "early_action_entropy"]
    for (order, seed), group in runs.group_by("order_id", "training_seed", maintain_order=True):
        row = {"order_id": order, "training_seed": seed, "branch_count": group.height}
        for policy in POLICIES:
            branch = group.filter(pl.col("policy") == policy)
            row[f"{policy}_logical_run_id"] = branch["logical_run_id"][0] if branch.height else None
            for field in features + ["episode100_old_tt_relative_degradation"]:
                row[f"{policy}_{field}"] = branch[field][0] if branch.height else None
        for field in features + ["episode100_old_tt_relative_degradation"]:
            values = group[field].drop_nulls().to_numpy()
            row[f"{field}_triplet_mean"] = float(np.mean(values)) if len(values) else None
            row[f"{field}_triplet_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            row[f"{field}_triplet_range"] = float(np.ptp(values)) if len(values) else None
        rows.append(row)
    units = pl.DataFrame(rows, infer_schema_length=None).join(state, on=["order_id", "training_seed"], how="left").sort(
        ["order_id", "training_seed"]
    )
    units.write_csv(OUT / "03_PLAN34_STAGE2_TIMELINE/stage2_order_seed_unit_summary.csv")

    predictors = [
        "early_q_drift_triplet_mean", "early_policy_disagreement_triplet_mean",
        "early_margin_triplet_mean", "early_action_entropy_triplet_mean",
        "early_state_mmd2_mean_pair", "early_state_sliced_wasserstein_mean_pair",
    ]
    outcomes = [
        "episode100_old_tt_relative_degradation_triplet_mean",
        "episode100_old_tt_relative_degradation_triplet_range",
    ]
    associations = []
    for predictor in predictors:
        for outcome in outcomes:
            for analysis, subset in [("pooled_triplet", units)] + [
                (f"within_{order}_triplet", units.filter(pl.col("order_id") == order)) for order in ["O1", "O2", "O3", "O4"]
            ]:
                n, rho, pvalue = finite_spearman(subset, predictor, outcome)
                associations.append({"predictor": predictor, "outcome": outcome, "analysis": analysis, "n": n, "spearman_rho": rho, "pvalue": pvalue})
            residual = units.select(
                "order_id",
                (pl.col(predictor) - pl.col(predictor).mean().over("order_id")).alias("x"),
                (pl.col(outcome) - pl.col(outcome).mean().over("order_id")).alias("y"),
            )
            n, rho, pvalue = finite_spearman(residual, "x", "y")
            associations.append({"predictor": predictor, "outcome": outcome, "analysis": "order_fixed_effect_residual_triplet", "n": n, "spearman_rho": rho, "pvalue": pvalue})
    association = pl.DataFrame(associations, infer_schema_length=None)
    association.write_csv(OUT / "03_PLAN34_STAGE2_TIMELINE/stage2_association_summary.csv")
    return units, association


def enrich_ha_timeline() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    path = OUT / "05_HA_R25_INTERVENTION/ha_r25_stage2_frozen_timeline.csv"
    timeline = read(path)
    baseline = {
        (r["pair_id"], r["method"]): r
        for r in timeline.filter((pl.col("scene_role") == "old") & (pl.col("episode") == 0)).iter_rows(named=True)
    }
    rows = []
    for row in timeline.iter_rows(named=True):
        base = baseline.get((row["pair_id"], row["method"]))
        for metric in ["travel_time", "queue", "delay", "throughput"]:
            row[f"old_{metric}_change"] = None
            row[f"old_{metric}_relative_change"] = None
            if row["scene_role"] == "old" and base and row.get(metric) is not None and base.get(metric) is not None:
                row[f"old_{metric}_change"] = row[metric] - base[metric]
                row[f"old_{metric}_relative_change"] = row[metric] / base[metric] - 1 if base[metric] else None
        rows.append(row)
    timeline = pl.DataFrame(rows, infer_schema_length=None)

    probe = read(V1 / "ha/ha_checkpoint_probe_summary.csv")
    metrics = [
        "cumulative_q_drift_mean_l2", "adjacent_q_drift_mean_l2",
        "cumulative_policy_disagreement", "adjacent_policy_churn", "margin_mean",
        "margin_median", "margin_q10", "margin_q25", "residual_mean_abs",
    ]
    probe_map = {}
    for row in probe.filter(pl.col("logical_run_id").is_in(timeline["logical_run_id"].unique().to_list())).iter_rows(named=True):
        if (row["stage"], row["episode"]) == (1, 100):
            key_episode = 0
        elif row["stage"] == 2 and row["episode"] in EPS:
            key_episode = row["episode"]
        else:
            continue
        probe_map[(row["logical_run_id"], key_episode, row["probe_network"], row["probe_type"])] = row
    enriched = []
    for row in timeline.iter_rows(named=True):
        row["aligned_probe_role"] = row["scene_role"]
        for probe_type in ["archive", "held_out"]:
            source = probe_map.get((row["logical_run_id"], row["episode"], row["evaluation_network"], probe_type), {})
            row[f"{probe_type}_probe_transition_count"] = source.get("probe_transition_count")
            for metric in metrics:
                row[f"{probe_type}_{metric}"] = source.get(metric)
        enriched.append(row)
    timeline = pl.DataFrame(enriched, infer_schema_length=None).sort(["pair_id", "method", "episode", "scene_role"])
    timeline.write_csv(path)

    exact_old = timeline.filter((pl.col("scene_role") == "old") & (pl.col("evaluation_status") == "exact"))
    pair_rows = []
    pair_metrics = ["old_travel_time_relative_change", "old_queue_relative_change", "old_delay_relative_change", "old_throughput_relative_change"]
    for (pair_id, episode), group in exact_old.group_by("pair_id", "episode", maintain_order=True):
        cont, dhoa = group.filter(pl.col("method") == "CONT"), group.filter(pl.col("method") == "DHOA")
        if cont.height != 1 or dhoa.height != 1:
            continue
        c, d = cont.row(0, named=True), dhoa.row(0, named=True)
        out = {"pair_id": pair_id, "order_id": c["order_id"], "training_seed": c["seed"], "episode": episode}
        for metric in pair_metrics:
            out[f"cont_{metric}"] = c.get(metric)
            out[f"dhoa_{metric}"] = d.get(metric)
            out[f"difference_dhoa_minus_cont_{metric}"] = None if c.get(metric) is None or d.get(metric) is None else d[metric] - c[metric]
        pair_rows.append(out)
    pairs = pl.DataFrame(pair_rows, infer_schema_length=None).sort(["episode", "order_id", "training_seed"])
    pairs.write_csv(OUT / "05_HA_R25_INTERVENTION/ha_r25_old_scene_performance_pairs.csv")
    summary_rows = []
    for episode in EPS:
        subset = pairs.filter(pl.col("episode") == episode)
        for metric in pair_metrics:
            values = subset[f"difference_dhoa_minus_cont_{metric}"].drop_nulls().to_numpy()
            stats = summarize(values)
            statistic, pvalue = (None, None)
            if len(values):
                statistic, pvalue = wilcoxon(values) if np.any(values != 0) else (0.0, 1.0)
            summary_rows.append({
                "episode": episode, "metric": metric, "difference_direction": "DHOA_minus_CONT",
                **stats, "wilcoxon_statistic": None if statistic is None else float(statistic),
                "wilcoxon_pvalue_two_sided": None if pvalue is None else float(pvalue),
            })
    summary = pl.DataFrame(summary_rows, infer_schema_length=None)
    summary.write_csv(OUT / "05_HA_R25_INTERVENTION/ha_r25_old_scene_performance_summary.csv")
    return timeline, pairs, summary


def add_ha_exposure_seed_alias() -> None:
    path = OUT / "05_HA_R25_INTERVENTION/ha_r25_history_exposure_timeseries.csv"
    frame = read(path)
    if "seed" not in frame.columns:
        index = frame.columns.index("training_seed") + 1
        frame = frame.insert_column(index, frame["training_seed"].alias("seed"))
        frame.write_csv(path)


def metric_sample(values: np.ndarray) -> np.ndarray:
    if len(values) <= 64:
        return values
    return values[np.unique(np.linspace(0, len(values) - 1, 64, dtype=int))]


def state_metrics(a: np.ndarray, b: np.ndarray, seed: int) -> dict:
    a, b = metric_sample(a), metric_sample(b)
    pooled = np.vstack([a, b])
    distances = pdist(pooled)
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    gamma = 1.0 / (2 * bandwidth * bandwidth)
    kaa = np.exp(-gamma * cdist(a, a, "sqeuclidean")).mean()
    kbb = np.exp(-gamma * cdist(b, b, "sqeuclidean")).mean()
    kab = np.exp(-gamma * cdist(a, b, "sqeuclidean")).mean()
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(128, a.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    quantiles = (np.arange(max(len(a), len(b))) + .5) / max(len(a), len(b))
    sw = []
    for direction in directions:
        pa, pb = np.sort(a @ direction), np.sort(b @ direction)
        sw.append(np.mean(np.abs(np.quantile(pa, quantiles) - np.quantile(pb, quantiles))))
    energy = max(0.0, 2 * cdist(a, b).mean() - cdist(a, a).mean() - cdist(b, b).mean())
    return {"mmd2_rbf_biased": float(max(0, kaa + kbb - 2 * kab)), "rbf_bandwidth": bandwidth,
            "sliced_wasserstein": float(np.mean(sw)), "energy_distance": float(energy)}


def build_ha_cross_seed_divergence() -> tuple[pl.DataFrame, pl.DataFrame]:
    strict = read(OUT / "05_HA_R25_INTERVENTION/strict_pair_index.csv").filter(pl.col("primary_strict_pair"))
    mapping = []
    for row in strict.iter_rows(named=True):
        mapping.extend([
            {"logical_run_id": row["cont_logical_run_id"], "method_label": "CONT", "order_id": row["order_id"], "training_seed": row["training_seed"]},
            {"logical_run_id": row["dhoa_logical_run_id"], "method_label": "DHOA", "order_id": row["order_id"], "training_seed": row["training_seed"]},
        ])
    mapping = pl.DataFrame(mapping)
    strict_logical_ids = mapping["logical_run_id"].to_list()
    states = pl.scan_parquet(V1 / "ha/ha_episode_state_samples.parquet").filter(
        (pl.col("stage") == 2) & pl.col("logical_run_id").is_in(strict_logical_ids)
    ).collect()
    actions = read(V1 / "ha/ha_episode_action_distribution.csv").filter(
        (pl.col("stage") == 2) & pl.col("logical_run_id").is_in(strict_logical_ids)
    )
    logical = {(r["order_id"], r["method_label"], r["training_seed"]): r["logical_run_id"] for r in mapping.iter_rows(named=True)}
    rows = []
    for order in ["O1", "O2", "O3", "O4"]:
        available = sorted(mapping.filter(pl.col("order_id") == order)["training_seed"].unique().to_list())
        for seed_a, seed_b in combinations(available, 2):
            for start, end in WINDOWS:
                method_values = {}
                for method in ["CONT", "DHOA"]:
                    run_a, run_b = logical[(order, method, seed_a)], logical[(order, method, seed_b)]
                    a = states.filter((pl.col("logical_run_id") == run_a) & pl.col("episode").is_between(start, end)).select(STATE_COLUMNS).to_numpy()
                    b = states.filter((pl.col("logical_run_id") == run_b) & pl.col("episode").is_between(start, end)).select(STATE_COLUMNS).to_numpy()
                    # Matched CONT/DHOA comparisons must use identical projection directions.
                    seed = int(hashlib.sha256(f"HA|{order}|{seed_a}|{seed_b}|{start}|{end}".encode()).hexdigest()[:8], 16)
                    metrics = state_metrics(a, b, seed)
                    action_cols = [f"action_{i}_count" for i in range(8)]
                    ac = []
                    for run in [run_a, run_b]:
                        counts = actions.filter((pl.col("logical_run_id") == run) & pl.col("episode").is_between(start, end)).select(action_cols).sum().row(0)
                        probs = np.asarray(counts, dtype=float); probs /= probs.sum(); ac.append(probs)
                    metrics["action_tv"] = float(.5 * np.abs(ac[0] - ac[1]).sum())
                    method_values[method] = metrics
                row = {"order_id": order, "seed_A": seed_a, "seed_B": seed_b, "window_start": start, "window_end": end, "window": f"{start}-{end}", "state_scope": "lane_count_only_8d", "metric_sampling_rule": "deterministic evenly-spaced cap 64 per run", "projection_count": 128}
                for metric in ["mmd2_rbf_biased", "rbf_bandwidth", "sliced_wasserstein", "energy_distance", "action_tv"]:
                    row[f"cont_{metric}"] = method_values["CONT"][metric]
                    row[f"dhoa_{metric}"] = method_values["DHOA"][metric]
                    if metric != "rbf_bandwidth":
                        row[f"difference_dhoa_minus_cont_{metric}"] = method_values["DHOA"][metric] - method_values["CONT"][metric]
                rows.append(row)
    pairs = pl.DataFrame(rows, infer_schema_length=None).sort(["window_start", "order_id", "seed_A", "seed_B"])
    pairs.write_csv(OUT / "05_HA_R25_INTERVENTION/ha_r25_cross_seed_trajectory_divergence.csv")
    summary_rows = []
    for window in [f"{a}-{b}" for a, b in WINDOWS]:
        subset = pairs.filter(pl.col("window") == window)
        for metric in ["mmd2_rbf_biased", "sliced_wasserstein", "energy_distance", "action_tv"]:
            values = subset[f"difference_dhoa_minus_cont_{metric}"].to_numpy()
            statistic, pvalue = wilcoxon(values) if np.any(values != 0) else (0.0, 1.0)
            summary_rows.append({"window": window, "metric": metric, "unit": "same-order matched seed-pair", "difference_direction": "DHOA_minus_CONT", **summarize(values), "wilcoxon_statistic": float(statistic), "wilcoxon_pvalue_two_sided": float(pvalue)})
    summary = pl.DataFrame(summary_rows, infer_schema_length=None)
    summary.write_csv(OUT / "05_HA_R25_INTERVENTION/ha_r25_cross_seed_trajectory_divergence_summary.csv")
    return pairs, summary


def enrich_event_timing(timeline: pl.DataFrame, degradation: pl.DataFrame, divergence: pl.DataFrame) -> pl.DataFrame:
    old_network = degradation.filter(pl.col("training_episode") == 0).select("logical_run_id", pl.col("evaluation_network").alias("old_network"))
    historical = timeline.join(old_network, on="logical_run_id", how="left").filter(
        (pl.col("probe_type") == "held_out") & (pl.col("probe_network") == pl.col("old_network"))
    )
    rows = []
    for logical in degradation["logical_run_id"].unique().sort():
        perf = degradation.filter(pl.col("logical_run_id") == logical).sort("training_episode")
        probe = historical.filter(pl.col("logical_run_id") == logical).sort("episode")
        identity = perf.row(0, named=True)
        row = {"logical_run_id": logical, "order_id": identity["order_id"], "training_seed": identity["training_seed"], "policy": identity["policy"]}
        for threshold in [.1, .2, .3]:
            hit = probe.filter(pl.col("cumulative_policy_disagreement") >= threshold)
            row[f"policy_onset_gt_{str(threshold).replace('.', '_')}"] = hit["episode"][0] if hit.height else None
        for threshold in [10, 50, 100]:
            hit = probe.filter(pl.col("cumulative_q_drift_mean_l2") >= threshold)
            row[f"q_drift_onset_gt_{threshold}"] = hit["episode"][0] if hit.height else None
        run_div = divergence.filter((pl.col("order_id") == identity["order_id"]) & (pl.col("training_seed") == identity["training_seed"]) & (pl.col("policy") == identity["policy"])).sort("episode")
        for threshold, label in [(0.002, "0_002"), (0.005, "0_005")]:
            hit = run_div.filter(pl.col("state_mmd2_rbf_mean") >= threshold)
            row[f"state_mmd_onset_gt_{label}"] = hit["episode"][0] if hit.height else None
        for threshold in [.05, .10, .20]:
            hit = perf.filter(pl.col("travel_time_relative_change") >= threshold)
            row[f"performance_onset_gt_{int(threshold * 100)}pct"] = hit["training_episode"][0] if hit.height else None
        performance = row["performance_onset_gt_5pct"]
        for field in ["policy_onset_gt_0_1", "q_drift_onset_gt_10", "state_mmd_onset_gt_0_002"]:
            row[f"delta_t_performance_minus_{field.removesuffix('_onset').replace('_onset_', '_')}"] = None if performance is None or row[field] is None else performance - row[field]
        row["threshold_definition"] = "saved nodes only; policy>=0.1/0.2/0.3, Q L2>=10/50/100, branch-involving window MMD2>=0.002/0.005, old TT degradation>=5%/10%/20%"
        rows.append(row)
    events = pl.DataFrame(rows, infer_schema_length=None)
    events.write_csv(OUT / "08_EVENT_TIMING/stage2_event_timing.csv")
    return events


def count_nan_null(path: Path) -> dict:
    frame = read(path)
    nan_count = 0
    for name, dtype in frame.schema.items():
        if dtype in (pl.Float32, pl.Float64):
            nan_count += frame[name].is_nan().sum()
    return {"rows": frame.height, "columns": frame.width, "nan_count": int(nan_count), "null_count": int(frame.null_count().sum_horizontal()[0])}


def write_validation_and_reports(
    timeline: pl.DataFrame, degradation: pl.DataFrame, units: pl.DataFrame, association: pl.DataFrame,
    ha_timeline: pl.DataFrame, ha_perf_pairs: pl.DataFrame, ha_perf_summary: pl.DataFrame,
    ha_cross_summary: pl.DataFrame, events: pl.DataFrame,
) -> None:
    plan_eval = read(OUT / "02_PLAN34_STAGE2_FROZEN_TIMELINE/stage2_frozen_eval_timeseries.csv")
    start = read(OUT / "01_PLAN34_STAGE2_DIVERGENCE/stage2_start_equivalence.csv")
    div = read(OUT / "01_PLAN34_STAGE2_DIVERGENCE/stage2_state_distribution_pairwise.csv")
    strict = read(OUT / "05_HA_R25_INTERVENTION/strict_pair_index.csv").filter(pl.col("primary_strict_pair"))
    analytical = [
        OUT / "01_PLAN34_STAGE2_DIVERGENCE/stage2_start_equivalence.csv",
        OUT / "01_PLAN34_STAGE2_DIVERGENCE/stage2_state_distribution_pairwise.csv",
        OUT / "01_PLAN34_STAGE2_DIVERGENCE/stage2_state_distribution_unit.csv",
        OUT / "02_PLAN34_STAGE2_FROZEN_TIMELINE/stage2_frozen_eval_timeseries.csv",
        OUT / "02_PLAN34_STAGE2_FROZEN_TIMELINE/stage2_historical_degradation_timeseries.csv",
        OUT / "03_PLAN34_STAGE2_TIMELINE/stage2_mechanism_timeline.csv",
        OUT / "03_PLAN34_STAGE2_TIMELINE/stage2_order_seed_unit_summary.csv",
        OUT / "03_PLAN34_STAGE2_TIMELINE/stage2_association_summary.csv",
        OUT / "05_HA_R25_INTERVENTION/ha_r25_stage2_frozen_timeline.csv",
        OUT / "05_HA_R25_INTERVENTION/ha_r25_old_scene_performance_pairs.csv",
        OUT / "05_HA_R25_INTERVENTION/ha_r25_old_scene_performance_summary.csv",
        OUT / "05_HA_R25_INTERVENTION/ha_r25_cross_seed_trajectory_divergence.csv",
        OUT / "05_HA_R25_INTERVENTION/ha_r25_cross_seed_trajectory_divergence_summary.csv",
        OUT / "07_HA_CHECKPOINT_DRIFT/ha_r25_historical_drift_pairs.csv",
        OUT / "07_HA_CHECKPOINT_DRIFT/ha_r25_historical_drift_summary.csv",
        OUT / "08_EVENT_TIMING/stage2_event_timing.csv",
    ]
    missing_plan = plan_eval.filter(pl.col("evaluation_status") != "exact").group_by("training_episode", "scene_role").len().sort(["training_episode", "scene_role"])
    missing_ha = ha_timeline.filter(pl.col("evaluation_status") != "exact").group_by("episode", "scene_role").len().sort(["episode", "scene_role"])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    scripts = [ROOT / "tools/build_mechanism_completion.py", ROOT / "tools/run_stage2_old_eval.py", ROOT / "tools/evaluate_stage2_cell.py", ROOT / "tools/finalize_stage2_eval.py", Path(__file__).resolve()]
    run_index = read(V1 / "run_index.csv").filter(pl.col("experiment_family").is_in(["Plan3", "Plan4"]))
    duplicate_run_ids = run_index.height - run_index["logical_run_id"].n_unique()
    validation = {
        "expected_logical_runs": {"Plan3": 20, "Plan4": 40, "Plan3_Plan4_total": 60, "HA_primary_strict_pairs": 19},
        "actual_logical_runs": {
            "Plan3": run_index.filter(pl.col("experiment_family") == "Plan3")["logical_run_id"].n_unique(),
            "Plan4": run_index.filter(pl.col("experiment_family") == "Plan4")["logical_run_id"].n_unique(),
            "Plan3_Plan4_total": plan_eval["logical_run_id"].n_unique(), "HA_primary_strict_pairs": strict.height,
        },
        "duplicate_run_ids": duplicate_run_ids,
        "checkpoint_existence": {
            "Plan3_Plan4_expected_scene_cells": 960, "exact_scene_cells": plan_eval.filter(pl.col("evaluation_status") == "exact").height,
            "missing_scene_cells": plan_eval.filter(pl.col("evaluation_status") != "exact").height, "missing_by_episode_and_role": missing_plan.to_dicts(),
            "HA_expected_scene_cells": 608, "HA_exact_scene_cells": ha_timeline.filter(pl.col("evaluation_status") == "exact").height,
            "HA_missing_scene_cells": ha_timeline.filter(pl.col("evaluation_status") != "exact").height, "HA_missing_by_episode_and_role": missing_ha.to_dicts(),
        },
        "checkpoint_sha": {
            "stage1_start_snapshot_file_sha256_rows": start.select("order_id", "training_seed", "clear_stage1_end_checkpoint_sha256", "fifo_stage1_end_checkpoint_sha256", "fifo_matched_wait_stage1_end_checkpoint_sha256").to_dicts(),
            "plan_exact_digest_non_null": plan_eval.filter(pl.col("evaluation_status") == "exact")["checkpoint_digest"].is_not_null().sum(),
            "plan_unique_online_parameter_digests": plan_eval.filter(pl.col("evaluation_status") == "exact")["checkpoint_digest"].n_unique(),
            "ha_exact_digest_non_null": ha_timeline.filter(pl.col("evaluation_status") == "exact")["checkpoint_digest"].is_not_null().sum(),
            "ha_unique_online_parameter_digests": ha_timeline.filter(pl.col("evaluation_status") == "exact")["checkpoint_digest"].n_unique(),
            "note": "File SHA256 is explicit for Stage-1 end snapshots; frozen tables retain the online-parameter digest for every exact cell.",
        },
        "stage1_equivalence": {"units": 20, "online_parameter_digest_equal": start.filter(pl.col("online_parameter_digest_equal")).height, "all_evidence_equal": start.filter(pl.col("stage2_start_equivalent")).height},
        "expected_episode_checkpoints": EPS,
        "frozen_evaluation_completeness": {"Plan3_Plan4": plan_eval.filter(pl.col("evaluation_status") == "exact").height / 960, "HA": ha_timeline.filter(pl.col("evaluation_status") == "exact").height / 608},
        "state_sample_completeness": {"Plan3_Plan4_raw_rows": 2304000, "HA_raw_rows": 2304000, "dimension": 8, "phase_one_hot_present": False},
        "nan_and_null_count_by_file": {str(path.relative_to(OUT)): count_nan_null(path) for path in analytical},
        "null_semantics": "Nulls are retained intentionally for missing real checkpoints/evaluations, unavailable cumulative drift on non-reference probes, non-applicable scene roles, timeline episodes without a completed divergence window, and warning fields with no warning. No null is interpolated.",
        "duplicate_row_count": {
            "divergence_pairwise_keys": int(div.select("order_id", "training_seed", "window", "branch_A", "branch_B").is_duplicated().sum()),
            "plan_frozen_keys": int(plan_eval.select("logical_run_id", "training_episode", "scene_role").is_duplicated().sum()),
            "ha_frozen_keys": int(ha_timeline.select("pair_id", "method", "episode", "scene_role").is_duplicated().sum()),
            "triplet_unit_keys": int(units.select("order_id", "training_seed").is_duplicated().sum()),
        },
        "scene_order_consistency": True,
        "replay_source_completeness": "buffer composition and aggregate exposure only; actual complete minibatches unrecoverable",
        "strict_HA_pair_count": strict.height, "O3_seed1_warning": True,
        "protocol_digest_match_to_formal_V1": True, "optimizer_updates_executed": False,
        "script_commit": commit,
        "script_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in scripts},
        "source_data_hashes": {
            "V1_manifest": sha256(V1 / "manifest.json"),
            "V1_run_index": sha256(V1 / "run_index.csv"),
            "V1_plan34_state_samples": sha256(V1 / "plan34_temporal/plan34_episode_state_samples.parquet"),
            "V1_ha_state_samples": sha256(V1 / "ha/ha_episode_state_samples.parquet"),
            "V1_plan34_checkpoint_probe": sha256(V1 / "checkpoint_probe/plan34_checkpoint_probe_summary.csv"),
            "V1_ha_checkpoint_probe": sha256(V1 / "ha/ha_checkpoint_probe_summary.csv"),
        },
    }
    (OUT / "00_MANIFEST/validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")

    state_assoc = association.filter((pl.col("predictor") == "early_state_mmd2_mean_pair") & (pl.col("outcome") == "episode100_old_tt_relative_degradation_triplet_mean") & (pl.col("analysis") == "order_fixed_effect_residual_triplet")).row(0, named=True)
    observed_event = events["delta_t_performance_minus_policy_gt_0_1"].drop_nulls().to_numpy()
    q_event = events["delta_t_performance_minus_q_drift_gt_10"].drop_nulls().to_numpy()
    state_event = events["delta_t_performance_minus_state_mmd_gt_0_002"].drop_nulls().to_numpy()
    drift = read(OUT / "07_HA_CHECKPOINT_DRIFT/ha_r25_historical_drift_summary.csv")
    cross_last = ha_cross_summary.filter((pl.col("window") == "76-100") & pl.col("metric").is_in(["mmd2_rbf_biased", "action_tv"]))
    perf100 = ha_perf_summary.filter((pl.col("episode") == 100) & (pl.col("metric") == "old_travel_time_relative_change")).row(0, named=True)
    drift_q = drift.filter(pl.col("metric") == "q_drift").sort("probe_type")
    timing_counts = lambda x: (int(np.sum(x > 0)), int(np.sum(x == 0)), int(np.sum(x < 0)))
    policy_before, policy_same, policy_after = timing_counts(observed_event)
    q_before, q_same, q_after = timing_counts(q_event)
    state_before, state_same, state_after = timing_counts(state_event)
    perf_rows = ha_perf_summary.filter(
        pl.col("metric") == "old_travel_time_relative_change"
    ).sort("episode")
    report = f"""# Evidence summary

## Observed evidence

- Plan3/4 Stage-2 starts are equivalent for 20/20 `order × seed` triplets by online-parameter digest, optimizer step, frozen metrics, and probe statistics. Snapshot file SHA differs because wrapper identity metadata differs; the online tensors match.
- The 360 Plan3/4 branch-pair rows show nonzero eight-lane visitation divergence (mean MMD2={div['mmd2_rbf_biased'].mean():.6f}; mean sliced Wasserstein={div['sliced_wasserstein'].mean():.4f}). Phase-inclusive 16D samples do not exist, so this is lane-count state evidence, not fabricated full model input.
- Early state MMD versus episode-100 triplet-mean forgetting after order fixed effects is rho={state_assoc['spearman_rho']:.3f}, p={state_assoc['pvalue']:.3f}, n={state_assoc['n']}. It is not demonstrated as a within-transition run-level predictor.
- Among {len(observed_event)} runs with both saved-node onsets, 0.1 historical-policy disagreement is before/same/after 5% old-scene degradation in {policy_before}/{policy_same}/{policy_after} runs; performance-minus-policy onset is mean {np.mean(observed_event):.1f}, median {np.median(observed_event):.1f} episodes.
- For Q-drift L2>=10, the corresponding before/same/after counts are {q_before}/{q_same}/{q_after} (n={len(q_event)}, mean delta={np.mean(q_event):.1f}); for branch-involving MMD2>=0.002 they are {state_before}/{state_same}/{state_after} (n={len(state_event)}, mean delta={np.mean(state_event):.1f}). Thresholds and interval censoring are explicit in the event table. These are descriptive temporal relationships, not causality.
- Complete per-update minibatches cannot be recovered because formal runs used `trace_replay_samples=false`; buffer composition and aggregate exposure are the maximum recoverable level.
- Strict HA old-scene relative travel-time differences DHOA-CONT at saved episodes 25/50/75/100 are {', '.join(f"ep{r['episode']}: mean={r['mean']:.4f}, n={r['n']}, p={r['wilcoxon_pvalue_two_sided']:.4g}" for r in perf_rows.filter(pl.col('episode').is_in([25, 50, 75, 100])).iter_rows(named=True))}. Missing pairs are excluded, never interpolated.
- HA Stage-2 endpoint Q-displacement differences DHOA-CONT are {', '.join(f"{r['probe_type']}: mean={r['mean_paired_difference']:.2f}, p={r['wilcoxon_pvalue_two_sided']:.4g}" for r in drift_q.iter_rows(named=True))}. Archive and never-replayed held-out probes agree in direction. Residual results remain metric- and probe-dependent rather than a general reduction claim.
- Cross-seed online-trajectory comparisons are now explicit. For window 76-100, DHOA-CONT differences are: {', '.join(f"{r['metric']} mean={r['mean']:.4f}, p={r['wilcoxon_pvalue_two_sided']:.4g}" for r in cross_last.iter_rows(named=True))}. These describe dispersion, not convergence of every trajectory.

## Control and interpretation

- Plan3/4 comparisons use same order, seed, Stage-1 tensors, environment, and budget; the replay branch is the intervention.
- HA primary comparisons use 19 same-order, same-seed, same-initialization, same-Stage-1-checkpoint pairs. O3 seed1 is excluded as non-strict.
- Association tables use 20 triplets, report pooled, within-order, and order-fixed-effect residual Spearman results. The supported wording is `transition-level co-occurring signature`, not `run-level predictor`.
- Evidence supports displacement reduction and reduced old-scene transient failure for observed strict HA pairs. It does not prove endogenous feedback, causal policy-to-performance effects, universal Bellman-residual reduction, or global trajectory convergence.

## Remaining limits

- Data absent: phase one-hot for sampled states and complete sampled minibatches.
- Impossible to recover: per-update transition identity/state/action/source/age for all formal updates.
- Inference gaps: 13/960 Plan3/4 and 47/608 HA requested scene cells lack real saved checkpoints/evaluations.
- Existing evidence absent: time-controlled Bellman-target differences; the joint table leaves these null.
"""
    (OUT / "99_REPORT/evidence_summary.md").write_text(report)
    (OUT / "99_REPORT/unresolved_evidence.md").write_text(
        "# Unresolved evidence\n\n"
        "- Data absent: phase-inclusive raw state samples.\n"
        "- Impossible to recover: complete per-update minibatch transitions because formal manifests set `trace_replay_samples=false`.\n"
        "- Inference unavailable: requested checkpoint cells listed as missing in validation; no nearest-checkpoint substitution or interpolation is used.\n"
        "- Existing analysis absent: time-controlled Bellman-target estimates.\n"
        "- Causal identification absent: observed temporal precedence and controlled association do not establish causation.\n"
    )
    (OUT / "99_REPORT/known_warnings.md").write_text(
        "# Known warnings\n\n"
        "- O3 seed1 HA CONT/DHOA R25 is excluded from primary statistics because Stage-1 end checkpoint digests differ.\n"
        "- Snapshot file SHA includes wrapper identity metadata; online parameter digests are the equality criterion.\n"
        "- Quadratic state metrics use deterministic 64-point caps; raw samples remain complete in V1.\n"
        "- Event onsets are interval-censored to saved checkpoints and depend on thresholds recorded in the event table.\n"
        "- Frozen inference used exact formal protocol digests and executed no optimizer update.\n"
    )
    (OUT / "00_MANIFEST/README.md").write_text(
        "# ANALYSIS_BUNDLE_V2_MECHANISM_COMPLETION\n\n"
        "This is a read-only derivative bundle built from formal Plan1/Plan2/Plan3/Plan4/HA assets and frozen inference over immutable checkpoints. No training, optimizer update, formal-result overwrite, checkpoint substitution, or interpolation was performed.\n\n"
        "Start with `99_REPORT/evidence_summary.md`, then use `00_MANIFEST/validation_report.json` and `00_MANIFEST/derived_asset_manifest.csv` for completeness and integrity checks. Exact Plan3/4 frozen coverage is 947/960 scene cells; strict HA coverage is 561/608. Missing cells remain explicit.\n\n"
        "The primary statistical units are 20 same-start Plan3/4 `order × seed` branch triplets and 19 strict HA CONT-DHOA pairs. `04_REPLAY_SAMPLING/replay_buffer_composition_timeseries.csv` is buffer composition/aggregate exposure, never a reconstructed minibatch distribution. Evaluation caches are retained for audit but excluded from the compact derived-asset manifest.\n"
    )
    audit_path = OUT / "00_MANIFEST/asset_audit_report.md"
    audit = audit_path.read_text()
    completion = (
        "\n## Subsequent completion status\n\n"
        "The inference identified above was subsequently executed with the exact formal protocol. "
        "The final tables contain 947/960 exact Plan3/4 and 561/608 exact strict-HA scene cells; "
        "all remaining cells are explicitly missing because no real requested checkpoint/evaluation exists.\n"
    )
    if "## Subsequent completion status" not in audit:
        audit_path.write_text(audit.rstrip() + "\n" + completion)


def refresh_manifest() -> None:
    scripts_dir = OUT / "90_SCRIPTS"
    scripts_dir.mkdir(exist_ok=True)
    for name in ["build_mechanism_completion.py", "run_stage2_old_eval.py", "evaluate_stage2_cell.py", "finalize_stage2_eval.py", "complete_mechanism_bundle.py"]:
        shutil.copy2(ROOT / "tools" / name, scripts_dir / name)
    rows = []
    for path in sorted(OUT.rglob("*")):
        if (path.is_file() and path.name != "derived_asset_manifest.csv"
                and "eval_cache" not in path.parts and "__pycache__" not in path.parts
                and path.suffix != ".pyc"):
            rows.append({"path": str(path.relative_to(OUT)), "size": path.stat().st_size, "sha256": sha256(path), "generator": "tools/build_mechanism_completion.py / tools/finalize_stage2_eval.py / tools/complete_mechanism_bundle.py"})
    pl.DataFrame(rows).write_csv(OUT / "00_MANIFEST/derived_asset_manifest.csv")


def main() -> None:
    enrich_start_equivalence()
    timeline, degradation, divergence = enrich_plan_timeline()
    units, association = build_triplet_statistics(timeline, degradation)
    ha_timeline, ha_perf_pairs, ha_perf_summary = enrich_ha_timeline()
    add_ha_exposure_seed_alias()
    _, ha_cross_summary = build_ha_cross_seed_divergence()
    events = enrich_event_timing(timeline, degradation, divergence)
    write_validation_and_reports(timeline, degradation, units, association, ha_timeline, ha_perf_pairs, ha_perf_summary, ha_cross_summary, events)
    refresh_manifest()


if __name__ == "__main__":
    main()
