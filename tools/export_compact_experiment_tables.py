#!/usr/bin/env python3
"""Export compact tables from frozen Plan 1, Plan 3/4, and HA analyses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "analysis/plan1_observation_conditioned_outcomes"
P34 = ROOT / ("data/output_data/analysis/plan34/"
              "plan34_b100_formal_60_plots_20260725/tables")
HA = ROOT / "data/output_data/ha_sodqn/analysis_e7705f7/E4"
OUT = ROOT / "analysis/compact_experiment_tables"
BOOTSTRAP_SEED = 20260803
BOOTSTRAP_RESAMPLES = 500


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, float_format="%.10g")


def consistency(values: pd.Series) -> float:
    return float((np.sign(values) == np.sign(values.mean())).mean())


def clustered_ci(seed_values: pd.Series,
                 rng: np.random.Generator) -> tuple[float, float]:
    """Bootstrap the five independent matched training-seed aggregates."""
    values = seed_values.dropna().to_numpy(float)
    draws = [float(rng.choice(values, len(values), replace=True).mean())
             for _ in range(BOOTSTRAP_RESAMPLES)]
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def bellman_target_table() -> pd.DataFrame:
    raw = pd.read_csv(P1 / "all_checkpoint_target_components.csv")
    raw = raw[raw.seed_a == raw.seed_b].copy()
    keys = ["data_scope", "scene_pair", "evaluator_scene", "evaluator_seed"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for identity, group in raw.groupby(keys, sort=True, observed=True):
        seed_target = group.groupby("seed_a")[
            "total_target_difference_b_minus_a"].mean()
        low, high = clustered_ci(seed_target, rng)
        rows.append({
            **dict(zip(keys, identity)), "time_control": "none",
            "reward_difference_b_minus_a": group.reward_difference_b_minus_a.mean(),
            "bootstrap_difference_b_minus_a":
                group.bootstrap_component_difference_b_minus_a.mean(),
            "total_target_difference_b_minus_a":
                group.total_target_difference_b_minus_a.mean(),
            "seed_consistency": consistency(seed_target),
            "cluster_ci95_low": low, "cluster_ci95_high": high,
        })
    columns = ["data_scope", "scene_pair", "time_control", "evaluator_scene",
               "evaluator_seed", "reward_difference_b_minus_a",
               "bootstrap_difference_b_minus_a",
               "total_target_difference_b_minus_a", "seed_consistency",
               "cluster_ci95_low", "cluster_ci95_high"]
    return pd.DataFrame(rows)[columns]


def scene_relation_matrix(target: pd.DataFrame) -> pd.DataFrame:
    groups = pd.read_csv(P1 / "strict_matching_groups.csv")
    outcomes = pd.read_csv(P1 / "strict_outcome_differences.csv", low_memory=False)
    outcomes = outcomes[(outcomes.time_control == "none") &
                        (outcomes.seed_a == outcomes.seed_b)]
    support = (groups.groupby(["data_scope", "scene_pair"], observed=True)
               .size().rename("support_overlap_exact_group_count").reset_index())
    relation = (outcomes.groupby(["data_scope", "scene_pair"], observed=True)[
        ["next_l1_difference_b_minus_a", "reward_difference_b_minus_a"]]
        .mean().reset_index().rename(columns={
            "next_l1_difference_b_minus_a":
                "strict_state_increment_l1_difference_b_minus_a"}))
    target_pair = (target.groupby(["data_scope", "scene_pair"], observed=True)[
        "total_target_difference_b_minus_a"].median().reset_index())
    return (support.merge(relation, on=["data_scope", "scene_pair"], validate="1:1")
            .merge(target_pair, on=["data_scope", "scene_pair"], validate="1:1")
            .sort_values(["data_scope", "scene_pair"]).reset_index(drop=True))


def action_conditioned_target_table() -> pd.DataFrame:
    raw = pd.read_csv(P1 / "all_checkpoint_target_components.csv")
    raw = raw[raw.seed_a == raw.seed_b].copy()
    checkpoint_keys = ["evaluator_scene", "evaluator_seed"]
    value_columns = ["reward_difference_b_minus_a",
        "bootstrap_component_difference_b_minus_a",
        "total_target_difference_b_minus_a"]
    per_checkpoint = (raw.groupby(
        ["data_scope", "scene_pair", "action"] + checkpoint_keys,
        observed=True)[value_columns].mean().reset_index())
    rows = []
    for identity, group in per_checkpoint.groupby(
            ["data_scope", "scene_pair", "action"], sort=True, observed=True):
        target = group.total_target_difference_b_minus_a
        overall_direction = np.sign(target.mean())
        rows.append({
            "data_scope": identity[0], "scene_pair": identity[1],
            "action": int(identity[2]),
            "reward_difference_b_minus_a":
                group.reward_difference_b_minus_a.mean(),
            "bootstrap_difference_b_minus_a":
                group.bootstrap_component_difference_b_minus_a.mean(),
            "target_difference_b_minus_a": target.mean(),
            "checkpoint_direction_consistency":
                float((np.sign(target) == overall_direction).mean()),
            "checkpoint_count": int(len(group)),
        })
    result = pd.DataFrame(rows)
    pair_stats = (result.groupby(["data_scope", "scene_pair"], observed=True)
        .target_difference_b_minus_a.agg(
            action_target_difference_std="std",
            action_target_min="min", action_target_max="max").reset_index())
    pair_stats["action_order_reversal"] = (
        (pair_stats.action_target_min < 0) & (pair_stats.action_target_max > 0))
    result = result.merge(pair_stats[["data_scope", "scene_pair",
        "action_target_difference_std", "action_order_reversal"]],
        on=["data_scope", "scene_pair"], validate="m:1")
    columns = ["data_scope", "scene_pair", "action",
        "reward_difference_b_minus_a", "bootstrap_difference_b_minus_a",
        "target_difference_b_minus_a", "action_target_difference_std",
        "action_order_reversal", "checkpoint_direction_consistency",
        "checkpoint_count"]
    return result[columns].sort_values(
        ["data_scope", "scene_pair", "action"]).reset_index(drop=True)


def plan34_transition_table() -> pd.DataFrame:
    stages = pd.read_csv(P34 / "stage_summary.csv")
    evaluations = pd.read_csv(P34 / "stage_scene_evaluations.csv")
    final = pd.read_csv(P34 / "final_retention.csv")
    run = ["logical_run_id", "order_id", "training_seed", "policy"]
    stage_metrics = stages[run + ["stage_index", "network",
        "zero_shot_normalized", "normalized_auc"]].rename(
            columns={"network": "current_scene"})
    old = evaluations[(evaluations.stage_index >= 2) &
        (evaluations.evaluation_scene_introduced_stage < evaluations.stage_index)].copy()
    previous = evaluations[run + ["evaluation_network", "stage_index",
                                  "travel_time"]].copy()
    previous["stage_index"] += 1
    previous = previous.rename(columns={"travel_time": "previous_stage_travel_time"})
    old = old.merge(previous, on=run + ["evaluation_network", "stage_index"],
                    validate="m:1")
    old = old.merge(stage_metrics, on=run + ["stage_index"], validate="m:1")
    old = old.drop(columns="final_retention_normalized")
    old = old.merge(final.rename(columns={"network": "evaluation_network"}),
                    on=run + ["evaluation_network"], validate="m:1")
    old["zero_shot_drop"] = old.zero_shot_normalized - 1
    old["incremental_forgetting_travel_time"] = (
        old.travel_time - old.previous_stage_travel_time)
    old = old.rename(columns={"evaluation_network": "old_scene",
        "normalized_auc": "adaptation_aulc_normalized_travel_time"})
    columns = ["order_id", "training_seed", "policy", "stage_index", "old_scene",
        "current_scene", "zero_shot_drop",
        "adaptation_aulc_normalized_travel_time",
        "incremental_forgetting_travel_time", "final_retention_normalized"]
    return old[columns].sort_values(columns[:5]).reset_index(drop=True)


def ha_improvement_table() -> pd.DataFrame:
    curves = pd.read_csv(HA / "adaptation_curves.csv")
    lower = pd.read_csv(HA / "lower_triangle.csv")
    runs = pd.read_csv(HA / "runs.csv")
    causal = runs[((runs.archive_mode == "NONE") & (runs.method == "CONT")) |
                  ((runs.archive_mode == "P1C") &
                   runs.method.isin(["DHOA", "CQ", "CQA"]))]
    ids = causal[["logical_run_id", "archive_mode", "method", "offline_ratio",
                  "order_id", "training_seed"]]
    curves = curves.merge(ids, on=["logical_run_id", "archive_mode", "method",
        "offline_ratio", "order_id", "training_seed"], validate="m:1")
    adaptation = (curves.groupby(["logical_run_id", "stage_index"], observed=True)
        .normalized_travel_time.mean().rename("current_adaptation_aulc").reset_index())
    lower = lower.merge(ids, on="logical_run_id", validate="m:1")
    lower = lower.sort_values(["logical_run_id", "evaluation_network", "stage_index"])
    lower["previous_travel_time"] = lower.groupby(
        ["logical_run_id", "evaluation_network"], observed=True).travel_time.shift(1)
    lower["incremental_forgetting"] = lower.travel_time - lower.previous_travel_time
    old = lower[lower.evaluation_network != lower.training_network]
    forgetting = (old.groupby(["logical_run_id", "stage_index"], observed=True)
        .incremental_forgetting.mean().rename("old_scene_forgetting_mean").reset_index())
    table = ids.merge(adaptation, on="logical_run_id", validate="1:m")
    table = table.merge(forgetting, on=["logical_run_id", "stage_index"],
                        how="left", validate="1:1")
    table.loc[table.stage_index == 1, "old_scene_forgetting_mean"] = 0
    baseline = table[table.method == "CONT"][["order_id", "training_seed",
        "stage_index", "current_adaptation_aulc", "old_scene_forgetting_mean"]]
    baseline = baseline.rename(columns={
        "current_adaptation_aulc": "cont_current_adaptation_aulc",
        "old_scene_forgetting_mean": "cont_old_scene_forgetting_mean"})
    table = table.merge(baseline,
        on=["order_id", "training_seed", "stage_index"], validate="m:1")
    table["adaptation_difference_vs_cont"] = (
        table.current_adaptation_aulc - table.cont_current_adaptation_aulc)
    table["forgetting_difference_vs_cont"] = (
        table.old_scene_forgetting_mean - table.cont_old_scene_forgetting_mean)
    columns = ["order_id", "training_seed", "stage_index", "method", "archive_mode",
        "offline_ratio", "current_adaptation_aulc", "old_scene_forgetting_mean",
        "adaptation_difference_vs_cont", "forgetting_difference_vs_cont"]
    return table[columns].sort_values(columns[:4]).reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target = bellman_target_table()
    relation = scene_relation_matrix(target)
    action_target = action_conditioned_target_table()
    plan34 = plan34_transition_table()
    ha = ha_improvement_table()
    write_csv(target, "bellman_target_aggregate.csv")
    write_csv(relation, "scene_relation_matrix.csv")
    write_csv(action_target, "action_conditioned_bellman_target_heterogeneity.csv")
    write_csv(plan34, "plan34_transition_level_results.csv")
    write_csv(ha, "ha_method_improvement.csv")
    semantics = {
        "source_only": True, "training_or_sumo_launched": False,
        "difference_direction": "B-minus-A; HA is method-minus-CONT",
        "bellman_time_control": "none (checkpoint targets lack clock-window conditioning)",
        "bootstrap_difference": "gamma*max Q_target(s_next,a) component difference",
        "cluster_ci": {"unit": "matched training-seed aggregate",
            "seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES},
        "support_overlap": "exact common state+phase+action group count",
        "strict_state_increment": "B-minus-A difference in mean next-state increment L1",
        "scene_matrix_target": "median over evaluator-specific aggregate target differences",
        "action_conditioned_aggregation": "matched behavior seeds are averaged within each of 20 evaluator checkpoints, then checkpoints are averaged",
        "action_target_difference_std": "sample standard deviation of the eight action-level aggregate target differences within a data scope and scene pair",
        "action_order_reversal": "true when aggregate B-minus-A target differences include both positive and negative actions within a data scope and scene pair",
        "checkpoint_direction_consistency": "fraction of 20 evaluator checkpoints whose action-level target-difference sign matches the across-checkpoint mean sign",
        "zero_shot_drop": "zero_shot_normalized_travel_time - 1",
        "incremental_forgetting": "old-scene travel time after current minus previous stage",
        "final_retention": "T4 travel time / Plan 1 same-scene reference",
        "ha_scope": "E4 causal CONT, P1C-DHOA-R25, P1C-CQ-R75, P1C-CQA-R75",
        "row_counts": {"bellman_target_aggregate": len(target),
            "scene_relation_matrix": len(relation),
            "action_conditioned_bellman_target_heterogeneity": len(action_target),
            "plan34_transition_level_results": len(plan34),
            "ha_method_improvement": len(ha)}}
    (OUT / "table_semantics.json").write_text(
        json.dumps(semantics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(semantics["row_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
