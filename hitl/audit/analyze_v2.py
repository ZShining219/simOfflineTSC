#!/usr/bin/env python3
"""Analysis for the counterfactual v2 rollouts.

Task 1: recompute action returns under the original DQN training reward
        (-12 * mean incoming waiting count per decision, gamma=0.95, 10 s
        cadence) and re-check whether the Q-value ranking error persists.
Task 2: compare 'indep' vs 'cont' continuation policies and quantify how much
        of the ranking error depends on the continuation choice.
Task 3: for clear error pairs, repeat across future seeds and evaluate
        whether stable and costly mis-rankings exist.
Task 4: stratify mis-ranking and loss by demand scene, load bin, phase and
        horizon.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V2 = REPO_ROOT / "data/output_data" / "action_ranking_audit" / "v2_rollouts"
DEFAULT_CF = REPO_ROOT / "data/output_data" / "allocation_accuracy_formal_cf"
DEFAULT_MANIFEST = (
    REPO_ROOT / "data/output_data" / "allocation_accuracy_formal_state" / "allocation_accuracy_state_manifest.csv"
)
ACTIONS = list(range(8))
HORIZON_CYCLES = {30: 3, 60: 6, 90: 9}
SCENES = ("sumohz1x1_config2", "sumohz1x1", "sumohz1x1_config4", "sumohz1x1_config3")


def load_jsonl_dir(path: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for file in sorted(path.glob("branches_shard*.jsonl")):
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)


def minmax(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        return np.full(len(arr), np.nan)
    lo, hi = arr.min(), arr.max()
    if np.isclose(lo, hi):
        return np.zeros(len(arr))
    return (arr - lo) / (hi - lo)


def cost_table(group: pd.DataFrame, horizon: int) -> pd.Series:
    """Replicate the original composite cost on a state's 8-action subset."""
    group = group.sort_values("candidate_action")
    q = minmax(group[f"queue_auc_{horizon}"])
    d = minmax(group[f"real_delay_{horizon}"])
    tp = minmax(group[f"throughput_{horizon}"])
    cost = (q + d + (1.0 - tp)) / 3.0
    return pd.Series(cost, index=group["candidate_action"].to_numpy())


def pairwise_inversions(order_a: np.ndarray, order_b: np.ndarray) -> Tuple[int, int]:
    inv = 0
    comparable = 0
    for i in range(len(order_a)):
        for j in range(i + 1, len(order_a)):
            db = order_b[i] - order_b[j]
            if db == 0:
                continue
            comparable += 1
            da = order_a[i] - order_a[j]
            if da == 0 or np.sign(da) != np.sign(db):
                inv += 1
    return inv, comparable


def state_metrics(
    q: np.ndarray,
    utility: np.ndarray,
    chosen: int,
    best: int,
    chosen_raw: float,
    best_raw: float,
) -> Dict[str, float]:
    inv, comp = pairwise_inversions(q, utility)
    chosen_rank = int(np.sum(utility > utility[chosen])) + 1
    return {
        "spearman": stats.spearmanr(q, utility).statistic,
        "kendall": stats.kendalltau(q, utility).statistic,
        "inversions": inv,
        "inversion_rate": inv / comp if comp else np.nan,
        "top1": float(chosen == best),
        "chosen_true_rank": float(chosen_rank),
        "bottom_half": float(chosen_rank > 4),
        "regret": float(best_raw - chosen_raw),
    }


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(values[0]), float(values[0])
    means = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def wilson(hits: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = hits / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def build_frame(branches: pd.DataFrame, predictions: pd.DataFrame,
                manifest: pd.DataFrame) -> pd.DataFrame:
    """Per (state, continuation): utilities and metrics for the CONT-DQN Q."""
    valid = branches[branches["safety_valid"].astype(str).str.lower().eq("true")].copy()
    cont_pred = predictions[predictions["method"] == "CONT-DQN"].copy()
    cont_pred = cont_pred[
        cont_pred["prediction_error"].isna()
        | cont_pred["prediction_error"].astype(str).str.strip().eq("")
    ]
    state_meta = manifest.set_index("state_id")
    records: List[Dict] = []
    for (scene, state_id, continuation), group in valid.groupby(["scene", "state_id", "continuation"]):
        if len(group) != len(ACTIONS):
            continue
        group = group.sort_values("candidate_action")
        qrow = cont_pred[(cont_pred["scene"] == scene) & (cont_pred["state_id"] == state_id)]
        if qrow.empty:
            continue
        q = np.asarray([float(qrow.iloc[0][f"q_{a}"]) for a in ACTIONS])
        chosen = int(qrow.iloc[0]["predicted_action"])
        rec: Dict[str, object] = {
            "scene": scene, "state_id": state_id, "continuation": continuation,
            "chosen_action": chosen,
        }
        for key in ("phase", "load_bin", "time_bin", "total_vehicle_count", "direction_imbalance",
                    "order", "training_seed", "source_run_id", "simulation_time_seconds"):
            rec[key] = state_meta.loc[state_id][key] if state_id in state_meta.index else group.iloc[0].get(key)
        for horizon in (30, 60, 90):
            cost = cost_table(group, horizon)
            cost = cost.reindex(ACTIONS)
            if cost.isna().any():
                continue
            utility_cost = -cost.to_numpy()
            ret = group.sort_values("candidate_action")[f"dqn_return_{horizon}"].to_numpy(dtype=float)
            best_cost = int(cost.idxmin())
            best_ret = int(np.argmax(ret))
            u_cost_metrics = state_metrics(q, utility_cost, chosen, best_cost,
                                           utility_cost[chosen], utility_cost[best_cost])
            u_ret_metrics = state_metrics(q, ret, chosen, best_ret,
                                          ret[chosen], ret[best_ret])
            rec[f"oracle_cost_{horizon}"] = best_cost
            rec[f"oracle_return_{horizon}"] = best_ret
            for prefix, m in (("cost", u_cost_metrics), ("ret", u_ret_metrics)):
                for name, value in m.items():
                    rec[f"{prefix}_{name}_{horizon}"] = value
            queue = group.sort_values("candidate_action")["queue_auc_" + str(horizon)].to_numpy(dtype=float)
            rec[f"queue_gap_{horizon}"] = float(queue[chosen] - queue[best_cost])
            ret_gap = ret[best_ret] - ret[chosen]
            rec[f"return_gap_{horizon}"] = float(ret_gap)
        records.append(rec)
    return pd.DataFrame(records)


def task1_summary(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for continuation in ("indep", "cont"):
        sub = frame[frame["continuation"] == continuation]
        for horizon in (30, 60, 90):
            ret_cols = [c for c in sub.columns if c.startswith(f"ret_") and c.endswith(f"_{horizon}")]
            if not ret_cols:
                continue
            valid = sub.dropna(subset=[f"ret_spearman_{horizon}"])
            if not len(valid):
                continue
            top1 = valid[f"ret_top1_{horizon}"].to_numpy()
            lo, hi = wilson(int(top1.sum()), len(top1))
            regret = valid[f"ret_regret_{horizon}"].to_numpy()
            rlo, rhi = bootstrap_ci(regret, rng)
            rows.append({
                "continuation": continuation, "horizon": horizon, "states": len(valid),
                "top1_pct": 100 * top1.mean(), "top1_lo": 100 * lo, "top1_hi": 100 * hi,
                "top2like_pct": 100 * (valid[f"ret_chosen_true_rank_{horizon}"] <= 2).mean(),
                "mean_true_rank": valid[f"ret_chosen_true_rank_{horizon}"].mean(),
                "bottom_half_pct": 100 * valid[f"ret_bottom_half_{horizon}"].mean(),
                "spearman_mean": valid[f"ret_spearman_{horizon}"].mean(),
                "kendall_mean": valid[f"ret_kendall_{horizon}"].mean(),
                "any_inversion_pct": 100 * (valid[f"ret_inversions_{horizon}"] > 0).mean(),
                "inversion_rate_mean": valid[f"ret_inversion_rate_{horizon}"].mean(),
                "regret_mean_reward_units": regret.mean(),
                "regret_ci_lo": rlo, "regret_ci_hi": rhi,
                "queue_gap_mean_veh_s": valid[f"queue_gap_{horizon}"].mean(),
                "return_gap_mean_reward_units": valid[f"return_gap_{horizon}"].mean(),
            })
    return pd.DataFrame(rows)


def task2_continuation(frame: pd.DataFrame, rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Paired indep-vs-cont comparison on states complete under both."""
    metric_cols = [
        "oracle_cost_60", "oracle_return_60",
        "cost_top1_60", "ret_top1_60", "cost_regret_60", "ret_regret_60",
        "cost_spearman_60", "ret_spearman_60", "cost_chosen_true_rank_60", "ret_chosen_true_rank_60",
    ]
    rows = []
    for metric in metric_cols:
        wide = frame.pivot_table(index=["scene", "state_id"], columns="continuation", values=metric)
        wide = wide.dropna(subset=["indep", "cont"])
        if not len(wide):
            continue
        delta = wide["cont"] - wide["indep"]
        lo, hi = bootstrap_ci(delta.to_numpy(), rng)
        rows.append({
            "metric": metric, "paired_states": len(wide),
            "indep_mean": float(wide["indep"].mean()),
            "cont_mean": float(wide["cont"].mean()),
            "mean_delta_cont_minus_indep": float(delta.mean()),
            "ci_lo": lo, "ci_hi": hi,
            "pct_states_changed": 100.0 * float((delta != 0).mean()),
        })
    cost_oracle = frame.pivot_table(index=["scene", "state_id"], columns="continuation", values="oracle_cost_60").dropna()
    ret_oracle = frame.pivot_table(index=["scene", "state_id"], columns="continuation", values="oracle_return_60").dropna()
    wrong_indep = frame[frame["continuation"] == "indep"].set_index(["scene", "state_id"])["ret_top1_60"]
    wrong_cont = frame[frame["continuation"] == "cont"].set_index(["scene", "state_id"])["ret_top1_60"]
    common = wrong_indep.index.intersection(wrong_cont.index)
    wi, wc = wrong_indep.loc[common], wrong_cont.loc[common]
    crosstab = {
        "paired_states": int(len(common)),
        "cost_oracle_same_pct": 100.0 * float((cost_oracle["indep"] == cost_oracle["cont"]).mean()),
        "return_oracle_same_pct": 100.0 * float((ret_oracle["indep"] == ret_oracle["cont"]).mean()),
        "wrong_under_both_pct": 100.0 * float(((wi == 0) & (wc == 0)).mean()),
        "wrong_only_indep_pct": 100.0 * float(((wi == 0) & (wc == 1)).mean()),
        "wrong_only_cont_pct": 100.0 * float(((wi == 1) & (wc == 0)).mean()),
        "correct_under_both_pct": 100.0 * float(((wi == 1) & (wc == 1)).mean()),
    }
    return pd.DataFrame(rows), pd.DataFrame([crosstab])


def task3_stability(branches: pd.DataFrame, pairs: Mapping[str, Sequence[int]],
                    manifest: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    valid = branches[branches["safety_valid"].astype(str).str.lower().eq("true")].copy()
    rows = []
    for state_id, pair in pairs.items():
        a_pred, a_oracle = int(pair[0]), int(pair[1])
        sub = valid[valid["state_id"] == state_id]
        for seed, seed_group in sub.groupby("future_seed"):
            got = seed_group.set_index("candidate_action")
            if a_pred not in got.index or a_oracle not in got.index:
                continue
            for horizon in (60,):
                cost_pair = minmax([got.loc[a_pred, f"queue_auc_{horizon}"],
                                    got.loc[a_oracle, f"queue_auc_{horizon}"]])
                delay_pair = minmax([got.loc[a_pred, f"real_delay_{horizon}"],
                                     got.loc[a_oracle, f"real_delay_{horizon}"]])
                tp_pair = minmax([got.loc[a_pred, f"throughput_{horizon}"],
                                  got.loc[a_oracle, f"throughput_{horizon}"]])
                cost = (cost_pair + delay_pair + (1.0 - tp_pair)) / 3.0
                if not np.all(np.isfinite(cost)):
                    continue
                rows.append({
                    "state_id": state_id, "future_seed": str(seed),
                    "action_pred": a_pred, "action_oracle": a_oracle,
                    f"cost_pred_{horizon}": float(cost[0]), f"cost_oracle_{horizon}": float(cost[1]),
                    f"oracle_still_better_{horizon}": float(cost[1] < cost[0]),
                    f"queue_gap_{horizon}": float(got.loc[a_pred, f"queue_auc_{horizon}"] - got.loc[a_oracle, f"queue_auc_{horizon}"]),
                    f"return_gap_{horizon}": float(got.loc[a_oracle, f"dqn_return_{horizon}"] - got.loc[a_pred, f"dqn_return_{horizon}"]),
                })
    per_seed = pd.DataFrame(rows)
    summary_rows = []
    for state_id, group in per_seed.groupby("state_id"):
        horizon = 60
        better = group[f"oracle_still_better_{horizon}"]
        summary_rows.append({
            "state_id": state_id,
            "seeds_evaluated": len(group),
            "seeds_oracle_better_pct": 100.0 * float(better.mean()),
            "mean_cost_gap": float((group[f"cost_pred_{horizon}"] - group[f"cost_oracle_{horizon}"]).mean()),
            "mean_queue_gap_veh_s": float(group[f"queue_gap_{horizon}"].mean()),
            "mean_return_gap_reward": float(group[f"return_gap_{horizon}"].mean()),
            "sign_stable": bool((better == 1.0).all() or (better == 0.0).all()),
        })
    return per_seed, pd.DataFrame(summary_rows)


def task4_strata(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    sub = frame[frame["continuation"] == "indep"].dropna(subset=["ret_spearman_60"])
    strata = [("scene", sub["scene"].astype(str)),
              ("load_bin", sub["load_bin"].astype(str)),
              ("phase", sub["phase"].astype(int).astype(str)),
              ("time_bin", sub["time_bin"].astype(str))]
    for horizon in (30, 60, 90):
        for name, series in strata:
            for level, idx in series.groupby(series).groups.items():
                cell = sub.loc[idx]
                n = len(cell)
                if n < 3:
                    continue
                lo, hi = wilson(int(cell[f"ret_top1_{horizon}"].sum()), n)
                rows.append({
                    "horizon": horizon, "stratum": name, "level": level, "states": n,
                    "top1_pct": 100 * cell[f"ret_top1_{horizon}"].mean(), "top1_lo": 100 * lo, "top1_hi": 100 * hi,
                    "spearman_mean": cell[f"ret_spearman_{horizon}"].mean(),
                    "bottom_half_pct": 100 * cell[f"ret_bottom_half_{horizon}"].mean(),
                    "regret_mean": cell[f"ret_regret_{horizon}"].mean(),
                    "queue_gap_mean": cell[f"queue_gap_{horizon}"].mean(),
                    "cost_top1_pct": 100 * cell[f"cost_top1_{horizon}"].mean(),
                    "cost_regret_mean": cell[f"cost_regret_{horizon}"].mean(),
                })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2)
    parser.add_argument("--cf-dir", type=Path, default=DEFAULT_CF)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/output_data" / "action_ranking_audit")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    cont = load_jsonl_dir(args.v2_dir / "continuations")
    multi = load_jsonl_dir(args.v2_dir / "multiseed") if (args.v2_dir / "multiseed").exists() else pd.DataFrame()
    pairs = json.loads((args.v2_dir / "multiseed_pairs.json").read_text()) if (args.v2_dir / "multiseed_pairs.json").exists() else {}
    predictions = pd.read_csv(args.cf_dir / "allocation_accuracy_agent_predictions.csv")
    manifest = pd.read_csv(args.manifest)

    frame = build_frame(cont, predictions, manifest)
    t1 = task1_summary(frame, rng)
    t2, t2_oracle = task2_continuation(frame, rng)
    t3_seed, t3_summary = task3_stability(multi, pairs, manifest) if len(multi) else (pd.DataFrame(), pd.DataFrame())
    t4 = task4_strata(frame, rng)

    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "v2_per_state_metrics.csv", index=False)
    t1.to_csv(args.out / "v2_task1_reward_return_summary.csv", index=False)
    t2.to_csv(args.out / "v2_task2_continuation_paired.csv", index=False)
    t2_oracle.to_csv(args.out / "v2_task2_oracle_agreement.csv", index=False)
    if len(t3_seed):
        t3_seed.to_csv(args.out / "v2_task3_seed_pair_detail.csv", index=False)
        t3_summary.to_csv(args.out / "v2_task3_seed_stability.csv", index=False)
    t4.to_csv(args.out / "v2_task4_strata.csv", index=False)

    print("=== TASK1 (DQN-reward discounted return, indep continuation) ===")
    print(t1[t1["continuation"] == "indep"].to_string(index=False))
    print("\n=== TASK2 paired deltas (cont - indep) ===")
    print(t2.to_string(index=False))
    print("\n=== TASK2 oracle agreement ===")
    print(t2_oracle.to_string(index=False))
    if len(t3_summary):
        print("\n=== TASK3 stable mis-ranking summary ===")
        print(t3_summary.describe().to_string())
        stable_wrong = t3_summary[(t3_summary["sign_stable"]) & (t3_summary["seeds_oracle_better_pct"] == 100.0)]
        print(f"pairs where oracle stays better on ALL seeds: {len(stable_wrong)}/{len(t3_summary)}")
    print("\n=== TASK4 strata (60 s) ===")
    print(t4[t4["horizon"] == 60].to_string(index=False))
    print(f"\n[done] outputs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
