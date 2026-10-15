#!/usr/bin/env python3
"""Action-ranking misordering audit on the formal counterfactual rollouts.

Reads the existing allocation_accuracy_formal_cf artifacts:
  - allocation_accuracy_agent_predictions.csv  (per method x state: q_0..q_7 + chosen action)
  - allocation_accuracy_all_action_rollouts.csv (per state x action: queue/delay/throughput + cost at 30/60/90 s)
  - allocation_accuracy_oracle_labels.csv      (pre-registered oracle labels)

For every (method, scene, state) it compares the Q-value action ordering with
the realised-outcome ordering (utility = -cost_h) and quantifies the realised
traffic loss of the chosen action. Pure post-hoc analysis; no simulation.
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
DEFAULT_SOURCE = REPO_ROOT / "data/output_data" / "allocation_accuracy_formal_cf"
DEFAULT_OUT = REPO_ROOT / "data/output_data" / "action_ranking_audit"

ACTION_COUNT = 8
PRIMARY_HORIZON = 60
HORIZONS = (30, 60, 90)
SCENE_ORDER = ("sumohz1x1_config2", "sumohz1x1", "sumohz1x1_config4", "sumohz1x1_config3")


def wilson_ci(hits: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 10000) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(values[0]), float(values[0])
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def pairwise_inversion(q: np.ndarray, u: np.ndarray) -> Tuple[int, int]:
    """Count action pairs where the Q order disagrees with the utility order."""
    inversions = 0
    comparable = 0
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            du = u[i] - u[j]
            if du == 0:
                continue
            comparable += 1
            dq = q[i] - q[j]
            if dq == 0 or np.sign(dq) != np.sign(du):
                inversions += 1
    return inversions, comparable


def true_rank(u: np.ndarray, action: int) -> int:
    """Competition rank of `action` under utility `u` (1 = strictly best)."""
    return int(np.sum(u > u[action])) + 1


def load_inputs(source_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(source_dir / "allocation_accuracy_agent_predictions.csv")
    rollouts = pd.read_csv(source_dir / "allocation_accuracy_all_action_rollouts.csv")
    labels = pd.read_csv(source_dir / "allocation_accuracy_oracle_labels.csv")
    return predictions, rollouts, labels


def build_state_table(rollouts: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per (scene, state_id, candidate_action) with realised outcomes."""
    df = rollouts.copy()
    df["safety_valid"] = df["safety_valid"].astype(str).str.lower().eq("true")
    df = df[df["safety_valid"]]
    df = df[df["rollout_error"].isna() | df["rollout_error"].astype(str).str.strip().eq("")]
    keep = ["scene", "state_id", "candidate_action", "source_run_id", "order", "training_seed"]
    for h in HORIZONS:
        keep += [f"cost_{h}", f"queue_auc_{h}", f"real_delay_{h}", f"throughput_{h}"]
    df = df[keep]
    lab = labels[["scene", "state_id", "oracle_action_60", "oracle_margin_60",
                  "labels_consistent_30_60_90", "high_confidence"]].copy()
    return df.merge(lab, on=["scene", "state_id"], how="left")


def per_state_metrics(pred_row: Mapping[str, object], state_actions: pd.DataFrame) -> Optional[Dict[str, object]]:
    q = np.asarray([float(pred_row[f"q_{a}"]) for a in range(ACTION_COUNT)], dtype=float)
    if len(state_actions) != ACTION_COUNT:
        return None
    state_actions = state_actions.sort_values("candidate_action")
    cost60 = state_actions["cost_60"].to_numpy(dtype=float)
    if np.isnan(cost60).any():
        return None
    u = -cost60
    predicted = int(pred_row["predicted_action"])
    oracle = int(pred_row["oracle_action_60"])
    inv, comparable = pairwise_inversion(q, u)
    row = state_actions[state_actions["candidate_action"] == predicted].iloc[0]
    best = state_actions[state_actions["candidate_action"] == oracle].iloc[0]
    out: Dict[str, object] = {
        "spearman_60": stats.spearmanr(q, u).statistic,
        "kendall_60": stats.kendalltau(q, u).statistic,
        "pairwise_inversions_60": inv,
        "pairwise_comparable_60": comparable,
        "inversion_rate_60": inv / comparable if comparable else np.nan,
        "ranking_perfect_60": inv == 0,
        "predicted_true_rank_60": true_rank(u, predicted),
        "pred_in_bottom_half": true_rank(u, predicted) > ACTION_COUNT / 2,
        "q_top2_gap": float(np.sort(q)[-1] - np.sort(q)[-2]),
        "cost_margin_60": float(np.sort(cost60)[1] - np.sort(cost60)[0]),
        "regret_cost_60": float(row["cost_60"] - best["cost_60"]),
        "regret_queue_auc_60": float(row["queue_auc_60"] - best["queue_auc_60"]),
        "regret_real_delay_60": float(row["real_delay_60"] - best["real_delay_60"]),
        "regret_throughput_60": float(row["throughput_60"] - best["throughput_60"]),
    }
    for h in (30, 90):
        cost_h = state_actions[f"cost_{h}"].to_numpy(dtype=float)
        u_h = -cost_h
        best_h = float(cost_h.min())
        out[f"spearman_{h}"] = stats.spearmanr(q, u_h).statistic
        out[f"regret_cost_{h}"] = float(row[f"cost_{h}"] - best_h)
        out[f"pred_is_best_{h}"] = float(row[f"cost_{h}"]) == best_h
    out["top1_correct"] = predicted == oracle
    out["chosen_in_top2"] = true_rank(u, predicted) <= 2
    out["high_confidence"] = bool(pred_row.get("high_confidence", False))
    out["labels_consistent"] = bool(pred_row.get("labels_consistent_30_60_90", False))
    return out


def aggregate(rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Aggregate per-state metrics at (method,) and (method, scene) levels."""
    records: List[Dict[str, object]] = []
    scopes = [("all_scenes", rows)]
    for scene in SCENE_ORDER:
        sub = rows[rows["scene"] == scene]
        if len(sub):
            scopes.append((scene, sub))
    for method in sorted(rows["method"].unique()):
        for scope, scope_rows_all in scopes:
            sub = scope_rows_all[scope_rows_all["method"] == method]
            if not len(sub):
                continue
            n = len(sub)
            top1_hits = int(sub["top1_correct"].sum())
            lo, hi = wilson_ci(top1_hits, n)
            reg = sub["regret_cost_60"].to_numpy()
            r_lo, r_hi = bootstrap_mean_ci(reg, rng)
            wrong = sub[~sub["top1_correct"]]
            rec: Dict[str, object] = {
                "method": method,
                "scope": scope,
                "states": n,
                "top1_pct": 100.0 * top1_hits / n,
                "top1_wilson_low": 100.0 * lo,
                "top1_wilson_high": 100.0 * hi,
                "top2_pct": 100.0 * float(sub["chosen_in_top2"].mean()),
                "mean_predicted_true_rank": float(sub["predicted_true_rank_60"].mean()),
                "pct_pred_bottom_half": 100.0 * float(sub["pred_in_bottom_half"].mean()),
                "mean_spearman_60": float(sub["spearman_60"].mean()),
                "median_spearman_60": float(sub["spearman_60"].median()),
                "pct_spearman_nonpos": 100.0 * float((sub["spearman_60"] <= 0).mean()),
                "mean_kendall_60": float(sub["kendall_60"].mean()),
                "pct_ranking_perfect": 100.0 * float(sub["ranking_perfect_60"].mean()),
                "mean_inversion_rate_60": float(sub["inversion_rate_60"].mean()),
                "pct_states_any_inversion": 100.0 * float((sub["pairwise_inversions_60"] > 0).mean()),
                "mean_regret_cost_60": float(reg.mean()),
                "regret_cost_60_ci_low": r_lo,
                "regret_cost_60_ci_high": r_hi,
                "median_regret_cost_60": float(np.median(reg)),
                "pct_regret_gt_0.10": 100.0 * float((reg > 0.10).mean()),
                "pct_regret_gt_0.25": 100.0 * float((reg > 0.25).mean()),
                "mean_dqueue_auc_60": float(sub["regret_queue_auc_60"].mean()),
                "mean_dreal_delay_60": float(sub["regret_real_delay_60"].mean()),
                "mean_dthroughput_60": float(sub["regret_throughput_60"].mean()),
                "mean_regret_cost_60_when_wrong": float(wrong["regret_cost_60"].mean()) if len(wrong) else 0.0,
                "mean_spearman_30": float(sub["spearman_30"].mean()),
                "mean_spearman_90": float(sub["spearman_90"].mean()),
                "mean_regret_cost_30": float(sub["regret_cost_30"].mean()),
                "mean_regret_cost_90": float(sub["regret_cost_90"].mean()),
            }
            records.append(rec)
    return pd.DataFrame(records)


def paired_method_deltas(rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Paired R25/R50 vs CONT-DQN deltas on states where both predicted."""
    out: List[Dict[str, object]] = []
    cont = rows[rows["method"] == "CONT-DQN"].set_index(["scene", "state_id"])
    for method in ("P1C-DHOA-R25", "P1C-DHOA-R50"):
        other = rows[rows["method"] == method].set_index(["scene", "state_id"])
        common = cont.index.intersection(other.index)
        if not len(common):
            continue
        d_reg = (other.loc[common, "regret_cost_60"] - cont.loc[common, "regret_cost_60"]).to_numpy()
        d_rank = (other.loc[common, "predicted_true_rank_60"] - cont.loc[common, "predicted_true_rank_60"]).to_numpy()
        d_spr = (other.loc[common, "spearman_60"] - cont.loc[common, "spearman_60"]).to_numpy()
        lo, hi = bootstrap_mean_ci(d_reg, rng)
        out.append({
            "comparison": f"{method} - CONT-DQN",
            "paired_states": int(len(common)),
            "mean_dregret_cost_60": float(d_reg.mean()),
            "dregret_ci_low": lo,
            "dregret_ci_high": hi,
            "pct_states_arm_lower_regret": 100.0 * float((d_reg < -1e-9).mean()),
            "pct_states_arm_higher_regret": 100.0 * float((d_reg > 1e-9).mean()),
            "mean_dtrue_rank": float(d_rank.mean()),
            "mean_dspearman": float(d_spr.mean()),
        })
    return pd.DataFrame(out)


def write_report(out_dir: Path, summary: pd.DataFrame, per_state: pd.DataFrame,
                 paired: pd.DataFrame, meta: Mapping[str, object]) -> None:
    lines = [
        "# 动作排序误排与反事实交通损失审计",
        "",
        "基于 `allocation_accuracy_formal_cf` 既有产物的事后分析：每个冻结状态（SUMO saveState + Python 状态共同恢复）",
        "枚举 8 个合法动作控制首个 10s 决策周期，随后交由同场景同 seed 的冻结独立 DQN 继续推演。",
        "效用口径：utility = -cost_h（cost = queue AUC / real_delay / 1-throughput 的 min-max 等权均值），主口径 h=60s，30/90s 为敏感性。",
        "",
        "## 数据覆盖",
        "",
        f"- 状态数 {meta['states']}（4 场景 × 64）；有效反事实分支 {meta['branches']}；",
        f"- 纳入方法：{', '.join(meta['methods'])}（各自有匹配 checkpoint 的状态数见表）；",
        f"- 高置信标签（30/60/90 一致且 60s margin≥0.05）：{meta['reliable']} 状态。",
        "",
        "## 排序一致性（Q 值序 vs 实际回报序）",
        "",
        "| method | scope | states | Top-1 % [Wilson] | Top-2 % | 选中动作真实名次均值 | 选入后半 % | Spearman 均值 | Kendall 均值 | 完美排序 % | 任一反序状态 % | 平均反序对率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['method']} | {r['scope']} | {int(r['states'])} | "
            f"{r['top1_pct']:.2f} [{r['top1_wilson_low']:.1f},{r['top1_wilson_high']:.1f}] | "
            f"{r['top2_pct']:.2f} | {r['mean_predicted_true_rank']:.2f} | {r['pct_pred_bottom_half']:.1f} | "
            f"{r['mean_spearman_60']:.3f} | {r['mean_kendall_60']:.3f} | "
            f"{r['pct_ranking_perfect']:.1f} | {r['pct_states_any_inversion']:.1f} | {r['mean_inversion_rate_60']:.3f} |"
        )
    lines += [
        "",
        "## 实际交通损失（chosen − oracle，60s 窗）",
        "",
        "| method | scope | states | 平均归一化 regret [boot95] | 中位 regret | regret>0.10 % | regret>0.25 % | Δqueue AUC (veh·s) | Δreal_delay (s) | Δthroughput (veh) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['method']} | {r['scope']} | {int(r['states'])} | "
            f"{r['mean_regret_cost_60']:.4f} [{r['regret_cost_60_ci_low']:.4f},{r['regret_cost_60_ci_high']:.4f}] | "
            f"{r['median_regret_cost_60']:.4f} | {r['pct_regret_gt_0.10']:.1f} | {r['pct_regret_gt_0.25']:.1f} | "
            f"{r['mean_dqueue_auc_60']:.1f} | {r['mean_dreal_delay_60']:.3f} | {r['mean_dthroughput_60']:.2f} |"
        )
    if len(paired):
        lines += ["", "## 半离线臂配对比较（同状态配对，相对 CONT-DQN）", "",
                  "| comparison | paired states | Δregret 均值 [boot95] | 该臂 regret 更低 % | 该臂 regret 更高 % | Δ真实名次均值 | ΔSpearman 均值 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for _, r in paired.iterrows():
            lines.append(
                f"| {r['comparison']} | {int(r['paired_states'])} | "
                f"{r['mean_dregret_cost_60']:.4f} [{r['dregret_ci_low']:.4f},{r['dregret_ci_high']:.4f}] | "
                f"{r['pct_states_arm_lower_regret']:.1f} | {r['pct_states_arm_higher_regret']:.1f} | "
                f"{r['mean_dtrue_rank']:.2f} | {r['mean_dspearman']:.3f} |"
            )
    lines += [
        "",
        "## 敏感性（30/90s 口径）",
        "",
        "| method | scope | mean Spearman@30 | mean Spearman@60 | mean Spearman@90 | mean regret@30 | mean regret@60 | mean regret@90 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['method']} | {r['scope']} | {r['mean_spearman_30']:.3f} | {r['mean_spearman_60']:.3f} | "
            f"{r['mean_spearman_90']:.3f} | {r['mean_regret_cost_30']:.4f} | {r['mean_regret_cost_60']:.4f} | {r['mean_regret_cost_90']:.4f} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "- “实际长期回报”是 60s 反事实窗内、由同场景同 seed 冻结独立 DQN 接管的复合代价，不是完整 episode 折现回报；30/90s 仅作标签稳健性对照。",
        "- 状态来自 CONT-FIFO 正式 run 的确定性重放快照（replay 校验 256/256 匹配），负荷档位覆盖以 mid/high 为主，不外推到低负荷或新需求。",
        "- R25/R50 仅在有匹配 checkpoint 的状态子集上解释（223/82 个状态）；与 CONT 的比较用同状态配对差。",
        "- 归一化 regret 是该状态 8 动作 min-max 代价差；原始 Δqueue/Δdelay/Δthroughput 为同一 chosen−oracle 差值，量纲见各列名。",
    ]
    (out_dir / "action_ranking_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    predictions, rollouts, labels = load_inputs(args.source)
    state_actions = build_state_table(rollouts, labels)

    # Merge prediction Q vectors with realised per-action outcomes.
    pred = predictions.copy()
    pred = pred[pred["prediction_error"].isna() | pred["prediction_error"].astype(str).str.strip().eq("")]
    merged = pred.merge(
        labels[["scene", "state_id", "oracle_action_60", "labels_consistent_30_60_90", "high_confidence"]],
        on=["scene", "state_id"], how="left", suffixes=("", "_label"),
    )

    records: List[Dict[str, object]] = []
    grouped = {k: g for k, g in state_actions.groupby(["scene", "state_id"])}
    skipped = 0
    for _, prow in merged.iterrows():
        key = (prow["scene"], prow["state_id"])
        sa = grouped.get(key)
        if sa is None:
            skipped += 1
            continue
        metrics = per_state_metrics(prow, sa)
        if metrics is None:
            skipped += 1
            continue
        rec = {
            "method": prow["method"],
            "scene": prow["scene"],
            "state_id": prow["state_id"],
            "order": prow.get("order", ""),
            "training_seed": prow.get("training_seed", ""),
            "predicted_action": int(prow["predicted_action"]),
            "oracle_action_60": int(prow["oracle_action_60"]),
        }
        rec.update(metrics)
        records.append(rec)
    per_state = pd.DataFrame(records)

    summary = aggregate(per_state, rng)
    paired = paired_method_deltas(per_state, rng)

    args.out.mkdir(parents=True, exist_ok=True)
    per_state.to_csv(args.out / "action_ranking_audit_per_state.csv", index=False)
    summary.to_csv(args.out / "action_ranking_audit_summary.csv", index=False)
    paired.to_csv(args.out / "action_ranking_audit_paired.csv", index=False)
    meta = {
        "states": int(labels.shape[0]),
        "branches": int(state_actions.shape[0]),
        "methods": sorted(per_state["method"].unique().tolist()),
        "reliable": int(labels["high_confidence"].astype(str).str.lower().eq("true").sum()),
        "skipped_prediction_rows": skipped,
    }
    (args.out / "action_ranking_audit_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.out, summary, per_state, paired, meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[done] outputs -> {args.out}")


if __name__ == "__main__":
    main()
