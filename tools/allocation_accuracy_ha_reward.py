#!/usr/bin/env python3
"""Read-only HA-reward-consistent three-level validation.

The stored counterfactual queue AUC is reward-equivalent to the training
reward in the current one-intersection, eight-incoming-lane setup:
reward_t = -12 * mean(waiting_count_8) = -1.5 * queue_sum_t.
This computes an undiscounted fixed-horizon equivalent oracle from existing
CSV files. It does not reconstruct gamma-discounted Bellman returns because
per-step rewards are not stored in the CSV.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/output_data/allocation_accuracy"
DEFAULT_OUTPUT = DEFAULT_INPUT / "ha_reward_validation"
METHOD = "P1C-DHOA-R25"
HORIZONS = (30, 60, 90)
REWARD_SCALE = -1.5


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    index = max(0, min(len(values) - 1, math.ceil(q * len(values)) - 1))
    return values[index]


def observed_actions(value):
    return {int(item) for item in json.loads(value or "[]")}


def rank_for_action(actions, action, field):
    values = {int(row["candidate_action"]): float(row[field]) for row in actions}
    ordered = sorted(values, key=lambda candidate: (values[candidate], candidate))
    best = values[ordered[0]]
    return ordered.index(action) + 1, values[action] - best, ordered[0], best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", default=METHOD)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "rollouts": input_dir / "allocation_accuracy_all_action_rollouts.csv",
        "predictions": input_dir / "allocation_accuracy_agent_predictions.csv",
        "evidence": input_dir / "experience_traceability_support" / "experience_traceability_state_evidence.csv",
        "states": input_dir / "allocation_accuracy_state_manifest.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    rollouts = defaultdict(list)
    for row in read_csv(paths["rollouts"]):
        if row.get("safety_valid", "").lower() == "true":
            rollouts[row["state_id"]].append(row)
    predictions = {
        row["state_id"]: row
        for row in read_csv(paths["predictions"])
        if row["method"] == args.method and row.get("predicted_action", "").strip()
    }
    evidence = {
        row["state_id"]: row
        for row in read_csv(paths["evidence"])
        if row["method"] == args.method
    }
    states = {row["state_id"]: row for row in read_csv(paths["states"])}

    details = []
    for state_id in sorted(states):
        if state_id not in predictions or len(rollouts[state_id]) != 8:
            continue
        prediction = predictions[state_id]
        evidence_row = evidence.get(state_id, {})
        predicted = int(prediction["predicted_action"])
        seen = observed_actions(evidence_row.get("training_observed_actions", "[]"))
        row = {
            "method": args.method,
            "state_id": state_id,
            "scene": states[state_id].get("scene", ""),
            "order": states[state_id].get("order", ""),
            "training_seed": states[state_id].get("training_seed", ""),
            "predicted_action": predicted,
            "training_reference_covered": evidence_row.get("training_reference_covered", "False"),
            "experience_supported": evidence_row.get("experience_supported", "False"),
            "training_observed_actions": json.dumps(sorted(seen)),
        }
        actions = rollouts[state_id]
        for horizon in HORIZONS:
            field = f"queue_auc_{horizon}"
            rank, queue_regret, oracle_action, best_queue = rank_for_action(actions, predicted, field)
            selected_queue = float(next(item for item in actions if int(item["candidate_action"]) == predicted)[field])
            row.update({
                f"ha_reward_oracle_action_{horizon}": oracle_action,
                f"ha_reward_top1_{horizon}": predicted == oracle_action,
                f"ha_reward_rank_{horizon}": rank,
                f"queue_regret_{horizon}": queue_regret,
                f"ha_reward_regret_{horizon}": queue_regret * abs(REWARD_SCALE),
                f"ha_reward_oracle_cumulative_{horizon}": REWARD_SCALE * best_queue,
                f"ha_reward_selected_cumulative_{horizon}": REWARD_SCALE * selected_queue,
                f"oracle_action_seen_in_training_{horizon}": oracle_action in seen,
            })
        details.append(row)

    summaries = []
    for horizon in HORIZONS:
        ranks = [int(row[f"ha_reward_rank_{horizon}"]) for row in details]
        regrets = [float(row[f"ha_reward_regret_{horizon}"]) for row in details]
        covered = [row for row in details if row["training_reference_covered"] == "True"]
        supported = sum(row["experience_supported"] == "True" for row in covered)
        oracle_seen = sum(bool(row[f"oracle_action_seen_in_training_{horizon}"]) for row in covered)
        oracle_seen_all = sum(bool(row[f"oracle_action_seen_in_training_{horizon}"]) for row in details)
        top1 = sum(bool(row[f"ha_reward_top1_{horizon}"]) for row in details)
        summaries.append({
            "method": args.method,
            "horizon_seconds": horizon,
            "states": len(details),
            "ha_reward_top1_count": top1,
            "ha_reward_top1_pct": 100 * top1 / len(details),
            "rank_at_2_pct": 100 * sum(value <= 2 for value in ranks) / len(ranks),
            "rank_at_3_pct": 100 * sum(value <= 3 for value in ranks) / len(ranks),
            "mean_rank": statistics.mean(ranks),
            "median_rank": statistics.median(ranks),
            "p90_rank": percentile(ranks, 0.90),
            "mean_ha_reward_regret": statistics.mean(regrets),
            "median_ha_reward_regret": statistics.median(regrets),
            "p90_ha_reward_regret": percentile(regrets, 0.90),
            "training_reference_covered_states": len(covered),
            "experience_supported_states": supported,
            "experience_support_pct_on_covered": 100 * supported / len(covered),
            "oracle_action_seen_states_on_covered": oracle_seen,
            "oracle_action_seen_pct_on_covered": 100 * oracle_seen / len(covered),
            "oracle_action_seen_states_all_states": oracle_seen_all,
            "oracle_action_seen_pct_all_states": 100 * oracle_seen_all / len(details),
        })

    oracle_consistent = sum(
        len({row[f"ha_reward_oracle_action_{h}"] for h in HORIZONS}) == 1
        for row in details
    )
    all_horizon_hits = sum(
        all(bool(row[f"ha_reward_top1_{h}"]) for h in HORIZONS)
        for row in details
    )
    audit = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_read_only_derived_audit",
        "method": args.method,
        "levels": {
            "level_1": "frozen action equals HA-reward-consistent full-action oracle",
            "level_2": "frozen action observed in training for exact state/phase key",
            "level_3": "HA-reward oracle action observed in training for exact state/phase key",
        },
        "reward_alignment": {
            "training_reward": "-12 * mean(all 8 incoming-lane lane_waiting_count)",
            "queue_equivalent_reward": "-1.5 * queue_sum",
            "reward_scale": REWARD_SCALE,
            "action_interval_seconds": 10,
            "gamma_configured": 0.95,
            "computed_return": "undiscounted fixed-horizon cumulative reward equivalent",
            "discounted_return_status": "not computable from existing aggregate CSV",
        },
        "counts": {
            "evaluation_states": len(details),
            "training_reference_covered_states": sum(row["training_reference_covered"] == "True" for row in details),
            "oracle_consistent_30_60_90_states": oracle_consistent,
            "frozen_action_hits_all_30_60_90_states": all_horizon_hits,
        },
        "source_paths": {
            "all_action_rollouts": str(paths["rollouts"].relative_to(ROOT)),
            "agent_predictions": str(paths["predictions"].relative_to(ROOT)),
            "experience_state_evidence": str(paths["evidence"].relative_to(ROOT)),
            "state_manifest": str(paths["states"].relative_to(ROOT)),
            "reward_definition_module": "sequential/agent.py; generator/lane_vehicle.py",
            "training_loop_module": "sequential/trainer.py",
            "counterfactual_module": "tools/allocation_accuracy_counterfactual.py",
        },
        "source_digests": {key: sha256(path) for key, path in paths.items()},
        "calculation_pseudocode": [
            "reward_t = -12 * mean(waiting_count_8) = -1.5 * queue_sum_t",
            "oracle = argmin_a(queue_auc_horizon[state, a])",
            "level_1 = frozen_action == oracle",
            "level_2 = frozen_action in training_observed_actions for covered exact key",
            "level_3 = oracle in training_observed_actions for covered exact key",
        ],
        "limitations": [
            "Continuation in the existing rollouts is frozen Independent DQN, not HA-DHOA self-closed-loop.",
            "The result is reward-equivalent for undiscounted fixed horizons; gamma=0.95 discounted return cannot be reconstructed from aggregate CSV.",
            "Experience support proves observed action lineage, not individual action optimality.",
            "Oracle-action-seen rates are reported both on covered states and all evaluation states.",
            "No checkpoint, training data, or SUMO simulation was modified or started.",
        ],
    }
    report = [
        "# HA-reward 一致性三层验证", "",
        "本审计复用已有全动作反事实 CSV，不重新训练、不重新运行 SUMO。", "",
        "## 口径", "",
        "HA-DHOA 的训练 reward 为 -12 × 八个入口车道 waiting count 的平均值。当前单路口八车道场景下，它等价于 -1.5 × queue_sum，所以固定时域内最大累计 reward 与最小 queue AUC 完全同序。", "",
        "现有 CSV 仅保存 30/60/90 秒累计 queue AUC，没有逐秒或逐决策 reward 序列。因此这里报告的是未折扣固定时域的 HA-reward 等价 Top-1，不宣称已经重建 gamma=0.95 的 Bellman 折扣回报。", "",
        "| 时域 | HA-reward Top-1 | Rank@2 | Rank@3 | 平均 rank | 平均 reward regret | 经验支持率 | Oracle动作出现率（覆盖状态） | Oracle动作出现率（全部状态） |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        report.append(
            f"| {row['horizon_seconds']} s | {row['ha_reward_top1_pct']:.2f}% ({row['ha_reward_top1_count']}/{row['states']}) | "
            f"{row['rank_at_2_pct']:.2f}% | {row['rank_at_3_pct']:.2f}% | {row['mean_rank']:.3f} | "
            f"{row['mean_ha_reward_regret']:.3f} | {row['experience_support_pct_on_covered']:.2f}% ({row['experience_supported_states']}/{row['training_reference_covered_states']}) | "
            f"{row['oracle_action_seen_pct_on_covered']:.2f}% ({row['oracle_action_seen_states_on_covered']}/{row['training_reference_covered_states']}) | "
            f"{row['oracle_action_seen_pct_all_states']:.2f}% ({row['oracle_action_seen_states_all_states']}/{row['states']}) |"
        )
    report += [
        "", f"- 30/60/90 秒 HA-reward oracle 完全一致：{oracle_consistent}/{len(details)}。",
        f"- 冻结动作在 30/60/90 秒均命中 HA-reward oracle：{all_horizon_hits}/{len(details)}。", "",
        "## 三层含义", "",
        "第一层衡量冻结 HA-DHOA 是否恢复了自身 queue/waiting reward 方向下的全动作参考排序；第二层衡量冻结动作是否有训练经验来源；第三层衡量训练探索是否实际尝试过 reward 参考动作。三者不能互相替代。", "",
        "当前后续接管仍是 Independent DQN，因此该结果是固定接管条件下的 HA-reward 一致性诊断，不是 HA-DHOA 自身闭环全局最优。",
    ]
    write_csv(output_dir / "ha_reward_validation_state.csv", details)
    write_csv(output_dir / "ha_reward_validation_summary.csv", summaries)
    (output_dir / "ha_reward_validation_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "ha_reward_validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
