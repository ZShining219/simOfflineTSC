#!/usr/bin/env python3
"""Three-way evaluation of high-discrepancy states.

R1  truncated return: discounted reward over 60 s, no tail value.
R2  bootstrap:        R1 + gamma^6 * max_a Q_eval(s_60, a), where the evaluator
                      is either the branch's source CONT-DQN checkpoint
                      ('feature') or the continuation checkpoint ('cont').
R3  full episode:     discounted reward until the SUMO episode end (3600 s).

For every (state, continuation, evaluator) the 8 candidate actions are ranked
under each evaluation and compared against the frozen CONT-DQN root-state Q
ranking (top-1, Spearman, chosen rank, regret).  Determinism cross-checks:
boundary-mode dqn_return_60 must equal the stored v2 continuations value, and
episodes-mode dqn_return_60 must equal the boundary-mode one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V2 = REPO_ROOT / "data/output_data" / "action_ranking_audit" / "v2_rollouts"
DEFAULT_PRED = (
    REPO_ROOT
    / "data/output_data"
    / "allocation_accuracy_formal_cf"
    / "allocation_accuracy_agent_predictions.csv"
)
GAMMA = 0.95
ACTIONS = list(range(8))
HORIZON_CYCLES = {30: 3, 60: 6, 90: 9}


def load_jsonl_dir(path: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for file in sorted(path.glob("branches_shard*.jsonl")):
        for line in file.open("r", encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def parse_q(record: pd.Series, key: str) -> np.ndarray:
    raw = record.get(key)
    if not isinstance(raw, str) or not raw:
        return np.full(len(ACTIONS), np.nan)
    return np.asarray(json.loads(raw), dtype=float)


def state_metrics(q: np.ndarray, utility: np.ndarray, chosen: int, best: int) -> Dict[str, float]:
    valid = np.isfinite(utility)
    if valid.sum() < 2:
        return {"top1": np.nan, "spearman": np.nan, "chosen_rank": np.nan,
                "regret": np.nan, "n_valid": int(valid.sum())}
    chosen_rank = int(np.sum(utility[valid] > utility[chosen])) + 1
    return {
        "top1": float(chosen == best),
        "spearman": stats.spearmanr(q[valid], utility[valid]).statistic,
        "chosen_rank": float(chosen_rank),
        "regret": float(utility[best] - utility[chosen]),
        "n_valid": int(valid.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "data/output_data" / "action_ranking_audit")
    args = parser.parse_args()

    boundary = load_jsonl_dir(args.v2_dir / "boundary")
    episodes = load_jsonl_dir(args.v2_dir / "episodes")
    stored = load_jsonl_dir(args.v2_dir / "continuations")
    pred = pd.read_csv(args.predictions)
    pred = pred[pred["method"] == "CONT-DQN"].copy()
    pred = pred[pred["prediction_error"].isna()
                | pred["prediction_error"].astype(str).str.strip().eq("")]
    pred = pred.set_index("state_id")

    boundary = boundary[boundary["safety_valid"].astype(str).str.lower().eq("true")]
    episodes = episodes[episodes["safety_valid"].astype(str).str.lower().eq("true")]

    # --- determinism cross-checks ------------------------------------------
    key_cols = ["state_id", "continuation", "candidate_action"]
    bmap = boundary.set_index(key_cols)["dqn_return_60"]
    smap = stored.set_index(key_cols)["dqn_return_60"]
    common = bmap.index.intersection(smap.index)
    diff_b = float((bmap.loc[common] - smap.loc[common]).abs().max())
    emap = episodes.set_index(key_cols)["dqn_return_60"]
    common2 = bmap.index.intersection(emap.index)
    diff_e = float((bmap.loc[common2] - emap.loc[common2]).abs().max())
    checks = {"branches_checked_boundary_vs_stored": int(len(common)),
              "max_abs_diff_boundary_vs_stored": diff_b,
              "branches_checked_episodes_vs_boundary": int(len(common2)),
              "max_abs_diff_episodes_vs_boundary": diff_e}

    rows: List[Dict] = []
    for (scene, state_id, continuation), group in boundary.groupby(
            ["scene", "state_id", "continuation"]):
        if len(group) != len(ACTIONS):
            continue
        group = group.sort_values("candidate_action")
        qrow = pred.loc[state_id]
        q = np.asarray([float(qrow[f"q_{a}"]) for a in ACTIONS])
        chosen = int(qrow["predicted_action"])

        ret60 = group["dqn_return_60"].to_numpy(dtype=float)
        ep_group = episodes[
            (episodes["state_id"] == state_id)
            & (episodes["continuation"] == continuation)
        ].sort_values("candidate_action")
        full = ep_group["dqn_return_full"].to_numpy(dtype=float) \
            if len(ep_group) == len(ACTIONS) else np.full(len(ACTIONS), np.nan)

        for horizon in (30, 60, 90):
            k = HORIZON_CYCLES[horizon]
            ret_h = group[f"dqn_return_{horizon}"].to_numpy(dtype=float)
            evals: Dict[str, np.ndarray] = {f"r1_ret{horizon}": ret_h}
            for tag in ("feature", "cont"):
                v_boundary = np.asarray(
                    [np.max(parse_q(rec, f"boundary_{horizon}_q_{tag}"))
                     for _, rec in group.iterrows()]
                )
                evals[f"r2_{tag}_{horizon}"] = ret_h + (GAMMA ** k) * v_boundary
            if horizon == 60:
                evals["r3_full"] = full
            for name, utility in evals.items():
                if not np.isfinite(utility).any():
                    continue
                best = int(np.nanargmax(utility))
                m = state_metrics(q, utility, chosen, best)
                rows.append({
                    "scene": scene, "state_id": state_id,
                    "continuation": continuation, "eval": name,
                    "best_action": best, **m,
                })

    frame = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "v2_bootstrap_evals.csv", index=False)
    pd.DataFrame([checks]).to_csv(args.out / "v2_bootstrap_checks.csv", index=False)

    summary = (
        frame.groupby(["continuation", "eval"])
        .agg(states=("top1", "count"),
             top1=("top1", "mean"),
             spearman=("spearman", "mean"),
             chosen_rank=("chosen_rank", "mean"),
             regret=("regret", "mean"))
        .reset_index()
    )
    summary["top1"] = (100 * summary["top1"]).round(1)
    summary[["spearman", "chosen_rank", "regret"]] = \
        summary[["spearman", "chosen_rank", "regret"]].round(3)
    summary.to_csv(args.out / "v2_bootstrap_summary.csv", index=False)

    print("=== determinism checks ===")
    for k, v in checks.items():
        print(f"{k}: {v}")
    print("\n=== per-eval ranking agreement with CONT-DQN root Q ===")
    print(summary.to_string(index=False))

    print("\n=== R1 vs R3 oracle agreement (60 s cut vs full episode) ===")
    r1 = frame[frame["eval"] == "r1_ret60"].set_index(["state_id", "continuation"])
    r3 = frame[frame["eval"] == "r3_full"].set_index(["state_id", "continuation"])
    both = r1.join(r3, lsuffix="_r1", rsuffix="_r3", how="inner")
    if len(both):
        same = (both["best_action_r1"] == both["best_action_r3"]).mean()
        print(f"states={len(both)}  oracle_same={same:.3f}  "
              f"top1_r1={both['top1_r1'].mean():.3f}  top1_r3={both['top1_r3'].mean():.3f}  "
              f"regret_r1={both['regret_r1'].mean():.3f}  regret_r3={both['regret_r3'].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
