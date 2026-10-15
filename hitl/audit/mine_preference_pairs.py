#!/usr/bin/env python3
"""Mine seed-confirmed counterfactual action-preference pairs.

Reads multiseed branch records (candidate_action rollouts over several future
seeds) and emits pairs (s, a+, a-) where a+ beats a- in the 60 s discounted
DQN return on at least --min-win-rate of the evaluated seeds.  Output feeds
the rank-auxiliary training in rank_aux_train.py (L = L_TD + lam * L_rank).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(paths: List[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        for file in sorted(path.glob("branches_shard*.jsonl")):
            for line in file.open("r", encoding="utf-8"):
                if line.strip():
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiseed-dir", type=Path, required=True, action="append",
                        help="dir containing branches_shard*.jsonl (repeatable)")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/output_data"
                        / "allocation_accuracy_formal_state" / "allocation_accuracy_state_manifest.csv")
    parser.add_argument("--min-seeds", type=int, default=4,
                        help="minimum evaluated seeds for a pair to count")
    parser.add_argument("--min-win-rate", type=float, default=0.8,
                        help="a+ must win on at least this fraction of seeds")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--predictions", type=Path, default=REPO_ROOT / "data/output_data"
                        / "allocation_accuracy_formal_cf" / "allocation_accuracy_agent_predictions.csv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = load_jsonl(list(args.multiseed_dir))
    frame = frame[frame["safety_valid"].astype(str).str.lower().eq("true")]
    ret_col = f"dqn_return_{args.horizon}"

    manifest = pd.read_csv(args.manifest).set_index("state_id")
    pred = pd.read_csv(args.predictions)
    pred = pred[pred["method"] == "CONT-DQN"]
    pred = pred[pred["prediction_error"].isna()
                | pred["prediction_error"].astype(str).str.strip().eq("")]
    predicted = pred.set_index("state_id")["predicted_action"].to_dict()

    pairs: List[Dict] = []
    rejected: List[Dict] = []
    for (state_id), group in frame.groupby("state_id"):
        by_seed: Dict[str, Dict[int, float]] = defaultdict(dict)
        for _, row in group.iterrows():
            by_seed[str(row["future_seed"])][int(row["candidate_action"])] = float(row[ret_col])
        actions = sorted({a for per in by_seed.values() for a in per})
        complete = {s: per for s, per in by_seed.items() if len(per) == len(actions)}
        if len(complete) < args.min_seeds:
            continue
        # per-action win statistics across seeds
        wins = {a: 0 for a in actions}
        gaps = {a: [] for a in actions}
        for per in complete.values():
            vals = np.asarray([per[a] for a in actions])
            best = actions[int(np.argmax(vals))]
            wins[best] += 1
            worst = actions[int(np.argmin(vals))]
            gaps[worst].append(float(per[best] - per[worst]))
        n = len(complete)
        a_plus = max(actions, key=lambda a: wins[a])
        win_rate = wins[a_plus] / n
        # a- = action that loses to a+ most often (mean return gap)
        mean_ret = {a: float(np.mean([per[a] for per in complete.values()])) for a in actions}
        mean_gap = {a: float(mean_ret[a_plus] - mean_ret[a]) for a in actions if a != a_plus}
        a_minus = max(mean_gap, key=mean_gap.get)
        entry = {
            "state_id": str(state_id),
            "scene": str(manifest.loc[state_id, "scene"]) if state_id in manifest.index else "",
            "a_plus": int(a_plus),
            "a_minus": int(a_minus),
            "a_plus_return": float(mean_ret[a_plus]),
            "a_minus_return": float(mean_ret[a_minus]),
            "seeds_evaluated": n,
            "win_rate": float(win_rate),
            "mean_return_gap": float(mean_gap[a_minus]),
            "pair_actions": actions,
            "predicted_action": int(predicted[state_id]) if state_id in predicted else None,
        }
        entry["dqn_was_right"] = bool(entry["predicted_action"] == a_plus)
        if win_rate >= args.min_win_rate and mean_gap[a_minus] > 0:
            pairs.append(entry)
        else:
            rejected.append(entry)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"pairs": pairs, "rejected": rejected,
                                    "min_win_rate": args.min_win_rate,
                                    "horizon": args.horizon}, ensure_ascii=False, indent=2))
    by_scene = pd.Series([p["scene"] for p in pairs]).value_counts().to_dict()
    print(f"confirmed pairs: {len(pairs)} (rejected {len(rejected)}); by scene {by_scene}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
