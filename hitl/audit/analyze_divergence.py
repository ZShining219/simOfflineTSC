#!/usr/bin/env python3
"""Analysis: are the actions that Ranking changed actually better?

Two questions on the closed-loop divergence snapshots of B vs C:
  Q1 divergence rate/characterization (from decisions_*.jsonl)
  Q2 counterfactual value of the change (from cf/divcf_shard*.jsonl):
     same snapshot, same future seed, same continuation; compare
     dqn_return_60 of the driver's action vs the shadow's action.
For B_drives snapshots positive delta = C's pick better; for C_drives
snapshots positive delta = B's pick better (sign normalized so that
delta > 0 always means "the arm-C checkpoint's choice wins").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n=5000):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    m = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def wilson(hits, n, z=1.959964):
    if n <= 0:
        return np.nan, np.nan
    p = hits / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True,
                    help="divergence probe dir with decisions_*.jsonl and cf/")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = args.out or args.dir

    # ---------- Q1: divergence characterization ----------
    dec = []
    for f in sorted(args.dir.glob("decisions_*.jsonl")):
        tag = f.stem.replace("decisions_", "").rsplit("_", 1)[0]
        for line in f.open("r", encoding="utf-8"):
            r = json.loads(line)
            r["tag"] = tag
            dec.append(r)
    dec = pd.DataFrame(dec)
    q1_rows = []
    for tag, d in dec.groupby("tag"):
        dv, ag = d[d["diverged"]], d[~d["diverged"]]
        lo, hi = wilson(int(dv.shape[0]), len(d))
        q1_rows.append({
            "tag": tag, "decisions": len(d), "diverged": len(dv),
            "rate": len(dv) / len(d), "rate_lo": lo, "rate_hi": hi,
            "div_queue_in": dv["queue_incoming"].mean(),
            "agr_queue_in": ag["queue_incoming"].mean(),
            "div_vehicles": dv["vehicles_in_net"].mean(),
            "agr_vehicles": ag["vehicles_in_net"].mean(),
            "div_sim_time": dv["sim_time"].mean(),
            "agr_sim_time": ag["sim_time"].mean(),
        })
        for ph, sub in dv.groupby("phase"):
            q1_rows.append({"tag": tag, "phase": int(ph), "div_share": len(sub) / len(dv)})
    q1 = pd.DataFrame(q1_rows)

    # ---------- Q2: counterfactual value ----------
    cf = []
    for cdir in ("cf", "cf_indep"):
        for f in sorted((args.dir / cdir).glob("divcf_shard*.jsonl")):
            for line in f.open("r", encoding="utf-8"):
                cf.append(json.loads(line))
    cf = pd.DataFrame(cf)
    cf = cf[cf["safety_valid"].astype(str).str.lower().eq("true")]

    # delta > 0 <=> the arm-C checkpoint's action is better
    def signed_delta(g: pd.DataFrame) -> float:
        m = g.set_index("candidate_action")["dqn_return_60"]
        if len(m) != 2:
            return np.nan
        tag = g["snapshot_tag"].iloc[0]
        a_driver = int(g["a_driver"].iloc[0])
        a_shadow = int(g["a_shadow"].iloc[0])
        d = float(m[a_shadow] - m[a_driver])
        return d if tag.startswith("B_") else -d  # C wins <=> delta>0

    per_state = []
    key = ["snapshot_tag", "state_id", "continuation", "future_seed"]
    for (tag, sid, cont, seed), g in cf.groupby(
            ["snapshot_tag", "state_id", "continuation", "future_seed"]):
        d = signed_delta(g)
        per_state.append({"snapshot_tag": tag, "state_id": sid,
                          "continuation": cont, "future_seed": str(seed),
                          "delta_c_minus_b": d,
                          "queue_incoming": g["queue_incoming"].iloc[0],
                          "sim_time": g["rollout_seconds"].iloc[0] * 0 +
                          float(g.get("final_sim_time", np.nan).iloc[0]
                                if "final_sim_time" in g else np.nan)})
    ps = pd.DataFrame(per_state)

    # per state: majority across seeds
    q2_rows = []
    for cont, sub in ps.groupby("continuation"):
        per_st = sub.groupby(["snapshot_tag", "state_id"])["delta_c_minus_b"]
        state_mean = per_st.mean()
        state_sign = per_st.apply(lambda s: float((s > 0).mean()))
        stable_c = state_sign[(state_sign >= 0.99)]
        stable_b = state_sign[(state_sign <= 0.01)]
        lo, hi = bootstrap_ci(state_mean.to_numpy(), rng)
        wl, wh = wilson(int((state_mean > 0).sum()), len(state_mean))
        q2_rows.append({
            "continuation": cont,
            "states": len(state_mean),
            "c_better_share": float((state_mean > 0).mean()),
            "c_better_lo": wl, "c_better_hi": wh,
            "mean_delta": float(state_mean.mean()),
            "ci_lo": lo, "ci_hi": hi,
            "median_delta": float(state_mean.median()),
            "stable_c_wins": len(stable_c), "stable_b_wins": len(stable_b),
            "mixed": int(len(state_mean) - len(stable_c) - len(stable_b)),
            "mean_abs_delta": float(state_mean.abs().mean()),
        })
        # seed-level
        lo2, hi2 = bootstrap_ci(sub["delta_c_minus_b"].to_numpy(), rng)
        q2_rows.append({
            "continuation": f"{cont} (seed-level)",
            "states": len(sub),
            "c_better_share": float((sub["delta_c_minus_b"] > 0).mean()),
            "mean_delta": float(sub["delta_c_minus_b"].mean()),
            "ci_lo": lo2, "ci_hi": hi2,
            "median_delta": float(sub["delta_c_minus_b"].median()),
            "mean_abs_delta": float(sub["delta_c_minus_b"].abs().mean()),
        })
    q2 = pd.DataFrame(q2_rows)

    # by queue stratum (indep cont, state-mean)
    ind = ps[ps["continuation"] == "indep"]
    if len(ind):
        st = ind.groupby(["state_id", "snapshot_tag"])["delta_c_minus_b"].mean().reset_index()
        qi = ind.groupby(["state_id", "snapshot_tag"])["queue_incoming"].first().reset_index()
        st = st.merge(qi, on=["state_id", "snapshot_tag"])
        st["qbin"] = pd.qcut(st["queue_incoming"], 3, labels=["low", "mid", "high"], duplicates="drop")
        strata = st.groupby("qbin", observed=True)["delta_c_minus_b"].agg(
            states="count", c_better_share=lambda s: float((s > 0).mean()),
            mean_delta="mean").reset_index()
    else:
        strata = pd.DataFrame()

    q1.to_csv(out / "divergence_q1_stats.csv", index=False)
    ps.to_csv(out / "divergence_cf_deltas.csv", index=False)
    q2.to_csv(out / "divergence_cf_summary.csv", index=False)
    strata.to_csv(out / "divergence_cf_by_queue.csv", index=False)

    print("=== Q1 divergence ===")
    print(q1[q1["phase"].isna() if "phase" in q1 else slice(None)].to_string(index=False))
    print("\n=== Q2: delta>0 means C's choice better ===")
    print(q2.to_string(index=False))
    print("\n=== by queue stratum (indep cont, state-mean) ===")
    print(strata.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
