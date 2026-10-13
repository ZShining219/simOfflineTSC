#!/usr/bin/env python3
"""Counterfactual action-preference auxiliary fine-tuning for DQN.

        L = L_TD + lam * L_rank

- L_TD:  standard DQN update on replay transitions (RMSprop, MSE, gamma=0.95,
         target network) -- identical math to agent/dqn.py.
- L_rank: logistic preference loss on seed-confirmed counterfactual pairs
         mined by mine_preference_pairs.py: softplus(-(Q(s,a+) - Q(s,a-))).

Keeps the original DQNNet(16->20->20->8) and the 8 discrete phase actions;
no attention module, no SUMO executor changes -- this script only does
offline gradient updates on a checkpoint plus frozen evaluation.

Usage (pilot):
    python dev/action_ranking_audit/rank_aux_train.py \
        --checkpoint <src.pt> --scene sumohz1x1 --pairs preference_pairs.json \
        --td-dir output_data/tsc/sumo_dqn/sumohz1x1/<run>/trajectory/episodes \
        --lam 1.0 --steps 3000
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
import os
os.chdir(REPO_ROOT)
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Same env fix as tools/allocation_accuracy_counterfactual.py: the agent
# package __init__ pulls world_sumo transitively and exits silently when the
# SUMO tools/binary paths are missing from the environment.
SUMO_ROOT = Path(os.environ.get("SUMO_ROOT", "/opt/host_colight/lib/python3.10/site-packages/sumo"))
os.environ.setdefault("SUMO_HOME", str(SUMO_ROOT / "tools"))
os.environ.setdefault("SUMO_BINARY", str(SUMO_ROOT / "bin" / "sumo"))

from agent.dqn import DQNNet  # noqa: E402
from agent import utils as agent_utils  # noqa: E402

GAMMA = 0.95
ACTION_COUNT = 8
STATE_DIM = 8
OBS_DIM = 16


def obs16(state8: np.ndarray, phase: np.ndarray) -> np.ndarray:
    onehot = agent_utils.idx2onehot(phase.astype(int), ACTION_COUNT)
    return np.concatenate([state8.astype(np.float32), onehot.astype(np.float32)], axis=1)


def load_student(checkpoint: Path) -> Tuple[DQNNet, DQNNet]:
    payload = torch.load(checkpoint, map_location="cpu")
    if isinstance(payload, dict) and payload.get("checkpoint_type") == "online_only":
        state_dict = payload["online_model_state_dict"]
    else:
        state_dict = payload
    model = DQNNet(OBS_DIM, ACTION_COUNT)
    model.load_state_dict(state_dict)
    target = DQNNet(OBS_DIM, ACTION_COUNT)
    target.load_state_dict(state_dict)
    for p in target.parameters():
        p.requires_grad_(False)
    return model, target


def load_td_data(td_dir: Path, max_episodes: int) -> Tuple[np.ndarray, ...]:
    states, actions, rewards, next_states, next_phases, phases, dones = ([] for _ in range(7))
    files = sorted(glob.glob(str(td_dir / "episode_*.npz")))[:max_episodes]
    for f in files:
        d = np.load(f)
        states.append(d["state"].reshape(-1, STATE_DIM))
        next_states.append(d["next_state"].reshape(-1, STATE_DIM))
        actions.append(d["action"].reshape(-1))
        rewards.append(d["reward"].reshape(-1))
        phases.append(d["current_phase"].reshape(-1))
        next_phases.append(d["next_phase"].reshape(-1))
        dones.append(d["terminated"].reshape(-1).astype(float)
                     if "terminated" in d else np.zeros(len(d["reward"].reshape(-1))))
    return (np.concatenate(states), np.concatenate(actions), np.concatenate(rewards),
            np.concatenate(next_states), np.concatenate(next_phases),
            np.concatenate(phases), np.concatenate(dones))


def load_oracle_matrix(v2_dir: Path, scene: str, horizon: int = 60) -> pd.DataFrame:
    """(state_id, candidate_action) -> dqn_return_{horizon} under indep continuation."""
    rows = []
    for f in sorted((v2_dir / "continuations").glob("branches_shard*.jsonl")):
        for line in f.open("r", encoding="utf-8"):
            r = json.loads(line)
            if (r["scene"] == scene and r["continuation"] == "indep"
                    and str(r["safety_valid"]).lower() == "true"):
                rows.append({"state_id": r["state_id"],
                             "candidate_action": int(r["candidate_action"]),
                             "ret": float(r[f"dqn_return_{horizon}"])})
    df = pd.DataFrame(rows)
    return df.pivot_table(index="state_id", columns="candidate_action", values="ret")


def scene_obs(manifest: pd.DataFrame, scene: str) -> Tuple[np.ndarray, List[str]]:
    sub = manifest[manifest["scene"] == scene].copy()
    state8 = np.asarray([json.loads(v) for v in sub["state_vector"]], dtype=np.float32)
    phase = sub["phase"].astype(int).to_numpy()
    return obs16(state8, phase), sub["state_id"].tolist()


def ranking_metrics(model: DQNNet, obs: np.ndarray, oracle: pd.DataFrame,
                    state_ids: List[str]) -> Dict[str, float]:
    with torch.no_grad():
        q = model(torch.as_tensor(obs, dtype=torch.float32)).numpy()
    metrics = {"top1": [], "spearman": [], "chosen_rank": [], "regret": []}
    for i, sid in enumerate(state_ids):
        if sid not in oracle.index:
            continue
        ret = oracle.loc[sid].to_numpy(dtype=float)
        if not np.all(np.isfinite(ret)):
            continue
        chosen = int(np.argmax(q[i]))
        best = int(np.argmax(ret))
        metrics["top1"].append(float(chosen == best))
        metrics["spearman"].append(stats_spearman(q[i], ret))
        metrics["chosen_rank"].append(float(np.sum(ret > ret[chosen])) + 1)
        metrics["regret"].append(float(ret[best] - ret[chosen]))
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def stats_spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import stats
    return float(stats.spearmanr(a, b).statistic)


def pair_loss_and_acc(model: DQNNet, pair_obs: np.ndarray,
                      a_plus: np.ndarray, a_minus: np.ndarray) -> Tuple[torch.Tensor, float]:
    q = model(torch.as_tensor(pair_obs, dtype=torch.float32))
    qp = q.gather(1, torch.as_tensor(a_plus).view(-1, 1)).squeeze(1)
    qm = q.gather(1, torch.as_tensor(a_minus).view(-1, 1)).squeeze(1)
    loss = F.softplus(-(qp - qm)).mean()
    acc = float((qp > qm).float().mean().item())
    return loss, acc


def reg_loss(model: DQNNet, pair_obs: np.ndarray, a_plus: np.ndarray,
             a_minus: np.ndarray, g_plus: np.ndarray,
             g_minus: np.ndarray) -> torch.Tensor:
    """Arm D: regress Q(s,a) toward measured counterfactual returns."""
    q = model(torch.as_tensor(pair_obs, dtype=torch.float32))
    qp = q.gather(1, torch.as_tensor(a_plus).view(-1, 1)).squeeze(1)
    qm = q.gather(1, torch.as_tensor(a_minus).view(-1, 1)).squeeze(1)
    t = torch.cat([
        torch.as_tensor(g_plus, dtype=torch.float32),
        torch.as_tensor(g_minus, dtype=torch.float32)])
    pred = torch.cat([qp, qm])
    return F.mse_loss(pred, t)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("A", "B", "C", "D"), default="C",
                   help="A=plain TD(1 batch/step); B=equal TD(2 batches/step); "
                        "C=TD+rank; D=TD+counterfactual return regression")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scene", type=str, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--td-dir", type=Path, required=True)
    p.add_argument("--v2-dir", type=Path,
                   default=REPO_ROOT / "output_data" / "action_ranking_audit" / "v2_rollouts")
    p.add_argument("--manifest", type=Path, default=REPO_ROOT / "output_data"
                   / "allocation_accuracy_formal_state" / "allocation_accuracy_state_manifest.csv")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-td", type=int, default=64)
    p.add_argument("--batch-rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--target-sync", type=int, default=100)
    p.add_argument("--max-episodes", type=int, default=100)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # --- data --------------------------------------------------------------
    manifest = pd.read_csv(args.manifest)
    pairs_all = json.loads(args.pairs.read_text())["pairs"]
    pairs = [x for x in pairs_all if x["scene"] == args.scene]
    pair_states = sorted({x["state_id"] for x in pairs})
    rng.shuffle(pair_states)
    n_val = max(1, int(round(len(pair_states) * args.val_frac)))
    val_states = set(pair_states[:n_val])
    train_pairs = [x for x in pairs if x["state_id"] not in val_states]
    val_pairs = [x for x in pairs if x["state_id"] in val_states]

    man = manifest.set_index("state_id")
    def pair_arrays(plist):
        obs = np.stack([
            obs16(np.asarray(json.loads(man.loc[x["state_id"], "state_vector"]),
                              dtype=np.float32).reshape(1, -1),
                  np.asarray([int(man.loc[x["state_id"], "phase"])]))
            for x in plist]).squeeze(1)
        return (obs,
                np.asarray([x["a_plus"] for x in plist]),
                np.asarray([x["a_minus"] for x in plist]),
                np.asarray([x.get("a_plus_return", np.nan) for x in plist], dtype=np.float32),
                np.asarray([x.get("a_minus_return", np.nan) for x in plist], dtype=np.float32))

    obs_tr, ap_tr, am_tr, gp_tr, gm_tr = pair_arrays(train_pairs)
    if val_pairs:
        obs_va, ap_va, am_va, _, _ = pair_arrays(val_pairs)
    else:
        obs_va = ap_va = am_va = None

    s8, act, rew, ns8, nph, ph, done = load_td_data(args.td_dir, args.max_episodes)
    obs_td = obs16(s8, ph)
    nobs_td = obs16(ns8, nph)

    oracle = load_oracle_matrix(args.v2_dir, args.scene)
    all_obs, all_ids = scene_obs(manifest, args.scene)

    # --- model --------------------------------------------------------------
    model, target = load_student(args.checkpoint)
    frozen_ref, _ = load_student(args.checkpoint)
    opt = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    def td_batch_loss():
        idx = rng.integers(0, len(obs_td), size=args.batch_td)
        b_obs = torch.as_tensor(obs_td[idx], dtype=torch.float32)
        b_nobs = torch.as_tensor(nobs_td[idx], dtype=torch.float32)
        b_a = torch.as_tensor(act[idx], dtype=torch.long)
        b_r = torch.as_tensor(rew[idx], dtype=torch.float32)
        b_d = torch.as_tensor(done[idx], dtype=torch.float32)
        with torch.no_grad():
            y = b_r + GAMMA * (1 - b_d) * target(b_nobs).max(dim=1).values
        q = model(b_obs)
        q_t = q.detach().clone()
        q_t[torch.arange(len(idx)), b_a] = y
        return F.mse_loss(q, q_t)

    val_idx = rng.integers(0, len(obs_td), size=min(512, len(obs_td)))
    v_obs = torch.as_tensor(obs_td[val_idx], dtype=torch.float32)
    v_nobs = torch.as_tensor(nobs_td[val_idx], dtype=torch.float32)
    v_a = torch.as_tensor(act[val_idx], dtype=torch.long)
    v_r = torch.as_tensor(rew[val_idx], dtype=torch.float32)
    v_d = torch.as_tensor(done[val_idx], dtype=torch.float32)

    def td_val_loss() -> float:
        """Bellman error of the CURRENT model against the FROZEN initial model's
        bootstrapped targets -- a fixed yardstick for drift during fine-tuning."""
        with torch.no_grad():
            y = v_r + GAMMA * (1 - v_d) * frozen_ref(v_nobs).max(dim=1).values
            q = model(v_obs)
            q_t = q.clone()
            q_t[torch.arange(len(val_idx)), v_a] = y
            return float(F.mse_loss(q, q_t))

    def eval_pack(tag: str) -> Dict:
        model.eval()
        with torch.no_grad():
            m = {"tag": tag}
            m.update({f"scene_{k}": v for k, v in
                      ranking_metrics(model, all_obs, oracle, all_ids).items()})
            l_tr, a_tr = pair_loss_and_acc(model, obs_tr, ap_tr, am_tr)
            m["pair_train_acc"] = a_tr
            m["pair_train_loss"] = float(l_tr)
            if obs_va is not None:
                l_va, a_va = pair_loss_and_acc(model, obs_va, ap_va, am_va)
                m["pair_val_acc"] = a_va
                m["pair_val_loss"] = float(l_va)
            m["td_val_loss"] = float(td_val_loss())
        model.train()
        return m

    args.out.mkdir(parents=True, exist_ok=True)
    history = [eval_pack("before")]
    model.train()
    ntd = len(obs_td)
    log = []
    for step in range(1, args.steps + 1):
        l_td = td_batch_loss()
        idx = rng.integers(0, len(obs_tr), size=args.batch_rank)
        if args.arm == "A":
            loss = l_td
            aux = {"aux_loss": 0.0}
        elif args.arm == "B":
            loss = l_td + td_batch_loss()
            aux = {"aux_loss": 0.0}
        elif args.arm == "C":
            l_rank, acc = pair_loss_and_acc(
                model, obs_tr[idx], ap_tr[idx], am_tr[idx])
            loss = l_td + args.lam * l_rank
            aux = {"aux_loss": float(l_rank), "aux_acc": acc}
        else:  # D
            l_reg = reg_loss(model, obs_tr[idx], ap_tr[idx], am_tr[idx],
                             gp_tr[idx], gm_tr[idx])
            loss = l_td + args.lam * l_reg
            aux = {"aux_loss": float(l_reg)}
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % args.target_sync == 0:
            target.load_state_dict(model.state_dict())
        if step % args.eval_every == 0 or step == args.steps:
            ep = eval_pack(f"step{step}")
            ep["step"] = step
            history.append(ep)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in ep.items()}), flush=True)

    history[-1]["tag"] = "after"
    out_ckpt = args.out / f"arm{args.arm}_lam{args.lam}.pt"
    torch.save(model.state_dict(), out_ckpt)
    result = {
        "config": vars(args) | {"checkpoint": str(args.checkpoint),
                                "pairs": str(args.pairs), "td_dir": str(args.td_dir),
                                "v2_dir": str(args.v2_dir), "manifest": str(args.manifest),
                                "out": str(args.out)},
        "n_pairs_scene": len(pairs), "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs), "n_td_transitions": int(ntd),
        "history": history, "checkpoint_out": str(out_ckpt),
    }
    (args.out / f"arm{args.arm}_lam{args.lam}_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[done] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
