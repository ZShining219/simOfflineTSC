#!/usr/bin/env python3
"""Append SEQ8 demand-suite sequential runs to ledger/runs.jsonl.

Scans data/output_data/sequential/seq8_* buckets (smoke + formal):
- logical run dirs (logical_run_manifest.json) -> train rows
- _shared_evaluation stores + resource_gate.json -> batch rows

Idempotent: skips (run_id, run_dir) pairs already present in runs.jsonl.
Field order matches dev/ledger/build_ledger.py output.
"""
import json, os, re, collections, datetime

ROOT = "/projects/simOfflineTSC"
SEQ = "data/output_data/sequential"
LEDGER = os.path.join(ROOT, "ledger", "runs.jsonl")
BUCKET_RE = re.compile(r"^seq8_(norep|rep|drift)_(formal|smoke)")

FIELDS = ["run_id", "line", "campaign", "arm", "alias_of", "net", "seed",
          "episodes_total", "episodes_increment", "resume_from", "ngpu",
          "status", "plan_id", "code_commit", "dirty", "queue_file", "run_dir",
          "tmux", "started", "eval_mode", "behavior_source", "metrics",
          "abort_reason", "error", "evidence_lost", "notes", "run_kind",
          "config_sha", "sumo_seed_mode", "sumo_seed", "eval_seed", "condition",
          "superseded_by", "runtime_env"]

STATUS_MAP = {"completed": "DONE", "failed": "FAILED", "running": "ABORTED",
              "aborted": "ABORTED", "interrupted": "ABORTED",
              "planned": "REGISTERED"}


def base_row():
    return {k: None for k in FIELDS} | {
        "line": "semi_offline", "campaign": "seq8_demand",
        "metrics": {}, "evidence_lost": False, "notes": "",
        "runtime_env": "colight", "sumo_seed_mode": "fixed_default",
    }


def iso(unix):
    if unix is None:
        return None
    return datetime.datetime.fromtimestamp(
        unix, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pack_tag(network):
    return re.sub(r"^sumohz1x1_?", "", network) or "base"


def seq_net(networks):
    tags = [pack_tag(n) for n in (networks or [])]
    return "hz1x1:" + ">".join(tags) if tags else "hz1x1"


def manifest_of(attempt):
    cmd = attempt.get("command") or []
    if "--manifest" in cmd:
        return os.path.basename(
            cmd[cmd.index("--manifest") + 1]).replace(".json", "")
    return None


def final_eval_metrics(attempt_dir, child):
    """Stage-N last local_episode eval on the last-stage scene -> summary."""
    nets = child.get("networks") or []
    eps = child.get("stage_episodes") or []
    if not nets or not eps:
        return {}, 0
    cs_path = os.path.join(attempt_dir, "current_state.json")
    if not os.path.isfile(cs_path):
        return {}, 0
    cs = json.load(open(cs_path))
    ops = (cs.get("completed_operations") or {})
    key = f"evaluation:stage_{len(eps)}:local_{eps[-1]}:{nets[-1]}"
    n_cells = sum(1 for k in ops if k.startswith("evaluation:"))
    art = (ops.get(key) or {}).get("artifact")
    if not art or not os.path.isfile(art):
        return {}, n_cells
    alias = json.load(open(art))
    com = json.load(open(alias["physical_committed_path"]))
    sp = com.get("summary_path")
    if not sp or not os.path.isfile(sp):
        return {}, n_cells
    s = json.load(open(sp))
    m = {k: round(s[k], 3) if isinstance(s.get(k), float) else s.get(k)
         for k in ("travel_time", "throughput", "unfinished_vehicles",
                   "queue", "delay") if s.get(k) is not None}
    return m, n_cells


def iter_logical_runs(bucket_abs):
    for name in sorted(os.listdir(bucket_abs)):
        d = os.path.join(bucket_abs, name)
        if os.path.isfile(os.path.join(d, "logical_run_manifest.json")):
            yield name, d


def build_run_row(lrid, ldir, kind, run_dir_rel):
    lm = json.load(open(os.path.join(ldir, "logical_run_manifest.json")))
    attempts = lm.get("attempts") or []
    eff_id = lm.get("effective_attempt") or (
        attempts[-1]["attempt_id"] if attempts else None)
    eff = next(
        (a for a in attempts if a.get("attempt_id") == eff_id),
        attempts[-1] if attempts else {})
    status = STATUS_MAP.get(lm.get("status"), "LEGACY_UNCLEAR")
    adir = eff.get("attempt_dir") or os.path.join(
        ldir, "attempts", eff_id or "attempt_1")
    child = {}
    cm = os.path.join(adir, "child_run_manifest.json")
    if os.path.isfile(cm):
        child = json.load(open(cm))
    seed = child.get("training_seed")
    stage_eps = child.get("stage_episodes") or []
    eps_total = sum(stage_eps) if stage_eps else None
    arm = re.sub(r"-SD\d+$", "", lrid) or None
    metrics, n_cells = final_eval_metrics(adir, child)
    notes = [f"attempts={len(attempts)}"
             + (f"（有效{eff_id}）" if len(attempts) > 1 else "")]
    stage = child.get("experiment_stage")
    if stage:
        notes.append(f"stage={stage}")
    if n_cells:
        notes.append(f"冻结评估cell×{n_cells}（去重存储于_shared_evaluation）")
    if child.get("archive_mode") == "SELF":
        notes.append(
            f"SELF自产档案 method={child.get('method')}"
            f" ratio={child.get('offline_ratio')}")
    if len(attempts) > 1:
        fails = sum(1 for a in attempts if a.get("status") == "failed")
        if fails:
            notes.append(f"失败前驱attempt×{fails}留档")
    row = base_row() | {
        "run_id": lrid,
        "arm": arm,
        "net": seq_net(child.get("networks")),
        "seed": seed,
        "episodes_total": eps_total,
        "episodes_increment": eps_total,
        "status": status,
        "plan_id": manifest_of(eff),
        "code_commit": (child.get("git_commit") or "")[:9] or None,
        "run_dir": run_dir_rel,
        "started": (iso(attempts[0].get("created_at_unix"))
                    if attempts else None),
        "eval_mode": "train_frozen_eval" if kind == "train" else "smoke",
        "metrics": metrics,
        "notes": "; ".join(notes),
        "run_kind": kind,
        "condition": child.get("condition"),
        "abort_reason": ("状态标记 interrupted（中止留档）"
                         if status == "ABORTED" else None),
    }
    return row


def batch_row(rid, run_dir_rel, notes, arm="evidence"):
    return base_row() | {
        "run_id": rid, "arm": arm, "status": "DONE",
        "run_dir": run_dir_rel, "eval_mode": "engineering_validation",
        "notes": notes, "run_kind": "batch",
    }


def main():
    existing = set()
    with open(LEDGER) as f:
        for line in f:
            r = json.loads(line)
            existing.add((r["run_id"], r.get("run_dir")))

    new = []
    seq_abs = os.path.join(ROOT, SEQ)
    for bucket in sorted(os.listdir(seq_abs)):
        bm = BUCKET_RE.match(bucket)
        if not bm:
            continue
        proto, stage_kind = bm.group(1), bm.group(2)
        babs = os.path.join(seq_abs, bucket)
        if not os.path.isdir(babs):
            continue
        kind = "train" if stage_kind == "formal" else "smoke"
        for lrid, ldir in iter_logical_runs(babs):
            row = build_run_row(lrid, ldir, kind, f"{SEQ}/{bucket}/{lrid}")
            new.append(row)
        se = os.path.join(babs, "_shared_evaluation")
        if os.path.isdir(se):
            na = len(os.listdir(os.path.join(se, "aliases"))) \
                if os.path.isdir(os.path.join(se, "aliases")) else 0
            np_ = len(os.listdir(os.path.join(se, "physical"))) \
                if os.path.isdir(os.path.join(se, "physical")) else 0
            new.append(batch_row(
                f"SEQ8-{proto.upper()}-SHARED-EVAL/{bucket}",
                f"{SEQ}/{bucket}/_shared_evaluation",
                f"{bucket} 冻结评估去重存储：{na} aliases / {np_} physical cells"))
        rg = os.path.join(babs, "resource_gate.json")
        if os.path.isfile(rg):
            g = json.load(open(rg))
            new.append(batch_row(
                f"SEQ8-{proto.upper()}-RESOURCE-GATE/{bucket}",
                f"{SEQ}/{bucket}/resource_gate.json",
                f"派发资源门禁：slots={g.get('concurrency_slots')}"
                f" valid={g.get('valid')}"))

    fresh = [r for r in new if (r["run_id"], r.get("run_dir")) not in existing]
    with open(LEDGER, "a") as f:
        for r in fresh:
            f.write(json.dumps(
                {k: r.get(k) for k in FIELDS}, ensure_ascii=False) + "\n")

    print(f"new_scanned={len(new)} appended={len(fresh)}")
    c = collections.Counter((r["run_kind"], r["status"]) for r in fresh)
    for k, v in sorted(c.items()):
        print(" ", k, v)
    no_metrics = [r["run_id"] for r in fresh
                  if r["run_kind"] == "train" and not r["metrics"]]
    print("train rows w/o metrics:", len(no_metrics), no_metrics[:10])


if __name__ == "__main__":
    main()
