#!/usr/bin/env python3
"""Append HA-SODQN hz1x1 semi-offline runs to ledger/runs.jsonl.

Scans data/output_data/ha_sodqn/ buckets:
- formal_e7705f7_20260726/   -> train rows (logical run dirs w/ logical_run_manifest.json)
- smoke_*/recovery_*/behavior_seed_audit_*/ -> smoke rows
- formal bucket _shared_evaluation store + engineering/coordination/analysis dirs -> batch rows

Idempotent: skips (run_id, run_dir) pairs already present in runs.jsonl.
Field order matches dev/ledger/build_ledger.py output.
"""
import json, os, re, collections, datetime, sys

ROOT = "/projects/simOfflineTSC"
HA = "data/output_data/ha_sodqn"
LEDGER = os.path.join(ROOT, "ledger", "runs.jsonl")
FORMAL = f"{HA}/formal_e7705f7_20260726"

SCENE_CODE = {"sumohz1x1_config2": "S1", "sumohz1x1": "S2",
              "sumohz1x1_config4": "S3", "sumohz1x1_config3": "S4"}

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
        "line": "semi_offline", "campaign": "ha_sodqn",
        "metrics": {}, "evidence_lost": False, "notes": "",
        "runtime_env": "colight", "sumo_seed_mode": "fixed_default",
    }

def iso(unix):
    if unix is None:
        return None
    return datetime.datetime.fromtimestamp(
        unix, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def seq_net(networks):
    codes = [SCENE_CODE.get(n, n) for n in (networks or [])]
    return "hz1x1:" + ">".join(codes) if codes else "hz1x1"

def manifest_of(attempt):
    cmd = attempt.get("command") or []
    if "--manifest" in cmd:
        return os.path.basename(cmd[cmd.index("--manifest") + 1]).replace(".json", "")
    return None

def final_eval_metrics(attempt_dir, child):
    """Stage-N last local_episode eval on the last-stage scene -> summary metrics."""
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

def build_run_row(lrid, ldir, kind, run_dir_rel, e4_members, extra_id=None):
    lm = json.load(open(os.path.join(ldir, "logical_run_manifest.json")))
    attempts = lm.get("attempts") or []
    eff_id = lm.get("effective_attempt") or (attempts[-1]["attempt_id"] if attempts else None)
    eff = next((a for a in attempts if a.get("attempt_id") == eff_id), attempts[-1] if attempts else {})
    status = STATUS_MAP.get(lm.get("status"), "LEGACY_UNCLEAR")
    adir = eff.get("attempt_dir") or os.path.join(ldir, "attempts", eff_id or "attempt_1")
    child = {}
    cm = os.path.join(adir, "child_run_manifest.json")
    if os.path.isfile(cm):
        child = json.load(open(cm))
    seed = child.get("training_seed")
    stage_eps = child.get("stage_episodes") or []
    eps_total = sum(stage_eps) if stage_eps else None
    arm = re.sub(r"-SD\d+$", "", lrid) or None
    metrics, n_cells = final_eval_metrics(adir, child)
    init_ckpt = child.get("initial_checkpoint") or ""
    bsrc = None
    mm = re.search(r"data/output_data/tsc/[^/]+/[^/]+/([^/]+)/checkpoints/", init_ckpt)
    if mm:
        bsrc = mm.group(1)
    notes = [f"attempts={len(attempts)}" + (f"（有效{eff_id}）" if len(attempts) > 1 else "")]
    stage = child.get("experiment_stage")
    if stage:
        notes.append(f"stage={stage}")
    if e4_members is not None:
        notes.append("E4审计白名单成员" if lrid in e4_members else
                     "非E4审计白名单（E0–E3阶段产物/扫参）")
    if n_cells:
        notes.append(f"冻结评估cell×{n_cells}（去重存储于_shared_evaluation）")
    if child.get("archive_digest"):
        notes.append(f"HOA_digest={child['archive_digest'][:12]}")
    if len(attempts) > 1:
        fails = sum(1 for a in attempts if a.get("status") == "failed")
        if fails:
            notes.append(f"失败前驱attempt×{fails}留档")
    row = base_row() | {
        "run_id": extra_id or lrid,
        "arm": arm,
        "net": seq_net(child.get("networks")),
        "seed": seed,
        "episodes_total": eps_total,
        "episodes_increment": eps_total,
        "status": status,
        "plan_id": manifest_of(eff),
        "code_commit": (child.get("git_commit") or "")[:9] or None,
        "run_dir": run_dir_rel,
        "started": iso(attempts[0].get("created_at_unix")) if attempts else None,
        "eval_mode": "train_frozen_eval" if kind == "train" else "smoke",
        "behavior_source": bsrc,
        "metrics": metrics,
        "notes": "; ".join(notes),
        "run_kind": kind,
        "condition": child.get("condition"),
        "abort_reason": ("状态标记 interrupted（审计/探针中止留档）"
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
    rows_old = []
    with open(LEDGER) as f:
        for line in f:
            r = json.loads(line)
            rows_old.append(r)
            existing.add((r["run_id"], r.get("run_dir")))

    audit = json.load(open(os.path.join(
        ROOT, HA, "engineering_20260726/E4_ha_audit_664ac2a.json")))
    e4_members = {r["logical_run_id"] for r in audit.get("runs", [])}

    new = []
    formal_abs = os.path.join(ROOT, FORMAL)
    for lrid, ldir in iter_logical_runs(formal_abs):
        row = build_run_row(lrid, ldir, "train", f"{FORMAL}/{lrid}", e4_members)
        new.append(row)

    se = os.path.join(formal_abs, "_shared_evaluation")
    n_alias = len(os.listdir(os.path.join(se, "aliases"))) if os.path.isdir(os.path.join(se, "aliases")) else 0
    n_phys = len(os.listdir(os.path.join(se, "physical"))) if os.path.isdir(os.path.join(se, "physical")) else 0
    new.append(batch_row(
        "HA-SHARED-EVAL/formal_e7705f7", f"{FORMAL}/_shared_evaluation",
        f"冻结评估内容寻址去重存储：{n_alias} aliases / {n_phys} physical cells；"
        "各 run 的评估 cell 归属见 attempt 内 current_state.completed_operations"))
    rg = os.path.join(formal_abs, "resource_gate.json")
    if os.path.isfile(rg):
        g = json.load(open(rg))
        new.append(batch_row(
            "HA-RESOURCE-GATE/formal_e7705f7", f"{FORMAL}/resource_gate.json",
            f"派发资源门禁：slots={g.get('concurrency_slots')} valid={g.get('valid')}"))

    # --- smoke / recovery / audit buckets -> smoke rows ---
    smoke_buckets = {
        "smoke_O2_SD0_6197fcc_20260726": "HA-SMOKE-6197fcc",
        "smoke_final_O2_SD0_e7705f7_20260726": "HA-SMOKE-e7705f7",
        "recovery_CQA_O2_SD0_52dfdc5_20260726": "HA-RECOVERY-52dfdc5",
        "behavior_seed_audit_O2_SD0_20260726": "HA-BSEEDAUDIT-20260726",
        "behavior_seed_audit_O2_SD0_20260726_r2": "HA-BSEEDAUDIT-20260726r2",
    }
    for bucket, tag in smoke_buckets.items():
        babs = os.path.join(ROOT, HA, bucket)
        if not os.path.isdir(babs):
            continue
        for lrid, ldir in iter_logical_runs(babs):
            row = build_run_row(lrid, ldir, "smoke", f"{HA}/{bucket}/{lrid}", None,
                                extra_id=f"{tag}-{lrid}")
            new.append(row)
        se_b = os.path.join(babs, "_shared_evaluation")
        if os.path.isdir(se_b):
            na = len(os.listdir(os.path.join(se_b, "aliases"))) if os.path.isdir(os.path.join(se_b, "aliases")) else 0
            np_ = len(os.listdir(os.path.join(se_b, "physical"))) if os.path.isdir(os.path.join(se_b, "physical")) else 0
            new.append(batch_row(
                f"{tag}-SHARED-EVAL", f"{HA}/{bucket}/_shared_evaluation",
                f"{bucket} 冻结评估去重存储：{na} aliases / {np_} physical cells"))

    # --- evidence batches ---
    if os.path.isdir(os.path.join(ROOT, HA, "engineering_20260726")):
        new.append(batch_row(
            "HA-ENG-20260726", f"{HA}/engineering_20260726",
            "E0–E4 manifest/ha_audit/selection/initial_state_catalog/archive_root_manifest 工程证据目录"))
    if os.path.isdir(os.path.join(ROOT, HA, "coordination_20260727")):
        new.append(batch_row(
            "HA-COORD-20260727", f"{HA}/coordination_20260727",
            "正式批派发协调证据：assignments/recovery_ledgers/resource_profiles/stage_gates/sync_ledgers"))
    if os.path.isdir(os.path.join(ROOT, HA, "analysis_e7705f7")):
        new.append(batch_row(
            "HA-ANALYSIS-e7705f7", f"{HA}/analysis_e7705f7",
            "E0–E4 分析产物：runs/paired_effects/adaptation_curves/mechanism/selection/lower_triangle CSV + PNG"))

    fresh = [r for r in new if (r["run_id"], r.get("run_dir")) not in existing]
    with open(LEDGER, "a") as f:
        for r in fresh:
            f.write(json.dumps({k: r.get(k) for k in FIELDS}, ensure_ascii=False) + "\n")

    print(f"existing={len(rows_old)} new_scanned={len(new)} appended={len(fresh)}")
    c = collections.Counter((r["run_kind"], r["status"]) for r in fresh)
    for k, v in sorted(c.items()):
        print(" ", k, v)
    no_metrics = [r["run_id"] for r in fresh if r["run_kind"] == "train" and not r["metrics"]]
    print("train rows w/o metrics:", len(no_metrics), no_metrics[:10])
    e4 = [r for r in fresh if r["run_kind"] == "train" and "E4审计白名单成员" in r["notes"]]
    print("E4 whitelist members:", len(e4))

if __name__ == "__main__":
    main()
