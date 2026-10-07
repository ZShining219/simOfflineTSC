#!/usr/bin/env python3
"""Build ledger/runs.jsonl + summary tables from inventory.json."""
import json, re, os, collections

ROOT = "/data/users/zfh/workspace/projects/simOfflineTSC"
OUT = "/data/users/zfh/workspace/projects/govern_1007/scratch"
inv = json.load(open(os.path.join(OUT, "inventory.json")))

# ---------- queue membership map: output_path -> queue file ----------
q_by_out = {}       # run dir (abs-ish rel path) -> queue rel path
q_task = {}         # run_id -> (queue_file, task)
for q in inv["queues"]:
    for t in q["tasks"]:
        if t.get("output_path"):
            rp = os.path.relpath(t["output_path"], "/data/users/zfh/workspace/projects/simOfflineTSC") \
                 if t["output_path"].startswith("/") else t["output_path"]
            q_by_out[rp] = q["file"]
        if t.get("run_id"):
            q_task[t["run_id"]] = (q["file"], t)

# state manifests status map
state_status = {}   # run_id -> (state_dir, status)
for s in inv["states"]:
    for t in s.get("tasks", []) or []:
        if t.get("run_id"):
            state_status[t["run_id"]] = (s["dir"], t.get("status"), t.get("exit_code"))

def queue_for(run):
    d = run["dir"]
    if d in q_by_out:
        return q_by_out[d]
    # match by prefix basename
    base = d.rstrip("/").split("/")[-1]
    for q in inv["queues"]:
        for t in q["tasks"]:
            if t.get("prefix") == base or (t.get("output_path") or "").endswith("/" + base):
                return q["file"]
    return None

# ---------- campaign / arm classification ----------
EVCOND = ("canonical", "empty", "wrong_location", "wrong_type", "both_wrong",
          "foreign", "relabeled", "normal", "rain", "closure", "blockage")

def classify(dir_rel, name, agent):
    """-> (line, campaign, arm, kind) ; kind in train/eval/smoke/batch"""
    n = name
    if "cross_algorithm/plan5_b100" in dir_rel:
        return "semi_offline", "plan5_b100", None, "train"
    if "arterial" in dir_rel or n.startswith(("stage0_", "stage1_", "stage2_", "arterial_")) \
       or agent == "shared_dqn":
        arm = re.sub(r"_seed\d+.*$", "", n)
        return "semi_offline", "arterial_1x6", arm, "train"
    if n.startswith(("p1_formal", "p1_pilot", "p1_eval", "p1m3")):
        return "semi_offline", "plan1_dqn", "dqn", "train" if "eval" not in n else "eval"
    if n.startswith(("m0f", "verify_sumo", "smoke_runtime")):
        return "engineering", "milestone0_infra", "dqn", "smoke"
    if n.startswith(("h1x1",)):
        return "attention", "att_entity_004", "h1x1_smoke", "smoke"
    if "paper_robustness" in dir_rel or "sumo_events/" in dir_rel or \
       "training_dashboard_examples" in dir_rel or "_paper_dashboard_examples" in dir_rel:
        return "engineering", "paper_infra_validation", None, "batch"
    # ---- attention line ----
    if n.startswith("att004_") or n.startswith("h1x1"):
        sub = n[len("att004_"):]
        arm = re.sub(r"_s\d+$", "", sub)
        if "smoke" in n or "h1x1" in n or "bert_regress" in n or "atsmoke" in n:
            return "attention", "att_entity_004", arm, "smoke"
        return "attention", "att_entity_004", arm, "train"
    m = re.match(r"fe_(.+?_s\d+)_ep(\d+)_([a-z_]+?)_v(\d+)$", n)
    if m:
        return "attention", "att_entity_004", m.group(1), "eval"
    m = re.match(r"fe_(att004_\w+?)_ep(\d+)_([a-z_]+?)_v(\d+)$", n)
    if m:
        return "attention", "att_entity_004", m.group(1), "eval"
    if re.match(r"^(a2s|a3s)_s\d+_(ce|cf|e0)", n):
        arm = n.split("_")[0]
        return "attention", "att_entity_004", arm, "eval"
    if n.startswith(("dense_", "mid15_", "mid20_", "single_", "dumplib_", "e200p_",
                     "bcmp_", "stl_", "ho_t", "diag_", "smoke_dumplib")):
        arm = re.sub(r"^(dense|mid15|mid20|single|dumplib|e200p|bcmp|stl|ho_t\d|diag|smoke_dumplib)_?", "", n)
        arm = re.sub(r"_ss?\d+.*$", "", arm)
        return "attention", "att_entity_004", arm or "?", "eval" if not n.startswith("smoke_") else "smoke"
    if n.startswith(("tarlp_", "tarlcf", "tarlrepr", "tarlutil")):
        arm = re.sub(r"^(tarlp|tarlcf|tarlrepr|tarlutil)_?", "", n)
        kind = "eval" if not n.startswith("tarlp_") else "train"
        if "smoke" in n:
            kind = "smoke"
        return "attention", "att_entity_003", arm, kind
    if n.startswith(("flxcf", "sga_flx", "flx_align", "align_", "retr_", "probe",
                     "mplight_event_random", "sga_colight_text_event_random",
                     "colight_event_random", "sga_flx_colight_text_event_200",
                     "sga_flx_colight_text_event_random")):
        arm = n
        kind = "eval" if n.startswith(("flxcf", "retr_", "probe")) else \
               ("smoke" if "smoke" in n or "livefire" in n else "train")
        return "attention", "att_entity_003", arm, kind
    if n.startswith("mplight_randE"):
        return "attention", "att_entity_003", "mplight_randE", "train"
    if n.startswith(("sgacf_", "sga_", "concat_colight", "colight_", "mplight_",
                     "sga_mplight")):
        kind = "train"
        if "smoke" in n:
            kind = "smoke"
        elif n.startswith(("sgacf_", "sga_closed", "sga_counterfactual", "sga_location",
                           "sga_wrong", "sga_colight_sga_v2", "sga_mplight_sga_v2",
                           "sga_v1_regress", "sga_cf2")):
            kind = "eval"
        return "attention", "att_entity_002", n, kind
    if n.startswith(("tarl_v1", "tarl_v21", "tarl_graph", "tarl_fixed", "tarl_mp",
                     "tarl_smoke", "tarl_dqn", "tarl_maxpressure")):
        camp = "tarl_v21_formal" if n.startswith("tarl_v21_formal") else "tarl_reproduction"
        kind = "train"
        if "smoke" in n or "pilot" in n or "audit" in n or "contract" in n or "probe" in n:
            kind = "smoke"
        return "attention", camp, n, kind
    if n in ("test", "readiness", "inspect_mplight", "frap_adaptation_probe_20260920",
             "frap_adaptation_probe2_20260920", "colight_adaptation_probe_20260920",
             "rewards_dashboard_probe_20260920", "p1m3_eval_smoke_20260722_1"):
        return "engineering", "misc_probes", n, "smoke"
    if n.startswith("colight_event_smoke"):
        return "attention", "att_entity_002", n, "smoke"
    return "uncertain", "LEGACY_UNCLEAR", n, "train"

STATUS_MAP = {"已完成": "DONE", "失败": "FAILED", "运行中": "RUNNING_STALE",
              "已中断": "ABORTED", "已创建": "REGISTERED"}

rows = []
def add(row):
    row.setdefault("alias_of", None); row.setdefault("resume_from", None)
    row.setdefault("ngpu", None); row.setdefault("plan_id", None)
    row.setdefault("dirty", None); row.setdefault("tmux", None)
    row.setdefault("eval_mode", None); row.setdefault("behavior_source", None)
    row.setdefault("metrics", {}); row.setdefault("abort_reason", None)
    row.setdefault("error", None); row.setdefault("evidence_lost", False)
    row.setdefault("episodes_total", None); row.setdefault("episodes_increment", None)
    row.setdefault("code_commit", None); row.setdefault("queue_file", None)
    row.setdefault("notes", "")
    row.setdefault("runtime_env", "colight")
    rows.append(row)

# ---------- tsc run dirs ----------
for r in inv["runs"]:
    d = r["dir"]
    if not d.startswith("data/output_data/tsc/"):
        continue
    rest = d[len("data/output_data/tsc/"):]
    parts = rest.split("/")
    if parts[0].startswith("_"):        # _paper_dashboard_examples component dirs
        continue
    quarantined = len(parts) == 4       # <agent>/<net>/<quarantine_dir>/<run>
    top = len(parts) == 3
    if not (top or quarantined):
        continue
    name = parts[-1]
    man = r["manifest"] or {}
    st = r["status"] or {}
    agent = man.get("agent") or (parts[0].replace("sumo_", ""))
    net = man.get("network") or parts[-2]
    seed = man.get("training_seed")
    if seed is None:
        m = re.search(r"_s(?:eed)?(\d+)", name)
        seed = int(m.group(1)) if m else None
    line, camp, arm, kind = classify(d, name, agent)
    status_zh = st.get("status")
    status = STATUS_MAP.get(status_zh, "LEGACY_UNCLEAR" if status_zh is None else status_zh)
    notes = []
    if quarantined:
        notes.append(f"quarantined under {parts[-2]}")
        if "aborted" in parts[-2] or "stragglers" in parts[-2] or "pre_pause" in parts[-2]:
            status = "ABORTED"
            notes.append("隔离桶语义=中止残留")
        elif "misresolved" in parts[-2] or "failed" in parts[-2] or "invalid" in parts[-2]:
            status = "FAILED"
            notes.append("隔离桶语义=失效/错配残留")
    if status == "RUNNING_STALE":
        eps = (r["ep"] or {}).get("train")
        status = "ABORTED"
        notes.append(f"run_status 残留'运行中'；无存活进程(2026-10-07 核查)；"
                     f"已写训练记录 {eps} 条")
    ep = r["ep"] or {}
    last_ev = ep.get("last_eval_full") or {}
    metrics = {}
    for k in ("travel_time", "throughput", "unfinished_vehicles", "queue", "delay"):
        if last_ev.get(k) is not None:
            metrics[k] = round(last_ev[k], 3) if isinstance(last_ev[k], float) else last_ev[k]
    episodes_total = ep.get("train") if ep.get("train") else (ep.get("eval") if kind == "eval" else ep.get("train"))
    add({
        "run_id": name,
        "line": line, "campaign": camp, "arm": arm,
        "net": net, "seed": seed,
        "episodes_total": episodes_total,
        "episodes_increment": episodes_total,
        "status": status,
        "code_commit": (man.get("baseline_commit") or "")[:9] or None,
        "run_dir": d,
        "started": st.get("started_at_utc"),
        "eval_mode": "frozen_eval" if kind == "eval" else ("smoke" if kind == "smoke" else "train_frozen_eval"),
        "metrics": metrics,
        "error": (st.get("error_message") or None) if status_zh == "失败" else None,
        "queue_file": queue_for(r),
        "notes": "; ".join(notes),
        "abort_reason": ("进程外部终止/状态残留" if status == "ABORTED" and status_zh == "运行中"
                         else ("状态文件标记'已中断'" if status == "ABORTED" else None)),
        "config_sha": man.get("config_hash"),
        "sumo_seed_mode": man.get("sumo_seed_mode"),
        "sumo_seed": man.get("sumo_seed"),
        "run_kind": kind,
    })

# ---------- artifacts eval attempts ----------
for r in inv["runs"]:
    d = r["dir"]
    if not (d.startswith("artifacts/") and "/attempts/" in d):
        continue
    st = r["status"] or {}
    name = d.split("/attempts/")[-1]
    pkg = d.split("/attempts/")[0].split("/")[-1]
    root = d.split("/")[1]
    camp = {"sga_concat_colight_200_v1": "att_entity_002",
            "att_entity_003": "att_entity_003",
            "arterial_experiments": "arterial_1x6"}.get(root, "LEGACY_UNCLEAR")
    line = "attention" if camp.startswith("att_") else "semi_offline"
    status_zh = st.get("status")
    status = STATUS_MAP.get(status_zh, "LEGACY_UNCLEAR" if status_zh is None else status_zh)
    quaran = "quarantine" in d or "partial_" in d or "INVALID" in d
    notes = [f"eval package={pkg}"]
    if status == "RUNNING_STALE":
        status = "ABORTED"; notes.append("run_status 残留'运行中'，无存活进程")
    if "INVALID" in d: status, notes = "FAILED", notes + ["invalidated: no_ckpt_load"]
    elif "partial_" in d:
        if status not in ("ABORTED",): notes.append("partial package (straggler)")
    elif "quarantine" in d: status = "ABORTED"; notes.append("quarantined")
    ab_reason = ("run_status 残留'运行中'，无存活进程" if "残留" in ";".join(notes)
                 else ("评估包隔离/straggler 残留" if status == "ABORTED" else None))
    m = re.search(r"_s(\d+)_ep(\d+)_([a-z_]+?)__eval_seed_(\d+)$", name)
    cond = m.group(3) if m else None
    evseed = int(m.group(4)) if m else None
    arm = re.sub(r"_200_s\d+$", "", pkg)
    add({
        "run_id": name, "line": line, "campaign": camp, "arm": arm,
        "net": "hz4x4" if line == "attention" else "sumoarterial1x6",
        "seed": None,
        "status": status, "run_dir": d, "started": st.get("started_at_utc"),
        "abort_reason": ab_reason,
        "eval_mode": "frozen_eval_attempt",
        "behavior_source": pkg,
        "metrics": {},
        "error": (st.get("error_message") or None) if status_zh == "失败" else None,
        "evidence_lost": False,
        "notes": "; ".join(notes) + (f"; cond={cond} eval_seed={evseed}" if m else ""),
        "run_kind": "eval",
        "eval_seed": evseed, "condition": cond,
    })

# ---------- artifacts top-level run dirs (quarantine etc, non-attempt) ----------
for r in inv["runs"]:
    d = r["dir"]
    if not d.startswith("artifacts/") or "/attempts/" in d:
        continue
    if d.endswith("run_state") or "/run_state" in d or d.rstrip("/").split("/")[-1].endswith("_state"):
        continue
    man = r["manifest"] or {}
    st = r["status"] or {}
    if not man and not st:
        continue
    name = d.split("/")[-1]
    root = d.split("/")[1]
    camp = {"sga_concat_colight_200_v1": "att_entity_002", "att_entity_003": "att_entity_003",
            "att_entity_004": "att_entity_004", "arterial_experiments": "arterial_1x6"}.get(root, "LEGACY_UNCLEAR")
    line = "attention" if camp.startswith("att_") else "semi_offline"
    status_zh = st.get("status")
    status = STATUS_MAP.get(status_zh, "LEGACY_UNCLEAR" if status_zh is None else status_zh)
    notes = []
    if "quarantine" in d:
        notes.append("quarantined")
        if "invalid" in d or "misresolved" in d or "dropped_stragglers" in d:
            status = "FAILED" if "invalid" in d or "misresolved" in d else "ABORTED"
        else:
            status = "ABORTED"
    if status == "RUNNING_STALE":
        status = "ABORTED"
        notes.append("run_status 残留'运行中'；无存活进程")
    ep = r["ep"] or {}
    add({
        "run_id": name + "__" + d.split("/")[-2] if not name else name,
        "line": line, "campaign": camp,
        "arm": re.sub(r"_200_s\d+$", "", name),
        "net": man.get("network") or "hz4x4",
        "seed": man.get("training_seed"),
        "episodes_total": ep.get("train"), "episodes_increment": ep.get("train"),
        "status": status, "run_dir": d, "started": st.get("started_at_utc"),
        "eval_mode": None, "metrics": {},
        "error": (st.get("error_message") or None) if status_zh == "失败" else None,
        "notes": "; ".join(notes), "run_kind": "train",
        "abort_reason": "队列中止/隔离" if status == "ABORTED" else None,
    })

# ---------- plan5 logical runs ----------
for p in inv["plan5"]:
    lm = p["manifest"] or {}
    lr = p["logical_run_id"]
    m = re.match(r"P5-(ANCHOR|CAL)-([A-Z]+)-(.*)", lr)
    run_type = m.group(1) if m else "?"
    algo = m.group(2) if m else "?"
    scene = re.search(r"-(S\d)-", lr)
    seed = re.search(r"-SD(\d+)", lr)
    status_map5 = {"planned": "REGISTERED", "completed": "DONE", "failed": "FAILED",
                   "running": "RUNNING"}
    eff = None
    natt = len(p["attempts"])
    summ_status = None
    for a in p["attempts"]:
        if a.get("summary_status"):
            summ_status = a["summary_status"]
    status = status_map5.get(summ_status or lm.get("status"), "LEGACY_UNCLEAR")
    row_status = None
    for a in p["attempts"]:
        rr = a.get("run_row") or {}
        row_status = rr.get("status") or row_status
        eff = eff or rr.get("source_commit")
    if status == "REGISTERED" and summ_status is None and row_status == "planned":
        status = "LEGACY_UNCLEAR"
    add({
        "run_id": lr, "line": "semi_offline", "campaign": "plan5_b100",
        "arm": algo + "_" + run_type,
        "net": {"S1": "sumohz1x1_config2", "S2": "sumohz1x1", "S3": "sumohz1x1_config4",
                "S4": "sumohz1x1_config3"}.get(scene.group(1) if scene else "", None),
        "seed": int(seed.group(1)) if seed else None,
        "episodes_total": 100, "episodes_increment": 100,
        "status": status,
        "plan_id": "plan5_b100",
        "code_commit": (eff or "")[:9] or None,
        "run_dir": f"data/output_data/cross_algorithm/plan5_b100/runs/{lr}",
        "notes": f"attempts={natt}; type={run_type}" + ("; manifest stale" if status == "LEGACY_UNCLEAR" else ""),
        "run_kind": "train" if run_type == "ANCHOR" else "calibration",
    })

# ---------- engineering batches ----------
for r in inv["runs"]:
    d = r["dir"]
    if "paper_robustness" in d or "sumo_events/" in d or "training_dashboard_examples" in d:
        pass
batch_roots = collections.OrderedDict()
for r in inv["runs"]:
    d = r["dir"]
    for top in ("data/output_data/paper_robustness/", "data/output_data/sumo_events/",
                "data/output_data/training_dashboard_examples/"):
        if d.startswith(top):
            root = top + d[len(top):].split("/")[0]
            batch_roots.setdefault(root, {"n": 0, "done": 0, "fail": 0, "other": 0})
            b = batch_roots[root]
            b["n"] += 1
            s = (r["status"] or {}).get("status")
            if s == "已完成": b["done"] += 1
            elif s == "失败": b["fail"] += 1
            else: b["other"] += 1
for root, b in batch_roots.items():
    add({
        "run_id": "BATCH/" + root.split("/")[-1],
        "line": "engineering", "campaign": "paper_infra_validation",
        "arm": None, "net": "hz4x4", "seed": 7,
        "status": "DONE" if b["fail"] == 0 else "DONE",
        "run_dir": root,
        "eval_mode": "engineering_validation",
        "notes": f"工程验证批次：内部 run 单元 {b['n']}（完成 {b['done']}，失败 {b['fail']}，其他 {b['other']}）；pytest/预检类微执行",
        "run_kind": "batch",
    })

# ---------- queue tasks with no run dir ----------
existing_dirs = {r["run_dir"] for r in rows}
for q in inv["queues"]:
    qf = q["file"]
    camp = "att_entity_004" if "att004" in qf or "att_entity_004" in qf or "e200" in qf else \
           "att_entity_003" if "att_entity_003" in qf else \
           "att_entity_002" if "sga_concat" in qf else \
           "arterial_1x6" if "arterial" in qf else "LEGACY_UNCLEAR"
    for t in q["tasks"]:
        op = t.get("output_path") or ""
        mrel = re.search(r"simOfflineTSC/(.*)$", op)
        rp = mrel.group(1) if mrel else op
        if rp and rp in existing_dirs:
            continue
        # eval 队列任务的产物是 artifacts 评估包，已由 attempt 行覆盖；
        # 仅保留 DROPPED（计划但未派发）以示范围裁剪。
        if "data/output_data" not in op and t.get("status") != "DROPPED":
            continue
        st = t.get("status")
        stt = state_status.get(t.get("run_id"), (None, None, None))
        if stt[1]:
            st = stt[1]
        status = {"DROPPED": "SCOPE_DISCARD", "SUCCEEDED": "DONE",
                  "INVALID": "FAILED", "FAILED": "FAILED", "SKIPPED_COMPLETED": "DONE",
                  "RUNNING": "ABORTED"}.get(st, "REGISTERED")
        is_eval_q = bool(re.search(r"(eval|condeval|probe)", qf))
        notes = [f"队列登记无对应产物目录(prefix={t.get('prefix')})"]
        if status == "SCOPE_DISCARD":
            notes.append("队列标记 DROPPED=未派发")
        dir_exists = bool(rp) and os.path.isdir(os.path.join(ROOT, rp))
        if status in ("REGISTERED", "SCOPE_DISCARD") or not dir_exists:
            notes.append("目标目录不存在（计划输出位置仅作记录）")
        add({
            "run_id": "__".join(t["run_id"].split("/")[-2:]), "line": "attention" if "att_" in camp else "semi_offline",
            "campaign": camp, "arm": t.get("prefix") or t["run_id"].split("/")[-1],
            "net": "hz4x4", "seed": int(t["seed"]) if str(t.get("seed") or "").isdigit() else None,
            "status": status, "queue_file": qf,
            "run_dir": rp if dir_exists else None,
            "eval_mode": "frozen_eval_attempt" if is_eval_q else None,
            "abort_reason": "队列 DROPPED/未执行" if status == "SCOPE_DISCARD" else (
                "状态残留 RUNNING 无存活进程" if status == "ABORTED" else None),
            "error": "队列终态 FAILED，无产物" if status == "FAILED" else None,
            "notes": "; ".join(notes), "run_kind": "eval" if is_eval_q else "train",
            "evidence_lost": status == "FAILED",
        })

# ---------- plan5 phase0 evidence batches ----------
for sub, desc in [("fixedtime", "FixedTime 参照（S1-S4×repeat×2）"),
                  ("probe/plan5_fixed_probe_v1", "plan5_fixed_probe_v1 冻结探针"),
                  ("probe/evaluation_runs", "探针评估执行"),
                  ("resume_equivalence", "resume 等价性验证"),
                  ("resource_gate", "1/4/8 并发门禁"),
                  ("resource_gate_attempt_1_failed", "并发门禁失败 attempt"),
                  ("resource_gate_attempt_2", "并发门禁 attempt2"),
                  ("resource_gate_attempt_3", "并发门禁 attempt3"),
                  ("environment", "plan5 micromamba 环境冻结"),
                  ("provenance", "源码 provenance/bundle")]:
    p = f"data/output_data/cross_algorithm/plan5_b100/{sub}"
    if os.path.isdir(os.path.join(ROOT, p)):
        add({"run_id": "P5-PHASE0/" + sub.replace("/", "_"), "line": "semi_offline",
             "campaign": "plan5_b100", "arm": "phase0", "net": None, "seed": None,
             "status": "DONE", "run_dir": p, "eval_mode": "engineering_validation",
             "notes": "Phase0 基础设施/门禁证据批次: " + desc, "run_kind": "batch"})

# ---------- supersede relations (known) ----------
name2row = collections.defaultdict(list)
for row in rows:
    name2row[row["run_id"]].append(row)
for row in rows:
    n = row["run_id"]
    if re.match(r"p1_formal_dqn_.*_400ep_20260722$", n) and row["status"] == "ABORTED":
        row["status"] = "SUPERSEDED"
        row["notes"] = (row["notes"] + "; " if row["notes"] else "") + \
            f"由同身份重跑 {n}_r2 替代"; row["superseded_by"] = n + "_r2"
    if n.startswith("att004_fixG_") and n.endswith("_aborted100"):
        row["notes"] += "; 由 fixG 200ep 正式波次 att004_fixG_s* 取代(不同预算口径,非同身份)"

# ---------- run_id uniqueness + quarantine aliasing ----------
cnt = collections.Counter(r["run_id"] for r in rows)
old_ids = {}
for row in rows:
    row["_old_id"] = row["run_id"]
for row in rows:
    if cnt[row["run_id"]] > 1 and row.get("run_dir"):
        row["run_id"] = row["run_dir"].replace("data/output_data/tsc/", "").replace("/", "__")
old_groups = collections.defaultdict(list)
for row in rows:
    old_groups[(row["campaign"], row["_old_id"])].append(row)
for (camp, oid), rs in old_groups.items():
    if len(rs) < 2:
        continue
    canon = next((r for r in rs if r["status"] == "DONE" and
                  (r["run_dir"] or "").startswith("data/output_data")), None)
    if canon:
        for r in rs:
            if r is not canon:
                r["alias_of"] = canon["run_id"]
                r["notes"] = (r["notes"] + "; " if r["notes"] else "") + \
                    f"同名隔离/残留副本，逻辑同一 run→{canon['run_id']}"
for row in rows:
    row.pop("_old_id", None)

# ---------- dedupe ----------
seen = {}
for row in rows:
    k = (row["run_id"], row["run_dir"])
    seen[k] = row
rows = list(seen.values())

# ---------- output ----------
fields = ["run_id", "line", "campaign", "arm", "alias_of", "net", "seed",
          "episodes_total", "episodes_increment", "resume_from", "ngpu",
          "status", "plan_id", "code_commit", "dirty", "queue_file", "run_dir",
          "tmux", "started", "eval_mode", "behavior_source", "metrics",
          "abort_reason", "error", "evidence_lost", "notes", "run_kind",
          "config_sha", "sumo_seed_mode", "sumo_seed", "eval_seed", "condition",
          "superseded_by", "runtime_env"]
with open(os.path.join(OUT, "runs.jsonl"), "w") as f:
    for row in rows:
        f.write(json.dumps({k: row.get(k) for k in fields}, ensure_ascii=False) + "\n")

c = collections.Counter((r["campaign"], r["status"]) for r in rows)
for (camp, st), n in sorted(c.items()):
    print(f"{camp:28s} {st:16s} {n}")
print("TOTAL", len(rows))
