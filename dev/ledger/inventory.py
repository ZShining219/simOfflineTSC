#!/usr/bin/env python3
"""Full experiment inventory scanner for simOfflineTSC.
Emits inventory.json with every identifiable run under data/output_data/,
plus queue/state registrations. Read-only."""
import json, os, re, sys, csv, glob

ROOT = "/data/users/zfh/workspace/projects/simOfflineTSC"
OUT = "/data/users/zfh/workspace/projects/govern_1007/scratch"

def jload(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None

def rel(p):
    return os.path.relpath(p, ROOT)

def last_records(path):
    """Return (n_train_eps, n_eval_eps, last_eval_row, last_train_row) from records.jsonl."""
    ntr = nev = 0
    last_ev = last_tr = None
    try:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                rt = r.get("record_type")
                if rt == "TRAIN":
                    ntr += 1; last_tr = r
                elif rt == "EVALUATION":
                    nev += 1; last_ev = r
    except Exception:
        return None
    return ntr, nev, last_ev, last_tr

runs = []

# ---- 1. generic run dirs: any dir containing run_manifest.json or run_status.json ----
seen_dirs = set()
for base in ["data/output_data", "artifacts", "final_result", "logs"]:
    bpath = os.path.join(ROOT, base)
    if not os.path.isdir(bpath):
        continue
    for dirpath, dirnames, filenames in os.walk(bpath):
        # prune gigantic env/venv dirs
        dirnames[:] = [d for d in dirnames if d not in
                       ("micromamba", "node_modules", "__pycache__", ".git", "site-packages")]
        if "run_manifest.json" in filenames or "run_status.json" in filenames:
            if dirpath in seen_dirs:
                continue
            seen_dirs.add(dirpath)
            man = jload(os.path.join(dirpath, "run_manifest.json")) or {}
            st = jload(os.path.join(dirpath, "run_status.json")) or {}
            ep = last_records(os.path.join(dirpath, "metrics", "records.jsonl"))
            runs.append({
                "dir": rel(dirpath),
                "manifest": man or None,
                "status": st or None,
                "ep": {"train": ep[0], "eval": ep[1],
                       "last_eval": {k: last_ev2 for k, last_ev2 in
                                     ((ep[2] or {}).items())} if ep and ep[2] else None,
                       "last_eval_full": ep[2] if ep else None,
                       "last_train": ep[3] if ep else None} if ep else None,
            })

print(f"generic run dirs: {len(runs)}", file=sys.stderr)

# ---- 2. plan5 logical runs ----
p5root = os.path.join(ROOT, "data/output_data/cross_algorithm/plan5_b100/runs")
p5 = []
if os.path.isdir(p5root):
    for lr in sorted(os.listdir(p5root)):
        ldir = os.path.join(p5root, lr)
        if not os.path.isdir(ldir):
            continue
        lm = jload(os.path.join(ldir, "logical_run_manifest.json")) or {}
        attempts = []
        adir = os.path.join(ldir, "attempts")
        if os.path.isdir(adir):
            for a in sorted(os.listdir(adir)):
                ap = os.path.join(adir, a)
                if not os.path.isdir(ap):
                    continue
                rr = jload(os.path.join(ap, "run_row.json"))
                rs = jload(os.path.join(ap, "run_summary.json"))
                rv = jload(os.path.join(ap, "run_validation.json"))
                attempts.append({"attempt": a, "run_row": rr,
                                 "has_summary": rs is not None,
                                 "summary_status": (rs or {}).get("status"),
                                 "validation": rv})
        p5.append({"logical_run_id": lr, "manifest": lm, "attempts": attempts})

print(f"plan5 logical runs: {len(p5)}", file=sys.stderr)

# ---- 3. queues + state manifests ----
queues = []
for qf in sorted(glob.glob(os.path.join(ROOT, "**/run_queue*.json"), recursive=True)):
    if "/.git/" in qf or "/micromamba/" in qf:
        continue
    d = jload(qf)
    if d is None:
        continue
    tasks = d.get("tasks", []) if isinstance(d, dict) else d
    queues.append({"file": rel(qf), "n_tasks": len(tasks),
                   "tasks": [{"run_id": t.get("run_id"), "status": t.get("status"),
                              "agent": t.get("agent"), "config_path": t.get("config_path"),
                              "output_path": t.get("output_path"),
                              "seed": (t.get("command") or [])[t.get("command", []).index("--seed") + 1]
                              if "--seed" in (t.get("command") or []) else None,
                              "prefix": (t.get("command") or [])[t.get("command", []).index("--prefix") + 1]
                              if "--prefix" in (t.get("command") or []) else None,
                              "env": {k: v for k, v in (t.get("env") or {}).items()
                                      if k.startswith("TARL") or k.startswith("CUDA") or k.startswith("SGA")},
                              "experiment_id": t.get("experiment_id"),
                              } for t in tasks]})

states = []
for sd in sorted(glob.glob(os.path.join(ROOT, "**/run_state_*"), recursive=True) +
                 glob.glob(os.path.join(ROOT, "artifacts/*/run_state"), recursive=True)):
    if not os.path.isdir(sd):
        continue
    mf = os.path.join(sd, "run_manifest.json")
    m = jload(mf)
    entry = {"dir": rel(sd), "files": sorted(os.listdir(sd)),
             "manifest_source": (m or {}).get("source_manifest"),
             "updated_at": (m or {}).get("updated_at")}
    if m:
        entry["tasks"] = [{"run_id": t.get("run_id"), "status": t.get("status"),
                           "state": t.get("state"), "finished": t.get("finished_at"),
                           "exit_code": t.get("exit_code")}
                          for t in m.get("tasks", [])]
    states.append(entry)

print(f"queues: {len(queues)}, states: {len(states)}", file=sys.stderr)

# ---- 4. dirs under tsc net level lacking run_status (incomplete) ----
incomplete = []
tsc = os.path.join(ROOT, "data/output_data/tsc")
if os.path.isdir(tsc):
    for agent in os.listdir(tsc):
        for net in os.listdir(os.path.join(tsc, agent)) if os.path.isdir(os.path.join(tsc, agent)) else []:
            nd = os.path.join(tsc, agent, net)
            if not os.path.isdir(nd):
                continue
            for rd in os.listdir(nd):
                fp = os.path.join(nd, rd)
                if os.path.isdir(fp) and fp not in seen_dirs:
                    incomplete.append(rel(fp))

out = {"runs": runs, "plan5": p5, "queues": queues, "states": states,
       "incomplete_dirs": incomplete}
with open(os.path.join(OUT, "inventory.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"incomplete dirs: {len(incomplete)}", file=sys.stderr)
print("wrote inventory.json", file=sys.stderr)
