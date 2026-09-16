#!/usr/bin/env python3
"""Run inference-only old-scene Stage-2 evaluations into a private cache."""
from __future__ import annotations
import json, os, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output_data/analysis/ANALYSIS_BUNDLE_V2_MECHANISM_COMPLETION/02_PLAN34_STAGE2_FROZEN_TIMELINE/eval_cache"
EPISODES = (0, 1, 5, 10, 25, 50, 75, 100)
SUMO_HOME = "/home/dev/miniforge3/envs/colight/lib/python3.10/site-packages/sumo"

def plan_jobs():
    root = ROOT / "output_data/sequential/plan34_b100_formal_60_20260723"
    for run in sorted(root.glob("plan34_b100_O*_seed*_*") ): # 60 runs
        manifest = json.loads((run / "attempts/attempt_1/child_run_manifest.json").read_text())
        order, seed = manifest["order_id"], int(manifest["training_seed"])
        networks = manifest["networks"]
        old = networks[0]
        for ep in EPISODES:
            name = "stage_02_policy_applied.pt" if ep == 0 else f"stage_02_episode_{ep:04d}.pt"
            snap = run / "attempts/attempt_1/checkpoints/online" / name
            if snap.is_file():
                yield ("Plan3/4", run.name, order, seed, ep, old, networks[1], snap)

def ha_jobs():
    root = ROOT / "output_data/ha_sodqn/formal_e7705f7_20260726"
    pairs = []
    pair_file = ROOT / "output_data/analysis/plan_ha_analysis_bundle_v1/ha/ha_cont_dhoa_pair_index.csv"
    import csv
    with pair_file.open() as f:
        for row in csv.DictReader(f):
            if row["pair_scope"] == "cross_order_R25" and not (row["order_id"] == "O3" and row["training_seed"] == "1"):
                pairs.extend((("CONT", row["cont_logical_run_id"]), ("DHOA", row["dhoa_logical_run_id"])))
    for method, logical in pairs:
        run = root / logical
        manifest = json.loads((run / "attempts/attempt_1/child_run_manifest.json").read_text())
        networks = manifest["networks"]
        for ep in EPISODES:
            name = "stage_02_policy_applied.pt" if ep == 0 else f"stage_02_episode_{ep:04d}.pt"
            snap = run / "attempts/attempt_1/checkpoints/online" / name
            if snap.is_file():
                yield ("HA", logical, manifest["order_id"], int(manifest["training_seed"]), ep, networks[0], networks[1], snap)

def run(job):
    family, logical, order, seed, ep, old, current, snap = job
    out = OUT / ("plan34_formal_protocol" if family == "Plan3/4" else "ha_formal_protocol")
    cmd = ["/home/dev/miniforge3/envs/colight/bin/python", "tools/evaluate_stage2_cell.py", "--snapshot", str(snap), "--network", old, "--output-root", str(out), "--training-network", current, "--episode", str(ep), "--seed", str(seed), "--controller-id", logical]
    env = dict(os.environ, SUMO_HOME=SUMO_HOME, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    return {"family": family, "logical_run_id": logical, "order_id": order, "seed": seed, "episode": ep, "returncode": result.returncode, "output": result.stdout[-1000:]}

def main():
    jobs = list(plan_jobs()) + list(ha_jobs())
    print("jobs", len(jobs), flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            try: result = future.result()
            except Exception as exc: result = {"returncode": -1, "error": repr(exc)}
            results.append(result)
            print(i, result.get("family"), result.get("logical_run_id"), result.get("episode"), result.get("returncode"), flush=True)
    (OUT / "batch_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
