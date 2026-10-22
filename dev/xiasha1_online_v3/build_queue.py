#!/usr/bin/env python3
"""Build artifacts/xiasha1_online_v3/run_queue.json (288 tasks).

v3 = xiasha1 几何修复版重跑：8 arms x 12 packs x 3 seeds，训练协议与
xiasha1_online_v2 完全一致（200 eps, pseudo_step, overlay
configs/tsc/xiasha1_online_v3/<arm>.yml）。run_id 前缀 xi3re_。

用法: python dev/xiasha1_online_v3/build_queue.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "xiasha1_online_v3"
PY = "/home/dev/miniforge3/envs/colight/bin/python"
ARMS = ["dqn", "ddqn", "dueling", "d3qn", "presslight", "frap", "mplight", "sac"]
PACKS = ["sparse_low", "dense_high", "dense_extreme", "rt_high", "lt_high",
         "st_dom", "asym_ns", "asym_we", "asym_nonly", "fluct_sin",
         "fluct_burst", "fluct_ramp"]
SEEDS = [0, 1, 2]
AGENT_DIR = {a: "dqn" for a in ("dqn", "ddqn", "dueling", "d3qn")}
ENV = {"LD_PRELOAD": "/home/dev/miniforge3/envs/colight/lib/libstdc++.so.6",
       "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1",
       "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}

tasks = []
for arm in ARMS:
    for pack in PACKS:
        for seed in SEEDS:
            rid = f"xi3re_{arm}_{pack}_s{seed}"
            outdir = (ROOT / "data" / "output_data" / "tsc" /
                      f"sumo_{AGENT_DIR.get(arm, arm)}" /
                      f"xiasha1_{pack}" / rid)
            tasks.append({
                "run_id": rid,
                "arm": arm,
                "pack": pack,
                "seed": seed,
                "command": [PY, "run.py", "-w", "sumo", "-a", arm,
                            "-n", f"xiasha1_{pack}", "--seed", str(seed),
                            "--interface", "libsumo", "--prefix", rid,
                            "--experiment-config",
                            f"configs/tsc/xiasha1_online_v3/{arm}.yml"],
                "cwd": str(ROOT),
                "env": dict(ENV),
                "output_path": str(outdir),
                "completion": {
                    "path": str(outdir / "run_status.json"),
                    "json_field": "status",
                    "equals": "已完成"}})

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "run_queue.json").write_text(
    json.dumps({"campaign": "xiasha1_online_v3", "tasks": tasks},
               ensure_ascii=False, indent=1) + "\n")
print(f"wrote {OUT/'run_queue.json'}: {len(tasks)} tasks")
