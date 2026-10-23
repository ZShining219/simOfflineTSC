#!/usr/bin/env python3
"""Build artifacts/hz1x1_sanity_v1/run_queue.json (60 tasks).

hz1x1 口径体检：5 臂(d3qn/frap/sac/presslight/mplight) × 4 原始真实数据包
(bc-tyc/qc-yn/kn-hz/sb-sx，非生成 demand pack) × 3 seeds，训练协议与
xiasha1_online_v3 逐位一致。目的：验证当前训练+eval 记录口径是否在
干净路网上产出形态正常的数据，隔离"口径问题 vs xiasha1 车流数据问题"。

用法: python dev/hz1x1_sanity/build_queue.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "hz1x1_sanity_v1"
PY = "/home/dev/miniforge3/envs/colight/bin/python"
ARMS = ["d3qn", "frap", "sac", "presslight", "mplight"]
PACKS = {  # packkey -> (-n 名称, 数据目录)
    "bctyc": ("sumohz1x1", "hangzhou_1x1_bc-tyc_18041610_1h"),
    "qcyn": ("sumohz1x1_config2", "hangzhou_1x1_qc-yn_18041608_1h"),
    "knhz": ("sumohz1x1_config3", "hangzhou_1x1_kn-hz_18041608_1h"),
    "sbsx": ("sumohz1x1_config4", "hangzhou_1x1_sb-sx_18041607_1h"),
}
SEEDS = [0, 1, 2]
AGENT_CMD = {"d3qn": "dqn"}
AGENT_DIR = {"d3qn": "dqn"}
ENV = {"LD_PRELOAD": "/home/dev/miniforge3/envs/colight/lib/libstdc++.so.6",
       "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1",
       "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}

tasks = []
for arm in ARMS:
    for key, (net_name, _dir) in PACKS.items():
        for seed in SEEDS:
            rid = f"hz1sn_{arm}_{key}_s{seed}"
            outdir = (ROOT / "data" / "output_data" / "tsc" /
                      f"sumo_{AGENT_DIR.get(arm, arm)}" / net_name / rid)
            tasks.append({
                "run_id": rid,
                "arm": arm,
                "pack": key,
                "network": net_name,
                "seed": seed,
                "command": [PY, "run.py", "-w", "sumo",
                            "-a", AGENT_CMD.get(arm, arm),
                            "-n", net_name, "--seed", str(seed),
                            "--interface", "libsumo", "--prefix", rid,
                            "--experiment-config",
                            f"configs/tsc/hz1x1_sanity_v1/{arm}.yml"],
                "cwd": str(ROOT),
                "env": dict(ENV),
                "output_path": str(outdir),
                "completion": {
                    "path": str(outdir / "run_status.json"),
                    "json_field": "status",
                    "equals": "已完成"}})

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "run_queue.json").write_text(
    json.dumps({"campaign": "hz1x1_sanity_v1", "tasks": tasks},
               ensure_ascii=False, indent=1) + "\n")
print(f"wrote {OUT/'run_queue.json'}: {len(tasks)} tasks")
