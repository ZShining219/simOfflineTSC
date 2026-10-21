#!/usr/bin/env python3
"""xiasha1 需求包 no-training 规则控制器采集队列生成件（一次性）。

生成：
  configs/sim/xiasha1_<pack>.cfg   12 个（指向各包 sumocfg/rou，供 run.py -n）
  artifacts/xiasha1_notrain/run_queue.json
      12 pack × {fixedtime, sotl, maxpressure} = 36 个 1-episode 采集 run

用法：
  python dev/demand_packs/build_xiasha_notrain_queue.py
  python tools/run_experiment_queue.py \
      --queue artifacts/xiasha1_notrain/run_queue.json \
      --state artifacts/xiasha1_notrain/run_state --parallel 6
"""
import glob
import json
import pathlib

PACK_GLOB = "data/raw_data/xiasha1_*_1h"
AGENTS = ("fixedtime", "sotl", "maxpressure")
OUT = pathlib.Path("artifacts/xiasha1_notrain")
PY = "/home/dev/miniforge3/envs/colight/bin/python"
ENV = {
    "LD_PRELOAD": "/home/dev/miniforge3/envs/colight/lib/libstdc++.so.6",
    "CUDA_VISIBLE_DEVICES": "-1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def main():
    packs = sorted(pathlib.Path(p).name for p in glob.glob(PACK_GLOB))
    assert len(packs) == 12, packs
    tasks = []
    for dname in packs:
        pack = dname.replace("xiasha1_", "").replace("_1h", "")
        cfg = {
            # All demand packs share the same physical xiasha1 network.
            # Keep the pack identity in the file name/CLI -n argument; the
            # simulator-level network key is the shared physical network.
            "network": "xiasha1",
            "interval": 1.0, "seed": 0, "dir": "data/",
            "combined_file": f"raw_data/{dname}/{dname}.sumocfg",
            "roadnetFile": f"raw_data/{dname}/{dname}_signal.net.xml",
            "flowFile": f"raw_data/{dname}/{dname}.rou.xml",
            "no_warning": True, "name": "debug",
            "yellow_length": 5, "gui": False,
        }
        cfg_name = dname.replace("_1h", "")
        pathlib.Path(f"configs/sim/{cfg_name}.cfg").write_text(
            json.dumps(cfg, indent=2) + "\n")
        for agent in AGENTS:
            short = {"fixedtime": "ft", "sotl": "sotl", "maxpressure": "mp"}[agent]
            rid = f"xi1_{short}_{pack}"
            out = (f"/projects/simOfflineTSC/data/output_data/tsc/"
                   f"sumo_{agent}/{cfg_name}/{rid}")
            tasks.append({
                "run_id": rid,
                "command": [PY, "run.py", "-w", "sumo", "-a", agent,
                            "-n", cfg_name, "--seed", "0",
                            "--interface", "libsumo", "--prefix", rid],
                "cwd": "/projects/simOfflineTSC",
                "env": ENV,
                "output_path": out,
                "completion": {
                    "path": f"{out}/run_status.json",
                    "json_field": "status",
                    "equals": "已完成",
                },
            })
    OUT.mkdir(parents=True, exist_ok=True)
    spec = {
        "schema_version": 1,
        "plan": "xiasha1_notrain_matrix",
        "expected_tasks": len(tasks),
        "note": "12 需求包 × fixedtime/sotl/maxpressure 规则控制器 1-episode "
                "采集；原始 128s 程序臂由 verify_xiasha_packs 另采",
        "tasks": tasks,
    }
    (OUT / "run_queue.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=1) + "\n")
    print(f"cfgs=12 queue tasks={len(tasks)} -> {OUT}/run_queue.json")


if __name__ == "__main__":
    main()
