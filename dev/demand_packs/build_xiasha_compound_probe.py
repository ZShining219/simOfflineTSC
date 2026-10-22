#!/usr/bin/env python3
"""xiasha1 compound_a/compound_b 区分度探测队列生成件（一次性）。

Probe：v2 每臂 dense_extreme 代表 seed 的 best ckpt（xi2ts 台账口径的
同一批 frozen checkpoint）跨评到两个新综合需求包 compound_a/compound_b，
判读「综合难度场景是否拉开学习臂差异」。另附 fixedtime/sotl/maxpressure
规则参照（run.py + xiasha1_timeseries overlay，1s eval_step_log）。

生成：
  artifacts/xiasha1_compound_probe/run_queue.json
      8 arm × 2 target pack = 16 个 frozen ckpt eval（prefix xi3x_）
      3 rule × 2 target pack = 6 个采集 run（prefix xi3ts_）

用法：
    python dev/demand_packs/build_xiasha_compound_probe.py
    python tools/run_experiment_queue.py \\
        --queue artifacts/xiasha1_compound_probe/run_queue.json \\
        --state artifacts/xiasha1_compound_probe/run_state --parallel 6
"""
import json
import pathlib

ROOT = pathlib.Path("/projects/simOfflineTSC")
SRC_QUEUE = ROOT / "artifacts/xiasha1_v2_bestckpt_ts/run_queue.json"
SOURCE_PACK = "dense_extreme"          # 最高量级既有包，最接近复合场景
TARGETS = ("compound_a", "compound_b")
RULE_AGENTS = {"fixedtime": "ft", "sotl": "sotl", "maxpressure": "mp"}
RULE_OVERLAY = "configs/tsc/xiasha1_timeseries.yml"
OUT = ROOT / "artifacts/xiasha1_compound_probe"
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
    src = json.loads(SRC_QUEUE.read_text(encoding="utf-8"))
    src_tasks = {t["arm"]: t for t in src["tasks"]
                 if t["pack"] == SOURCE_PACK}
    assert len(src_tasks) == 8, sorted(src_tasks)
    tasks = []
    for target in TARGETS:
        net = f"xiasha1_{target}"
        for arm, t in sorted(src_tasks.items()):
            rid = f"xi3x_{arm}_{target}"
            out = (f"/projects/simOfflineTSC/data/output_data/tsc/"
                   f"sumo_{arm}/{net}/{rid}")
            tasks.append({
                "run_id": rid,
                "arm": arm,
                "source_pack": SOURCE_PACK,
                "target_pack": target,
                "rep_train_seed": t["rep_train_seed"],
                "eval_seed": 0,
                "sumo_seed_mode": "fixed_default",
                "eval_overlay": t["eval_overlay"],
                "checkpoint": t["checkpoint"],
                "best_episode": t["best_episode"],
                "best_travel_time": t["best_travel_time"],
                "source_run_dir": t["source_run_dir"],
                "command": [
                    PY, "tools/eval_baseline_checkpoint.py",
                    "--agent", arm, "--network", net,
                    "--config", t["eval_overlay"],
                    "--checkpoint", t["checkpoint"],
                    "--seed", "0", "--prefix", rid,
                ],
                "cwd": "/projects/simOfflineTSC",
                "env": ENV,
                "output_path": out,
                "completion": {
                    "path": f"{out}/run_status.json",
                    "json_field": "status",
                    "equals": "已完成",
                },
            })
        for agent, short in RULE_AGENTS.items():
            rid = f"xi3ts_{short}_{target}"
            out = (f"/projects/simOfflineTSC/data/output_data/tsc/"
                   f"sumo_{agent}/{net}/{rid}")
            tasks.append({
                "run_id": rid,
                "arm": agent,
                "source_pack": None,
                "target_pack": target,
                "command": [
                    PY, "run.py", "-w", "sumo", "-a", agent,
                    "-n", net, "--seed", "0", "--interface", "libsumo",
                    "--prefix", rid,
                    "--experiment-config",
                    f"/projects/simOfflineTSC/{RULE_OVERLAY}",
                ],
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
        "plan": "xiasha1_compound_probe_v1",
        "expected_tasks": len(tasks),
        "note": "复合需求包区分度探测：8 臂 dense_extreme best-ckpt 跨评 "
                "compound_a/compound_b + ft/sotl/mp 规则参照；1s step log；"
                "SUMO fixed_default；legacy pseudo_step。",
        "tasks": tasks,
    }
    (OUT / "run_queue.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=1) + "\n")
    print(f"tasks={len(tasks)} -> {OUT}/run_queue.json")


if __name__ == "__main__":
    main()
