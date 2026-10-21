#!/usr/bin/env python3
"""xiasha1 no-training 控制器矩阵收割 + 对比图（一次性分析件）。

输入：
  artifacts/xiasha1_notrain/run_state/run_manifest.json（36 个 run.py 采集 run，
    fixedtime/sotl/maxpressure × 12 包，读 metrics/records.jsonl FINAL_EVALUATION）
  data/raw_data/xiasha1_*_1h/*.sumocfg（原始 128s 程序臂：本脚本用
    libsumo 被动重跑，不干预信号，指标采集代码路径与 run.py 一致）

产出 data/output_data/analysis/xiasha1_notrain/：
  notrain_matrix.json / notrain_matrix.csv
  fig_notrain_2x2.png    ATT / queue / waiting_s / throughput 四指标分组柱状
  fig_notrain_heatmap.png pack×arm ATT 热力图（行=包按压力排序）
  fig_notrain_backlog.png 终点积压堆叠柱（uninserted + in-net unfinished）

列口径（v2，跨臂可比）：
  waiting_s   终点仍在网车辆的累计等待均值（秒）——orig=summary
              meanWaitingTime，ctl=records waiting_time（world 同语义）
  real_delay_s 逐车实际-自由流时延（秒），仅 ctl 有（orig statistic 无对应）
  lane_delay_idx 原 run.py 'delay'：Σ_12 进口道(1-v̄/vmax) 决策步均值，无量纲
  uninserted  未注入（orig=stat waiting；ctl=loaded-throughput-unfinished）
  unfinished_in_net 终点网内未到达（orig=running；ctl=unfinished_vehicles）

用法：LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 \
      python dev/demand_packs/harvest_xiasha_notrain.py
"""
import argparse
import csv
import glob
import json
import os
import pathlib
import sys
import tempfile
import xml.etree.ElementTree as ET
STATE = [
    pathlib.Path("artifacts/xiasha1_notrain/run_state/run_manifest.json"),
    pathlib.Path("artifacts/xiasha1_notrain/run_state_retry/run_manifest.json"),
]
OUT_DIR = pathlib.Path("data/output_data/analysis/xiasha1_notrain")
ARMS = ("orig_prog", "fixedtime", "sotl", "maxpressure")
ARM_LABEL = {"orig_prog": "orig(128s)", "fixedtime": "fixedtime(30s)",
             "sotl": "sotl", "maxpressure": "maxpressure"}


IN_EDGES = ("E2J", "N2J", "S2J", "W2J")
OUT_EDGES = ("J2E", "J2N", "J2S", "J2W")
IN_LANES = [f"{e}_{i}" for e in IN_EDGES for i in range(3)]
OUT_LANES = [f"{e}_{i}" for e, n in zip(OUT_EDGES, (1, 2, 2, 1))
             for i in range(n)]


def run_orig(pdir):
    """原始程序臂：libsumo 被动跑包（不动 tlLogic），指标采集路径与
    world_sumo 完全一致：
      att      = 到达车 in-net 时长均值（depart->arrive 台账）
      queue    = 每 10s 决策步采样 Σ_12 进口道 lane_waiting_count
                 （车 wait>0 即计：曾停车且未离开观察域，复刻 observe()）
      waiting_s= 终点仍在网车辆 accumulated waiting 均值
      real_delay_s = tripinfo timeLoss 均值（与 ctl real_delay 同为
                 '实际-期望'秒级时延，口径最接近）
    """
    import libsumo as traci
    cfg = next(pathlib.Path(pdir).glob("*.sumocfg"))
    obs_lanes = IN_LANES + OUT_LANES
    with tempfile.TemporaryDirectory() as td:
        stat = pathlib.Path(td) / "stat.xml"
        trip = pathlib.Path(td) / "trip.xml"
        cwd = os.getcwd()
        os.chdir(pdir)
        try:
            traci.start(["sumo", "-c", cfg.name,
                         "--statistic-output", str(stat),
                         "--tripinfo-output", str(trip),
                         "--no-warnings", "true"])
            dep, arr, waiting = {}, {}, {}
            prev_obs, q_sum, q_n = set(), 0.0, 0
            while traci.simulation.getTime() < 3600:
                traci.simulationStep()
                t = traci.simulation.getTime()
                for v in traci.simulation.getDepartedIDList():
                    dep[v] = t
                for v in traci.simulation.getArrivedIDList():
                    arr[v] = t - dep[v]
                obs = set()
                for ln in obs_lanes:
                    obs.update(traci.lane.getLastStepVehicleIDs(ln))
                q_lane = {ln: traci.lane.getLastStepVehicleIDs(ln)
                          for ln in IN_LANES}
                for ln, vs in q_lane.items():
                    for v in vs:
                        if v in waiting:
                            waiting[v] += 1.0
                        elif traci.vehicle.getWaitingTime(v) > 0:
                            waiting[v] = traci.vehicle.getWaitingTime(v)
                for v in prev_obs - obs:
                    waiting.pop(v, None)
                prev_obs = obs
                if int(t + 0.5) % 10 == 0:
                    q_sum += sum(1 for vs in q_lane.values() for v in vs
                                 if waiting.get(v, 0) > 0)
                    q_n += 1
            ids = traci.vehicle.getIDList()
            wait = (sum(traci.vehicle.getAccumulatedWaitingTime(v)
                        for v in ids) / len(ids)) if ids else 0.0
            traci.close()
        finally:
            os.chdir(cwd)
        st = ET.parse(str(stat)).getroot()
        veh = st.find("vehicles").attrib
        loaded = int(veh["loaded"])
        # 台账自洽校验（终点边界可差 ≤1 辆"已加载未注入未计 waiting"）
        resid = loaded - len(dep) - int(veh["waiting"])
        assert 0 <= resid <= 1, (pdir, loaded, len(dep), veh["waiting"])
        assert int(veh["running"]) == len(dep) - len(arr), (
            pdir, veh["running"], len(dep), len(arr))
        losses = [float(x.attrib["timeLoss"])
                  for x in ET.parse(str(trip)).getroot().iter("tripinfo")]
        return {
            "loaded": loaded, "inserted": len(dep),
            "uninserted": loaded - len(dep),
            "running": len(dep) - len(arr),
            "throughput": len(arr),
            "att": sum(arr.values()) / len(arr) if arr else 0.0,
            "waiting_s": wait,
            "queue": q_sum / q_n if q_n else 0.0,
            "real_delay_s": sum(losses) / len(losses) if losses else 0.0,
            "teleports": int(st.find("teleports").attrib["total"]),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    runs = []
    for sp in STATE:
        if sp.is_file():
            mani = json.loads(sp.read_text())
            runs += [t for t in mani["tasks"]
                     if t.get("status") == "SUCCEEDED"]
    assert len(runs) == 36, f"only {len(runs)} done"

    # pack -> arm -> metrics
    matrix = {}
    for t in runs:
        rid = t["run_id"]                      # xi1_<short>_<pack>
        _, short, pack = rid.split("_", 2)
        agent = {"ft": "fixedtime", "sotl": "sotl",
                 "mp": "maxpressure"}[short]
        rec_path = pathlib.Path(t["output_path"]) / "metrics/records.jsonl"
        rec = [json.loads(l) for l in rec_path.read_text().splitlines()
               if '"FINAL_EVALUATION"' in l]
        assert len(rec) == 1, (rid, len(rec))
        r = rec[0]
        matrix.setdefault(pack, {})[agent] = {
            "att": r["travel_time"], "queue": r["queue"],
            "lane_delay_idx": r["delay"], "throughput": r["throughput"],
            "waiting_s": r["waiting_time"],
            "real_delay_s": r.get("real_delay"),
            "unfinished_in_net": r.get("unfinished_vehicles"),
            "run_id": rid}

    for pdir in sorted(glob.glob("data/raw_data/xiasha1_*_1h")):
        pack = pathlib.Path(pdir).name.replace(
            "xiasha1_", "").replace("_1h", "")
        o = run_orig(pdir)
        matrix.setdefault(pack, {})["orig_prog"] = {
            "att": o["att"], "queue": o["queue"],
            "lane_delay_idx": None,
            "waiting_s": o["waiting_s"], "real_delay_s": o["real_delay_s"],
            "throughput": o["throughput"],
            "uninserted": o["uninserted"],
            "unfinished_in_net": o["running"],
            "teleports": o["teleports"], "run_id": "sumo_direct"}
        for arm in ARMS:
            if arm == "orig_prog":
                continue
            m = matrix[pack].get(arm)
            if m is not None and m.get("unfinished_in_net") is not None:
                m["uninserted"] = (o["loaded"] - m["throughput"]
                                   - m["unfinished_in_net"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    packs = sorted(matrix)
    (OUT_DIR / "notrain_matrix.json").write_text(json.dumps(
        {"packs": packs, "arms": list(ARMS), "matrix": matrix},
        ensure_ascii=False, indent=2))
    with open(OUT_DIR / "notrain_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pack", "arm", "att", "queue", "lane_delay_idx",
                    "waiting_s", "real_delay_s", "throughput",
                    "uninserted", "unfinished_in_net", "teleports",
                    "run_id"])
        for p in packs:
            for a in ARMS:
                m = matrix[p].get(a) or {}
                w.writerow([p, a, m.get("att"), m.get("queue"),
                            m.get("lane_delay_idx"), m.get("waiting_s"),
                            m.get("real_delay_s"), m.get("throughput"),
                            m.get("uninserted"), m.get("unfinished_in_net"),
                            m.get("teleports"), m.get("run_id")])

    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import matplotlib.pyplot as plt

    colors = {"orig_prog": "#4c72b0", "fixedtime": "#dd8452",
              "sotl": "#55a868", "maxpressure": "#c44e52"}
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    specs = [("att", "Average travel time (s)"),
             ("queue", "Mean queue (veh)"),
             ("waiting_s", "Waiting of in-net veh, end (s)"),
             ("throughput", "Throughput (veh)")]
    x = np.arange(len(packs)); wdt = 0.2
    for ax, (key, ylab) in zip(axes.flat, specs):
        for i, arm in enumerate(ARMS):
            vals = [matrix[p].get(arm, {}).get(key) for p in packs]
            vals = [np.nan if v is None else v for v in vals]
            ax.bar(x + (i - 1.5) * wdt, vals, wdt, label=ARM_LABEL[arm],
                   color=colors[arm])
        ax.set_xticks(x); ax.set_xticklabels(packs, rotation=45, ha="right",
                                           fontsize=8)
        ax.set_ylabel(ylab); ax.legend(fontsize=8)
    fig.suptitle("xiasha1 demand packs × rule controllers — 1 episode each")
    fig.savefig(OUT_DIR / "fig_notrain_2x2.png", dpi=160)
    plt.close(fig)

    att = np.array([[matrix[p].get(a, {}).get("att", np.nan)
                     for a in ARMS] for p in packs])
    order = np.argsort(-att[:, 0])
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(att[order], cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=9)
    ax.set_yticks(range(len(packs)))
    ax.set_yticklabels([packs[i] for i in order], fontsize=9)
    for i in range(len(packs)):
        for j in range(len(ARMS)):
            v = att[order][i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=8)
    ax.set_title("ATT (s) — pack × controller")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(OUT_DIR / "fig_notrain_heatmap.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    for i, arm in enumerate(ARMS):
        unins = [matrix[p].get(arm, {}).get("uninserted") or 0
                 for p in packs]
        innet = [matrix[p].get(arm, {}).get("unfinished_in_net") or 0
                 for p in packs]
        xc = x + (i - 1.5) * wdt
        ax.bar(xc, unins, wdt, color=colors[arm],
               label=ARM_LABEL[arm] + " uninserted")
        ax.bar(xc, innet, wdt, bottom=unins, color=colors[arm],
               alpha=0.45, label=ARM_LABEL[arm] + " in-net")
    ax.set_xticks(x); ax.set_xticklabels(packs, rotation=45, ha="right",
                                       fontsize=8)
    ax.set_ylabel("Backlog vehicles at end")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("End-of-run backlog: uninserted (blocked at insertion)"
                 " + in-net unfinished")
    fig.savefig(OUT_DIR / "fig_notrain_backlog.png", dpi=160)
    plt.close(fig)
    print(f"matrix+figs -> {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
