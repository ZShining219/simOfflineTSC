#!/usr/bin/env python3
"""xiasha1 需求包套件仿真验证与区分度图证（一次性验证件）。

对 data/raw_data/xiasha1_*_1h/ 下每个包执行一次完整 0–3600s SUMO 仿真
（信号为包内复制的 12 相位 128s 固定程序，不介入控制），采集
--summary-output 逐步统计，产出：

  data/output_data/analysis/xiasha1_pack_verify/
      fig_queue_timeseries.png   12 包 halting(排队) 曲线 + 均值柱
      fig_arrival_profile.png    12 包累计到达曲线（放行速率差异）
      verify_metrics.json        每包 loaded/inserted/arrived/waiting/
                                 tele/ebrake/平均排队/最大排队 汇总表

用法：python dev/demand_packs/verify_xiasha_packs.py [--sumo <binary>]
"""
import argparse
import glob
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

SUMO_DEFAULT = ("/home/dev/miniforge3/envs/colight/lib/python3.10/"
                "site-packages/sumo/bin/sumo")
OUT_DIR = pathlib.Path("data/output_data/analysis/xiasha1_pack_verify")


def run_pack(pdir, sumo):
    cfg = next(pathlib.Path(pdir).glob("*.sumocfg"))
    with tempfile.TemporaryDirectory() as td:
        summary = pathlib.Path(td) / "summary.xml"
        stat = pathlib.Path(td) / "stat.xml"
        proc = subprocess.run(
            [sumo, "-c", cfg.name, "--duration-log.disable", "true",
             "--summary-output", str(summary),
             "--statistic-output", str(stat), "--no-warnings", "true"],
            cwd=pdir, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{cfg}: sumo failed: {proc.stderr[-500:]}")
        steps = []
        for ev, el in ET.iterparse(str(summary), events=("end",)):
            if el.tag == "step":
                steps.append(dict(
                    t=float(el.get("time")),
                    halting=float(el.get("halting", 0)),
                    running=float(el.get("running", 0)),
                    arrived=float(el.get("arrived", 0)),
                    waiting=float(el.get("waiting", 0))))
                el.clear()
        st = ET.parse(str(stat)).getroot()
        veh = st.find("vehicles").attrib
        tele = st.find("teleports").attrib
        safety = st.find("safety").attrib
        return steps, dict(
            loaded=int(veh["loaded"]), inserted=int(veh["inserted"]),
            running=int(veh["running"]), waiting=int(veh["waiting"]),
            teleports=int(tele["total"]),
            emergency_braking=int(safety["emergencyBraking"]),
            collisions=int(safety["collisions"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumo", default=SUMO_DEFAULT)
    args = ap.parse_args()
    packs = {}
    for pdir in sorted(glob.glob("data/raw_data/xiasha1_*_1h")):
        name = pathlib.Path(pdir).name.replace("xiasha1_", "").replace(
            "_1h", "")
        steps, totals = run_pack(pdir, args.sumo)
        h = [s["halting"] for s in steps]
        totals["halting_mean"] = sum(h) / len(h)
        totals["halting_max"] = max(h)
        totals["n_pack"] = int(sum(1 for _ in
                                 open(pathlib.Path(pdir, "pack_spec.json"))))
        packs[name] = dict(steps=steps, totals=totals)
        print(f"{name:16s} loaded={totals['loaded']:5d} "
              f"inserted={totals['inserted']:5d} "
              f"wait_end={totals['waiting']:4d} "
              f"q_mean={totals['halting_mean']:6.1f} "
              f"q_max={totals['halting_max']:6.1f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({n: p["totals"] for n, p in packs.items()},
              open(OUT_DIR / "verify_metrics.json", "w"),
              ensure_ascii=False, indent=2)

    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(packs, key=lambda n: -packs[n]["totals"]["halting_mean"])
    cmap = plt.get_cmap("tab20")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)
    for i, n in enumerate(order):
        st = packs[n]["steps"]
        axes[0].plot([s["t"] for s in st], [s["halting"] for s in st],
                     lw=1.0, alpha=0.9, color=cmap(i),
                     label=f"{n} (μ={packs[n]['totals']['halting_mean']:.0f})")
    axes[0].set_title("(a) Halting vehicles vs time — 12 packs, same 128s program")
    axes[0].set_xlabel("sim time (s)"); axes[0].set_ylabel("halting vehicles")
    axes[0].legend(fontsize=7, ncols=2)
    worder = sorted(packs, key=lambda n: -packs[n]["totals"]["waiting"])
    for i, n in enumerate(worder):
        st = packs[n]["steps"]
        axes[1].plot([s["t"] for s in st], [s["waiting"] for s in st],
                     lw=1.0, alpha=0.9, color=cmap(order.index(n)),
                     label=f"{n} (end={packs[n]['totals']['waiting']:.0f})")
    axes[1].set_title("(b) Insertion-blocked vehicles vs time — oversaturation signal")
    axes[1].set_xlabel("sim time (s)"); axes[1].set_ylabel("waiting-to-insert")
    axes[1].legend(fontsize=7, ncols=2)
    axes[2].bar(range(len(order)),
                [packs[n]["totals"]["halting_mean"] for n in order],
                color=[cmap(i) for i in range(len(order))])
    axes[2].set_xticks(range(len(order)))
    axes[2].set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    axes[2].set_title("(c) Time-mean queue (halting) per pack")
    axes[2].set_ylabel("mean halting vehicles")
    fig.savefig(OUT_DIR / "fig_queue_timeseries.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for i, n in enumerate(order):
        st = packs[n]["steps"]
        ax.plot([s["t"] for s in st], [s["arrived"] for s in st],
                lw=1.0, alpha=0.9, color=cmap(i), label=n)
    ax.set_title("Cumulative arrivals — throughput differentiation")
    ax.set_xlabel("sim time (s)"); ax.set_ylabel("arrived vehicles")
    ax.legend(fontsize=7, ncols=2)
    fig.savefig(OUT_DIR / "fig_arrival_profile.png", dpi=160)
    plt.close(fig)
    print(f"figures+metrics -> {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
