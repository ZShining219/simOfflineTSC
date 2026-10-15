#!/usr/bin/env python3
"""在杭州 1×1（bc-tyc 基准）路网结构上批量生成语义化流量需求包。

设计约束（实测得出）：
- 路网每臂仅 2 车道、2 个流向：lane0=直行、lane1=左转；net.xml 无右转
  连接（dir 集合只有 s/l/t）。因此"高右转"类需求在该拓扑上物理不可表达，
  本批不生成右转包；若需要须先重建 roadnet（netconvert + 配相）。
- 基线需求取自 hangzhou_1x1_bc-tyc_18041610_1h（sumohz1x1）：3600s、
  2021 veh、四臂、每臂 straight+left 两条 route、左转占比 ~15%。

分类（4 大类 15 包，命名 sumohz1x1_<cat>_<sub>）：
  sparse  稀疏强度:   low(×0.40) ultra(×0.18)
  dense   高流量:     high(×1.40) extreme(×1.80, 近饱和压力场景)
  lt      左转结构:   high(全臂35%) extreme(全臂55%) wsing(仅西臂55%)
  st      直行主导:   dom(左转压至5%)
  asym    方向不对称: we(东西×1.9/南北×0.35) ns(反向) wonly(西臂×3其余×0.5)
  fluct   时变波动:   sin(±45%,T=600s) burst(1800-2400×2.5)
                      ramp(0.5→2.0线性爬坡) recede(前半×1.6后半退潮)
                      —— 全部归一化到基线总流量，只变时序结构

生成物（每包一个 raw 目录 + 一个 sim cfg）：
  data/raw_data/hangzhou_1x1_<cat>_<sub>_1h/
      <name>.net.xml + net_or/nod/edg/con/tll/roadnet.json/
      signal_plan_template.txt（全部复制基准包，文件名不改内部逻辑）
      <name>.rou.xml（新生成）flow.json（同步生成）<name>.sumocfg（路径改写）
      pack_spec.json（参数与意图自述）
  configs/sim/sumohz1x1_<cat>_<sub>.cfg

用法：python dev/demand_packs/build_hz1x1_demands.py [--only name ...]
"""
import argparse
import json
import pathlib
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

BASE_DIR = pathlib.Path(
    "data/raw_data/hangzhou_1x1_bc-tyc_18041610_1h")
BASE = "hangzhou_1x1_bc-tyc_18041610_1h"
HORIZON = 3600
# 臂 → (直行 route, 左转 route)；值为 (from_edge, to_edge)
ARMS = {
    "W": (("road_0_1_0", "road_1_1_0"), ("road_0_1_0", "road_1_1_1")),
    "S": (("road_1_0_1", "road_1_1_1"), ("road_1_0_1", "road_1_1_2")),
    "E": (("road_2_1_2", "road_1_1_2"), ("road_2_1_2", "road_1_1_3")),
    "N": (("road_1_2_3", "road_1_1_3"), ("road_1_2_3", "road_1_1_0")),
}
# 基准每臂流入 veh/h（flow.json 实测）
ARM_BASE = {"W": 586, "S": 566, "E": 427, "N": 442}
LEFT_BASE = {"W": 88 / 586, "S": 83 / 566, "E": 67 / 427, "N": 64 / 442}
WE = ("W", "E")
NS = ("S", "N")


def _scale_all(k):
    return dict(arm={a: k for a in ARMS})


def _set_left(share_map):
    return dict(left_share={a: s for a, s in share_map.items()})


def _arms(mod):
    return dict(arm={a: m for a, m in mod.items()})


def _shape(fn, note):
    return dict(shape=fn, shape_note=note)


PACKS = {
    # ---- A 稀疏强度 ----
    "sparse_low": dict(cat="sparse", scale=0.40,
                       note="总流入×0.40（~800veh/h），低密度常态"),
    "sparse_ultra": dict(cat="sparse", scale=0.18,
                        note="总流入×0.18（~360veh/h），近空载边界"),
    # ---- B 高流量 ----
    "dense_high": dict(cat="dense", scale=1.40,
                       note="总流入×1.40（~2800veh/h），逼近通行能力"),
    "dense_extreme": dict(cat="dense", scale=1.80,
                         note="总流入×1.80（~3640veh/h）压力场景，预期排队累积"),
    # ---- C 左转结构（总流入不变，仅改直/左分配）----
    "lt_high": dict(cat="lt", left=0.35,
                   note="全臂左转占比 15%→35%，转向冲突加重"),
    "lt_extreme": dict(cat="lt", left=0.55,
                      note="全臂左转占比→55%，超高左转"),
    "lt_wsing": dict(cat="lt", left=0.55, left_arms=["W"],
                    note="仅西臂左转→55%（单臂转向集中），其余臂不变"),
    "st_dom": dict(cat="st", left=0.05,
                  note="直行主导：全臂左转压至 5%"),
    # ---- D 方向不对称（总流入不变）----
    "asym_we": dict(cat="asym", arm={"W": 1.9, "E": 1.9, "S": 0.35, "N": 0.35},
                   note="东西向重载潮汐（W+E×1.9，S+N×0.35）"),
    "asym_ns": dict(cat="asym", arm={"W": 0.35, "E": 0.35, "S": 1.9, "N": 1.9},
                   note="南北向重载潮汐（与 we 对称镜像）"),
    "asym_wonly": dict(cat="asym", arm={"W": 3.0, "E": 0.5, "S": 0.5, "N": 0.5},
                      note="单臂集中：西臂×3.0，其余×0.5"),
    # ---- E 时变波动（总量≈基线，λ(t) 结构不同）----
    "fluct_sin": dict(cat="fluct", shape=lambda t: 1 + 0.45 * np.sin(
        2 * np.pi * t / 600), shape_note="正弦±45%周期600s",
        note="周期波动需求，考验相位切换节奏跟踪"),
    "fluct_burst": dict(cat="fluct",
                       shape=lambda t: np.where(
                           (t >= 1800) & (t < 2400), 2.5, 0.7),
                       shape_note="1800–2400s 脉冲×2.5，其余×0.7",
                       note="中段突发尖峰（事件/事故后集聚）"),
    "fluct_ramp": dict(cat="fluct",
                      shape=lambda t: 0.4 + 1.2 * t / HORIZON,
                      shape_note="线性爬坡 ×0.4→×1.6",
                      note="单调爬坡型需求增长（早高峰形态）"),
    "fluct_recede": dict(cat="fluct",
                        shape=lambda t: np.where(t < 1800, 1.6, 0.4),
                        shape_note="前半×1.6后半×0.4退潮",
                        note="先高后塌（晚高峰退潮/活动散场）"),
}


def build_lambda(spec):
    """返回 per-route λ(t) 函数组 {route_pair: callable(t)->veh/s}。"""
    scale = spec.get("scale", 1.0)
    left = spec.get("left")  # 全臂或单臂左转占比
    left_arms = spec.get("left_arms", list(ARMS))
    arm = spec.get("arm", {a: 1.0 for a in ARMS})
    shape = spec.get("shape", lambda t: 1.0)

    rates = {}
    for a, (st, lt) in ARMS.items():
        share = left if (left is not None and a in left_arms) else LEFT_BASE[a]
        arm_rate = ARM_BASE[a] / HORIZON * arm.get(a, 1.0) * scale
        rates[st] = arm_rate * (1 - share)
        rates[lt] = arm_rate * share
    return {r: (lambda t, base=b, s=shape: base * float(s(t)))
            for r, b in rates.items()}


def gen_route(rates, seed):
    """逐 route 用 NHPP thinning 产生出发时刻。"""
    rng = np.random.default_rng(seed)
    out = []
    for (a, b), rate_fn in rates.items():
        lam_max = max(rate_fn(t) for t in np.linspace(0, HORIZON, 721))
        t = 0.0
        while True:
            t += rng.exponential(1.0 / lam_max) if lam_max > 0 else 1e9
            if t >= HORIZON:
                break
            if rng.random() <= rate_fn(t) / lam_max:
                out.append((float(t), (a, b)))
    out.sort()
    return out


def emit_pack(name, spec, vehicles, vtype_xml, base_dir):
    pdir = pathlib.Path("data/raw_data") / f"hangzhou_1x1_{name}_1h"
    pdir.mkdir(parents=True, exist_ok=True)
    pname = pdir.name
    # 复制路网侧文件（前缀改包名；roadnet.json/signal_plan_template.txt 名不变）
    for suf in (".net.xml", ".net_or.xml", ".nod.xml", ".edg.xml",
                ".con.xml", ".tll.xml"):
        shutil.copyfile(base_dir / (BASE + suf), pdir / (pname + suf))
    for plain in ("roadnet.json", "signal_plan_template.txt"):
        shutil.copyfile(base_dir / plain, pdir / plain)
    # rou.xml
    rou = [f'\t{vtype_xml}']
    for i, (dep, (a, b)) in enumerate(vehicles):
        rou.append(
            f'\t\t<vehicle depart="{dep:.1f}" id="{i}">\n'
            f'\t\t\t<route edges="{a} {b}"/>\n\t\t</vehicle>')
    (pdir / f"{pname}.rou.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '\t<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n'
        + "\n".join(rou) + "\n\t</routes>\n")
    # flow.json（与 rou 镜像，参数同基准）
    base_flow = json.load(open(base_dir / "flow.json"))
    vparams = base_flow[0]["vehicle"]
    fj = [{"vehicle": vparams, "route": [a, b], "interval": 5,
           "startTime": int(dep), "endTime": int(dep)}
          for dep, (a, b) in vehicles]
    (pdir / "flow.json").write_text(json.dumps(fj, indent=1))
    # sumocfg（路径指向本包文件）
    (pdir / f"{pname}.sumocfg").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<configuration>\n\t<input>\n'
        f'\t\t<net-file value="{pname}.net.xml"/>\n'
        f'\t\t<route-files value="{pname}.rou.xml"/>\n\t</input>\n\t<time>\n'
        '\t\t<begin value="0"/>\n\t\t<end value="3600"/>\n\t</time>\n'
        '</configuration>\n')
    # pack_spec.json
    spec_out = dict(spec)
    spec_out.pop("shape", None)
    spec_out["pack"] = name
    spec_out["n_vehicles"] = len(vehicles)
    spec_out["seed"] = vehicles_seed(name)
    spec_out["base"] = BASE
    spec_out["horizon_s"] = HORIZON
    (pdir / "pack_spec.json").write_text(
        json.dumps(spec_out, ensure_ascii=False, indent=2))
    # cfg
    cfg_name = f"sumohz1x1_{name}"
    cfg = {
        "network": "hz1x1", "interval": 1.0, "seed": 0, "dir": "data/",
        "combined_file": f"raw_data/{pname}/{pname}.sumocfg",
        "roadnetFile": f"raw_data/{pname}/{pname}.net.xml",
        "flowFile": f"raw_data/{pname}/{pname}.rou.xml",
        "convertroadnetFile": f"raw_data/{pname}/roadnet.json",
        "convertflowFile": f"raw_data/{pname}/flow.json",
        "no_warning": True, "name": "debug", "yellow_length": 5,
        "gui": False,
    }
    pathlib.Path("configs/sim").mkdir(exist_ok=True)
    (pathlib.Path("configs/sim") / f"{cfg_name}.cfg").write_text(
        json.dumps(cfg, indent=2) + "\n")
    return pdir, cfg_name


def vehicles_seed(name):
    import hashlib
    return int(hashlib.md5(("hz1x1dem:" + name).encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    root = pathlib.Path(BASE_DIR)
    vtype = None
    tree = ET.parse(root / f"{BASE}.rou.xml")
    vtype_el = tree.getroot().find("vType")
    vtype = ET.tostring(vtype_el, encoding="unicode").strip()
    names = args.only or list(PACKS)
    summary = []
    for name in names:
        spec = PACKS[name]
        rates = build_lambda(spec)
        veh = gen_route(rates, vehicles_seed(name))
        pdir, cfg = emit_pack(name, spec, veh, vtype, root)
        summary.append((name, spec["cat"], len(veh)))
        print(f"{name:16s} cat={spec['cat']:6s} n={len(veh):5d} -> {pdir}  cfg={cfg}")

    if args.only is not None:
        print("\n(--only 模式，跳过 collection manifest 重写)")
        return
    manifest = {
        "schema_version": 1,
        "collection": "hz1x1_demand_packs_v1",
        "base_pack": BASE,
        "topology": "hz1x1（每臂 lane0=直行/lane1=左转，无右转连接）",
        "caveat": "高右转需求在该拓扑物理不可表达，本批不含右转包",
        "horizon_s": HORIZON,
        "packs": [
            {"pack": n, "dir": f"hangzhou_1x1_{n}_1h",
             "cfg": f"sumohz1x1_{n}", "cat": c, "n_vehicles": v}
            for n, c, v in summary
        ],
    }
    mpath = pathlib.Path("data/raw_data/hangzhou_1x1_demand_packs.json")
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"manifest -> {mpath}")
    print("\n" + json.dumps([dict(pack=n, cat=c, n=v) for n, c, v in summary],
                            indent=1))


if __name__ == "__main__":
    sys.exit(main())
