#!/usr/bin/env python3
"""在下沙 1×1（xiasha1*1 实测基准）路网结构上批量生成语义化流量需求包。

设计约束（实测得出）：
- 中心路口 J + N/S/E/W 四边界节点；进口每臂 3 车道：lane0=右+直共用、
  lane1=直行、lane2=左转；出口 J2N/J2S 各 2 车道、J2E/J2W 各 1 车道。
- 受控转向仅 r/s/l（无 't' 连接，观测的 5 条 J 处掉头记录不可路由，
  本批不含掉头包；如需须先改 net 加 dir="t" 连接并重配相信号）。
- 信号 tlLogic J：12 相位 128s 周期（NS 直 27+3+2、NS 左 22+3、全红 5、
  EW 直 34+3+2、EW 左 19+3、全红 5），右转恒绿。
- 基准需求取自 xiasha1_sumo_vehicles.rou.xml（实测语义事件表转换）：
  350 veh / 约 867s 观测窗（等效 ~1450 veh/h），12 条 OD，NS 主导 ~61%。

分类（4 大类 12 包，目录命名 xiasha1_<sub>_1h）：
  sparse  稀疏强度:   low(×0.5)
  dense   高流量:     high(×1.5) extreme(×2.0, 过饱和压力)
  turn    转向结构:   rt_high(全臂右→35%) lt_high(全臂左→35%) st_dom(直→90%)
  asym    方向不对称: ns(N,S×1.8/E,W×0.4) we(镜像) nonly(N×3.0 其余×0.5, 不归一)
  fluct   时变波动:   sin(±45%,T=600s) burst(1800-2400×2.5)
                      ramp(0.4→1.6 线性爬坡) —— 全部归一化到基线总流量

生成物（每包一个 raw 目录）：
  data/raw_data/xiasha1_<sub>_1h/
      xiasha1_<sub>_1h_signal.net.xml + _signal.add.xml 复制基准；
      _logic_lines.add.xml 剔除基准里 file= 指向仓内绝对路径的
      instantInductionLoop 检测器（纯输出仪表，防每次仿真覆写仓内文件）
      xiasha1_<sub>_1h.rou.xml（新生成，逐车 + departLane 语义分配）
      xiasha1_<sub>_1h.sumocfg（路径改写，0–3600s）
      pack_spec.json（参数与意图自述）
  data/raw_data/xiasha1_demand_packs.json（集合清单）

用法：python dev/demand_packs/build_xiasha_demands.py [--only name ...]
"""
import argparse
import json
import pathlib
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

BASE_DIR = pathlib.Path("data/raw_data/xiasha1*1")
BASE_NET = "xiasha1_sumo_signal.net.xml"
BASE_ROU = "xiasha1_sumo_vehicles.rou.xml"
BASE_LOGIC = "xiasha1_sumo_logic_lines.add.xml"
BASE_SIGNAL_ADD = "xiasha1_sumo_signal.add.xml"
HORIZON = 3600
# 进口臂 → 出口边 → 转向（受控连接实测映射，无 u_turn）
MOVES = {
    "E2J": {"J2N": "r", "J2W": "s", "J2S": "l"},
    "N2J": {"J2W": "r", "J2S": "s", "J2E": "l"},
    "S2J": {"J2E": "r", "J2N": "s", "J2W": "l"},
    "W2J": {"J2S": "r", "J2E": "s", "J2N": "l"},
}
ARMS = tuple(MOVES)
# 基准每 OD 观测车数（demand_audit 守恒口径）
OD_BASE = {
    "E2J": {"J2N": 33, "J2S": 11, "J2W": 58},
    "N2J": {"J2E": 23, "J2S": 91, "J2W": 5},
    "S2J": {"J2E": 18, "J2N": 71, "J2W": 5},
    "W2J": {"J2E": 27, "J2N": 5, "J2S": 3},
}

PACKS = {
    # ---- A 稀疏强度 ----
    "sparse_low": dict(cat="sparse", scale=0.5,
                       note="总流入×0.50（~730veh/h），低密度常态"),
    # ---- B 高流量 ----
    "dense_high": dict(cat="dense", scale=1.5,
                       note="总流入×1.50（~2180veh/h），逼近通行能力"),
    "dense_extreme": dict(cat="dense", scale=2.0,
                          note="总流入×2.00（~2900veh/h）过饱和压力，预期排队累积"),
    # ---- C 转向结构（总流入不变，仅改 r/s/l 分配）----
    "rt_high": dict(cat="turn", right=0.35,
                    note="全臂右转占比→35%（xiasha 特有维度），右直共用 lane0 排队竞争"),
    "lt_high": dict(cat="turn", left=0.35,
                    note="全臂左转占比→35%，lane2 独占有相位承压"),
    "st_dom": dict(cat="turn", straight=0.90,
                   note="直行主导：全臂直行→90%，右/左≈5%"),
    # ---- D 方向不对称 ----
    "asym_ns": dict(cat="asym", arm={"N2J": 1.8, "S2J": 1.8, "E2J": 0.4, "W2J": 0.4},
                    note="南北重载潮汐，撞上 NS 短直绿 27s 的信号-需求失配"),
    "asym_we": dict(cat="asym", arm={"E2J": 1.8, "W2J": 1.8, "N2J": 0.4, "S2J": 0.4},
                    note="东西重载镜像：EW 长直绿 34s + J2E/J2W 单车道出口瓶颈"),
    "asym_nonly": dict(cat="asym", arm={"N2J": 3.0, "S2J": 0.5, "E2J": 0.5, "W2J": 0.5},
                       normalize=False,
                       note="单臂集中：N 臂×3.0 其余×0.5，总量不归一（~1.35×基线）"),
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
    # ---- F 综合难度（多轴复合，scale×arm×转向×shape 连乘）----
    # 基线臂计数 N119/S94/E102/W35（合350/867s窗）；参数挑选目标：
    #   均值总流 ~2400-2600veh/h（对标 dense_high~extreme / 论文高方差~2435），
    #   北臂占比 ~58%（失衡撞 NS 短直绿 27s），转向 s55/l20/r25（论文均衡档）。
    "compound_a": dict(
        cat="compound", scale=1.8,
        arm={"N2J": 1.6, "S2J": 0.6, "E2J": 0.6, "W2J": 0.6},
        normalize=False, left=0.20, right=0.25,
        shape=lambda t: np.where((t >= 1800) & (t < 2400), 2.5, 0.7),
        shape_note="全臂 1800–2400s ×2.5，其余 ×0.7（均值恰=1.0）",
        note="饱和量级(~2460veh/h)+北臂失衡(~58%)+均衡转向+中段全局突发"
             "（瞬时~6150veh/h，事件/事故后全域集聚）"),
    "compound_b": dict(
        cat="compound", scale=1.8,
        arm={"N2J": 1.6, "S2J": 0.6, "E2J": 0.6, "W2J": 0.6},
        normalize=False, left=0.20, right=0.25,
        shape=lambda t, e0: (2.5 if 1800 <= t < 2400 else 0.7)
                            if e0 == "N2J"
                            else (0.5 if 1800 <= t < 2400 else 0.7),
        shape_per_arm=True,
        shape_note="N臂 1800–2400s ×2.5、轻臂同窗×0.5，其余时段×0.7"
                   "（N均值=1.0、轻臂0.667）",
        note="同量级(~2320veh/h)但突发只打北臂：窗口内 N 瞬时~3560veh/h "
             "撞短直绿，考相位权重在线切换；窗口外 N~1000veh/h"),
}


def load_base():
    """从基准 rou.xml 提取 vType、观测窗、车道/位置先验。"""
    tree = ET.parse(BASE_DIR / BASE_ROU)
    root = tree.getroot()
    vtype = ET.tostring(root.find("vType"), encoding="unicode").strip()
    routes = {rt.get("id"): rt.get("edges").split()
              for rt in root.findall("route")}
    deps, dep_pos, arr_pos, st_lane0 = [], {}, {}, {}
    pos_cnt, arr_cnt = {a: Counter() for a in ARMS}, {}
    lane_cnt = {a: Counter() for a in ARMS}
    for v in root.findall("vehicle"):
        e0, e1 = routes[v.get("route")]
        deps.append(float(v.get("depart")))
        pos_cnt[e0][v.get("departPos")] += 1
        arr_cnt.setdefault((e0, e1), Counter())[v.get("arrivalPos")] += 1
        if MOVES[e0][e1] == "s":
            lane_cnt[e0][v.get("departLane")] += 1
    for a in ARMS:
        dep_pos[a] = pos_cnt[a].most_common(1)[0][0]
        st_lane0[a] = lane_cnt[a]["0"] / max(1, sum(lane_cnt[a].values()))
    arr_exit = {}
    for (e0, e1), c in arr_cnt.items():
        arr_exit.setdefault(e1, Counter()).update(c)
    arr_pos = {e: c.most_common(1)[0][0] for e, c in arr_exit.items()}
    span = max(deps) - min(deps)
    return vtype, span, dep_pos, arr_pos, st_lane0


def build_lambda(spec, obs_span):
    """返回 per-OD λ(t) 函数组 {(entry,exit): callable(t)->veh/s}。"""
    scale = spec.get("scale", 1.0)
    right, left = spec.get("right"), spec.get("left")
    straight = spec.get("straight")
    arm = spec.get("arm", {a: 1.0 for a in ARMS})
    shape_raw = spec.get("shape", lambda t: 1.0)
    normalize = spec.get("normalize", True)
    # 标量 shape(t)：全 OD 统一乘子；per-arm shape(t, e0)：突发可只打
    # 指定进口臂（shape_per_arm=True，与 normalize 互斥——逐臂归一化
    # 会抹掉刻意设计的臂间强度差）
    if spec.get("shape_per_arm"):
        if normalize:
            raise ValueError("shape_per_arm 与 normalize 不兼容")
        shape = shape_raw
    else:
        shape = lambda t, e0: shape_raw(t)

    # shape 归一化：∫shape dt / HORIZON == 1 → 总期望车数≈基线×scale
    if normalize:
        grid = np.linspace(0, HORIZON, 721)
        shape_norm = HORIZON / np.trapz(
            np.array([float(shape(t, None)) for t in grid]), grid)
    else:
        shape_norm = 1.0

    rates = {}
    for e0, outs in OD_BASE.items():
        n_arm = sum(outs.values()) * arm.get(e0, 1.0) * scale
        shares = {}
        for e1, n in outs.items():
            shares[e1] = n / sum(outs.values())
        if straight is not None:
            rest = (1 - straight) / 2
            shares = {e1: {"r": rest, "l": rest}.get(MOVES[e0][e1], straight)
                      for e1 in outs}
        else:
            if right is not None or left is not None:
                tgt_r = right if right is not None else sum(
                    n for e1, n in outs.items()
                    if MOVES[e0][e1] == "r") / sum(outs.values())
                tgt_l = left if left is not None else sum(
                    n for e1, n in outs.items()
                    if MOVES[e0][e1] == "l") / sum(outs.values())
                tgt_s = max(0.0, 1 - tgt_r - tgt_l)
                shares = {e1: {"r": tgt_r, "l": tgt_l}.get(
                    MOVES[e0][e1], tgt_s) for e1 in outs}
        for e1 in outs:
            rates[(e0, e1)] = n_arm / obs_span * shares[e1]
    return {r: (lambda t, base=b, s=shape, k=shape_norm, a=r[0]:
                base * float(s(t, a)) * k) for r, b in rates.items()}


def gen_route(rates, seed):
    """逐 OD 用 NHPP thinning 产生出发时刻。"""
    rng = np.random.default_rng(seed)
    out = []
    for od, rate_fn in rates.items():
        lam_max = max(rate_fn(t) for t in np.linspace(0, HORIZON, 721))
        t = 0.0
        while True:
            t += rng.exponential(1.0 / lam_max) if lam_max > 0 else 1e9
            if t >= HORIZON:
                break
            if rng.random() <= rate_fn(t) / lam_max:
                out.append((float(t), od))
    out.sort()
    return out


def assign_lane(e0, e1, rng, st_lane0):
    m = MOVES[e0][e1]
    if m == "l":
        return "2"
    if m == "r":
        return "0"
    return "0" if rng.random() < st_lane0[e0] else "1"


def emit_pack(name, spec, vehicles, meta):
    vtype_xml, dep_pos, arr_pos, st_lane0 = meta
    pdir = pathlib.Path("data/raw_data") / f"xiasha1_{name}_1h"
    pdir.mkdir(parents=True, exist_ok=True)
    pname = pdir.name
    # 复制路网侧文件（前缀改包名）
    shutil.copyfile(BASE_DIR / BASE_NET, pdir / f"{pname}_signal.net.xml")
    shutil.copyfile(BASE_DIR / BASE_SIGNAL_ADD, pdir / f"{pname}_signal.add.xml")
    # 基准 logic_lines.add.xml 里的 instantInductionLoop file= 指向仓内
    # 绝对路径（xiasha1*1/xiasha1_sumo_logic_line_events.xml），每跑一次
    # 仿真都会覆写该仓内文件——剔除检测器（纯输出仪表，不影响动力学）
    (pdir / f"{pname}_logic_lines.add.xml").write_text(
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<!-- instantInductionLoop detectors stripped: baseline file= attrs "
        "pointed at an absolute in-repo path; output-only instrumentation, "
        "no effect on dynamics. -->\n<additional/>\n")
    # rou.xml：逐车 + departLane 语义分配（左→2 右→0 直→0/1 按臂实测比）
    rng = np.random.default_rng(vehicles_seed(name) ^ 0x5EED)
    rou = [f"\t{vtype_xml}"]
    for i, (dep, (e0, e1)) in enumerate(vehicles):
        lane = assign_lane(e0, e1, rng, st_lane0)
        rou.append(
            f'\t\t<vehicle depart="{dep:.2f}" departLane="{lane}" '
            f'departPos="{dep_pos[e0]}" departSpeed="10.00" '
            f'arrivalPos="{arr_pos[e1]}" id="{i}" type="xiasha_passenger">\n'
            f"\t\t\t<route edges=\"{e0} {e1}\"/>\n\t\t</vehicle>")
    (pdir / f"{pname}.rou.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '\t<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n'
        + "\n".join(rou) + "\n\t</routes>\n")
    # sumocfg（路径指向本包文件）
    (pdir / f"{pname}.sumocfg").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<configuration>\n\t<input>\n'
        f'\t\t<net-file value="{pname}_signal.net.xml"/>\n'
        f'\t\t<route-files value="{pname}.rou.xml"/>\n'
        f'\t\t<additional-files value="{pname}_logic_lines.add.xml"/>\n'
        "\t</input>\n\t<time>\n\t\t<begin value=\"0\"/>\n"
        '\t\t<end value="3600"/>\n\t</time>\n</configuration>\n')
    # pack_spec.json
    spec_out = dict(spec)
    spec_out.pop("shape", None)
    spec_out["pack"] = name
    spec_out["n_vehicles"] = len(vehicles)
    spec_out["seed"] = vehicles_seed(name)
    spec_out["base"] = f"{BASE_DIR}/{BASE_ROU}"
    spec_out["horizon_s"] = HORIZON
    (pdir / "pack_spec.json").write_text(
        json.dumps(spec_out, ensure_ascii=False, indent=2))
    return pdir


def vehicles_seed(name):
    import hashlib
    return int(hashlib.md5(("xiasha1dem:" + name).encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    vtype, span, dep_pos, arr_pos, st_lane0 = load_base()
    print(f"base span={span:.1f}s dep_pos={dep_pos} arr_pos={arr_pos} "
          f"st_lane0={ {a: round(p, 3) for a, p in st_lane0.items()} }")
    names = args.only or list(PACKS)
    summary = []
    for name in names:
        spec = PACKS[name]
        rates = build_lambda(spec, span)
        veh = gen_route(rates, vehicles_seed(name))
        pdir = emit_pack(name, spec, veh,
                         (vtype, dep_pos, arr_pos, st_lane0))
        summary.append((name, spec["cat"], len(veh), spec.get("note", "")))
        print(f"{name:16s} cat={spec['cat']:6s} n={len(veh):5d} -> {pdir}")

    if args.only is not None:
        print("\n(--only 模式，跳过 collection manifest 重写)")
        return
    manifest = {
        "schema_version": 1,
        "collection": "xiasha1_demand_packs_v1",
        "base_pack": f"{BASE_DIR}/{BASE_ROU}",
        "topology": "xiasha1*1（进口 lane0=右+直共用/lane1=直/lane2=左；"
                    "J2N,J2S 出口 2 车道、J2E,J2W 出口 1 车道；无 u_turn 连接）",
        "caveat": "J 处掉头在受控网物理不可表达（观测 5 条 u_turn 记录仍丢弃），"
                  "本批不含掉头包；出口拓扑不做改造",
        "horizon_s": HORIZON,
        "packs": [
            {"pack": n, "dir": f"xiasha1_{n}_1h",
             "sumocfg": f"xiasha1_{n}_1h/xiasha1_{n}_1h.sumocfg",
             "cat": c, "n_vehicles": v, "note": nt}
            for n, c, v, nt in summary
        ],
    }
    mpath = pathlib.Path("data/raw_data/xiasha1_demand_packs.json")
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"manifest -> {mpath}")


if __name__ == "__main__":
    sys.exit(main())
