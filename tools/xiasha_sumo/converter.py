from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "data/raw_data/xiasha1*1"
DEFAULT_INPUT = DEFAULT_DIR / "原始数据/语义事件表(13min).csv"
DEFAULT_NET = DEFAULT_DIR / "原始数据/real_scene.net.xml"
DEFAULT_CYCLE = DEFAULT_DIR / "原始数据/红绿灯周期"

PHASES = [("NS直行绿",27,"G"),("NS直行黄",3,"y"),("NS过渡红",2,"r"),
          ("NS左转绿",22,"G"),("NS左转黄",3,"y"),("南北东西全红",5,"r"),
          ("EW直行绿",34,"G"),("EW直行黄",3,"y"),("EW过渡红",2,"r"),
          ("EW左转绿",19,"G"),("EW左转黄",3,"y"),("东西南北全红",5,"r")]

def parse_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    reasons = Counter(); valid = []
    for row in rows:
        missing = [k for k in ("entry_edge", "exit_edge", "entry_time") if not row.get(k, "").strip()]
        if missing:
            reasons["missing_" + "_and_".join(missing)] += 1
            continue
        try:
            t = float(row["entry_time"])
        except ValueError:
            reasons["invalid_entry_time"] += 1; continue
        row["_entry_time"] = t; valid.append(row)
    shift = max(0.0, -min((r["_entry_time"] for r in valid), default=0.0))
    for r in valid: r["_depart"] = r["_entry_time"] + shift
    return rows, valid, shift, reasons

def write_routes(rows, path: Path, flows=False, window=60.0):
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="xiasha_passenger", vClass="passenger", accel="2.6", decel="4.5", sigma="0.5", length="5", minGap="2.5", maxSpeed="13.89")
    if flows:
        groups = defaultdict(list)
        for r in rows:
            begin = math.floor(r["_depart"] / window) * window
            groups[(r["entry_edge"], r["exit_edge"], begin)].append(r)
        for i, ((entry, exit_, begin), group) in enumerate(sorted(groups.items(), key=lambda x: x[0][2])):
            ET.SubElement(root, "flow", id=f"flow_{i:04d}", type="xiasha_passenger", route=f"route_{i:04d}", begin=f"{begin:.2f}", end=f"{begin+window:.2f}", number=str(len(group)))
            ET.SubElement(root, "route", id=f"route_{i:04d}", edges=f"{entry} {exit_}")
    else:
        for i, r in enumerate(sorted(rows, key=lambda x: x["_depart"])):
            route = f"{r['entry_edge']} {r['exit_edge']}"
            ET.SubElement(root, "vehicle", id=r.get("vehicle_id") or f"vehicle_{i:04d}", type="xiasha_passenger", route=f"route_{i:04d}", depart=f"{r['_depart']:.2f}")
            ET.SubElement(root, "route", id=f"route_{i:04d}", edges=route)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

def direction(entry, exit_):
    if (entry, exit_) in {("N2J","J2W"),("S2J","J2E"),("E2J","J2N"),("W2J","J2S")}: return "right"
    if entry[0] == exit_[-1]: return "straight"
    return "left"

def write_signal_network(net_path: Path, out_path: Path, add_path: Path):
    tree = ET.parse(net_path); root = tree.getroot(); controlled=[]
    junction = next((n for n in root.findall("junction") if n.get("id") == "J"), None)
    if junction is None:
        raise ValueError("network does not contain central junction J")
    junction.set("type", "traffic_light")
    for c in root.findall("connection"):
        frm = c.get("from", ""); to = c.get("to", "")
        if frm in {"N2J","S2J","E2J","W2J"} and to.startswith("J2"):
            c.set("tl", "J"); c.set("linkIndex", str(len(controlled))); controlled.append(c)
    # Keep right turns green and control straight/left by approach group.
    states=[]
    for name, duration, mark in PHASES:
        chars=[]
        for c in controlled:
            turn=direction(c.get("from",""), c.get("to","")); approach=c.get("from","")[0]
            ns = approach in {"N","S"}; target = ("NS" if ns else "EW")
            is_left = turn == "left"; is_straight = turn == "straight"
            active = (("直行" in name and is_straight and ((target=="NS") == name.startswith("NS"))) or
                      ("左转" in name and is_left and ((target=="NS") == name.startswith("NS"))))
            chars.append("G" if turn == "right" else (mark if active else "r"))
        states.append("".join(chars))
    add = ET.Element("additional"); tl=ET.SubElement(add, "tlLogic", id="J", type="static", programID="xiasha_cycle", offset="0")
    for (name,duration,_), state in zip(PHASES, states): ET.SubElement(tl, "phase", duration=str(duration), state=state, name=name)
    ET.ElementTree(add).write(add_path, encoding="utf-8", xml_declaration=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return len(controlled), states

def write_cfg(path, net, routes, add):
    root=ET.Element("configuration"); inp=ET.SubElement(root,"input"); ET.SubElement(inp,"net-file",value=net.name); ET.SubElement(inp,"route-files",value=routes.name); ET.SubElement(inp,"additional-files",value=add.name); ET.SubElement(root,"time"); ET.ElementTree(root).write(path,encoding="utf-8",xml_declaration=True)

def main(argv=None):
    p=argparse.ArgumentParser(description="Convert xiasha semantic events to SUMO vehicles, flows and signal files")
    p.add_argument("--mode",choices=["all","vehicles","flows"],default="all"); p.add_argument("--all",action="store_true",help="generate all artifacts"); p.add_argument("--input",type=Path,default=DEFAULT_INPUT); p.add_argument("--output-dir",type=Path,default=DEFAULT_DIR); p.add_argument("--network",type=Path,default=DEFAULT_NET); p.add_argument("--window",type=float,default=60.0); p.add_argument("--shift",type=float,default=None)
    a=p.parse_args(argv); mode="all" if a.all else a.mode; out=a.output_dir; out.mkdir(parents=True,exist_ok=True); raw,valid,shift,reasons=parse_rows(a.input); shift=a.shift if a.shift is not None else shift
    for r in valid: r["_depart"]=r["_entry_time"]+shift
    veh=out/"xiasha1_sumo_vehicles.rou.xml"; flows=out/"xiasha1_sumo_flows.rou.xml"; net=out/"xiasha1_sumo_signal.net.xml"; add=out/"xiasha1_sumo_signal.add.xml"
    if mode in {"all","vehicles"}: write_routes(valid,veh)
    if mode in {"all","flows"}: write_routes(valid,flows,True,a.window)
    controlled=0
    if mode=="all": controlled,_=write_signal_network(a.network,net,add); write_cfg(out/"xiasha1_sumo_vehicles.sumocfg",net,veh,add); write_cfg(out/"xiasha1_sumo_flows.sumocfg",net,flows,add)
    report={"input":str(a.input),"total_records":len(raw),"valid_records":len(valid),"skipped_records":len(raw)-len(valid),"skip_reasons":dict(reasons),"shift_seconds":shift,"flow_window_seconds":a.window,"controlled_connections":controlled,"phase_durations_total":sum(x[1] for x in PHASES),"exit_time_usage":"statistics only; not used for departure"}
    (out/"xiasha1_sumo_conversion_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
