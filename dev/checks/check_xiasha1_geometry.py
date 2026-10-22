#!/usr/bin/env python3
"""xiasha1 几何守卫：防 106.4m 车道 + departPos 截断观测问题复发。

检查项：
  1. 16 个 xiasha1 net.xml 的受控进口车道 length == 280.00；
  2. 15 个 xiasha1 rou.xml 无 departPos（车道起点插入）；
  3. world/world_sumo.py 的 max_distance 为动态取值（无裸数字硬编码）。

用法: python dev/checks/check_xiasha1_geometry.py
退出码 0=全过, 1=存在违例。供 v3 冒烟及后续 pack 生成后复跑。
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw_data"
EXPECTED_LEN = "280.00"

NETS = [RAW / "xiasha1*1" / "xiasha1_sumo_signal.net.xml",
        RAW / "xiasha1*1" / "xiasha1_sumo_lane_aligned_base.net.xml"]
ROUS = [RAW / "xiasha1*1" / "xiasha1_sumo_vehicles.rou.xml"]
for p in sorted(RAW.glob("xiasha1_*_1h")):
    name = p.name
    NETS.append(p / f"{name}_signal.net.xml")
    ROUS.append(p / f"{name}.rou.xml")

fails = []
for net in NETS:
    if not net.exists():
        fails.append(f"缺文件 {net.relative_to(ROOT)}")
        continue
    lanes = [e for e in ET.parse(str(net)).getroot().iter("lane")
             if re.match(r"^[NESW]2J_", e.get("id", ""))]
    bad = [e.get("id") for e in lanes
           if e.get("length") != EXPECTED_LEN]
    if bad:
        fails.append(f"{net.relative_to(ROOT)}: 进口车道长度异常 {bad}")

for rou in ROUS:
    if not rou.exists():
        fails.append(f"缺文件 {rou.relative_to(ROOT)}")
        continue
    n = sum(1 for v in ET.parse(str(rou)).getroot().iter("vehicle")
            if v.get("departPos") not in (None, "0", "0.00", "0.0"))
    if n:
        fails.append(f"{rou.relative_to(ROOT)}: {n} 辆车仍带非零 departPos")

world = (ROOT / "world" / "world_sumo.py").read_text(encoding="utf-8")
if re.search(r"self\.max_distance\s*=\s*\d", world):
    fails.append("world_sumo.py: max_distance 仍为硬编码数值")

if fails:
    print("FAIL:")
    for f_ in fails:
        print(" -", f_)
    sys.exit(1)
print(f"OK: {len(NETS)} nets x {EXPECTED_LEN}, {len(ROUS)} rous 起点插入, "
      "max_distance 动态")
