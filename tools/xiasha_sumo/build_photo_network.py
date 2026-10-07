"""Build the photo-aligned Xiasha SUMO network.

The source geometry lives in ``data/raw_data/xiasha_photo``.  This script is
deliberately small so the geometry can be edited in XML and regenerated without
touching the historical ``xiasha1*1`` network.  It also writes an audit JSON
with the command and basic geometry facts used by overlay checks.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def _netconvert() -> str:
    explicit = os.environ.get("NETCONVERT")
    if explicit:
        return explicit
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home and (Path(sumo_home) / "bin" / "netconvert").exists():
        return str(Path(sumo_home) / "bin" / "netconvert")
    for candidate in (
        shutil.which("netconvert"),
        "/home/dev/miniforge3/envs/colight/lib/python3.10/site-packages/sumo/bin/netconvert",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("netconvert was not found; set NETCONVERT or SUMO_HOME")


def build(output: Path) -> dict:
    root = Path(__file__).resolve().parents[2]
    source = root / "data" / "raw_data" / "xiasha_photo"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _netconvert(),
        "--node-files", str(source / "xiasha_photo.nod.xml"),
        "--edge-files", str(source / "xiasha_photo.edg.xml"),
        "--connection-files", str(source / "xiasha_photo.con.xml"),
        "--output-file", str(output),
        "--no-turnarounds",
        "--default.junctions.radius", "8",
        "--default.junctions.keep-clear", "true",
        # Pedestrian margins and zebra markings are supplied by the visual
        # additional layer. Keeping them out of the vehicle lane index space
        # makes the east-west through lanes remain collinear.
        # Crossings are drawn in xiasha_photo.visual.add.xml.  Leaving
        # crossings.guess disabled keeps SUMO from adding pedestrian
        # connections to the traffic-light link list used by the vehicle
        # signal converter.
    ]
    proc = subprocess.run(command, cwd=root, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr or proc.stdout)
    try:
        network_label = str(output.relative_to(root))
    except ValueError:
        network_label = str(output)
    audit = {
        "network": network_label,
        "source": {
            "nodes": str((source / "xiasha_photo.nod.xml").relative_to(root)),
            "edges": str((source / "xiasha_photo.edg.xml").relative_to(root)),
            "connections": str((source / "xiasha_photo.con.xml").relative_to(root)),
        },
        "coordinate_system": "local metres; J=(120,120); north=+y; east=+x",
        "geometry_assumption": "aerial-image proportional approximation; not survey-grade georeferencing",
        "arm_length_m": 120.0,
        "junction_box_m": 28.0,
        "vehicle_lanes_per_approach": 3,
        "lane_width_m": 3.2,
        "stop_line_setback_m": 6.0,
        "sidewalk_width_m": 2.2,
        "median_width_m": 3.2,
        "approach_lanes": {"north_south": 3, "east_west": 2},
        "link_indices": list(range(16)),
        "sidewalks_and_crossings": True,
        "netconvert": command,
    }
    audit_path = output.with_name(output.stem + ".build.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[2] / "data/raw_data/xiasha_photo/xiasha_photo.net.xml"
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    audit = build(args.output.resolve())
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
