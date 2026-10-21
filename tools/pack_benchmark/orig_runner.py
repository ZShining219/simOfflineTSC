"""Direct native-program (orig) arm: passive SUMO run with raw outputs retained."""
import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from .spec import REPO_ROOT


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def discover_sumo_binary(explicit=None):
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"SUMO binary does not exist: {path}")
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / "sumo"
        if candidate.is_file():
            return candidate.resolve()
    try:
        import sumo  # noqa: F401
        candidate = Path(sumo.__file__).resolve().parent / "bin" / "sumo"
        if candidate.is_file():
            return candidate
    except Exception:
        pass
    candidate = shutil.which("sumo")
    if candidate:
        return Path(candidate).resolve()
    raise FileNotFoundError(
        "Cannot find the SUMO binary. Pass --sumo-binary or set SUMO_HOME."
    )


def sumo_version(binary):
    try:
        proc = subprocess.run(
            [str(binary), "--version"], capture_output=True, text=True, timeout=30
        )
    except Exception:
        return None
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    for line in text.splitlines():
        if "Eclipse SUMO" in line:
            return line.strip()
    return text.strip().splitlines()[0] if text.strip() else None


def lanes_from_net(net_path, edge_ids):
    root = ET.parse(str(net_path)).getroot()
    by_edge = {edge_id: [] for edge_id in edge_ids}
    for edge in root.iter("edge"):
        edge_id = edge.get("id")
        if edge_id in by_edge:
            by_edge[edge_id] = [lane.get("id") for lane in edge.findall("lane")]
    missing = [edge_id for edge_id, lanes in by_edge.items() if not lanes]
    if missing:
        raise ValueError(f"Edges not found in net: {missing}")
    lanes = []
    for edge_id in edge_ids:
        lanes.extend(by_edge[edge_id])
    return lanes


def lane_lengths(net_path, lanes):
    root = ET.parse(str(net_path)).getroot()
    lengths = {}
    for edge in root.iter("edge"):
        for lane in edge.findall("lane"):
            lane_id = lane.get("id")
            if lane_id in lanes:
                lengths[lane_id] = float(lane.get("length", 0.0))
    return lengths


def _mean(values):
    return sum(values) / len(values) if values else None


def run_orig_pack(pack, spec, raw_dir, sumo_binary=None):
    """Run one pack with the native tlLogic untouched and retain raw outputs."""
    import libsumo as traci

    binary = discover_sumo_binary(sumo_binary)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    net_path = pack["dir"] / Path(pack["sumocfg"]).with_suffix(".net.xml").name
    # The sumocfg names the net; resolve it from the config instead of guessing.
    cfg_root = ET.parse(str(pack["sumocfg"])).getroot()
    net_name = None
    route_name = None
    for item in cfg_root.iter():
        if item.tag == "net-file":
            net_name = item.get("value")
        elif item.tag == "route-files":
            route_name = item.get("value")
    net_path = pack["dir"] / (net_name or "")
    if not net_path.is_file():
        # Fall back to the pack manifest's conventional name.
        net_path = pack["dir"] / f"{pack['dir'].name}_signal.net.xml"

    in_lanes = lanes_from_net(net_path, spec["_in_edges"])
    out_lanes = lanes_from_net(net_path, spec["_out_edges"])
    obs_lanes = in_lanes + out_lanes
    lengths = lane_lengths(net_path, in_lanes)
    max_in_lane_length = max(lengths.values()) if lengths else None

    stat_path = raw_dir / "statistic.xml"
    trip_path = raw_dir / "tripinfo.xml"
    summary_path = raw_dir / "summary.xml"
    queue_path = raw_dir / "queue_samples.csv"
    command_path = raw_dir / "command.json"
    metrics_path = raw_dir / "orig_metrics.json"

    horizon = float(spec["_horizon_s"])
    command = [
        str(binary), "-c", Path(pack["sumocfg"]).name,
        "--no-warnings", "true", "--duration-log.disable", "true",
        "--statistic-output", str(stat_path),
        "--tripinfo-output", str(trip_path),
        "--summary-output", str(summary_path),
    ]
    env_identity = {
        key: os.environ.get(key)
        for key in ("SUMO_HOME", "LD_PRELOAD", "CONDA_PREFIX", "PATH")
    }
    started = utc_now()
    cwd = os.getcwd()
    dep, arr, waiting = {}, {}, {}
    prev_obs = set()
    q_sum, q_n = 0.0, 0
    samples = []
    try:
        os.chdir(pack["dir"])
        traci.start(command)
        while traci.simulation.getTime() < horizon:
            traci.simulationStep()
            t = traci.simulation.getTime()
            for v in traci.simulation.getDepartedIDList():
                dep[v] = t
            for v in traci.simulation.getArrivedIDList():
                if v in dep:
                    arr[v] = t - dep[v]
            obs = set()
            for lane in obs_lanes:
                obs.update(traci.lane.getLastStepVehicleIDs(lane))
            lane_vehicles = {
                lane: traci.lane.getLastStepVehicleIDs(lane) for lane in in_lanes
            }
            for vehicles in lane_vehicles.values():
                for v in vehicles:
                    if v in waiting:
                        waiting[v] += 1.0
                    elif traci.vehicle.getWaitingTime(v) > 0:
                        waiting[v] = traci.vehicle.getWaitingTime(v)
            for v in prev_obs - obs:
                waiting.pop(v, None)
            prev_obs = obs
            if int(t + 0.5) % int(spec["_decision_interval_s"]) == 0:
                queue_count = sum(
                    1 for vehicles in lane_vehicles.values() for v in vehicles
                    if waiting.get(v, 0) > 0
                )
                q_sum += queue_count
                q_n += 1
                inserted_now = len(dep)
                arrived_now = len(arr)
                active_now = list(traci.vehicle.getIDList())
                waiting_now = _mean([
                    traci.vehicle.getAccumulatedWaitingTime(v)
                    for v in active_now
                ]) if active_now else 0.0
                samples.append({
                    "time_s": t,
                    "queue_veh": queue_count,
                    "throughput_arrived_veh": arrived_now,
                    "unfinished_departed_veh": inserted_now - arrived_now,
                    "inserted_veh": inserted_now,
                    "att_arrived_s": _mean(list(arr.values())),
                    "waiting_end_in_net_s": waiting_now,
                })
        active = list(traci.vehicle.getIDList())
        waiting_end = _mean([
            traci.vehicle.getAccumulatedWaitingTime(v) for v in active
        ]) if active else 0.0
        waiting_n = len(active)
        traci.close()
    finally:
        os.chdir(cwd)
        try:
            traci.close()
        except Exception:
            pass

    sample_fields = (
        "time_s", "queue_veh", "throughput_arrived_veh",
        "unfinished_departed_veh", "inserted_veh", "att_arrived_s",
        "waiting_end_in_net_s",
    )
    with queue_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields)
        writer.writeheader()
        writer.writerows(samples)

    stat = ET.parse(str(stat_path)).getroot()
    vehicles = stat.find("vehicles").attrib
    teleports = stat.find("teleports").attrib
    loaded = int(vehicles["loaded"])
    inserted = int(vehicles["inserted"])
    stat_waiting = int(vehicles["waiting"])
    stat_running = int(vehicles["running"])
    losses = [
        float(x.attrib["timeLoss"]) for x in ET.parse(str(trip_path)).getroot().iter(
            "tripinfo"
        )
    ]
    durations = [
        float(x.attrib["duration"]) for x in ET.parse(str(trip_path)).getroot().iter(
            "tripinfo"
        )
    ]
    trips = {
        "att_arrived_s": _mean(list(arr.values())),
        "att_n": len(arr),
        "throughput_arrived_veh": len(arr),
        "unfinished_departed_veh": len(dep) - len(arr),
        "uninserted_loaded_veh": loaded - inserted,
        "backlog_total_veh": (len(dep) - len(arr)) + (loaded - inserted),
        "queue_decision_mean_veh": (q_sum / q_n) if q_n else None,
        "queue_n": q_n,
        "waiting_end_in_net_s": waiting_end,
        "waiting_n": waiting_n,
        "real_delay_tripinfo_s": _mean(losses),
        "teleports_count": int(teleports["total"]),
        "att_tripinfo_s": _mean(durations),
        "loaded": loaded,
        "inserted": inserted,
        "stat_waiting": stat_waiting,
        "stat_running": stat_running,
    }
    warnings = []
    residual = loaded - len(dep) - stat_waiting
    if not 0 <= residual <= 1:
        warnings.append(
            f"loaded/inserted/waiting residual is {residual} (expected 0..1)"
        )
    if stat_running != len(dep) - len(arr):
        warnings.append(
            f"statistic running {stat_running} != departed-arrived "
            f"{len(dep) - len(arr)}"
        )
    if max_in_lane_length is not None and max_in_lane_length > 200:
        warnings.append(
            "controlled in-lane length exceeds the world's 200 m observation "
            "distance; orig queue may not exactly match run.py queue"
        )
    if waiting_n and waiting_n < 20:
        warnings.append(f"waiting_end denominator is small (n={waiting_n})")

    command_payload = {
        "argv": command,
        "cwd": str(pack["dir"]),
        "env": env_identity,
        "sumo_binary": str(binary),
        "sumo_version": sumo_version(binary),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "horizon_s": horizon,
        "in_lanes": in_lanes,
        "out_lanes": out_lanes,
        "max_in_lane_length_m": max_in_lane_length,
    }
    command_path.write_text(
        json.dumps(command_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "pack_id": pack["pack_id"],
        "arm_id": "orig_prog",
        "source_kind": "direct_sumo",
        "control_semantics": "native_tls_program",
        "status": "ok",
        "metrics": trips,
        "warnings": warnings,
        "provenance": {
            "sumocfg": str(Path(pack["sumocfg"]).relative_to(REPO_ROOT)),
            "net": str(net_path.relative_to(REPO_ROOT)),
            "statistic_xml": str(stat_path.relative_to(REPO_ROOT)),
            "tripinfo_xml": str(trip_path.relative_to(REPO_ROOT)),
            "queue_samples": str(queue_path.relative_to(REPO_ROOT)),
            "sumo_version": command_payload["sumo_version"],
            "sumo_seed_mode": "fixed_default",
            "sumo_seed": None,
        },
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload
