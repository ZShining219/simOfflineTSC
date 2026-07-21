import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .models import APPROACHES, SumoScenario, VehicleDemand


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_sumocfg(input_path: Path) -> Path:
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.suffix != ".sumocfg":
            raise ValueError(f"Expected a .sumocfg file, got: {input_path}")
        return input_path
    if not input_path.is_dir():
        raise FileNotFoundError(f"SUMO package does not exist: {input_path}")
    candidates = sorted(input_path.glob("*.sumocfg"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one .sumocfg in {input_path}, found {len(candidates)}"
        )
    return candidates[0]


def _config_value(root: ET.Element, section: str, name: str) -> str:
    element = root.find(f"./{section}/{name}")
    if element is None or not element.get("value"):
        raise ValueError(f"SUMO config is missing {section}/{name}")
    return element.get("value", "")


def _resolve_config_paths(config_dir: Path, raw_value: str) -> Tuple[Path, ...]:
    values = [value.strip() for value in raw_value.split(",") if value.strip()]
    paths = tuple((config_dir / value).resolve() for value in values)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SUMO config references missing files: {missing}")
    return paths


def _load_network(net_path: Path):
    root = ET.parse(net_path).getroot()
    junctions = {
        item.get("id", ""): (float(item.get("x", "0")), float(item.get("y", "0")))
        for item in root.findall("junction")
    }
    signal_ids = {item.get("id", "") for item in root.findall("tlLogic")}
    signal_ids.discard("")
    if len(signal_ids) != 1:
        raise ValueError(
            f"Single-scenario profiler requires exactly one signal junction; found {len(signal_ids)}"
        )
    signal_id = next(iter(signal_ids))
    if signal_id not in junctions:
        raise ValueError(f"Signal junction {signal_id} is absent from the network junctions")

    edges = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        if not edge_id or edge.get("function") == "internal":
            continue
        from_id = edge.get("from")
        to_id = edge.get("to")
        if not from_id or not to_id:
            continue
        edges[edge_id] = (from_id, to_id)

    incoming = {
        edge_id: endpoints
        for edge_id, endpoints in edges.items()
        if endpoints[1] == signal_id
    }
    if len(incoming) != 4:
        raise ValueError(
            f"hz1x1 adapter requires four incoming edges; found {len(incoming)}"
        )

    approach_edges: Dict[str, str] = {}
    center_x, center_y = junctions[signal_id]
    for edge_id, (from_id, _) in incoming.items():
        source_x, source_y = junctions[from_id]
        dx = source_x - center_x
        dy = source_y - center_y
        if abs(dx) >= abs(dy):
            approach = "W" if dx < 0 else "E"
        else:
            approach = "S" if dy < 0 else "N"
        if approach in approach_edges:
            raise ValueError(f"Multiple incoming edges map to approach {approach}")
        approach_edges[approach] = edge_id
    if set(approach_edges) != set(APPROACHES):
        raise ValueError(f"Unable to identify all cardinal approaches: {approach_edges}")
    return signal_id, junctions, edges, approach_edges


def _movement_for_pair(
    incoming_edge: str,
    outgoing_edge: str,
    signal_id: str,
    junctions: Dict[str, Tuple[float, float]],
    edges: Dict[str, Tuple[str, str]],
) -> str:
    incoming_from, incoming_to = edges[incoming_edge]
    outgoing_from, outgoing_to = edges[outgoing_edge]
    if incoming_to != signal_id or outgoing_from != signal_id:
        raise ValueError(f"Edges do not form a movement through {signal_id}")
    center_x, center_y = junctions[signal_id]
    source_x, source_y = junctions[incoming_from]
    target_x, target_y = junctions[outgoing_to]
    in_x, in_y = center_x - source_x, center_y - source_y
    out_x, out_y = target_x - center_x, target_y - center_y
    in_norm = math.hypot(in_x, in_y)
    out_norm = math.hypot(out_x, out_y)
    dot = (in_x * out_x + in_y * out_y) / (in_norm * out_norm)
    cross = (in_x * out_y - in_y * out_x) / (in_norm * out_norm)
    if dot < -0.5:
        return "u_turn"
    if abs(cross) < 0.5 and dot > 0:
        return "through"
    return "left" if cross > 0 else "right"


def _route_edges(vehicle: ET.Element, route_definitions: Dict[str, Tuple[str, ...]]):
    route_ref = vehicle.get("route")
    if route_ref:
        if route_ref not in route_definitions:
            raise ValueError(f"Vehicle references unknown route: {route_ref}")
        return route_definitions[route_ref]
    route_element = vehicle.find("route")
    if route_element is None or not route_element.get("edges"):
        raise ValueError(f"Vehicle {vehicle.get('id')} has no usable route")
    return tuple(route_element.get("edges", "").split())


def _movement_pair(
    route_edges: Sequence[str], signal_id: str, edges: Dict[str, Tuple[str, str]]
) -> Tuple[str, str]:
    candidates = []
    for incoming_edge, outgoing_edge in zip(route_edges, route_edges[1:]):
        if incoming_edge not in edges or outgoing_edge not in edges:
            continue
        if edges[incoming_edge][1] == signal_id and edges[outgoing_edge][0] == signal_id:
            candidates.append((incoming_edge, outgoing_edge))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one movement through {signal_id}, found {len(candidates)} in {route_edges}"
        )
    return candidates[0]


def _load_vehicles(
    route_paths: Iterable[Path],
    begin: float,
    end: float,
    signal_id: str,
    junctions,
    edges,
    edge_to_approach,
) -> List[VehicleDemand]:
    vehicles: List[VehicleDemand] = []
    seen_ids = set()
    for route_path in route_paths:
        root = ET.parse(route_path).getroot()
        if root.findall("flow"):
            raise ValueError(
                f"SUMO <flow> elements are not supported in the hz1x1 first version: {route_path}"
            )
        route_definitions = {
            route.get("id", ""): tuple(route.get("edges", "").split())
            for route in root.findall("route")
            if route.get("id") and route.get("edges")
        }
        for vehicle in root.findall("vehicle"):
            vehicle_id = vehicle.get("id", "")
            if not vehicle_id:
                raise ValueError(f"Vehicle without id in {route_path}")
            scoped_id = f"{route_path.name}:{vehicle_id}"
            if scoped_id in seen_ids:
                raise ValueError(f"Duplicate vehicle id in route file: {scoped_id}")
            seen_ids.add(scoped_id)
            try:
                depart = float(vehicle.get("depart", ""))
            except ValueError as exc:
                raise ValueError(f"Vehicle {vehicle_id} has a non-numeric depart time") from exc
            if not begin <= depart < end:
                continue
            vehicle_route = _route_edges(vehicle, route_definitions)
            incoming_edge, outgoing_edge = _movement_pair(vehicle_route, signal_id, edges)
            movement = _movement_for_pair(
                incoming_edge, outgoing_edge, signal_id, junctions, edges
            )
            vehicles.append(
                VehicleDemand(
                    vehicle_id=scoped_id,
                    depart=depart,
                    approach=edge_to_approach[incoming_edge],
                    movement=movement,
                )
            )
    vehicles.sort(key=lambda item: (item.depart, item.vehicle_id))
    return vehicles


def load_sumo_scenario(input_path) -> SumoScenario:
    sumocfg_path = _resolve_sumocfg(Path(input_path))
    package_dir = sumocfg_path.parent
    config_root = ET.parse(sumocfg_path).getroot()
    net_path = _resolve_config_paths(
        package_dir, _config_value(config_root, "input", "net-file")
    )
    if len(net_path) != 1:
        raise ValueError("Single-scenario profiler requires exactly one SUMO net file")
    route_paths = _resolve_config_paths(
        package_dir, _config_value(config_root, "input", "route-files")
    )
    begin = float(_config_value(config_root, "time", "begin"))
    end = float(_config_value(config_root, "time", "end"))
    if end <= begin:
        raise ValueError(f"Invalid SUMO analysis window: begin={begin}, end={end}")
    if not math.isclose(end - begin, 3600.0):
        raise ValueError(
            f"hz1x1 first version requires a one-hour SUMO window; got {end - begin} seconds"
        )

    signal_id, junctions, edges, approach_edges = _load_network(net_path[0])
    edge_to_approach = {edge_id: approach for approach, edge_id in approach_edges.items()}
    vehicles = _load_vehicles(
        route_paths,
        begin,
        end,
        signal_id,
        junctions,
        edges,
        edge_to_approach,
    )
    if not vehicles:
        raise ValueError("No vehicle departures were found in the SUMO analysis window")

    all_inputs = (sumocfg_path, net_path[0], *route_paths)
    hashes = {path.name: _sha256(path) for path in all_inputs}
    return SumoScenario(
        scenario_id=package_dir.name,
        package_dir=package_dir,
        sumocfg_path=sumocfg_path,
        net_path=net_path[0],
        route_paths=route_paths,
        begin=begin,
        end=end,
        signal_junction_id=signal_id,
        approach_edges=approach_edges,
        vehicles=vehicles,
        input_sha256=hashes,
    )
