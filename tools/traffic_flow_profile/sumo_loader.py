import hashlib
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import (
    APPROACHES,
    NetworkDemandScenario,
    RoutedVehicle,
    SumoPackage,
    SumoScenario,
    VehicleDemand,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_sumocfg(input_path: Path) -> Tuple[Path, bool]:
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.suffix != ".sumocfg":
            raise ValueError(f"Expected a .sumocfg file, got: {input_path}")
        return input_path, False
    if not input_path.is_dir():
        raise FileNotFoundError(f"SUMO package does not exist: {input_path}")
    candidates = sorted(input_path.glob("*.sumocfg"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one .sumocfg in {input_path}, found {len(candidates)}; "
            "pass a .sumocfg path explicitly when a directory contains multiple scenarios"
        )
    return candidates[0], True


def _config_value(
    root: ET.Element, section: str, name: str, default: Optional[str] = None
) -> str:
    element = root.find(f"./{section}/{name}")
    if element is None or not element.get("value"):
        if default is not None:
            return default
        raise ValueError(f"SUMO config is missing {section}/{name}")
    return element.get("value", "")


def _resolve_config_paths(config_dir: Path, raw_value: str) -> Tuple[Path, ...]:
    values = [value.strip() for value in raw_value.split(",") if value.strip()]
    paths = tuple((config_dir / value).resolve() for value in values)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SUMO config references missing files: {missing}")
    return paths


def _load_network(net_path: Path, additional_paths: Iterable[Path]):
    root = ET.parse(net_path).getroot()
    junctions = {
        item.get("id", ""): (float(item.get("x", "0")), float(item.get("y", "0")))
        for item in root.findall("junction")
        if item.get("id")
    }
    signal_candidates = [
        item.get("id", "") for item in root.findall("tlLogic") if item.get("id")
    ]
    signal_candidates.extend(
        item.get("id", "")
        for item in root.findall("junction")
        if (item.get("type") or "").startswith("traffic_light") and item.get("id")
    )
    for additional_path in additional_paths:
        additional_root = ET.parse(additional_path).getroot()
        signal_candidates.extend(
            item.get("id", "")
            for item in additional_root.findall("tlLogic")
            if item.get("id")
        )
    signal_ids = tuple(
        signal_id
        for signal_id in dict.fromkeys(signal_candidates)
        if signal_id in junctions
    )

    edges = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        if not edge_id or edge.get("function") == "internal":
            continue
        from_id = edge.get("from")
        to_id = edge.get("to")
        if from_id and to_id:
            edges[edge_id] = (from_id, to_id)

    incoming_edges = {
        signal_id: tuple(
            edge_id for edge_id, endpoints in edges.items() if endpoints[1] == signal_id
        )
        for signal_id in signal_ids
    }
    return signal_ids, junctions, edges, incoming_edges


def _route_edges(element: ET.Element, route_definitions: Dict[str, Tuple[str, ...]]):
    route_ref = element.get("route")
    if route_ref:
        if route_ref not in route_definitions:
            raise ValueError(f"Demand element references unknown route: {route_ref}")
        return route_definitions[route_ref]
    route_element = element.find("route")
    if route_element is not None and route_element.get("edges"):
        return tuple(route_element.get("edges", "").split())
    return tuple()


def _numeric_depart(raw_value: Optional[str], demand_id: str) -> float:
    try:
        return float(raw_value or "")
    except ValueError as exc:
        raise ValueError(f"Demand {demand_id} has a non-numeric depart time") from exc


def _flow_departures(flow: ET.Element) -> List[float]:
    flow_id = flow.get("id", "<unnamed-flow>")
    begin = _numeric_depart(flow.get("begin", "0"), flow_id)
    raw_end = flow.get("end")
    end = _numeric_depart(raw_end, flow_id) if raw_end is not None else None

    if flow.get("probability") is not None:
        raise ValueError(
            f"Flow {flow_id} uses stochastic probability demand, which cannot be expanded "
            "into an exact deterministic profile"
        )

    if flow.get("number") is not None:
        try:
            number = int(flow.get("number", "0"))
        except ValueError as exc:
            raise ValueError(f"Flow {flow_id} has an invalid number") from exc
        if number <= 0:
            return []
        if end is None:
            period = flow.get("period")
            if period is None:
                raise ValueError(f"Flow {flow_id} with number requires end or period")
            try:
                step = float(period)
            except ValueError as exc:
                raise ValueError(f"Flow {flow_id} has a non-numeric period") from exc
        else:
            step = (end - begin) / number
        if step < 0:
            raise ValueError(f"Flow {flow_id} ends before it begins")
        return [begin + index * step for index in range(number)]

    raw_period = flow.get("period")
    raw_rate = flow.get("vehsPerHour")
    if raw_period is not None:
        try:
            period = float(raw_period)
        except ValueError as exc:
            raise ValueError(
                f"Flow {flow_id} uses a non-numeric period and cannot be expanded exactly"
            ) from exc
    elif raw_rate is not None:
        try:
            rate = float(raw_rate)
        except ValueError as exc:
            raise ValueError(f"Flow {flow_id} has an invalid vehsPerHour") from exc
        period = 3600.0 / rate if rate > 0 else 0.0
    else:
        raise ValueError(
            f"Flow {flow_id} requires number, numeric period, or vehsPerHour"
        )
    if end is None or period <= 0:
        raise ValueError(f"Flow {flow_id} requires a positive period and an end time")
    departures = []
    depart = begin
    while depart < end and not math.isclose(depart, end):
        departures.append(depart)
        depart += period
    return departures


def _load_routed_vehicles(route_paths: Iterable[Path]) -> List[RoutedVehicle]:
    vehicles: List[RoutedVehicle] = []
    seen_ids = set()
    for route_path in route_paths:
        root = ET.parse(route_path).getroot()
        route_definitions = {
            route.get("id", ""): tuple(route.get("edges", "").split())
            for route in root.findall("route")
            if route.get("id") and route.get("edges")
        }

        for element in root.findall("vehicle"):
            vehicle_id = element.get("id", "")
            if not vehicle_id:
                raise ValueError(f"Vehicle without id in {route_path}")
            scoped_id = f"{route_path.name}:{vehicle_id}"
            if scoped_id in seen_ids:
                raise ValueError(f"Duplicate vehicle id in route file: {scoped_id}")
            seen_ids.add(scoped_id)
            vehicles.append(
                RoutedVehicle(
                    vehicle_id=scoped_id,
                    depart=_numeric_depart(element.get("depart"), vehicle_id),
                    route_edges=_route_edges(element, route_definitions),
                )
            )

        for element in root.findall("trip"):
            trip_id = element.get("id", "")
            if not trip_id:
                raise ValueError(f"Trip without id in {route_path}")
            scoped_id = f"{route_path.name}:{trip_id}"
            if scoped_id in seen_ids:
                raise ValueError(f"Duplicate trip id in route file: {scoped_id}")
            seen_ids.add(scoped_id)
            vehicles.append(
                RoutedVehicle(
                    vehicle_id=scoped_id,
                    depart=_numeric_depart(element.get("depart"), trip_id),
                    route_edges=tuple(),
                )
            )

        for element in root.findall("flow"):
            flow_id = element.get("id", "")
            if not flow_id:
                raise ValueError(f"Flow without id in {route_path}")
            route_edges = _route_edges(element, route_definitions)
            for index, depart in enumerate(_flow_departures(element)):
                scoped_id = f"{route_path.name}:{flow_id}:{index}"
                if scoped_id in seen_ids:
                    raise ValueError(f"Duplicate expanded flow id: {scoped_id}")
                seen_ids.add(scoped_id)
                vehicles.append(
                    RoutedVehicle(
                        vehicle_id=scoped_id,
                        depart=depart,
                        route_edges=route_edges,
                    )
                )
    vehicles.sort(key=lambda item: (item.depart, item.vehicle_id))
    return vehicles


def _cardinal_approach(
    source: Tuple[float, float], center: Tuple[float, float]
) -> str:
    dx = source[0] - center[0]
    dy = source[1] - center[1]
    if abs(dx) >= abs(dy):
        return "W" if dx < 0 else "E"
    return "S" if dy < 0 else "N"


def _approach_mapping(package: SumoPackage, signal_id: str):
    incoming = package.incoming_edges.get(signal_id, tuple())
    if not incoming:
        raise ValueError(f"Signal junction {signal_id} has no incoming edges")
    center = package.junctions[signal_id]
    cardinal_by_edge = {
        edge_id: _cardinal_approach(
            package.junctions[package.edges[edge_id][0]], center
        )
        for edge_id in incoming
    }
    if len(set(cardinal_by_edge.values())) == len(incoming):
        approach_edges = {
            approach: edge_id for edge_id, approach in cardinal_by_edge.items()
        }
        approach_order = tuple(
            approach for approach in APPROACHES if approach in approach_edges
        )
    else:
        approach_edges = {edge_id: edge_id for edge_id in incoming}
        approach_order = tuple(
            sorted(
                incoming,
                key=lambda edge_id: math.atan2(
                    package.junctions[package.edges[edge_id][0]][1] - center[1],
                    package.junctions[package.edges[edge_id][0]][0] - center[0],
                ),
            )
        )
    edge_to_approach = {
        edge_id: approach for approach, edge_id in approach_edges.items()
    }
    return approach_edges, approach_order, edge_to_approach


def _movement_for_pair(
    incoming_edge: str,
    outgoing_edge: str,
    signal_id: str,
    package: SumoPackage,
) -> str:
    incoming_from, incoming_to = package.edges[incoming_edge]
    outgoing_from, outgoing_to = package.edges[outgoing_edge]
    if incoming_to != signal_id or outgoing_from != signal_id:
        raise ValueError(f"Edges do not form a movement through {signal_id}")
    center_x, center_y = package.junctions[signal_id]
    source_x, source_y = package.junctions[incoming_from]
    target_x, target_y = package.junctions[outgoing_to]
    in_x, in_y = center_x - source_x, center_y - source_y
    out_x, out_y = target_x - center_x, target_y - center_y
    in_norm = math.hypot(in_x, in_y)
    out_norm = math.hypot(out_x, out_y)
    if in_norm == 0 or out_norm == 0:
        raise ValueError(f"Zero-length movement geometry through {signal_id}")
    dot = (in_x * out_x + in_y * out_y) / (in_norm * out_norm)
    cross = (in_x * out_y - in_y * out_x) / (in_norm * out_norm)
    if dot < -0.5:
        return "u_turn"
    if abs(cross) < 0.5 and dot > 0:
        return "through"
    return "left" if cross > 0 else "right"


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "scenario"


def load_sumo_package(input_path) -> SumoPackage:
    requested_path = Path(input_path)
    sumocfg_path, requested_as_directory = _resolve_sumocfg(requested_path)
    package_dir = sumocfg_path.parent
    package_id = (
        package_dir.name
        if requested_as_directory or sumocfg_path.stem == package_dir.name
        else f"{package_dir.name}__{sumocfg_path.stem}"
    )
    config_root = ET.parse(sumocfg_path).getroot()
    net_paths = _resolve_config_paths(
        package_dir, _config_value(config_root, "input", "net-file")
    )
    if len(net_paths) != 1:
        raise ValueError("Traffic profiler requires exactly one SUMO net file per config")
    route_paths = _resolve_config_paths(
        package_dir, _config_value(config_root, "input", "route-files")
    )
    raw_additional = _config_value(
        config_root, "input", "additional-files", ""
    )
    additional_paths = (
        _resolve_config_paths(package_dir, raw_additional)
        if raw_additional
        else tuple()
    )
    all_vehicles = _load_routed_vehicles(route_paths)
    begin = float(_config_value(config_root, "time", "begin", "0"))
    raw_end = _config_value(config_root, "time", "end", "")
    configured_end = float(raw_end) if raw_end else None
    if configured_end is not None and configured_end >= 0:
        end = configured_end
    elif all_vehicles:
        end = max(vehicle.depart for vehicle in all_vehicles) + 1.0
    else:
        raise ValueError("SUMO config has no end time and no demand departures")
    if end <= begin:
        raise ValueError(f"Invalid SUMO analysis window: begin={begin}, end={end}")
    vehicles = [
        vehicle for vehicle in all_vehicles if begin <= vehicle.depart < end
    ]
    signal_ids, junctions, edges, incoming_edges = _load_network(
        net_paths[0], additional_paths
    )
    all_inputs = (sumocfg_path, net_paths[0], *route_paths, *additional_paths)
    hashes = {path.name: _sha256(path) for path in all_inputs}
    return SumoPackage(
        package_id=package_id,
        package_dir=package_dir,
        sumocfg_path=sumocfg_path,
        net_path=net_paths[0],
        route_paths=route_paths,
        additional_paths=additional_paths,
        begin=begin,
        end=end,
        signal_ids=signal_ids,
        junctions=junctions,
        edges=edges,
        incoming_edges=incoming_edges,
        vehicles=vehicles,
        input_sha256=hashes,
    )


def build_signal_scenario(
    package: SumoPackage, signal_id: str, scenario_id: Optional[str] = None
) -> SumoScenario:
    if signal_id not in package.signal_ids:
        raise ValueError(
            f"Unknown signal junction {signal_id}; available signals: {package.signal_ids}"
        )
    unrouted = [vehicle.vehicle_id for vehicle in package.vehicles if not vehicle.route_edges]
    if unrouted:
        raise ValueError(
            f"Signal profiling requires explicit routes; {len(unrouted)} demand elements "
            "use trips or route-less flows"
        )
    approach_edges, approach_order, edge_to_approach = _approach_mapping(
        package, signal_id
    )
    demands: List[VehicleDemand] = []
    for vehicle in package.vehicles:
        crossing_index = 0
        for incoming_edge, outgoing_edge in zip(
            vehicle.route_edges, vehicle.route_edges[1:]
        ):
            if incoming_edge not in package.edges or outgoing_edge not in package.edges:
                continue
            if (
                package.edges[incoming_edge][1] != signal_id
                or package.edges[outgoing_edge][0] != signal_id
            ):
                continue
            demands.append(
                VehicleDemand(
                    vehicle_id=f"{vehicle.vehicle_id}@{signal_id}:{crossing_index}",
                    depart=vehicle.depart,
                    approach=edge_to_approach[incoming_edge],
                    movement=_movement_for_pair(
                        incoming_edge, outgoing_edge, signal_id, package
                    ),
                )
            )
            crossing_index += 1
    demands.sort(key=lambda item: (item.depart, item.vehicle_id))
    if scenario_id is None:
        scenario_id = f"{package.package_id}__signal_{_safe_identifier(signal_id)}"
    return SumoScenario(
        scenario_id=scenario_id,
        package_id=package.package_id,
        package_dir=package.package_dir,
        sumocfg_path=package.sumocfg_path,
        net_path=package.net_path,
        route_paths=package.route_paths,
        additional_paths=package.additional_paths,
        begin=package.begin,
        end=package.end,
        signal_junction_id=signal_id,
        approach_edges=approach_edges,
        approach_order=approach_order,
        vehicles=demands,
        input_sha256=package.input_sha256,
    )


def build_network_scenario(package: SumoPackage) -> NetworkDemandScenario:
    return NetworkDemandScenario(
        scenario_id=f"{package.package_id}__network",
        package_id=package.package_id,
        package_dir=package.package_dir,
        sumocfg_path=package.sumocfg_path,
        net_path=package.net_path,
        route_paths=package.route_paths,
        additional_paths=package.additional_paths,
        begin=package.begin,
        end=package.end,
        signal_ids=package.signal_ids,
        junctions=package.junctions,
        edges=package.edges,
        vehicles=package.vehicles,
        input_sha256=package.input_sha256,
    )


def load_sumo_scenario(input_path, signal_id: Optional[str] = None) -> SumoScenario:
    package = load_sumo_package(input_path)
    if signal_id is None:
        if len(package.signal_ids) != 1:
            raise ValueError(
                f"SUMO network contains {len(package.signal_ids)} signal junctions; "
                "pass signal_id or use load_sumo_scenarios()"
            )
        return build_signal_scenario(
            package, package.signal_ids[0], scenario_id=package.package_dir.name
        )
    return build_signal_scenario(package, signal_id)


def load_sumo_scenarios(
    input_path, signal_ids: Optional[Sequence[str]] = None
) -> List[SumoScenario]:
    package = load_sumo_package(input_path)
    selected = package.signal_ids if signal_ids is None else tuple(signal_ids)
    return [build_signal_scenario(package, signal_id) for signal_id in selected]


def load_sumo_network_scenario(input_path) -> NetworkDemandScenario:
    return build_network_scenario(load_sumo_package(input_path))
