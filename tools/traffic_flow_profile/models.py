from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


APPROACHES: Tuple[str, ...] = ("W", "S", "E", "N")
MOVEMENTS: Tuple[str, ...] = ("left", "through", "right")


@dataclass(frozen=True)
class VehicleDemand:
    vehicle_id: str
    depart: float
    approach: str
    movement: str


@dataclass(frozen=True)
class RoutedVehicle:
    vehicle_id: str
    depart: float
    route_edges: Tuple[str, ...]


@dataclass(frozen=True)
class SumoPackage:
    package_id: str
    package_dir: Path
    sumocfg_path: Path
    net_path: Path
    route_paths: Tuple[Path, ...]
    additional_paths: Tuple[Path, ...]
    begin: float
    end: float
    signal_ids: Tuple[str, ...]
    junctions: Dict[str, Tuple[float, float]]
    edges: Dict[str, Tuple[str, str]]
    incoming_edges: Dict[str, Tuple[str, ...]]
    vehicles: List[RoutedVehicle]
    input_sha256: Dict[str, str]

    @property
    def duration(self) -> float:
        return self.end - self.begin


@dataclass(frozen=True)
class NetworkDemandScenario:
    scenario_id: str
    package_id: str
    package_dir: Path
    sumocfg_path: Path
    net_path: Path
    route_paths: Tuple[Path, ...]
    additional_paths: Tuple[Path, ...]
    begin: float
    end: float
    signal_ids: Tuple[str, ...]
    junctions: Dict[str, Tuple[float, float]]
    edges: Dict[str, Tuple[str, str]]
    vehicles: List[RoutedVehicle]
    input_sha256: Dict[str, str]

    @property
    def duration(self) -> float:
        return self.end - self.begin


@dataclass(frozen=True)
class SumoScenario:
    scenario_id: str
    package_id: str
    package_dir: Path
    sumocfg_path: Path
    net_path: Path
    route_paths: Tuple[Path, ...]
    additional_paths: Tuple[Path, ...]
    begin: float
    end: float
    signal_junction_id: str
    approach_edges: Dict[str, str]
    approach_order: Tuple[str, ...]
    vehicles: List[VehicleDemand]
    input_sha256: Dict[str, str]

    @property
    def duration(self) -> float:
        return self.end - self.begin
