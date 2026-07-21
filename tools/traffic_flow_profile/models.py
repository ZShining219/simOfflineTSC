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
class SumoScenario:
    scenario_id: str
    package_dir: Path
    sumocfg_path: Path
    net_path: Path
    route_paths: Tuple[Path, ...]
    begin: float
    end: float
    signal_junction_id: str
    approach_edges: Dict[str, str]
    vehicles: List[VehicleDemand]
    input_sha256: Dict[str, str]

    @property
    def duration(self) -> float:
        return self.end - self.begin
