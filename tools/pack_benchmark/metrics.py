"""Single source of truth for benchmark metric definitions.

Every figure and table must use these metric_id values.  Do not add an
ad-hoc column in a plotting script; add it here with an explicit population,
time basis, unit and comparability flag.
"""
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Metric:
    metric_id: str
    label: str
    unit: str
    definition: str
    population: str
    time_basis: str
    higher_is_better: bool
    cross_arm_comparable: bool
    source_fields: Dict[str, str]
    denominator: Optional[str] = None
    available_sources: Optional[Tuple[str, ...]] = None

    def to_dict(self):
        return asdict(self)


METRICS = {
    "att_arrived_s": Metric(
        metric_id="att_arrived_s",
        label="Average travel time (arrived vehicles)",
        unit="s",
        definition=(
            "Mean depart-to-arrive duration of vehicles that completed their "
            "trip by the end of the episode."
        ),
        population="arrived vehicles",
        time_basis="per vehicle",
        higher_is_better=False,
        cross_arm_comparable=True,
        source_fields={
            "run_dir": "FINAL_EVALUATION.travel_time",
            "direct_sumo": "mean(arrival_time - depart_time) over arrived vehicles",
        },
        denominator="att_n",
    ),
    "throughput_arrived_veh": Metric(
        metric_id="throughput_arrived_veh",
        label="Throughput (arrived vehicles)",
        unit="veh",
        definition="Number of vehicles that completed their trip by episode end.",
        population="arrived vehicles",
        time_basis="cumulative end-of-episode",
        higher_is_better=True,
        cross_arm_comparable=True,
        source_fields={
            "run_dir": "FINAL_EVALUATION.throughput",
            "direct_sumo": "len(arrived ledger)",
        },
    ),
    "unfinished_departed_veh": Metric(
        metric_id="unfinished_departed_veh",
        label="Unfinished vehicles (departed, not arrived)",
        unit="veh",
        definition=(
            "Vehicles that departed but had not arrived by episode end. "
            "Includes vehicles temporarily omitted from SUMO's active ID list."
        ),
        population="departed vehicles",
        time_basis="end-of-episode snapshot",
        higher_is_better=False,
        cross_arm_comparable=True,
        source_fields={
            "run_dir": "FINAL_EVALUATION.unfinished_vehicles",
            "direct_sumo": "len(departed ledger) - len(arrived ledger)",
        },
    ),
    "uninserted_loaded_veh": Metric(
        metric_id="uninserted_loaded_veh",
        label="Uninserted vehicles (loaded, not departed)",
        unit="veh",
        definition=(
            "Loaded demand vehicles still waiting to be inserted at episode end. "
            "For run.py controller arms this is derived as "
            "loaded - throughput - unfinished; loaded comes from the pack's "
            "direct native-program run."
        ),
        population="loaded demand",
        time_basis="end-of-episode snapshot",
        higher_is_better=False,
        cross_arm_comparable=True,
        source_fields={
            "run_dir": "derived: orig.loaded - throughput - unfinished",
            "direct_sumo": "statistic.xml loaded - inserted",
        },
    ),
    "backlog_total_veh": Metric(
        metric_id="backlog_total_veh",
        label="Total backlog (uninserted + in-net unfinished)",
        unit="veh",
        definition="uninserted_loaded_veh + unfinished_departed_veh.",
        population="loaded demand",
        time_basis="end-of-episode snapshot",
        higher_is_better=False,
        cross_arm_comparable=True,
        source_fields={
            "run_dir": "derived",
            "direct_sumo": "derived",
        },
    ),
    "queue_decision_mean_veh": Metric(
        metric_id="queue_decision_mean_veh",
        label="Mean queue (in-lane waiting vehicles)",
        unit="veh",
        definition=(
            "Mean over 10 s decision steps of the sum of lane_waiting_count "
            "over the controlled approach lanes (vehicles with accumulated "
            "wait > 0 within the world's observation distance)."
        ),
        population="controlled approach lanes",
        time_basis="mean over decision steps",
        higher_is_better=False,
        cross_arm_comparable=True,
        source_fields={
            "run_dir": "FINAL_EVALUATION.queue",
            "direct_sumo": "world-equivalent in-lane waiting count sampling",
        },
    ),
    "waiting_end_in_net_s": Metric(
        metric_id="waiting_end_in_net_s",
        label="Mean accumulated waiting of in-net vehicles at end",
        unit="s",
        definition=(
            "At the final simulation step, mean accumulated waiting time of "
            "vehicles still in the network. Excludes vehicles never inserted. "
            "The denominator is recorded as waiting_n; small denominators are "
            "flagged."
        ),
        population="vehicles in network at end",
        time_basis="end-of-episode snapshot",
        higher_is_better=False,
        cross_arm_comparable=True,
        source_fields={
            "run_dir": "FINAL_EVALUATION.waiting_time",
            "direct_sumo": "mean(getAccumulatedWaitingTime) over active vehicles",
        },
        denominator="waiting_n",
    ),
    "real_delay_world_s": Metric(
        metric_id="real_delay_world_s",
        label="Real delay (world lane-trajectory approximation)",
        unit="s",
        definition=(
            "Mean per-vehicle sum over traversed lanes of actual lane time "
            "minus free-flow lane time, with negative values clipped at zero. "
            "Covers vehicles with a trajectory, including active vehicles. "
            "Not the same estimator as SUMO tripinfo timeLoss."
        ),
        population="vehicles with trajectory",
        time_basis="per vehicle",
        higher_is_better=False,
        cross_arm_comparable=False,
        source_fields={"run_dir": "FINAL_EVALUATION.real_delay"},
        available_sources=("run_dir",),
    ),
    "real_delay_tripinfo_s": Metric(
        metric_id="real_delay_tripinfo_s",
        label="SUMO tripinfo time loss (completed trips)",
        unit="s",
        definition=(
            "Mean timeLoss reported by SUMO tripinfo for completed trips. "
            "Only available for the direct native-program arm; do not compare "
            "it directly with real_delay_world_s."
        ),
        population="arrived vehicles",
        time_basis="per vehicle",
        higher_is_better=False,
        cross_arm_comparable=False,
        source_fields={"direct_sumo": "tripinfo.xml timeLoss"},
        available_sources=("direct_sumo",),
    ),
    "lane_delay_index": Metric(
        metric_id="lane_delay_index",
        label="Approximate lane delay index",
        unit="dimensionless",
        definition=(
            "Mean over decision steps and controlled approach lanes of "
            "1 - mean_lane_speed / lane_speed_limit. Only recorded by run.py "
            "controller arms."
        ),
        population="controlled approach lanes",
        time_basis="mean over decision steps",
        higher_is_better=False,
        cross_arm_comparable=False,
        source_fields={"run_dir": "FINAL_EVALUATION.delay"},
        available_sources=("run_dir",),
    ),
    "teleports_count": Metric(
        metric_id="teleports_count",
        label="Teleports",
        unit="count",
        definition="SUMO statistic-output teleport total.",
        population="network",
        time_basis="cumulative end-of-episode",
        higher_is_better=False,
        cross_arm_comparable=False,
        source_fields={"direct_sumo": "statistic.xml teleports.total"},
        available_sources=("direct_sumo",),
    ),
    "phase_switches": Metric(
        metric_id="phase_switches",
        label="Phase switches (action changes)",
        unit="count",
        definition=(
            "Number of decision steps whose action differs from the previous "
            "decision action. Diagnostic of controller timing, not a traffic "
            "performance metric."
        ),
        population="controller decisions",
        time_basis="cumulative",
        higher_is_better=False,
        cross_arm_comparable=False,
        source_fields={"run_dir": "FINAL_EVALUATION.phase_switches"},
        available_sources=("run_dir",),
    ),
    "phase_switch_frequency": Metric(
        metric_id="phase_switch_frequency",
        label="Phase switch frequency",
        unit="ratio",
        definition="phase_switches / max(1, decision_step - 1).",
        population="controller decisions",
        time_basis="ratio",
        higher_is_better=False,
        cross_arm_comparable=False,
        source_fields={"run_dir": "FINAL_EVALUATION.phase_switch_frequency"},
        available_sources=("run_dir",),
    ),
}

METRIC_ORDER = tuple(METRICS)
LEGACY_ALIASES = {
    "att": "att_arrived_s",
    "throughput": "throughput_arrived_veh",
    "queue": "queue_decision_mean_veh",
    "waiting_s": "waiting_end_in_net_s",
    "unfinished_in_net": "unfinished_departed_veh",
    "uninserted": "uninserted_loaded_veh",
    "lane_delay_idx": "lane_delay_index",
    "teleports": "teleports_count",
}


def get_metric(metric_id):
    try:
        return METRICS[metric_id]
    except KeyError as error:
        raise ValueError(f"Unknown metric_id: {metric_id}") from error


def metric_definitions():
    return {metric_id: metric.to_dict() for metric_id, metric in METRICS.items()}


def legacy_metric_for(arm_kind, legacy_field):
    """Map a legacy notrain_matrix column to the canonical metric id."""
    if legacy_field == "real_delay_s":
        return (
            "real_delay_world_s"
            if arm_kind == "run_dir"
            else "real_delay_tripinfo_s"
        )
    return LEGACY_ALIASES.get(legacy_field)
