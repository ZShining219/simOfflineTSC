"""Stable semantic profiles shared by experiment plotting commands."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlotProfile:
    """Map an experiment plan onto the common plotting vocabulary."""

    name: str
    network_field: str
    algorithm_field: str
    seed_field: str
    progress_field: str
    progress_label: str
    algorithm_order: tuple
    network_order: tuple = ()


PROFILES = {
    "plan1": PlotProfile(
        name="plan1",
        network_field="network",
        algorithm_field="agent",
        seed_field="training_seed",
        progress_field="episode",
        progress_label="Completed training episode",
        algorithm_order=("dqn", "fixedtime", "maxpressure"),
        network_order=(
            "sumohz1x1_config2", "sumohz1x1",
            "sumohz1x1_config4", "sumohz1x1_config3",
        ),
    ),
    "plan2": PlotProfile(
        name="plan2",
        network_field="evaluation_network",
        algorithm_field="algorithm",
        seed_field="offline_training_seed",
        progress_field="training_update",
        progress_label="Offline gradient update",
        algorithm_order=("batch_dqn", "cql_dqn"),
        network_order=(
            "sumohz1x1_config2", "sumohz1x1",
            "sumohz1x1_config4", "sumohz1x1_config3",
        ),
    ),
}


def get_profile(name):
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"Unknown experiment plotting profile: {name}") from error
