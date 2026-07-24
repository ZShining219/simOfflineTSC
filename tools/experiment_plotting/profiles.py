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
    "s1_s4_diagnostics": PlotProfile(
        name="s1_s4_diagnostics",
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
}


S1_S4_SCENE_LABELS = {
    'sumohz1x1_config2': 'S1 (sumohz1x1_config2)',
    'sumohz1x1': 'S2 (sumohz1x1)',
    'sumohz1x1_config4': 'S3 (sumohz1x1_config4)',
    'sumohz1x1_config3': 'S4 (sumohz1x1_config3)',
}

S1_S4_ACTION_SEMANTICS = {
    0: '东西向直行同时放行',
    1: '南北向直行同时放行',
    2: '东西向左转同时放行',
    3: '南北向左转同时放行',
    4: '西进口直行与左转同时放行',
    5: '东进口直行与左转同时放行',
    6: '南进口直行与左转同时放行',
    7: '北进口直行与左转同时放行',
}


def get_profile(name):
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"Unknown experiment plotting profile: {name}") from error
