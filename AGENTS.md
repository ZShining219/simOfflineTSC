# Repository Guidelines

## Project overview

This repository is a Python traffic-signal-control research framework derived from LibSignal. It provides Gym-compatible environments and traditional/RL agents for CityFlow and SUMO simulations.

## Layout

- `run.py`: main experiment entry point and CLI argument definitions.
- `agent/`: traffic-signal-control agent implementations.
- `world/`: simulator-specific world integrations.
- `trainer/` and `task/`: training loops and task orchestration.
- `common/`, `utils/`, and `generator/`: shared configuration, logging, metrics, and generation utilities.
- `configs/`: simulator and TSC experiment configuration.
- `data/raw_data/`: input networks and traffic-flow data.
- `dataset/`: dataset integrations.
- `final_result/`: result-processing utilities or experiment artifacts.

## Development workflow

- Use Python 3.9 when matching the original environment.
- Install core dependencies with `pip install -r requirements.txt`; simulator dependencies must be installed separately.
- Run an experiment with `python run.py` and select the simulator, agent, and network through CLI options.
- Inspect available options with `python run.py --help` when the simulator dependencies are available.
- Keep generated outputs, caches, IDE files, and large experiment artifacts out of Git.

## Change guidelines

- Preserve the registry-based architecture in `common/registry.py` and existing registration conventions.
- Keep simulator-specific behavior isolated in the corresponding `world/` implementation.
- Prefer configuration changes in `configs/` over hard-coded experiment parameters.
- Avoid committing generated data or results unless they are intentional, small, and needed for reproducibility.
- When changing an agent or trainer, verify at least one representative configuration for the affected simulator.

## Validation

There is no dedicated automated test suite in this source snapshot. For code-only changes, run targeted import or syntax checks. For behavioral changes, run the smallest applicable experiment and report the simulator, agent, network, seed, and command used.
