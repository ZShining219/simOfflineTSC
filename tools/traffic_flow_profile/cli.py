import argparse
from pathlib import Path

from . import __version__
from .metrics import calculate_metrics
from .report import write_report
from .sumo_loader import load_sumo_scenario


DEFAULT_OUTPUT_ROOT = Path("data/output_data/traffic_flow_profile")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create a fixed single-scenario traffic-demand profile for a SUMO package."
    )
    parser.add_argument(
        "input",
        help="SUMO package directory containing one .sumocfg, or a .sumocfg path.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Output root; scenario id is appended (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument("--dpi", type=int, default=160, help="PNG resolution (default: 160).")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    scenario = load_sumo_scenario(args.input)
    metrics = calculate_metrics(scenario)
    output_dir, paths = write_report(metrics, args.output_root, dpi=args.dpi)
    standard = metrics["standard_metrics"]
    print(f"Scenario: {scenario.scenario_id}")
    print(f"Vehicles: {standard['vehicle_count']}")
    print(f"PHF: {standard['peak_hour_factor']:.3f}")
    print(f"Output: {output_dir}")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
