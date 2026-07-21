import argparse
from pathlib import Path

from . import __version__
from .metrics import calculate_metrics
from .report import write_report
from .sumo_loader import load_sumo_scenario


DEFAULT_OUTPUT_ROOT = Path("data/output_data/traffic_flow_profile")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create comparable traffic-demand profiles for one or more SUMO packages."
    )
    parser.add_argument(
        "input",
        nargs="+",
        help=(
            "One or more SUMO package directories or .sumocfg paths. "
            "Multiple inputs share plot scales for direct comparison."
        ),
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
    scenarios = [load_sumo_scenario(input_path) for input_path in args.input]
    metrics_collection = [calculate_metrics(scenario) for scenario in scenarios]

    from .plotting import calculate_shared_plot_limits

    plot_limits = calculate_shared_plot_limits(metrics_collection)
    if len(metrics_collection) > 1:
        print(
            "Shared plot limits: "
            f"5 min={plot_limits['five_minute']:g}, "
            f"approach={plot_limits['approach']:g}, "
            f"movement={plot_limits['movement']:g}, "
            f"15 min={plot_limits['fifteen_minute']:g}"
        )

    for scenario, metrics in zip(scenarios, metrics_collection):
        output_dir, paths = write_report(
            metrics, args.output_root, dpi=args.dpi, plot_limits=plot_limits
        )
        standard = metrics["standard_metrics"]
        print(f"Scenario: {scenario.scenario_id}")
        print(f"Vehicles: {standard['vehicle_count']}")
        print(f"PHF: {standard['peak_hour_factor']:.3f}")
        print(f"Output: {output_dir}")
        for name, path in paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
