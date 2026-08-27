import argparse
from pathlib import Path

from . import __version__
from .metrics import calculate_metrics, calculate_network_metrics
from .report import write_report
from .sumo_loader import (
    build_network_scenario,
    build_signal_scenario,
    load_sumo_package,
)


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
    signal_group = parser.add_mutually_exclusive_group()
    signal_group.add_argument(
        "--signal-id",
        action="append",
        dest="signal_ids",
        help="Profile one signal junction; repeat to select multiple junctions.",
    )
    signal_group.add_argument(
        "--all-signals",
        action="store_true",
        help="Profile every signal junction in each input network.",
    )
    parser.add_argument(
        "--network-profile",
        action="store_true",
        help="Also create a network-level unique-departure profile.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="PNG resolution (default: 160).")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    packages = [load_sumo_package(input_path) for input_path in args.input]
    profile_entries = []
    for package in packages:
        if args.network_profile:
            network_scenario = build_network_scenario(package)
            profile_entries.append(
                (network_scenario, calculate_network_metrics(network_scenario))
            )

        if args.all_signals:
            selected_signals = package.signal_ids
        elif args.signal_ids:
            selected_signals = tuple(args.signal_ids)
        elif args.network_profile:
            selected_signals = tuple()
        else:
            if len(package.signal_ids) != 1:
                raise ValueError(
                    f"Input {package.sumocfg_path} contains {len(package.signal_ids)} "
                    "signal junctions; use --signal-id, --all-signals, or --network-profile"
                )
            selected_signals = package.signal_ids

        for signal_id in selected_signals:
            preserve_legacy_id = (
                not args.all_signals
                and not args.signal_ids
                and not args.network_profile
                and len(package.signal_ids) == 1
                and len(args.input) == 1
            )
            scenario = build_signal_scenario(
                package,
                signal_id,
                scenario_id=package.package_dir.name if preserve_legacy_id else None,
            )
            profile_entries.append((scenario, calculate_metrics(scenario)))

    if not profile_entries:
        raise ValueError("No traffic profiles were selected")
    scenario_ids = [scenario.scenario_id for scenario, _ in profile_entries]
    duplicate_ids = sorted(
        scenario_id
        for scenario_id in set(scenario_ids)
        if scenario_ids.count(scenario_id) > 1
    )
    if duplicate_ids:
        raise ValueError(
            f"Selected inputs produce duplicate output scenario ids: {duplicate_ids}"
        )
    metrics_collection = [metrics for _, metrics in profile_entries]

    from .plotting import calculate_shared_plot_limits

    limits_by_scope = {
        scope: calculate_shared_plot_limits(
            [
                metrics
                for metrics in metrics_collection
                if metrics["scenario"].get("scope", "signal") == scope
            ]
        )
        for scope in {metrics["scenario"].get("scope", "signal") for metrics in metrics_collection}
    }
    if len(metrics_collection) > 1:
        for scope, plot_limits in sorted(limits_by_scope.items()):
            print(
                f"Shared {scope} plot limits: "
                f"5 min={plot_limits['five_minute']:g}, "
                f"approach={plot_limits['approach']:g}, "
                f"movement={plot_limits['movement']:g}, "
                f"15 min={plot_limits['fifteen_minute']:g}"
            )

    for scenario, metrics in profile_entries:
        scope = metrics["scenario"].get("scope", "signal")
        output_dir, paths = write_report(
            metrics,
            args.output_root,
            dpi=args.dpi,
            plot_limits=limits_by_scope[scope],
        )
        standard = metrics["standard_metrics"]
        print(f"Scenario: {scenario.scenario_id} ({scope})")
        print(f"Vehicles: {standard['vehicle_count']}")
        print(f"PHF: {standard['peak_hour_factor']:.3f}")
        print(f"Output: {output_dir}")
        for name, path in paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
