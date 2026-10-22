"""CLI for the pack x controller benchmark recorder/plotter."""
import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .ckpt_eval_ts import collect_ckpt_timeseries, write_ckpt_ts_queue
from .collect import collect_dataset
from .plot import load_dataset, plot_dataset
from .report import validate_dataset, write_manifest, write_validation
from .spec import load_benchmark_spec
from .timeseries import (
    collect_timeseries,
    plot_timeseries,
    plot_timeseries_focus,
    write_replay_queue,
)


DEFAULT_OUTPUT_ROOT = Path("data/output_data/analysis/pack_benchmark")


def _analysis_dir(args):
    if args.dataset:
        path = Path(args.dataset).expanduser().resolve()
        return path.parent
    return (Path(args.output_root) / args.analysis_id).expanduser().resolve()


def _dataset_path(args):
    if args.dataset:
        return Path(args.dataset).expanduser().resolve()
    return _analysis_dir(args) / "benchmark.json"


def _print_summary(dataset):
    print(f"benchmark: {dataset['benchmark_id']}  analysis: {dataset['analysis_id']}")
    print(f"packs: {len(dataset['packs'])}  arms: {len(dataset['arms'])}  cells: {len(dataset['cells'])}")
    for warning in dataset.get("warnings") or []:
        print(f"WARNING: {warning}")


def cmd_collect(args):
    spec = load_benchmark_spec(args.spec)
    output_dir = _analysis_dir(args)
    dataset, output_dir = collect_dataset(
        spec, output_dir, orig_mode=args.orig_mode,
        allow_partial=args.allow_partial, sumo_binary=args.sumo_binary,
    )
    _print_summary(dataset)
    print(f"wrote {output_dir / 'benchmark.json'}")
    return 0


def cmd_validate(args):
    dataset = load_dataset(_dataset_path(args))
    validation = validate_dataset(dataset, legacy_matrix=args.legacy_matrix)
    path = write_validation(validation, _analysis_dir(args))
    print(
        f"validation: {validation['status']}  "
        f"cells {validation['observed_cells']}/{validation['expected_cells']}"
    )
    for error in validation["errors"]:
        print(f"ERROR: {error}")
    if validation.get("legacy_comparison"):
        legacy = validation["legacy_comparison"]
        print(
            f"legacy comparison: checked {legacy['checked_values']} values, "
            f"{legacy['mismatch_count']} mismatches"
        )
    print(f"wrote {path}")
    return 0 if validation["status"] == "PASS" else 1


def cmd_plot(args):
    dataset = load_dataset(_dataset_path(args))
    output_dir = _analysis_dir(args)
    figures = plot_dataset(dataset, output_dir, dpi=args.dpi)
    manifest = write_manifest(dataset, output_dir, figure_manifest=figures)
    print(f"wrote {len(figures['figures'])} figures -> {output_dir / 'figures'}")
    print(f"wrote {manifest}")
    return 0


def cmd_all(args):
    spec = load_benchmark_spec(args.spec)
    output_dir = _analysis_dir(args)
    dataset, output_dir = collect_dataset(
        spec, output_dir, orig_mode=args.orig_mode,
        allow_partial=args.allow_partial, sumo_binary=args.sumo_binary,
    )
    _print_summary(dataset)
    validation = validate_dataset(dataset, legacy_matrix=args.legacy_matrix)
    write_validation(validation, output_dir)
    figures = plot_dataset(dataset, output_dir, dpi=args.dpi)
    manifest = write_manifest(dataset, output_dir, figure_manifest=figures)
    print(f"validation: {validation['status']}")
    for error in validation["errors"]:
        print(f"ERROR: {error}")
    print(f"wrote {len(figures['figures'])} figures and {manifest}")
    return 0 if validation["status"] == "PASS" else 1


def cmd_timeseries_queue(args):
    spec = load_benchmark_spec(args.spec)
    path = write_replay_queue(spec, args.queue, args.overlay)
    print(f"wrote replay queue: {path}")
    return 0


def cmd_timeseries_collect(args):
    spec = load_benchmark_spec(args.spec)
    output_dir = _analysis_dir(args)
    dataset, timeseries_path, manifest = collect_timeseries(
        spec, output_dir, args.reference_dataset,
        orig_mode=args.orig_mode, sumo_binary=args.sumo_binary,
    )
    validation = manifest["replay_validation"]
    print(
        f"timeseries: {manifest['rows']} rows -> {timeseries_path}; "
        f"replay validation checked {validation['checked_values']} values, "
        f"{validation['mismatch_count']} mismatches"
    )
    for mismatch in validation["mismatches"]:
        print(f"MISMATCH: {mismatch}")
    return 0 if validation["mismatch_count"] == 0 else 1


def cmd_timeseries_plot(args):
    dataset = load_dataset(_dataset_path(args))
    output_dir = _analysis_dir(args)
    manifest = plot_timeseries(dataset, output_dir, dpi=args.dpi)
    print(f"wrote {len(manifest['figures'])} within-episode figures -> {output_dir / 'figures'}")
    return 0


def cmd_timeseries_plot_focus(args):
    dataset = load_dataset(_dataset_path(args))
    output_dir = _analysis_dir(args)
    manifest = plot_timeseries_focus(dataset, output_dir, dpi=args.dpi)
    print(f"wrote {len(manifest['figures'])} focus figures -> {output_dir / 'figures'}")
    return 0


def cmd_ckpt_ts_queue(args):
    spec = load_benchmark_spec(args.spec)
    path = write_ckpt_ts_queue(spec, args.queue)
    print(f"wrote checkpoint-eval queue: {path}")
    return 0


def cmd_ckpt_ts_collect(args):
    spec = load_benchmark_spec(args.spec)
    output_dir = _analysis_dir(args)
    dataset, timeseries_path, manifest = collect_ckpt_timeseries(
        spec, output_dir, args.queue,
        orig_mode=args.orig_mode, sumo_binary=args.sumo_binary,
    )
    validation = manifest["frozen_replay_validation"]
    print(
        f"timeseries: {manifest['rows']} rows -> {timeseries_path}; "
        f"frozen replay validation checked {validation['checked_values']} "
        f"values, {validation['mismatch_count']} mismatches"
    )
    for mismatch in validation["mismatches"]:
        print(f"MISMATCH: {mismatch}")
    return 0 if validation["mismatch_count"] == 0 else 1


def build_parser():
    parser = argparse.ArgumentParser(
        description="Record and plot a pack x controller benchmark with fixed metric ids."
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sub):
        sub.add_argument("--analysis-id", required=True,
                         help="New analysis directory name under the output root.")
        sub.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
        sub.add_argument("--dataset", default=None,
                         help="Optional explicit benchmark.json for plot/validate.")
        sub.add_argument("--dpi", type=int, default=160)
        sub.add_argument("--legacy-matrix", default=None,
                         help="Optional legacy notrain_matrix.json for regression comparison.")
        sub.add_argument("--allow-partial", action="store_true",
                         help="Development only: allow an incomplete pack x arm matrix.")

    collect = subparsers.add_parser("collect", help="Collect run dirs and direct runs into a dataset.")
    collect.add_argument("--spec", required=True)
    collect.add_argument("--orig-mode", choices=("rerun", "reuse", "skip"), default="rerun")
    collect.add_argument("--sumo-binary", default=None)
    common(collect)
    collect.set_defaults(func=cmd_collect)

    validate = subparsers.add_parser("validate", help="Validate an existing dataset.")
    common(validate)
    validate.set_defaults(func=cmd_validate)

    plot = subparsers.add_parser("plot", help="Plot an existing dataset.")
    common(plot)
    plot.set_defaults(func=cmd_plot)

    all_cmd = subparsers.add_parser("all", help="Collect, validate and plot.")
    all_cmd.add_argument("--spec", required=True)
    all_cmd.add_argument("--orig-mode", choices=("rerun", "reuse", "skip"), default="rerun")
    all_cmd.add_argument("--sumo-binary", default=None)
    common(all_cmd)
    all_cmd.set_defaults(func=cmd_all)

    ts_queue = subparsers.add_parser(
        "timeseries-queue",
        help="Write a replay queue for within-episode step logging.",
    )
    ts_queue.add_argument("--spec", required=True)
    ts_queue.add_argument(
        "--queue", default="artifacts/xiasha1_timeseries/run_queue.json"
    )
    ts_queue.add_argument(
        "--overlay", default="configs/tsc/xiasha1_timeseries.yml"
    )
    ts_queue.set_defaults(func=cmd_timeseries_queue)

    ts_collect = subparsers.add_parser(
        "timeseries-collect",
        help="Collect replay step logs and validate them against the canonical dataset.",
    )
    ts_collect.add_argument("--spec", required=True)
    ts_collect.add_argument("--reference-dataset", required=True)
    ts_collect.add_argument(
        "--orig-mode", choices=("rerun", "reuse"), default="rerun"
    )
    ts_collect.add_argument("--sumo-binary", default=None)
    common(ts_collect)
    ts_collect.set_defaults(func=cmd_timeseries_collect)

    ts_plot = subparsers.add_parser(
        "timeseries-plot",
        help="Plot within-episode curves from a collected timeseries CSV.",
    )
    common(ts_plot)
    ts_plot.set_defaults(func=cmd_timeseries_plot)

    ts_focus = subparsers.add_parser(
        "timeseries-plot-focus",
        help="Zoomed learned-band curves, delta-vs-maxpressure and "
             "discriminability heatmaps from a collected timeseries CSV.",
    )
    common(ts_focus)
    ts_focus.set_defaults(func=cmd_timeseries_plot_focus)

    ckpt_queue = subparsers.add_parser(
        "ckpt-ts-queue",
        help="Write a frozen best-checkpoint replay queue with step logging.",
    )
    ckpt_queue.add_argument("--spec", required=True)
    ckpt_queue.add_argument("--queue", required=True)
    ckpt_queue.set_defaults(func=cmd_ckpt_ts_queue)

    ckpt_collect = subparsers.add_parser(
        "ckpt-ts-collect",
        help="Collect frozen-checkpoint step logs and validate them against "
             "the source training run's best-episode evaluation records.",
    )
    ckpt_collect.add_argument("--spec", required=True)
    ckpt_collect.add_argument("--queue", required=True,
                              help="The dispatch queue json (source of ckpt provenance).")
    ckpt_collect.add_argument(
        "--orig-mode", choices=("rerun", "reuse"), default="rerun"
    )
    ckpt_collect.add_argument("--sumo-binary", default=None)
    common(ckpt_collect)
    ckpt_collect.set_defaults(func=cmd_ckpt_ts_collect)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
