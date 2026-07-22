import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .aggregations import (
    AUC_FIELDS, CORE_FIELDS, LEARNING_SPEED_FIELDS, build_run_summaries,
    calculate_first_100_auc, calculate_learning_speed, normalize_records, write_csv,
)
from .loaders import REQUIRED_RUN_LIST_FIELDS, load_run_list
from .plotting import render_all
from .plan2 import run_plan2_analysis
from .validators import compare_dqn_run_configs, validate_run


DEFAULT_OUTPUT_ROOT = Path("data/output_data/analysis/plan1")
DEFAULT_PLAN2_OUTPUT_ROOT = Path("data/output_data/analysis/plan2")


def _json_dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _analysis_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise argparse.ArgumentTypeError(
            "analysis-id must contain only letters, digits, dot, underscore and hyphen"
        )
    return value


def build_parser():
    parser = argparse.ArgumentParser(description="Validated reusable experiment plotting")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan1 = subparsers.add_parser("plan1", help="Generate Plan 1 tables and figures")
    plan1.add_argument("--run-list", required=True, help="Explicit CSV run-list")
    plan1.add_argument("--analysis-id", required=True, type=_analysis_id)
    plan1.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    plan1.add_argument("--dpi", type=int, default=160)
    plan2 = subparsers.add_parser("plan2", help="Validate and aggregate Plan 2 runs")
    plan2.add_argument("--run-list", required=True, help="Explicit Plan 2 CSV run-list")
    plan2.add_argument("--analysis-id", required=True, type=_analysis_id)
    plan2.add_argument("--output-root", default=str(DEFAULT_PLAN2_OUTPUT_ROOT))
    plan2.add_argument(
        "--allow-incomplete", action="store_true",
        help="Development-only: permit non-144k schedules and incomplete five-seed cells",
    )
    return parser


def run_plan1(args):
    run_list_path, all_specs, included_specs = load_run_list(args.run_list)
    output_dir = Path(args.output_root).expanduser().resolve() / args.analysis_id
    if output_dir.exists():
        raise FileExistsError(
            f"Analysis output already exists: {output_dir}. Use a new analysis-id."
        )
    inputs_dir = output_dir / "inputs"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    inputs_dir.mkdir(parents=True)
    tables_dir.mkdir()
    figures_dir.mkdir()
    try:
        shutil.copy2(run_list_path, inputs_dir / "runs.csv")
        validated = [validate_run(spec) for spec in included_specs]
        config_comparison = compare_dqn_run_configs(validated)
        metric_rows, action_rows = normalize_records(validated)
        summaries, comparisons = build_run_summaries(validated, metric_rows)
        auc_rows = calculate_first_100_auc(metric_rows)
        learning_speed_rows = calculate_learning_speed(metric_rows)

        metric_fields = (
            "run_key", "role", "agent", "network", "training_seed", "run_dir",
            *CORE_FIELDS, "action_distribution_json",
        )
        write_csv(tables_dir / "metrics.csv", metric_rows, metric_fields)
        write_csv(tables_dir / "action_distribution.csv", action_rows, (
            "run_key", "role", "agent", "network", "training_seed", "run_dir",
            "record_type", "episode", "action", "fraction",
        ))
        write_csv(tables_dir / "run_summary.csv", summaries)
        write_csv(tables_dir / "final_best_comparison.csv", comparisons)
        write_csv(tables_dir / "first_100_auc.csv", auc_rows, AUC_FIELDS)
        write_csv(
            tables_dir / "learning_speed.csv", learning_speed_rows,
            LEARNING_SPEED_FIELDS,
        )
        _json_dump(tables_dir / "config_comparison.json", config_comparison)
        figures = render_all(
            metric_rows, action_rows, comparisons, auc_rows, figures_dir, dpi=args.dpi,
        )
        manifest = {
            "schema_version": 1,
            "tool": "tools.experiment_plotting",
            "tool_version": __version__,
            "analysis_type": "plan1",
            "analysis_id": args.analysis_id,
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "run_list_source": str(run_list_path),
            "run_list_fields": list(REQUIRED_RUN_LIST_FIELDS),
            "listed_run_count": len(all_specs),
            "included_run_count": len(included_specs),
            "runs": [{
                "run_key": run["spec"].run_key,
                "run_dir": str(run["spec"].run_dir),
                "config_sha256": run["resolved_config_sha256"],
                "metric_record_count": len(run["records"]),
            } for run in validated],
            "tables": sorted(str(path.relative_to(output_dir)) for path in tables_dir.iterdir()),
            "figures": sorted(str(path.relative_to(output_dir)) for path in figures),
            "config_compatible": config_comparison["compatible"],
        }
        _json_dump(output_dir / "plotting_manifest.json", manifest)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return output_dir


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "plan1":
        output_dir = run_plan1(args)
        print(f"Plan 1 analysis completed: {output_dir}")
    elif args.command == "plan2":
        output_dir = run_plan2_analysis(args)
        print(f"Plan 2 analysis completed: {output_dir}")


if __name__ == "__main__":
    main()
