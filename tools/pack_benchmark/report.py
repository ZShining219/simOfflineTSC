"""Validation, manifest and provenance for a benchmark dataset."""
import json
import math
import platform
import subprocess
from importlib import metadata
from pathlib import Path

from . import __version__
from .metrics import METRICS, legacy_metric_for
from .spec import REPO_ROOT, sha256_file


def _load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _git_identity():
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.splitlines()
    except Exception:
        return None, None, []
    dirty_files = [line[3:] for line in status if line.strip()]
    return commit or None, bool(dirty_files), dirty_files


def _version(name):
    try:
        return metadata.version(name)
    except Exception:
        return None


def validate_dataset(dataset, legacy_matrix=None):
    errors = []
    warnings = list(dataset.get("warnings") or [])
    pack_ids = [pack["pack_id"] for pack in dataset["packs"]]
    arm_ids = [arm["arm_id"] for arm in dataset["arms"]]
    expected = {(pack, arm) for pack in pack_ids for arm in arm_ids}
    seen = set()
    cells = {}
    for cell in dataset["cells"]:
        key = (cell["pack_id"], cell["arm_id"])
        if key in seen:
            errors.append(f"duplicate cell: {key}")
        seen.add(key)
        cells[key] = cell
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing:
        errors.append(f"missing cells: {missing}")
    if extra:
        errors.append(f"unexpected cells: {extra}")

    for key, cell in sorted(cells.items()):
        metrics = cell.get("metrics") or {}
        loaded = metrics.get("loaded")
        throughput = metrics.get("throughput_arrived_veh")
        unfinished = metrics.get("unfinished_departed_veh")
        uninserted = metrics.get("uninserted_loaded_veh")
        backlog = metrics.get("backlog_total_veh")
        if None not in (loaded, throughput, unfinished, uninserted):
            if throughput + unfinished + uninserted != loaded:
                errors.append(
                    f"{key}: loaded {loaded} != throughput {throughput} + "
                    f"unfinished {unfinished} + uninserted {uninserted}"
                )
            if uninserted < 0:
                errors.append(f"{key}: negative uninserted {uninserted}")
            if backlog != unfinished + uninserted:
                errors.append(
                    f"{key}: backlog {backlog} != unfinished {unfinished} + "
                    f"uninserted {uninserted}"
                )
        for metric_id in ("att_arrived_s", "throughput_arrived_veh",
                          "unfinished_departed_veh", "queue_decision_mean_veh",
                          "waiting_end_in_net_s"):
            value = metrics.get(metric_id)
            if value is None:
                errors.append(f"{key}: required metric {metric_id} is missing")
            elif not isinstance(value, (int, float)) or not math.isfinite(value):
                errors.append(f"{key}: metric {metric_id} is not finite: {value}")

    legacy = None
    if legacy_matrix:
        legacy = _compare_legacy(dataset, cells, legacy_matrix)
        errors.extend(legacy["failures"])

    validation = {
        "schema_version": 1,
        "benchmark_id": dataset["benchmark_id"],
        "analysis_id": dataset["analysis_id"],
        "status": "PASS" if not errors else "FAIL",
        "expected_cells": len(expected),
        "observed_cells": len(seen),
        "missing_cells": missing,
        "unexpected_cells": extra,
        "errors": errors,
        "warnings": warnings,
        "legacy_comparison": legacy,
    }
    return validation


def _compare_legacy(dataset, cells, legacy_matrix):
    reference = _load_json(legacy_matrix)
    matrix = reference.get("matrix") or {}
    arm_kind = {arm["arm_id"]: arm["kind"] for arm in dataset["arms"]}
    checked = 0
    failures = []
    mismatches = []
    for pack_id, arms in sorted(matrix.items()):
        for arm_id, legacy in sorted(arms.items()):
            cell = cells.get((pack_id, arm_id))
            if cell is None:
                failures.append(f"legacy cell missing in canonical dataset: {pack_id}/{arm_id}")
                continue
            for legacy_field, legacy_value in sorted(legacy.items()):
                if legacy_field in {"run_id"} or legacy_value is None:
                    continue
                metric_id = legacy_metric_for(arm_kind.get(arm_id), legacy_field)
                if not metric_id:
                    continue
                value = (cell.get("metrics") or {}).get(metric_id)
                if value is None:
                    failures.append(
                        f"{pack_id}/{arm_id}: canonical {metric_id} is missing "
                        f"for legacy {legacy_field}"
                    )
                    continue
                checked += 1
                if isinstance(legacy_value, int) and isinstance(value, int):
                    equal = legacy_value == value
                else:
                    equal = abs(float(legacy_value) - float(value)) <= 1e-6
                if not equal:
                    mismatches.append({
                        "pack_id": pack_id,
                        "arm_id": arm_id,
                        "legacy_field": legacy_field,
                        "metric_id": metric_id,
                        "legacy_value": legacy_value,
                        "canonical_value": value,
                        "abs_diff": abs(float(legacy_value) - float(value)),
                    })
    failures.extend(
        f"{item['pack_id']}/{item['arm_id']}: legacy {item['legacy_field']} "
        f"{item['legacy_value']} != canonical {item['metric_id']} "
        f"{item['canonical_value']}"
        for item in mismatches
    )
    return {
        "reference": str(Path(legacy_matrix).resolve().relative_to(REPO_ROOT)),
        "checked_values": checked,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "failures": failures,
    }


def write_validation(validation, output_dir):
    output_dir = Path(output_dir)
    manifests = output_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    path = manifests / "validation.json"
    path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def write_manifest(dataset, output_dir, figure_manifest=None):
    output_dir = Path(output_dir)
    commit, dirty, dirty_files = _git_identity()
    orig_cells = [
        cell for cell in dataset["cells"] if cell["source_kind"] == "direct_sumo"
    ]
    sumo_version = None
    for cell in orig_cells:
        sumo_version = (cell.get("provenance") or {}).get("sumo_version")
        if sumo_version:
            break
    tables = []
    for path in sorted((output_dir / "tables").glob("*.csv")):
        tables.append({
            "path": str(path.relative_to(output_dir)),
            "sha256": sha256_file(path),
        })
    figures = []
    if figure_manifest:
        for figure in figure_manifest.get("figures", []):
            figures.append({
                "figure_id": figure["figure_id"],
                "title": figure["title"],
                "paths": figure["paths"],
                "metric_ids": figure["metric_ids"],
                "sha256": {
                    path: sha256_file(output_dir / path) for path in figure["paths"]
                },
            })
    validation_path = output_dir / "manifests" / "validation.json"
    validation = None
    if validation_path.is_file():
        validation = {
            "path": str(validation_path.relative_to(output_dir)),
            "sha256": sha256_file(validation_path),
        }
    manifest = {
        "schema_version": 1,
        "benchmark_id": dataset["benchmark_id"],
        "analysis_id": dataset["analysis_id"],
        "validation": validation,
        "created_at_utc": dataset["created_at_utc"],
        "tool": {"name": "tools.pack_benchmark", "version": __version__},
        "code": {
            "commit": commit,
            "dirty": dirty,
            "dirty_files": dirty_files[:20],
            "dirty_file_count": len(dirty_files),
            "note": (
                "run_manifest.baseline_commit is a hard-coded constant, not the "
                "startup HEAD; this field records the collection-time HEAD."
            ),
        },
        "environment": {
            "sumo_version": sumo_version,
            "python": platform.python_version(),
            "matplotlib": _version("matplotlib"),
            "numpy": _version("numpy"),
            "PyYAML": _version("PyYAML"),
        },
        "spec": dataset.get("spec"),
        "horizon_s": dataset["horizon_s"],
        "decision_interval_s": dataset["decision_interval_s"],
        "seed": dataset["seed"],
        "episode": dataset["episode"],
        "cells": len(dataset["cells"]),
        "packs": len(dataset["packs"]),
        "arms": len(dataset["arms"]),
        "dataset_sha256": sha256_file(output_dir / "benchmark.json"),
        "tables": tables,
        "figures": figures,
        "metric_definitions": dataset["metric_definitions"],
        "warnings": dataset.get("warnings") or [],
    }
    path = output_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
