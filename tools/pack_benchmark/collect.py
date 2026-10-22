"""Collect a canonical pack x controller benchmark dataset."""
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import __version__
from .metrics import METRICS, METRIC_ORDER, metric_definitions
from .orig_runner import run_orig_pack
from .spec import REPO_ROOT, run_dir_for, sha256_file


IDENTITY_FIELDS = (
    "benchmark_id", "analysis_id", "pack_id", "pack_category", "network",
    "arm_id", "arm_kind", "arm_label", "control_semantics", "source_kind",
    "status", "run_id", "prefix", "run_dir", "seed", "episode",
    "horizon_s", "decision_interval_s", "sumo_seed_mode", "sumo_seed",
    "config_sha256", "records_sha256",
)
DENOMINATOR_FIELDS = (
    "att_n", "queue_n", "waiting_n", "loaded", "inserted",
    "stat_waiting", "stat_running", "att_tripinfo_s",
)
FLAG_FIELDS = ("warnings",)


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def json_load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _rel(path):
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(path).resolve())


def _read_records(run_dir):
    path = Path(run_dir) / "metrics" / "records.jsonl"
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return records


def pack_input_hashes(pack):
    root = ET.parse(str(pack["sumocfg"])).getroot()
    hashes = {"sumocfg": sha256_file(pack["sumocfg"])}
    for tag, key in (
        ("net-file", "net"),
        ("route-files", "route"),
        ("additional-files", "additional"),
    ):
        for item in root.iter(tag):
            value = item.get("value") or ""
            for name in value.split(","):
                name = name.strip()
                if not name:
                    continue
                path = pack["dir"] / name
                if path.is_file():
                    hashes[f"{key}:{name}"] = sha256_file(path)
    return hashes


def collect_run_cell(spec, arm, pack):
    run_dir = run_dir_for(spec, arm, pack)
    required = (
        run_dir / "run_manifest.json",
        run_dir / "run_status.json",
        run_dir / "config" / "resolved_config.yaml",
        run_dir / "config" / "config_hashes.json",
        run_dir / "config" / "model_resolved.json",
        run_dir / "metrics" / "records.jsonl",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run directory is incomplete: {missing}")
    manifest = json_load(run_dir / "run_manifest.json")
    status = json_load(run_dir / "run_status.json")
    if status.get("status") != "已完成" or status.get("exit_code") != 0:
        raise ValueError(f"Run did not complete successfully: {run_dir}")
    identity = {
        "agent": arm["agent"],
        "network": pack["network"],
        "training_seed": spec["_seed"],
    }
    mismatches = [
        f"{key}: run={manifest.get(key)!r}, expected={value!r}"
        for key, value in identity.items()
        if manifest.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"Run identity mismatch at {run_dir}: " + "; ".join(mismatches))

    resolved_path = run_dir / "config" / "resolved_config.yaml"
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    config_hash = sha256_file(resolved_path)
    if manifest.get("config_hash") != config_hash:
        raise ValueError(f"Config hash mismatch at {run_dir}")
    config_hashes = json_load(run_dir / "config" / "config_hashes.json")
    archived = (config_hashes.get("files") or {}).get("resolved_config.yaml")
    if archived and archived != config_hash:
        raise ValueError(f"Config archive hash mismatch at {run_dir}")

    records = _read_records(run_dir)
    finals = [row for row in records if row.get("record_type") == "FINAL_EVALUATION"]
    if len(finals) != 1:
        raise ValueError(f"Expected exactly one FINAL_EVALUATION at {run_dir}, got {len(finals)}")
    final = finals[0]
    expected_decisions = spec["_horizon_s"] // spec["_decision_interval_s"]
    if final.get("episode") != spec["_episode"]:
        raise ValueError(f"Unexpected final episode at {run_dir}: {final.get('episode')}")
    if final.get("simulation_step") != spec["_horizon_s"]:
        raise ValueError(f"Unexpected simulation_step at {run_dir}: {final.get('simulation_step')}")
    if final.get("decision_step") != expected_decisions:
        raise ValueError(f"Unexpected decision_step at {run_dir}: {final.get('decision_step')}")

    trainer = resolved.get("trainer") or {}
    signal_control = trainer.get("signal_control")
    control_semantics = (
        signal_control.get("version") if isinstance(signal_control, dict)
        else "legacy_pseudo_step"
    )
    expected_semantics = arm.get("expected_control_semantics")
    if expected_semantics and control_semantics != expected_semantics:
        raise ValueError(
            f"Control semantics mismatch at {run_dir}: "
            f"run={control_semantics}, spec={expected_semantics}"
        )
    model_resolved = json_load(run_dir / "config" / "model_resolved.json")
    agent_runtime = (model_resolved.get("agents") or [{}])[0]
    controller_params = dict(resolved.get("model") or {})
    controller_params.update(agent_runtime.get("controller_parameters") or {})

    metrics = {
        "att_arrived_s": final.get("travel_time"),
        "att_n": final.get("throughput"),
        "throughput_arrived_veh": final.get("throughput"),
        "unfinished_departed_veh": final.get("unfinished_vehicles"),
        "queue_decision_mean_veh": final.get("queue"),
        "queue_n": final.get("decision_step"),
        "waiting_end_in_net_s": final.get("waiting_time"),
        "waiting_n": final.get("unfinished_vehicles"),
        "real_delay_world_s": final.get("real_delay"),
        "lane_delay_index": final.get("delay"),
        "phase_switches": final.get("phase_switches"),
        "phase_switch_frequency": final.get("phase_switch_frequency"),
    }
    warnings = []
    if control_semantics == "legacy_pseudo_step":
        warnings.append(
            "legacy pseudo_step: current_phase_time is not reset on phase change; "
            f"observed phase_switches={final.get('phase_switches')}/"
            f"{final.get('decision_step')}"
        )
        warnings.append(
            "legacy ctl path has no real yellow clearance for this run "
            "(yellow_phase_time is 2 s and action_interval is 10 s)"
        )
    if not _finite(metrics["waiting_end_in_net_s"]):
        warnings.append("waiting_end_in_net_s is not finite")
    elif metrics["waiting_n"] is not None and metrics["waiting_n"] < 20:
        warnings.append(
            f"waiting_end denominator is small (n={metrics['waiting_n']}); "
            "mean over remaining vehicles is sensitive"
        )
    warnings.append(
        "real_delay_world_s is a lane-trajectory approximation and is not "
        "directly comparable with real_delay_tripinfo_s"
    )
    return {
        "cell_id": f"{pack['pack_id']}__{arm['id']}",
        "pack_id": pack["pack_id"],
        "pack_category": pack["category"],
        "network": pack["network"],
        "arm_id": arm["id"],
        "arm_kind": arm["kind"],
        "arm_label": arm.get("label") or arm["id"],
        "control_semantics": control_semantics,
        "source_kind": "run_dir",
        "status": "ok",
        "run_id": manifest.get("run_id"),
        "prefix": manifest.get("prefix"),
        "run_dir": _rel(run_dir),
        "seed": spec["_seed"],
        "episode": spec["_episode"],
        "horizon_s": spec["_horizon_s"],
        "decision_interval_s": spec["_decision_interval_s"],
        "sumo_seed_mode": manifest.get("sumo_seed_mode"),
        "sumo_seed": manifest.get("sumo_seed"),
        "config_sha256": config_hash,
        "records_sha256": sha256_file(run_dir / "metrics" / "records.jsonl"),
        "controller_params": controller_params,
        "metrics": metrics,
        "warnings": warnings,
        "provenance": {
            "run_manifest": _rel(run_dir / "run_manifest.json"),
            "run_status": _rel(run_dir / "run_status.json"),
            "resolved_config": _rel(resolved_path),
            "model_resolved": _rel(run_dir / "config" / "model_resolved.json"),
            "records": _rel(run_dir / "metrics" / "records.jsonl"),
            "input_hashes": pack_input_hashes(pack),
        },
    }


def collect_orig_cell(spec, arm, pack, raw_root, orig_mode="rerun", sumo_binary=None):
    pack_raw = Path(raw_root) / pack["pack_id"]
    metrics_path = pack_raw / "orig_metrics.json"
    if orig_mode == "reuse" and metrics_path.is_file():
        payload = json_load(metrics_path)
    elif orig_mode == "skip":
        raise ValueError(f"--orig-mode skip cannot provide cell for {pack['pack_id']}")
    else:
        payload = run_orig_pack(pack, spec, pack_raw, sumo_binary=sumo_binary)
    payload["cell_id"] = f"{pack['pack_id']}__{arm['id']}"
    payload["pack_id"] = pack["pack_id"]
    payload["pack_category"] = pack["category"]
    payload["network"] = pack["network"]
    payload["arm_id"] = arm["id"]
    payload["arm_kind"] = arm["kind"]
    payload["arm_label"] = arm.get("label") or arm["id"]
    payload["seed"] = spec["_seed"]
    payload["episode"] = spec["_episode"]
    payload["horizon_s"] = spec["_horizon_s"]
    payload["decision_interval_s"] = spec["_decision_interval_s"]
    payload["run_id"] = "sumo_direct"
    payload["prefix"] = "orig_prog"
    payload["sumo_seed_mode"] = (
        (payload.get("provenance") or {}).get("sumo_seed_mode") or "fixed_default"
    )
    payload["sumo_seed"] = (payload.get("provenance") or {}).get("sumo_seed")
    payload["controller_params"] = {"program": "native_tls"}
    payload["run_dir"] = _rel(pack_raw)
    payload["config_sha256"] = sha256_file(pack["sumocfg"])
    stat_path = pack_raw / "statistic.xml"
    payload["records_sha256"] = sha256_file(stat_path) if stat_path.is_file() else None
    payload["provenance"]["input_hashes"] = pack_input_hashes(pack)
    return payload


def _derive_backlog(cell, loaded):
    metrics = cell["metrics"]
    metrics["loaded"] = loaded
    throughput = metrics.get("throughput_arrived_veh")
    unfinished = metrics.get("unfinished_departed_veh")
    if throughput is None or unfinished is None:
        metrics["uninserted_loaded_veh"] = None
        metrics["backlog_total_veh"] = None
        cell.setdefault("warnings", []).append("cannot derive backlog without throughput/unfinished")
        return
    uninserted = loaded - throughput - unfinished
    if uninserted < 0:
        cell.setdefault("warnings", []).append(
            f"derived uninserted is negative ({uninserted}); check loaded/ledgers"
        )
    metrics["uninserted_loaded_veh"] = uninserted
    metrics["backlog_total_veh"] = unfinished + uninserted


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _wide_row(cell):
    row = {field: cell.get(field) for field in IDENTITY_FIELDS}
    row["controller_params_json"] = json.dumps(
        cell.get("controller_params") or {}, sort_keys=True
    )
    for metric_id in METRIC_ORDER:
        row[metric_id] = cell["metrics"].get(metric_id)
    for field in DENOMINATOR_FIELDS:
        row[field] = cell["metrics"].get(field)
    row["warnings"] = "; ".join(cell.get("warnings") or [])
    return row


def write_dataset(dataset, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    cells = dataset["cells"]
    wide_fields = (
        list(IDENTITY_FIELDS[:7])
        + ["controller_params_json"]
        + list(IDENTITY_FIELDS[7:])
        + list(METRIC_ORDER)
        + list(DENOMINATOR_FIELDS)
        + list(FLAG_FIELDS)
    )
    _write_csv(tables / "cells.csv", [_wide_row(cell) for cell in cells], wide_fields)
    long_fields = (
        "cell_id", "pack_id", "arm_id", "metric_id", "value", "unit",
        "definition", "population", "time_basis", "cross_arm_comparable",
        "denominator", "denominator_value", "available", "warnings",
    )
    long_rows = []
    for cell in cells:
        for metric_id in METRIC_ORDER:
            metric = METRICS[metric_id]
            value = cell["metrics"].get(metric_id)
            denominator_value = (
                cell["metrics"].get(metric.denominator) if metric.denominator else None
            )
            long_rows.append({
                "cell_id": cell["cell_id"],
                "pack_id": cell["pack_id"],
                "arm_id": cell["arm_id"],
                "metric_id": metric_id,
                "value": value,
                "unit": metric.unit,
                "definition": metric.definition,
                "population": metric.population,
                "time_basis": metric.time_basis,
                "cross_arm_comparable": metric.cross_arm_comparable,
                "denominator": metric.denominator,
                "denominator_value": denominator_value,
                "available": value is not None,
                "warnings": "; ".join(cell.get("warnings") or []),
            })
    _write_csv(tables / "metrics_long.csv", long_rows, long_fields)
    definitions = []
    for metric_id, metric in METRICS.items():
        row = metric.to_dict()
        row["source_fields_json"] = json.dumps(row.pop("source_fields"), sort_keys=True)
        row["available_sources"] = ",".join(row.get("available_sources") or ())
        definitions.append(row)
    _write_csv(tables / "metric_definitions.csv", definitions, list(definitions[0]))
    pack_rows = [
        {
            **pack,
            "input_hashes": json.dumps(pack.get("input_hashes") or {}, sort_keys=True),
        }
        for pack in dataset["packs"]
    ]
    _write_csv(tables / "packs.csv", pack_rows, list(pack_rows[0]))
    _write_csv(tables / "arms.csv", dataset["arms"], list(dataset["arms"][0]))
    return {
        "benchmark_json": str(output_dir / "benchmark.json"),
        "tables": sorted(str(path) for path in tables.glob("*.csv")),
    }


def collect_dataset(spec, output_dir, orig_mode="rerun", allow_partial=False,
                    sumo_binary=None):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Analysis directory already exists and is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_root = output_dir / "raw" / "orig_prog"
    packs = spec["_packs"]
    arms = spec["_arms"]
    direct_arms = [arm for arm in arms if arm["kind"] == "direct_sumo"]
    run_arms = [arm for arm in arms if arm["kind"] == "run_dir"]
    direct_arm = direct_arms[0]

    cells = []
    loaded_by_pack = {}
    orig_by_pack = {}
    for pack in packs:
        cell = collect_orig_cell(
            spec, direct_arm, pack, raw_root, orig_mode=orig_mode,
            sumo_binary=sumo_binary,
        )
        loaded = cell["metrics"].get("loaded")
        if loaded is None:
            raise ValueError(f"Orig run for {pack['pack_id']} has no loaded count")
        loaded_by_pack[pack["pack_id"]] = loaded
        orig_by_pack[pack["pack_id"]] = cell
        cells.append(cell)
    for pack in packs:
        for arm in run_arms:
            cell = collect_run_cell(spec, arm, pack)
            _derive_backlog(cell, loaded_by_pack[pack["pack_id"]])
            cells.append(cell)
    for pack in packs:
        cell = orig_by_pack[pack["pack_id"]]
        _derive_backlog(cell, loaded_by_pack[pack["pack_id"]])

    expected = len(packs) * len(arms)
    if len(cells) != expected and not allow_partial:
        raise ValueError(f"Expected {expected} cells, collected {len(cells)}")
    for cell in cells:
        cell["benchmark_id"] = spec["benchmark_id"]
        cell["analysis_id"] = output_dir.name
    warnings = []
    for pack in packs:
        if loaded_by_pack[pack["pack_id"]] != pack["n_vehicles"]:
            warnings.append(
                f"{pack['pack_id']}: loaded {loaded_by_pack[pack['pack_id']]} "
                f"!= manifest n_vehicles {pack['n_vehicles']}"
            )
    dataset = {
        "schema_version": 1,
        "benchmark_id": spec["benchmark_id"],
        "analysis_id": output_dir.name,
        "created_at_utc": utc_now(),
        "tool": {"name": "tools.pack_benchmark", "version": __version__},
        "spec": {"path": _rel(spec["_path"]), "sha256": spec["_sha256"]},
        "horizon_s": spec["_horizon_s"],
        "decision_interval_s": spec["_decision_interval_s"],
        "seed": spec["_seed"],
        "episode": spec["_episode"],
        "packs": [
            {
                "pack_id": pack["pack_id"],
                "category": pack["category"],
                "n_vehicles": pack["n_vehicles"],
                "loaded": loaded_by_pack[pack["pack_id"]],
                "dir": _rel(pack["dir"]),
                "sumocfg": _rel(pack["sumocfg"]),
                "input_hashes": pack_input_hashes(pack),
            }
            for pack in packs
        ],
        "arms": [
            {
                "arm_id": arm["id"],
                "kind": arm["kind"],
                "label": arm.get("label") or arm["id"],
                "agent": arm.get("agent"),
                "expected_control_semantics": arm.get("expected_control_semantics"),
                "role": arm.get("role"),
            }
            for arm in arms
        ],
        "metric_definitions": metric_definitions(),
        "cells": cells,
        "warnings": warnings,
    }
    outputs = write_dataset(dataset, output_dir)
    collection_manifest = {
        "schema_version": 1,
        "benchmark_id": spec["benchmark_id"],
        "analysis_id": output_dir.name,
        "created_at_utc": dataset["created_at_utc"],
        "spec": {"path": _rel(spec["_path"]), "sha256": spec["_sha256"]},
        "pack_manifest": {
            "path": _rel(spec["_pack_manifest_path"]),
            "sha256": spec["_pack_manifest_sha256"],
        },
        "orig_mode": orig_mode,
        "cell_count": len(cells),
        "expected_cell_count": expected,
        "outputs": outputs,
        "warnings": warnings,
    }
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests" / "collection_manifest.json").write_text(
        json.dumps(collection_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset, output_dir
