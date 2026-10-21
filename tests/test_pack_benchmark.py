import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.pack_benchmark.collect import collect_run_cell, _derive_backlog
from tools.pack_benchmark.metrics import LEGACY_ALIASES, METRICS, get_metric
from tools.pack_benchmark.plot import plot_dataset
from tools.pack_benchmark.report import validate_dataset


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_run(root):
    run_dir = Path(root) / "run"
    config_dir = run_dir / "config"
    metrics_dir = run_dir / "metrics"
    config_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    resolved = {
        "trainer": {"action_interval": 10, "yellow_length": 5},
        "model": {"name": "fixedtime", "t_fixed": 30},
    }
    resolved_path = config_dir / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    config_hash = sha256(resolved_path)
    write_json(config_dir / "config_hashes.json", {
        "algorithm": "sha256", "files": {"resolved_config.yaml": config_hash},
    })
    write_json(config_dir / "model_resolved.json", {
        "agents": [{"controller_parameters": {"t_fixed": 30}}],
    })
    write_json(run_dir / "run_manifest.json", {
        "agent": "fixedtime", "network": "xiasha1_pack_a", "prefix": "xi1_ft_pack_a",
        "training_seed": 0, "config_hash": config_hash, "run_id": "test/run",
        "sumo_seed_mode": "fixed_default",
    })
    write_json(run_dir / "run_status.json", {
        "status": "已完成", "exit_code": 0,
    })
    record = {
        "schema_version": 5, "record_type": "FINAL_EVALUATION",
        "agent": "fixedtime", "network": "xiasha1_pack_a", "training_seed": 0,
        "episode": 1, "simulation_step": 3600, "decision_step": 360,
        "travel_time": 10.0, "throughput": 80, "unfinished_vehicles": 3,
        "queue": 2.5, "waiting_time": 7.0, "real_delay": 4.0, "delay": 0.2,
        "phase_switches": 100, "phase_switch_frequency": 100 / 359,
    }
    (metrics_dir / "records.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    return run_dir


def make_pack(root):
    pack_dir = Path(root) / "pack"
    pack_dir.mkdir()
    for name in ("net.net.xml", "route.rou.xml", "add.add.xml"):
        (pack_dir / name).write_text("<xml/>\n", encoding="utf-8")
    sumocfg = pack_dir / "pack.sumocfg"
    sumocfg.write_text(
        "<configuration><input>"
        "<net-file value=\"net.net.xml\"/>"
        "<route-files value=\"route.rou.xml\"/>"
        "<additional-files value=\"add.add.xml\"/>"
        "</input></configuration>\n",
        encoding="utf-8",
    )
    return {
        "pack_id": "pack_a", "category": "test", "network": "xiasha1_pack_a",
        "dir": pack_dir, "sumocfg": sumocfg, "n_vehicles": 100,
    }


def make_dataset(root):
    cells = []
    for pack_id in ("pack_a", "pack_b"):
        for arm_id, kind in (("orig_prog", "direct_sumo"), ("fixedtime", "run_dir")):
            metrics = {
                "att_arrived_s": 10.0, "throughput_arrived_veh": 80,
                "unfinished_departed_veh": 10, "uninserted_loaded_veh": 10,
                "backlog_total_veh": 20, "queue_decision_mean_veh": 2.0,
                "waiting_end_in_net_s": 5.0, "loaded": 100,
            }
            cells.append({
                "cell_id": f"{pack_id}__{arm_id}", "pack_id": pack_id,
                "pack_category": "test", "network": f"xiasha1_{pack_id}",
                "arm_id": arm_id, "arm_kind": kind, "arm_label": arm_id,
                "control_semantics": "legacy_pseudo_step",
                "source_kind": kind, "status": "ok",
                "metrics": metrics,
            })
    return {
        "schema_version": 1, "benchmark_id": "test", "analysis_id": "test",
        "created_at_utc": "2026-10-21T00:00:00Z",
        "horizon_s": 3600, "decision_interval_s": 10, "seed": 0, "episode": 1,
        "packs": [
            {"pack_id": "pack_a"}, {"pack_id": "pack_b"},
        ],
        "arms": [
            {"arm_id": "orig_prog", "kind": "direct_sumo", "label": "orig"},
            {"arm_id": "fixedtime", "kind": "run_dir", "label": "fixedtime"},
        ],
        "metric_definitions": {key: value.to_dict() for key, value in METRICS.items()},
        "cells": cells, "warnings": [],
    }


class PackBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_metric_registry_ids_and_aliases(self):
        self.assertEqual(len(METRICS), len(set(METRICS)))
        for alias, metric_id in LEGACY_ALIASES.items():
            self.assertIn(metric_id, METRICS, alias)
            get_metric(metric_id)

    def test_collect_run_cell_maps_record_and_warns_on_legacy_timing(self):
        run_dir = make_run(self.root)
        pack = make_pack(self.root)
        spec = {"_seed": 0, "_episode": 1, "_horizon_s": 3600,
                "_decision_interval_s": 10}
        arm = {
            "id": "fixedtime", "kind": "run_dir", "agent": "fixedtime",
            "label": "fixedtime", "expected_control_semantics": "legacy_pseudo_step",
            "run_dir_template": str(run_dir),
        }
        cell = collect_run_cell(spec, arm, pack)
        self.assertEqual(10.0, cell["metrics"]["att_arrived_s"])
        self.assertEqual(80, cell["metrics"]["throughput_arrived_veh"])
        self.assertEqual(2.5, cell["metrics"]["queue_decision_mean_veh"])
        self.assertEqual(7.0, cell["metrics"]["waiting_end_in_net_s"])
        self.assertEqual(4.0, cell["metrics"]["real_delay_world_s"])
        self.assertEqual("legacy_pseudo_step", cell["control_semantics"])
        self.assertTrue(any("legacy pseudo_step" in note for note in cell["warnings"]))
        _derive_backlog(cell, loaded=100)
        self.assertEqual(17, cell["metrics"]["uninserted_loaded_veh"])
        self.assertEqual(20, cell["metrics"]["backlog_total_veh"])
        self.assertTrue(run_dir.is_dir())

    def test_validate_dataset_accepts_complete_accounting_and_rejects_negative(self):
        dataset = make_dataset(self.root)
        validation = validate_dataset(dataset)
        self.assertEqual("PASS", validation["status"], validation["errors"])
        dataset["cells"][0]["metrics"]["uninserted_loaded_veh"] = -1
        validation = validate_dataset(dataset)
        self.assertEqual("FAIL", validation["status"])
        self.assertTrue(any("negative uninserted" in error for error in validation["errors"]))

    def test_plot_dataset_writes_five_figures_and_captions(self):
        dataset = make_dataset(self.root)
        figure_manifest = plot_dataset(dataset, self.root, dpi=40)
        self.assertEqual(5, len(figure_manifest["figures"]))
        self.assertTrue((self.root / "captions.md").is_file())
        for figure in figure_manifest["figures"]:
            for path in figure["paths"]:
                self.assertTrue((self.root / path).is_file(), path)


if __name__ == "__main__":
    unittest.main()
