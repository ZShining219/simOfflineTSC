import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.experiment_plotting.cli import (
    _write_g0_stop_report, build_parser, run_plan1,
)
from tools.experiment_plotting.aggregations import (
    aggregate_cross_scene, aggregate_episode100_pair_gate, calculate_policy_disagreement,
    calculate_state_distribution_diagnostics, select_transition_pairs,
)
from tools.experiment_plotting.loaders import load_run_list
from tools.experiment_plotting.profiles import get_profile
from tools.experiment_plotting.validators import validate_run
from utils.logger import METRIC_FIELDS


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_run(root, name, role="pilot", network="n1", seed=0):
    run_dir = Path(root) / name
    config_dir = run_dir / "config"
    metrics_dir = run_dir / "metrics"
    evaluation_dir = run_dir / "evaluation"
    trajectory_dir = run_dir / "trajectory"
    for directory in (config_dir, metrics_dir, evaluation_dir, trajectory_dir):
        directory.mkdir(parents=True)

    resolved_config = {
        "command": {
            "task": "tsc", "world": "sumo", "agent": "dqn",
            "network": network, "prefix": name, "seed": seed,
            "interface": "libsumo", "delay_type": "apx",
        },
        "world": {
            "interval": 1.0, "combined_file": f"{network}.sumocfg",
            "roadnetFile": f"{network}.net.xml", "flowFile": f"{network}.rou.xml",
            "convertroadnetFile": f"{network}.road.json",
            "convertflowFile": f"{network}.flow.json", "gui": False,
        },
        "trainer": {"episodes": 100, "batch_size": 64},
        "model": {"gamma": 0.95},
        "logger": {"schema": 2},
        "config_record": {
            "created_at_utc": "2026-07-22T00:00:00Z",
            "sources": [
                "configs/tsc/base.yml", "configs/tsc/dqn.yml",
                f"configs/sim/{network}.cfg",
            ],
        },
    }
    with open(config_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False)
    runtime = {
        "schema_version": 1,
        "agent": "dqn",
        "reproducibility_probe": {"python_random": [seed]},
        "agents": [{
            "rank": 0, "action_dim": 2,
            "online_model_state_hash": str(seed), "target_model_state_hash": str(seed),
            "model": {"input_dim": 4, "hidden_layers": [20, 20]},
            "target_model": {"input_dim": 4, "hidden_layers": [20, 20]},
            "optimizer": {"class": "RMSprop"}, "loss": {"class": "MSELoss"},
        }],
    }
    write_json(config_dir / "model_resolved.json", runtime)
    hashes = {}
    for path in config_dir.iterdir():
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(config_dir / "config_hashes.json", {"algorithm": "sha256", "files": hashes})
    config_hash = hashlib.sha256((config_dir / "resolved_config.yaml").read_bytes()).hexdigest()

    write_json(run_dir / "run_manifest.json", {
        "schema_version": 1, "run_id": name, "task": "tsc", "world": "sumo",
        "agent": "dqn", "network": network, "prefix": name,
        "training_seed": seed, "sumo_seed_mode": "fixed_default",
        "baseline_commit": "a" * 40, "created_at_utc": "2026-07-22T00:00:00Z",
        "config_hash": config_hash,
    })
    write_json(run_dir / "run_status.json", {
        "schema_version": 1, "run_id": name, "status": "已完成",
        "started_at_utc": "2026-07-22T00:00:00Z",
        "finished_at_utc": "2026-07-22T00:01:00Z", "exit_code": 0,
        "error_type": None, "error_message": None,
    })

    records = []
    for episode, record_type in ((1, "TRAIN"), (10, "TRAIN"), (10, "FINAL_EVALUATION")):
        record = {field: None for field in METRIC_FIELDS}
        record.update({
            "schema_version": 2, "record_type": record_type,
            "agent": "dqn", "network": network, "training_seed": seed,
            "episode": episode, "simulation_step": 3600, "decision_step": 360,
            "global_decision_step": episode * 360, "gradient_updates": episode,
            "travel_time": 100.0 - episode, "reward_mean": -10.0 + episode,
            "reward_sum": -3600.0, "queue": 20.0 - episode,
            "delay": 0.3, "real_delay": 30.0, "throughput": 1000 + episode,
            "loss_mean": 2.0, "epsilon": 0.1, "wall_time_seconds": 1.0,
            "waiting_time": 15.0, "unfinished_vehicles": 4,
            "action_distribution": {"0": 0.6, "1": 0.4},
            "phase_switches": 50, "phase_switch_frequency": 0.14,
            "replay_size": 5000, "replay_capacity": 5000, "target_updates": 2,
        })
        records.append(record)
    with open(metrics_dir / "records.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    write_json(evaluation_dir / "summary.json", {
        "schema_version": 1, "selection_metric": "travel_time",
        "selection_rule": "minimum_then_earliest_episode", "evaluations": [records[-1]],
        "final_episode": 10, "final_travel_time": 90.0,
        "final_checkpoint": "checkpoints/evaluation/episode_0010.pt",
        "best_episode": 10, "best_travel_time": 90.0,
        "best_checkpoint": "checkpoints/evaluation/episode_0010.pt",
    })
    if role in {"pilot", "formal"}:
        episodes_dir = trajectory_dir / "episodes"
        episodes_dir.mkdir()
        shard = episodes_dir / "episode_0001.npz"
        shard.write_bytes(b"test trajectory shard")
        shard_hash = hashlib.sha256(shard.read_bytes()).hexdigest()
        with open(trajectory_dir / "index.jsonl", "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "schema_version": 1, "episode_id": 1,
                "file": "episodes/episode_0001.npz", "transition_count": 36000,
                "first_global_step": 1, "last_global_step": 36000,
                "sha256": shard_hash,
            }, sort_keys=True) + "\n")
        write_json(trajectory_dir / "manifest.json", {
            "schema_version": 1, "storage": "episode_npz", "network": network,
            "scene_id": network, "behavior_training_seed": seed,
            "sumo_seed_mode": "fixed_default", "config_hash": config_hash,
            "expected_decisions_per_episode": 360, "action_dim": 2,
        })
        write_json(trajectory_dir / "validation.json", {
            "schema_version": 1, "valid": True, "errors": [],
            "episode_count": 1, "transition_count": 36000,
            "expected_decisions_per_episode": 360,
            "evaluation_transition_count": 0,
        })
    return run_dir


def write_run_list(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "role", "agent", "network", "training_seed", "run_dir", "include",
        ))
        writer.writeheader()
        writer.writerows(rows)


class Arguments:
    command = "plan1"
    dpi = 50


class Plan1PlottingTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_s1_s4_integrated_diagnostic_aggregations(self):
        cross_raw = []
        for source in ('S1', 'S2', 'S3', 'S4'):
            for target in ('S1', 'S2', 'S3', 'S4'):
                for seed in range(5):
                    degradation = abs(int(source[1]) - int(target[1]))
                    cross_raw.append({
                        'source_scene': source, 'source_network': f'n{source[1]}',
                        'target_scene': target, 'target_network': f'n{target[1]}',
                        'training_seed': seed, 'travel_time': 100 + degradation,
                        'queue': 10 + degradation, 'real_delay': 20 + degradation,
                        'approximate_delay': .2 + degradation / 100,
                        'throughput': 1000 - degradation,
                        'phase_switch_frequency': .2, 'reward_mean': -10 - degradation,
                        'reward_definition': 'synthetic',
                    })
        summary, relative, directional = aggregate_cross_scene(cross_raw)
        self.assertEqual(16 * 7, len(summary))
        self.assertEqual(16 * 6, len(relative))
        self.assertEqual(12, len(directional))
        self.assertTrue(all(row['relative_degradation'] == 0
                            for row in relative if row['source_scene'] == row['target_scene']))
        self.assertTrue(all(row['absolute_difference'] == 0
                            for row in relative if row['source_scene'] == row['target_scene']))
        self.assertTrue(all(row['near_zero_reference'] is False for row in relative))

        predictions = []
        for scene in ('S1', 'S2', 'S3', 'S4'):
            for seed in range(5):
                for probe_index in range(8):
                    top = (int(scene[1]) + probe_index) % 8
                    q_values = [0.0] * 8
                    q_values[top] = 2.0
                    predictions.append({
                        'probe_id': f'p{probe_index}', 'probe_source_scene': 'S1',
                        'model_id': f'{scene}_seed{seed}', 'model_scene': scene,
                        'model_network': f'n{scene[1]}', 'training_seed': seed,
                        'top_action': top, 'normalized_q_margin': 1.0,
                        'q_values': q_values,
                    })
        model_pairs, scene_pairs, threshold = calculate_policy_disagreement(predictions)
        self.assertEqual(190 * 2, len(model_pairs))
        self.assertEqual(.1, threshold)
        all_scene = [row for row in scene_pairs
                     if row['probe_source_scene'] == 'ALL'
                     and row['metric'] == 'action_disagreement']
        self.assertEqual(16, len(all_scene))

        probes = []
        for scene in ('S1', 'S2', 'S3', 'S4'):
            for sample in range(8):
                values = [float(int(scene[1]) + sample)] * 8 + [1.0] + [0.0] * 7
                probes.append({'scene': scene, 'model_input': values})
        stats, state_distances = calculate_state_distribution_diagnostics(probes)
        self.assertEqual(64, len(stats))
        self.assertEqual(16, len(state_distances))

        action_rows = []
        for left in ('S1', 'S2', 'S3', 'S4'):
            for right in ('S1', 'S2', 'S3', 'S4'):
                action_rows.append({
                    'aggregation': 'scene_mean', 'metric': 'total_variation',
                    'source_scene': left, 'target_scene': right,
                    'distance': abs(int(left[1]) - int(right[1])) / 10,
                })
        evidence, selected = select_transition_pairs(
            relative, action_rows, scene_pairs, state_distances
        )
        self.assertEqual(6, len(evidence))
        self.assertTrue(all('action_total_variation_excess_over_within' in row
                            for row in evidence))
        self.assertTrue(all(row['source_scene'] != row['target_scene'] for row in selected))

    def test_plan1_analysis_validates_normalizes_and_renders_both_formats(self):
        first = write_run(self.root, "run_a", network="n1", seed=0)
        second = write_run(self.root, "run_b", network="n2", seed=1)
        run_list = self.root / "runs.csv"
        write_run_list(run_list, [{
            "role": "pilot", "agent": "dqn", "network": "n1",
            "training_seed": 0, "run_dir": first, "include": "true",
        }, {
            "role": "pilot", "agent": "dqn", "network": "n2",
            "training_seed": 1, "run_dir": second, "include": "true",
        }])
        args = Arguments()
        args.run_list = str(run_list)
        args.analysis_id = "test_analysis"
        args.output_root = str(self.root / "analysis")
        output = run_plan1(args)
        self.assertTrue((output / "plotting_manifest.json").is_file())
        self.assertTrue((output / "inputs" / "runs.csv").is_file())
        self.assertTrue((output / "tables" / "metrics.csv").is_file())
        self.assertTrue((output / "tables" / "first_100_auc.csv").is_file())
        self.assertTrue((output / "tables" / "learning_speed.csv").is_file())
        self.assertTrue((output / "tables" / "efficiency.csv").is_file())
        self.assertTrue((output / "tables" / "aulc.csv").is_file())
        self.assertTrue((output / "figures" / "results" / "final_travel_time.png").is_file())
        self.assertTrue((output / "figures" / "results" / "final_travel_time.pdf").is_file())
        self.assertTrue((output / "figures" / "diagnostics" / "individual_travel_time.png").is_file())
        with open(output / "plotting_manifest.json", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual(2, manifest["included_run_count"])
        self.assertTrue(manifest["config_compatible"])
        self.assertEqual(2, manifest["schema_version"])
        self.assertEqual("plan1", manifest["plot_profile"])
        self.assertTrue(manifest["figure_groups"]["results"])
        self.assertTrue(manifest["figure_groups"]["diagnostics"])
        result_stems = {
            Path(path).stem for path in manifest["figure_groups"]["results"]
        }
        diagnostic_stems = {
            Path(path).stem for path in manifest["figure_groups"]["diagnostics"]
        }
        self.assertEqual({
            "final_travel_time", "evaluation_learning_curve",
            "learning_speed_auc", "action_concentration", "training_cost",
            "travel_time_vs_transitions", "travel_time_vs_gradient_updates",
            "travel_time_vs_wall_time", "replay_fill_fraction",
            "update_to_data_ratio", "raw_travel_time_aulc",
            "normalized_control_aulc",
        }, result_stems)
        self.assertIn("action_distribution", diagnostic_stems)
        self.assertIn("individual_travel_time", diagnostic_stems)
        with open(output / "tables" / "first_100_auc.csv", newline="", encoding="utf-8") as handle:
            auc_rows = list(csv.DictReader(handle))
        self.assertEqual({"TRAIN", "EVALUATION"}, {
            row["curve_source"] for row in auc_rows
        })

    def test_generic_analyze_command_keeps_named_profile_explicit(self):
        args = build_parser().parse_args([
            "analyze", "--profile", "plan1", "--run-list", "runs.csv",
            "--analysis-id", "analysis_1",
        ])
        self.assertEqual("analyze", args.command)
        self.assertEqual("plan1", args.profile)
        self.assertEqual("runs.csv", args.run_list)
        self.assertEqual(
            "sumohz1x1_config2", get_profile("plan1").network_order[0],
        )

    def test_duplicate_and_failed_runs_are_rejected(self):
        run_dir = write_run(self.root, "run_a")
        run_list = self.root / "runs.csv"
        row = {
            "role": "pilot", "agent": "dqn", "network": "n1",
            "training_seed": 0, "run_dir": run_dir, "include": "true",
        }
        write_run_list(run_list, [row, row])
        with self.assertRaisesRegex(ValueError, "Duplicate run_dir"):
            load_run_list(run_list)

        write_json(run_dir / "run_status.json", {
            "schema_version": 1, "run_id": "run_a", "status": "失败",
            "started_at_utc": None, "finished_at_utc": None, "exit_code": 1,
            "error_type": "RuntimeError", "error_message": "failed",
        })
        write_run_list(run_list, [row])
        _, _, included = load_run_list(run_list)
        with self.assertRaisesRegex(ValueError, "did not complete"):
            validate_run(included[0])

    def test_corrupt_config_and_missing_pilot_trajectory_are_rejected(self):
        run_dir = write_run(self.root, "run_a")
        run_list = self.root / "runs.csv"
        row = {
            "role": "pilot", "agent": "dqn", "network": "n1",
            "training_seed": 0, "run_dir": run_dir, "include": "true",
        }
        write_run_list(run_list, [row])
        _, _, included = load_run_list(run_list)
        os.unlink(run_dir / "trajectory" / "validation.json")
        with self.assertRaises(FileNotFoundError):
            validate_run(included[0])

        run_dir = write_run(self.root, "run_b")
        (run_dir / "config" / "resolved_config.yaml").write_text("changed\n", encoding="utf-8")
        row["run_dir"] = run_dir
        write_run_list(run_list, [row])
        _, _, included = load_run_list(run_list)
        with self.assertRaisesRegex(IOError, "verification failed"):
            validate_run(included[0])

    def test_formal_requires_v2_and_v2_rejects_incomplete_evaluations(self):
        run_dir = write_run(self.root, "formal_old", role="formal")
        run_list = self.root / "runs.csv"
        row = {
            "role": "formal", "agent": "dqn", "network": "n1",
            "training_seed": 0, "run_dir": run_dir, "include": "true",
        }
        write_run_list(run_list, [row])
        _, _, included = load_run_list(run_list)
        with self.assertRaisesRegex(ValueError, "requires evaluation summary schema v2"):
            validate_run(included[0])

        config_path = run_dir / "config" / "resolved_config.yaml"
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["trainer"].update({
            "episodes": 10, "steps": 3600, "action_interval": 10,
            "evaluation_episodes": list(range(11)),
            "resumable_checkpoint_episodes": [0, 10],
        })
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        manifest_path = run_dir / "run_manifest.json"
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["config_hash"] = config_hash
        write_json(manifest_path, manifest)
        config_dir = run_dir / "config"
        hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in config_dir.iterdir() if path.name != "config_hashes.json"
        }
        write_json(config_dir / "config_hashes.json", {
            "algorithm": "sha256", "files": hashes,
        })
        summary_path = run_dir / "evaluation" / "summary.json"
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["schema_version"] = 2
        write_json(summary_path, summary)
        with self.assertRaisesRegex(ValueError, "TRAIN episodes are incomplete"):
            validate_run(included[0])

    def test_episode100_gate_uses_same_seed_reference_and_metric_direction(self):
        network = {'S2': 'n2', 'S3': 'n3', 'S4': 'n4'}
        directions = (('S3', 'S4'), ('S4', 'S3'),
                      ('S2', 'S3'), ('S3', 'S2'))
        gaps = {('S3', 'S4'): .15, ('S4', 'S3'): .12,
                ('S2', 'S3'): .02, ('S3', 'S2'): .03}
        rows = []
        for scene in ('S2', 'S3', 'S4'):
            for seed in range(5):
                baseline = 100.0 + seed
                rows.append({
                    'source_scene': scene, 'source_network': network[scene],
                    'target_scene': scene, 'target_network': network[scene],
                    'training_seed': seed, 'travel_time': baseline,
                    'queue': 10.0, 'real_delay': 20.0,
                    'approximate_delay': 2.0, 'throughput': 1000.0,
                    'unfinished_vehicles': 5.0,
                })
        for source, target in directions:
            for seed in range(5):
                baseline = 100.0 + seed
                gap = gaps[(source, target)]
                rows.append({
                    'source_scene': source, 'source_network': network[source],
                    'target_scene': target, 'target_network': network[target],
                    'training_seed': seed, 'travel_time': baseline * (1 + gap),
                    'queue': 10.0 * (1 + gap),
                    'real_delay': 20.0 * (1 + gap),
                    'approximate_delay': 2.0 * (1 + gap),
                    'throughput': 1000.0 * (1 - gap),
                    'unfinished_vehicles': 5.0 * (1 + gap),
                })
        summary, relative, gate = aggregate_episode100_pair_gate(rows, directions)
        self.assertTrue(gate['passed'])
        self.assertAlmostEqual(12.0, gate[
            'high_minus_low_worst_percentage_points'])
        s3_s4_seed4 = next(
            row for row in relative
            if row['source_scene'] == 'S3' and row['target_scene'] == 'S4'
            and row['training_seed'] == 4 and row['metric'] == 'travel_time'
        )
        self.assertEqual(104.0, s3_s4_seed4['same_seed_target_reference'])
        throughput = next(
            row for row in relative
            if row['source_scene'] == 'S3' and row['target_scene'] == 'S4'
            and row['training_seed'] == 0 and row['metric'] == 'throughput'
        )
        self.assertAlmostEqual(.15, throughput['relative_degradation'])
        self.assertEqual(24, len(summary))

    def test_failed_g0_writes_terminal_evidence_without_sequential_placeholders(self):
        summary = []
        relative = []
        gaps = {('S3', 'S4'): .0276, ('S4', 'S3'): .0535,
                ('S2', 'S3'): .0224, ('S3', 'S2'): .0015}
        for source, target in gaps:
            summary.append({
                'source_scene': source, 'target_scene': target,
                'metric': 'travel_time',
                'mean_relative_degradation': gaps[(source, target)],
                'sample_sd_relative_degradation': .01,
            })
            for seed in range(5):
                relative.append({
                    'source_scene': source, 'target_scene': target,
                    'metric': 'travel_time', 'training_seed': seed,
                    'relative_degradation': gaps[(source, target)],
                })
        controllers = []
        summaries = []
        for source, target in gaps:
            for seed in range(5):
                controller_id = f'{source}_{target}_{seed}'
                controllers.append({
                    'controller_id': controller_id,
                    'source_scene': source, 'target_scene': target,
                })
                summaries.append({
                    'controller_id': controller_id, 'training_seed': seed,
                    'evaluation_traffic_seed': 10000,
                })
        package = {
            'manifest': {'episode_count': 20, 'decision_record_count': 7200},
            'collection': {'controllers': controllers}, 'summaries': summaries,
        }
        gate = {
            'passed': False,
            'high_minus_low_worst_percentage_points': 3.11,
        }
        args = type('Args', (), {
            'evaluation_package': str(self.root / 'immutable_package')
        })()
        _write_g0_stop_report(
            self.root, args, package, summary, relative, gate
        )
        final_report = (self.root / 'reports' / 'g_final_report.md').read_text(
            encoding='utf-8'
        )
        self.assertIn('结论分类：E', final_report)
        self.assertIn('G1、smoke test 和 40 个正式 sequential runs 均未执行',
                      final_report)
        with (self.root / 'run_status.csv').open(newline='', encoding='utf-8') as handle:
            status = list(csv.DictReader(handle))
        self.assertEqual(20, sum(row['status'] == 'completed' for row in status))
        self.assertEqual(3, sum(row['status'] == 'not_run' for row in status))
        self.assertTrue((self.root / 'commands' / 'reproduction_commands.sh').exists())
        self.assertFalse((self.root / 'manifests' / 'sequential_manifest.json').exists())
        self.assertFalse((self.root / 'processed' / 'sequential_raw.csv').exists())

        gate['passed'] = True
        passed_output = self.root / 'passed'
        _write_g0_stop_report(
            passed_output, args, package, summary, relative, gate
        )
        self.assertFalse((passed_output / 'reports' / 'g_final_report.md').exists())


if __name__ == "__main__":
    unittest.main()
