"""统一登记（lab register）与回填（lab harvest）。"""
import json

from lab import evidence, ledger, registry


def _queue(tmp_path, tasks, campaign="demo_v1"):
    path = tmp_path / "run_queue.json"
    path.write_text(json.dumps({"campaign": campaign, "tasks": tasks}))
    return path


def _task(run_id, seed=0):
    return {"run_id": run_id, "arm": "dqn", "seed": seed, "command": [
        "python", "run.py", "-w", "sumo", "-a", "dqn", "-n", "sumohz1x1", "--seed", str(seed),
        "--sumo_seed", "7", "--prefix", run_id, "--experiment-config", "configs/tsc/x/dqn.yml"]}


def test_register_is_idempotent_and_leaves_provenance_empty(tmp_path):
    led = tmp_path / "runs.jsonl"
    led.write_text(json.dumps({"run_id": "old", "campaign": "demo_v1"}) + "\n")
    queue = _queue(tmp_path, [_task("old"), _task("r1", seed=3)])
    new, skipped = registry.register(queue, apply=True, ledger_path=led, notes="意图说明")
    assert skipped == ["old"] and [r["run_id"] for r in new] == ["r1"]
    row = ledger.load(led)["r1"]
    assert row["status"] == "REGISTERED" and row["net"] == "sumohz1x1" and row["seed"] == 3
    assert row["sumo_seed"] == 7 and row["run_dir"] == "data/output_data/tsc/sumo_dqn/sumohz1x1/r1"
    assert row["code_commit"] is None and row["config_sha"] is None  # 运行时再如实回填
    assert "意图说明" in row["notes"] and list(row)[:3] == ["run_id", "line", "campaign"]
    assert registry.register(queue, apply=True, ledger_path=led)[0] == []


def _run_dir(root, status="已完成"):
    run = root / "data" / "output_data" / "tsc" / "sumo_dqn" / "sumohz1x1" / "r1"
    (run / "evaluation").mkdir(parents=True)
    (run / "run_status.json").write_text(json.dumps({
        "status": status, "started_at_utc": "2026-11-12T00:00:00Z",
        "finished_at_utc": "2026-11-12T00:10:00Z", "error_message": "boom" if status == "失败" else None}))
    (run / "run_manifest.json").write_text(json.dumps({
        "config_hash": "c" * 64, "machine": "73",
        "code": {"commit": "a" * 40, "dirty": False}, "data": {"fingerprint": "d" * 64}}))
    entries = [{"episode": 0, "travel_time": 50.0, "throughput": 10, "unfinished_vehicles": 2,
                "queue": 1.0, "delay": 0.2},
               {"episode": 1, "travel_time": 40.0, "throughput": 12, "unfinished_vehicles": 1,
                "queue": 0.8, "delay": 0.1}]
    (run / "evaluation" / "summary.json").write_text(json.dumps({
        "evaluations": entries, "best_episode": 1, "best_travel_time": 40.0,
        "best_checkpoint": "checkpoints/evaluation/episode_0001.pt",
        "selection_rule": "minimum_then_earliest_episode", "final_travel_time": 40.0}))
    return run


def test_harvest_fills_runtime_provenance_and_standard_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(evidence, "worktree_roots", lambda: ())
    led = tmp_path / "runs.jsonl"
    led.write_text("")
    registry.register(_queue(tmp_path, [_task("r1")]), apply=True, ledger_path=led)
    rows = ledger.read_lines(led)
    rows[0]["config_sha"] = "registered-guess"
    ledger.write_lines(rows, led)
    _run_dir(tmp_path)

    changes = registry.harvest("demo_v1", apply=True, ledger_path=led)
    row = ledger.load(led)["r1"]
    assert set(changes) == {"r1"}
    assert row["status"] == "DONE" and row["code_commit"] == "a" * 40 and row["dirty"] is False
    assert row["config_sha"] == "c" * 64 and row["config_sha_registered"] == "registered-guess"
    assert row["data_fingerprint"] == "d" * 64 and row["machine"] == "73"
    assert row["metric_source"] == "best_ckpt" and row["metrics"]["travel_time"] == 40.0
    assert row["started"] == "2026-11-12T00:00:00Z"
    assert registry.harvest("demo_v1", apply=True, ledger_path=led) == {}  # 幂等


def test_harvest_records_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(evidence, "worktree_roots", lambda: ())
    led = tmp_path / "runs.jsonl"
    led.write_text("")
    registry.register(_queue(tmp_path, [_task("r1")]), apply=True, ledger_path=led)
    _run_dir(tmp_path, status="失败")
    registry.harvest("demo_v1", apply=True, ledger_path=led)
    row = ledger.load(led)["r1"]
    assert row["status"] == "FAILED" and row["error"] == "boom" and "metric_source" not in row
