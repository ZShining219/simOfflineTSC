"""Best-checkpoint frozen replay with within-episode step logging.

For every learned arm in the spec (arms carrying ``v2_arm`` and
``eval_overlay``), select the representative training seed of the
``xiasha1_online_v2`` run matrix — the seed whose ``best_travel_time`` in
``evaluation/summary.json`` is smallest — and replay its best checkpoint on
its own training pack for one frozen 3600 s episode.  The replay keeps the
training-time evaluation conditions (legacy pseudo_step, SUMO fixed_default
seed, greedy policy, no learning) and enables ``eval_step_log_interval=1``
so the per-step records can be plotted as within-episode curves.

Replay validation compares the frozen episode's FINAL_EVALUATION metrics
against the source training run's greedy-evaluation record at the same
(best) episode; identical conditions should reproduce the same values.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .collect import collect_dataset
from .spec import REPO_ROOT, run_dir_for, sha256_file
from .timeseries import _env, _read_jsonl, _timeseries_table, utc_now  # noqa: F401


def _learned_arms(spec):
    return [arm for arm in spec["_arms"] if arm.get("v2_arm")]


def select_representatives(spec):
    """Pick the representative (arm, pack) -> best-checkpoint selection.

    Representative seed = the training seed whose evaluation/summary.json
    ``best_travel_time`` is smallest (same TT-first selection semantics used
    to pick the best checkpoint itself).
    """
    source = spec.get("ckpt_eval") or {}
    v2_queue_path = REPO_ROOT / source["v2_queue"]
    v2 = json.loads(v2_queue_path.read_text(encoding="utf-8"))
    run_dirs = {
        (task["arm"], task["pack"], int(task["seed"])): Path(task["output_path"])
        for task in v2["tasks"]
    }
    selections = {}
    for arm in _learned_arms(spec):
        v2_arm = arm["v2_arm"]
        for pack in spec["_packs"]:
            pack_id = pack["pack_id"]
            candidates = []
            for task in v2["tasks"]:
                if task["arm"] != v2_arm or task["pack"] != pack_id:
                    continue
                run_dir = run_dirs[(v2_arm, pack_id, int(task["seed"]))]
                summary_path = run_dir / "evaluation" / "summary.json"
                if not summary_path.is_file():
                    continue
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                best = summary.get("best_checkpoint")
                if not best:
                    continue
                checkpoint = (run_dir / best).resolve()
                if not checkpoint.is_file():
                    continue
                candidates.append({
                    "train_seed": int(task["seed"]),
                    "best_episode": summary.get("best_episode"),
                    "best_travel_time": float(summary["best_travel_time"]),
                    "checkpoint": checkpoint,
                    "source_run_dir": run_dir,
                })
            if not candidates:
                raise FileNotFoundError(
                    f"No usable best checkpoint for arm={v2_arm} pack={pack_id}"
                )
            rep = min(candidates, key=lambda item: (item["best_travel_time"],
                                                    item["train_seed"]))
            selections[(v2_arm, pack_id)] = rep
    return selections


def write_ckpt_ts_queue(spec, queue_path):
    if Path(sys.prefix).name != "colight":
        raise SystemExit(
            "Run this generator with the colight environment so the queue "
            "commands use the correct Python interpreter."
        )
    source = spec.get("ckpt_eval") or {}
    prefix = source.get("run_prefix") or "xi2ts"
    selections = select_representatives(spec)
    tasks = []
    for arm in _learned_arms(spec):
        for pack in spec["_packs"]:
            pack_id = pack["pack_id"]
            sel = selections[(arm["v2_arm"], pack_id)]
            run_id = f"{prefix}_{arm['id']}_{pack_id}"
            output_dir = run_dir_for(spec, arm, pack)
            command = [
                sys.executable,
                "tools/eval_baseline_checkpoint.py",
                "--agent", arm["agent"],
                "--network", pack["network"],
                "--config", arm["eval_overlay"],
                "--checkpoint", str(sel["checkpoint"]),
                "--seed", str(spec["_seed"]),
                "--prefix", run_id,
            ]
            tasks.append({
                "run_id": run_id,
                "arm": arm["id"],
                "v2_arm": arm["v2_arm"],
                "pack": pack_id,
                "rep_train_seed": sel["train_seed"],
                "eval_seed": spec["_seed"],
                "sumo_seed_mode": source.get("sumo_seed_mode") or "fixed_default",
                "eval_overlay": arm["eval_overlay"],
                "checkpoint": str(sel["checkpoint"]),
                "best_episode": sel["best_episode"],
                "best_travel_time": sel["best_travel_time"],
                "source_run_dir": str(sel["source_run_dir"]),
                "command": command,
                "cwd": str(REPO_ROOT),
                "env": _env(),
                "output_path": str(output_dir),
                "completion": {
                    "path": str(output_dir / "run_status.json"),
                    "json_field": "status",
                    "equals": "已完成",
                },
            })
    queue_path = Path(queue_path)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "plan": spec["benchmark_id"],
        "expected_tasks": len(tasks),
        "note": (
            "Frozen best-checkpoint replay with 1 s eval_step_log; one "
            "representative training seed per (arm, pack), SUMO "
            "fixed_default seed, legacy pseudo_step semantics."
        ),
        "tasks": tasks,
    }
    queue_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return queue_path


def _source_eval_record(source_run_dir, best_episode):
    """Find the greedy-evaluation record at best_episode in the training run."""
    path = Path(source_run_dir) / "metrics" / "records.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing training records: {path}")
    for record in _read_jsonl(path):
        if (record.get("episode") == best_episode
                and record.get("record_type") in ("EVALUATION", "FINAL_EVALUATION")):
            return record
    raise ValueError(
        f"No evaluation record at episode {best_episode} in {path}"
    )


def _compare_frozen_replay(replay_dataset, spec, queue_path):
    """Compare each learned cell's FINAL_EVALUATION against the source
    training run's greedy-evaluation record at the selected best episode."""
    queue = json.loads(Path(queue_path).read_text(encoding="utf-8"))
    tasks = {
        (task["arm"], task["pack"]): task for task in queue["tasks"]
    }
    fields = (
        ("att_arrived_s", "travel_time"),
        ("throughput_arrived_veh", "throughput"),
        ("unfinished_departed_veh", "unfinished_vehicles"),
        ("queue_decision_mean_veh", "queue"),
        ("waiting_end_in_net_s", "waiting_time"),
        ("real_delay_world_s", "real_delay"),
    )
    learned_ids = {arm["id"] for arm in _learned_arms(spec)}
    mismatches = []
    checked = 0
    provenance = {}
    for cell in replay_dataset["cells"]:
        if cell["arm_id"] not in learned_ids:
            continue
        task = tasks.get((cell["arm_id"], cell["pack_id"]))
        if task is None:
            mismatches.append({
                "pack_id": cell["pack_id"], "arm_id": cell["arm_id"],
                "field": "task", "reference": None, "replay": "present",
            })
            continue
        reference = _source_eval_record(
            task["source_run_dir"], task["best_episode"]
        )
        summary_path = (
            REPO_ROOT / cell["run_dir"] / "baseline_eval_summary.json"
        )
        eval_summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file() else {}
        )
        provenance[f"{cell['pack_id']}__{cell['arm_id']}"] = {
            "source_run_dir": str(
                Path(task["source_run_dir"]).resolve().relative_to(REPO_ROOT)
            ),
            "rep_train_seed": task["rep_train_seed"],
            "best_episode": task["best_episode"],
            "best_travel_time": task["best_travel_time"],
            "checkpoint": eval_summary.get("checkpoint"),
            "checkpoint_sha256": eval_summary.get("checkpoint_sha256"),
            "sumo_seed_mode": eval_summary.get("sumo_seed_mode"),
        }
        for cell_field, record_field in fields:
            checked += 1
            replay_value = (cell.get("metrics") or {}).get(cell_field)
            reference_value = reference.get(record_field)
            if replay_value is None and reference_value is None:
                continue
            try:
                equal = abs(float(replay_value) - float(reference_value)) <= 1e-6
            except (TypeError, ValueError):
                equal = replay_value == reference_value
            if not equal:
                mismatches.append({
                    "pack_id": cell["pack_id"],
                    "arm_id": cell["arm_id"],
                    "field": cell_field,
                    "reference": reference_value,
                    "replay": replay_value,
                })
    return {
        "checked_values": checked,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "ckpt_eval_provenance": provenance,
    }


def collect_ckpt_timeseries(spec, output_dir, queue_path, orig_mode="rerun",
                            sumo_binary=None):
    output_dir = Path(output_dir)
    replay_dataset, output_dir = collect_dataset(
        spec, output_dir, orig_mode=orig_mode, sumo_binary=sumo_binary
    )
    validation = _compare_frozen_replay(replay_dataset, spec, queue_path)
    timeseries_path, rows, step_files = _timeseries_table(
        replay_dataset, output_dir
    )
    manifest = {
        "schema_version": 1,
        "tool": {"name": "tools.pack_benchmark", "version": __version__},
        "benchmark_id": replay_dataset["benchmark_id"],
        "analysis_id": replay_dataset["analysis_id"],
        "created_at_utc": utc_now(),
        "queue": {
            "path": str(Path(queue_path).resolve().relative_to(REPO_ROOT)),
            "sha256": sha256_file(queue_path),
        },
        "replay_dataset": {
            "path": "benchmark.json",
            "sha256": sha256_file(output_dir / "benchmark.json"),
        },
        "frozen_replay_validation": {
            "checked_values": validation["checked_values"],
            "mismatch_count": validation["mismatch_count"],
            "mismatches": validation["mismatches"],
        },
        "ckpt_eval_provenance": validation["ckpt_eval_provenance"],
        "rows": len(rows),
        "step_log_files": step_files,
        "resolution": {
            "run_dir": "1 s (eval_step_log_interval=1)",
            "direct_sumo": "10 s (decision samples)",
        },
    }
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests" / "timeseries_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "timeseries_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return replay_dataset, timeseries_path, manifest
