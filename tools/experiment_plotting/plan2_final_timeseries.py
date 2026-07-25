"""Thin, resumable Plan 2 final-checkpoint decision-record reproduction.

This module does not define a new experiment protocol.  It loads the frozen
online Q-network from an existing Plan 2 evaluation checkpoint and directly
calls :meth:`TSCTrainer.evaluate_once` with the original evaluation seed.
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import logging
import os
from pathlib import Path

import torch
import yaml

import agent  # noqa: F401 - registry side effects
import agent.offline_dqn  # noqa: F401 - registry side effects
import dataset  # noqa: F401 - registry side effects
import trainer  # noqa: F401 - registry side effects
import world  # noqa: F401 - registry side effects
from common import interface
from common.registry import Registry
from trainer.offline_tsc_trainer import isolated_random_seed
from trainer.tsc_trainer import TSCTrainer
from utils.logger import hash_torch_state_dict


SUMMARY_FIELDS = (
    "travel_time", "reward_mean", "queue", "delay", "real_delay",
    "throughput", "waiting_time", "unfinished_vehicles",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def reproduce(run_dir, output_dir):
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        audit_path = output_dir / "audit.json"
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text())
            source = str(run_dir)
            checkpoint = run_dir / "checkpoints" / "evaluation" / "update_144000.pt"
            if (audit.get("status") == "completed"
                    and audit.get("source_run_dir") == source
                    and audit.get("checkpoint_sha256") == _sha256(checkpoint)
                    and int(audit.get("decision_record_count", -1)) == 360):
                return audit
        raise FileExistsError(
            f"Existing reproduction is incomplete or has different identity: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    (output_dir / "config").mkdir()
    (output_dir / "logger").mkdir()

    summary = json.loads((run_dir / "evaluation" / "summary.json").read_text())
    final_update = int(summary["final_update"])
    if final_update != 144000:
        raise ValueError(f"Expected final update 144000, got {final_update}")
    original = next(
        item for item in summary["evaluations"]
        if int(item["training_update"]) == final_update
    )
    checkpoint = (run_dir / summary["final_checkpoint"]).resolve()
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != original["checkpoint_sha256"]:
        raise ValueError("Final checkpoint hash differs from original evaluation record")

    config = yaml.safe_load((run_dir / "config" / "resolved_config.yaml").read_text())
    config.pop("config_record", None)
    command = config["command"]
    command.update({
        "task": "tsc", "dataset": "onfly",
        "output_path": str(output_dir),
        "prefix": output_dir.name, "sumo_seed": None,
    })
    config["model"].update({
        "train_model": False, "test_model": False, "load_model": False,
    })
    config["logger"]["save_model"] = False

    simulator_source = (run_dir / "config" / "simulator_source.cfg").read_bytes()
    interface.Command_Setting_Interface(config)
    interface.Logger_param_Interface(config)
    interface.World_param_Interface(config, simulator_source)
    interface.Logger_path_Interface(config)
    interface.Trainer_param_Interface(config)
    interface.ModelAgent_param_Interface(config)
    _write_json(output_dir / "run_manifest.json", {
        "config_hash": "decision_record_reproduction",
        "source_run_dir": str(run_dir),
    })

    logger = logging.getLogger(f"plan2.reproduce.{output_dir.name}")
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(output_dir / "logger" / "evaluation.log"))
    trainer_instance = None
    records = []
    try:
        trainer_instance = TSCTrainer(logger, cpu=True)
        payload = torch.load(checkpoint, map_location="cpu")
        if (payload.get("checkpoint_type") != "evaluation"
                or payload.get("training_mode") != "pure_offline"
                or int(payload.get("training_update", -1)) != final_update
                or payload.get("algorithm") != command["agent"]):
            raise ValueError("Invalid Plan 2 final evaluation checkpoint identity")
        trainer_instance.agents[0].model.load_state_dict(
            payload["online_model_state_dict"]
        )
        model_hash = hash_torch_state_dict(trainer_instance.agents[0].model.state_dict())
        if model_hash != original["online_model_state_hash"]:
            raise ValueError("Loaded online model hash differs from original evaluation")
        context = {
            "controller_id": output_dir.name,
            "agent": command["agent"], "network": command["network"],
            "training_seed": int(command["seed"]),
            "evaluation_seed": int(original["evaluation_seed"]),
            "checkpoint_episode": final_update,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "evaluation_schema_version": 1,
            "evaluation_record_type": "DECISION_RECORD_REPRODUCTION",
            "attempt_output_dir": str(output_dir),
        }
        with isolated_random_seed(int(original["evaluation_seed"])):
            reproduced = trainer_instance.evaluate_once(context, records.append)
        if len(records) != 360:
            raise ValueError(f"Expected 360 decision records, got {len(records)}")
        with (output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        comparison = {}
        for field in SUMMARY_FIELDS:
            old, new = original[field], reproduced[field]
            comparison[field] = {"original": old, "reproduced": new, "difference": new - old}
        audit = {
            "schema_version": 1, "status": "completed",
            "purpose": "decision_record_reproduction",
            "source_run_dir": str(run_dir), "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha, "final_update": final_update,
            "evaluation_seed": int(original["evaluation_seed"]),
            "decision_record_count": len(records), "comparison": comparison,
            "original_summary": original, "reproduced_summary": reproduced,
        }
        _write_json(output_dir / "audit.json", audit)
        return audit
    finally:
        if trainer_instance is not None and trainer_instance.world is not None:
            trainer_instance.world.close()
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()


def _label(dataset_kind, dataset_stage):
    if dataset_kind == "leave_one_out":
        return "non_current_full"
    return {"Q1": "only_q1", "Q4": "only_q4", "full": "full"}[dataset_stage]


def _reproduce_job(item):
    row, output = item
    audit = reproduce(row["run_dir"], output)
    return row, Path(output).name, audit


def collect(run_list, algorithm, output_root, workers=1):
    output_root = Path(output_root).resolve() / algorithm
    output_root.mkdir(parents=True, exist_ok=True)
    with Path(run_list).open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["include"].lower() == "true" and row["algorithm"] == algorithm]
    if len(rows) != 80:
        raise ValueError(f"Expected 80 {algorithm} runs, got {len(rows)}")
    jobs = []
    for row in rows:
        label = _label(row["dataset_kind"], row["dataset_stage"])
        identity = (f"{row['evaluation_network']}__{label}__"
                    f"seed{int(row['offline_training_seed'])}")
        jobs.append((row, str(output_root / identity)))
    if workers == 1:
        results = [_reproduce_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_reproduce_job, jobs))
    completed = []
    for row, identity, audit in results:
        label = _label(row["dataset_kind"], row["dataset_stage"])
        completed.append({
            "controller_id": identity, "algorithm": algorithm,
            "network": row["evaluation_network"], "policy": label,
            "offline_training_seed": int(row["offline_training_seed"]),
            "run_dir": row["run_dir"],
            "checkpoint_sha256": audit["checkpoint_sha256"],
            "evaluation_seed": audit["evaluation_seed"],
            "decision_record_count": audit["decision_record_count"],
        })
    _write_json(output_root / "collection_manifest.json", {
        "schema_version": 1, "status": "completed",
        "purpose": "plan2_final_decision_record_reproduction",
        "algorithm": algorithm, "run_list": str(Path(run_list).resolve()),
        "controller_count": len(completed),
        "decision_record_count": sum(x["decision_record_count"] for x in completed),
        "controllers": completed,
    })
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-list")
    parser.add_argument("--algorithm", choices=("batch_dqn", "cql_dqn"))
    parser.add_argument("--output-root")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.run_list:
        if not args.algorithm or not args.output_root:
            parser.error("--run-list requires --algorithm and --output-root")
        if args.workers <= 0:
            parser.error("--workers must be positive")
        completed = collect(
            args.run_list, args.algorithm, args.output_root, args.workers
        )
        print(json.dumps({"status": "completed", "controller_count": len(completed)}))
        return
    if not args.run_dir or not args.output_dir:
        parser.error("single reproduction requires --run-dir and --output-dir")
    audit = reproduce(args.run_dir, args.output_dir)
    print(json.dumps({
        "status": audit["status"],
        "decision_record_count": audit["decision_record_count"],
        "comparison": audit["comparison"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
