"""台账指标标准口径（最佳 checkpoint 评估）与旧公式识别。"""
import json

import pytest

from lab import metrics


def _run(tmp_path, evals, *, offline=False, records=None, final=None):
    run = tmp_path / "run"
    (run / "evaluation").mkdir(parents=True)
    step = "training_update" if offline else "episode"
    entries = [{step: i, "travel_time": tt, "throughput": 100 + i, "unfinished_vehicles": 1,
                "queue": 0.5, "delay": 0.1} for i, tt in enumerate(evals)]
    best = min(entries, key=lambda e: (e["travel_time"], e[step]))
    summary = {"evaluations": entries, "best_travel_time": best["travel_time"],
               f"best_{'update' if offline else 'episode'}": best[step],
               "best_checkpoint": f"ckpt_{best[step]}.pt",
               "selection_rule": "minimum_then_earliest",
               "final_travel_time": final if final is not None else evals[-1]}
    (run / "evaluation" / "summary.json").write_text(json.dumps(summary))
    if records is not None:
        (run / "metrics").mkdir()
        (run / "metrics" / "records.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records))
    return str(run)


@pytest.mark.parametrize("offline", [False, True])
def test_best_checkpoint_matches_trainer_rule(tmp_path, offline):
    run = _run(tmp_path, [40.0, 35.0, 35.0, 38.0], offline=offline)
    values, detail = metrics.best_checkpoint(run)
    assert values["travel_time"] == 35.0 and values["throughput"] == 101  # 并列取最早
    assert detail["step"] == 1 and detail["checkpoint"] == "ckpt_1.pt"


def test_best_checkpoint_absent(tmp_path):
    assert metrics.best_checkpoint(str(tmp_path)) is None
    assert metrics.best_checkpoint(None) is None


def test_legacy_formulas(tmp_path):
    evals = [50.0 - i for i in range(25)]
    records = []
    for i, tt in enumerate(evals):
        records.append({"record_type": "TRAIN", "travel_time": tt + 5, "throughput": 1})
        records.append({"record_type": "EVALUATION", "travel_time": tt, "throughput": 1})
    run = _run(tmp_path, evals, records=records, final=30.123)
    eval_mean = sum(evals[-20:]) / 20
    mixed_mean = sum(r["travel_time"] for r in records[-20:]) / 20
    assert metrics.legacy_formula(min(evals), run) == metrics.STANDARD
    assert metrics.legacy_formula(30.1234, run) == "final_eval"   # 3 位小数舍入仍可识别
    assert metrics.legacy_formula(eval_mean, run) == "eval_last20"
    assert metrics.legacy_formula(mixed_mean, run) == "mixed_last20"
    assert metrics.legacy_formula(99.0, run) == "unknown"


def test_standardize_keeps_legacy_values(tmp_path):
    run = _run(tmp_path, [40.0, 35.0, 38.0])
    row = {"run_kind": "train", "status": "DONE", "metrics": {"travel_time": 38.0}}
    update = metrics.standardize(row, run)
    assert update["metric_source"] == metrics.STANDARD
    assert update["metrics"]["travel_time"] == 35.0
    assert update["metrics_legacy"] == {"travel_time": 38.0}
    assert update["metric_source_legacy"] == "legacy:final_eval"
    assert metrics.verify({**row, **update}, run)
    assert not metrics.verify({**row, **update, "metrics": {"travel_time": 36.0}}, run)


def test_standardize_eval_rows_and_unreachable_runs(tmp_path):
    run = _run(tmp_path, [40.0, 35.0])
    eval_row = {"run_kind": "eval", "status": "DONE", "metrics": {"travel_time": 41.0}}
    assert metrics.standardize(eval_row, run) == {"metric_source": metrics.EVAL_ATTEMPT}
    missing = {"run_kind": "train", "status": "DONE", "metrics": {"travel_time": 41.0}}
    assert metrics.standardize(missing, None) == {"metric_source": "legacy:unknown"}
    assert metrics.standardize({"run_kind": "train", "status": "DONE"}, None) == {}
