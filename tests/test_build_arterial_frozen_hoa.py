import json
from pathlib import Path

import pytest

from tools.build_arterial_frozen_hoa import build


def _fixtures(tmp_path, valid=True):
    tasks = []; runs = []
    for scene in ("300_0.3", "300_0.6", "700_0.3", "700_0.6"):
        for seed in range(5):
            run_id = f"{scene}-{seed}"; root = tmp_path / run_id
            tasks.append({"run_id": run_id, "output_path": str(root)})
            runs.append({
                "run_id": run_id, "scene_id": scene, "collector_seed": seed,
                "episode_count": 400, "decision_count": 144000,
                "transition_count": 864000, "archive_hash": f"hash-{run_id}",
                "config_hash": "config", "git_commit": "commit", "valid": True,
            })
    queue = tmp_path / "queue.json"; audit = tmp_path / "audit.json"
    queue.write_text(json.dumps({"tasks": tasks}))
    audit.write_text(json.dumps({
        "valid": valid, "schema_consistent": True, "runs": runs,
        "schema_hashes": {"state_schema_hash": ["s"],
                            "action_schema_hash": ["a"],
                            "reward_schema_hash": ["r"]},
    }))
    return audit, queue


def test_builds_exact_scene_and_global_indexes(tmp_path):
    audit, queue = _fixtures(tmp_path)
    destination, payload = build(audit, queue, tmp_path / "hoa")
    assert destination.exists()
    assert payload["num_runs"] == 20
    assert payload["num_transitions"] == 17_280_000
    assert len(payload["scenes"]) == 4
    for scene in payload["scenes"]:
        manifest = json.loads(Path(scene["manifest_path"]).read_text())
        assert manifest["collector_training_seeds"] == [0, 1, 2, 3, 4]
        assert manifest["num_runs"] == 5
        assert manifest["num_transitions"] == 4_320_000
        assert len(scene["history_paths"]) == 5


def test_rejects_invalid_audit(tmp_path):
    audit, queue = _fixtures(tmp_path, valid=False)
    with pytest.raises(ValueError, match="valid, schema-consistent"):
        build(audit, queue, tmp_path / "hoa")
