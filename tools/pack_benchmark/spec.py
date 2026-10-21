"""Load and validate a benchmark spec."""
import hashlib
import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_benchmark_spec(path):
    spec_path = _resolve(path).resolve()
    with spec_path.open(encoding="utf-8") as handle:
        spec = yaml.safe_load(handle) or {}
    if spec.get("schema_version") != 1:
        raise ValueError(f"Unsupported benchmark spec schema: {spec.get('schema_version')}")
    for key in ("benchmark_id", "horizon_s", "decision_interval_s", "seed",
                "episode", "pack_manifest", "pack_dir_template", "arms"):
        if key not in spec:
            raise ValueError(f"Benchmark spec is missing required key: {key}")
    horizon = int(spec["horizon_s"])
    interval = int(spec["decision_interval_s"])
    if horizon <= 0 or interval <= 0 or horizon % interval:
        raise ValueError("horizon_s must be positive and divisible by decision_interval_s")

    manifest_path = _resolve(spec["pack_manifest"]).resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        pack_manifest = json.load(handle)
    packs = []
    for entry in pack_manifest.get("packs", []):
        pack_dir = _resolve(spec["pack_dir_template"].format(**entry)).resolve()
        if not pack_dir.is_dir():
            raise FileNotFoundError(f"Pack directory does not exist: {pack_dir}")
        sumocfg = pack_dir / Path(entry["sumocfg"]).name
        if not sumocfg.is_file():
            raise FileNotFoundError(f"Pack sumocfg does not exist: {sumocfg}")
        packs.append({
            "pack_id": entry["pack"],
            "category": entry.get("cat"),
            "n_vehicles": int(entry["n_vehicles"]),
            "dir": pack_dir,
            "sumocfg": sumocfg,
            "network": f"xiasha1_{entry['pack']}",
        })
    if not packs:
        raise ValueError("Pack manifest has no packs")

    arms = list(spec["arms"])
    ids = [arm["id"] for arm in arms]
    if len(ids) != len(set(ids)):
        raise ValueError("Benchmark arm ids must be unique")
    direct = [arm for arm in arms if arm.get("kind") == "direct_sumo"]
    if len(direct) != 1:
        raise ValueError("Benchmark spec must have exactly one direct_sumo arm")
    for arm in arms:
        if arm.get("kind") not in {"direct_sumo", "run_dir"}:
            raise ValueError(f"Unknown arm kind: {arm.get('kind')}")
        if arm["kind"] == "run_dir":
            if not arm.get("run_dir_template") or not arm.get("agent"):
                raise ValueError(f"run_dir arm {arm['id']} needs run_dir_template and agent")
    spec["_path"] = spec_path
    spec["_sha256"] = sha256_file(spec_path)
    spec["_pack_manifest_path"] = manifest_path
    spec["_pack_manifest_sha256"] = sha256_file(manifest_path)
    spec["_packs"] = packs
    spec["_arms"] = arms
    spec["_horizon_s"] = horizon
    spec["_decision_interval_s"] = interval
    spec["_seed"] = int(spec["seed"])
    spec["_episode"] = int(spec["episode"])
    spec["_in_edges"] = tuple(spec.get("in_edges") or ())
    spec["_out_edges"] = tuple(spec.get("out_edges") or ())
    return spec


def run_dir_for(spec, arm, pack):
    return _resolve(
        arm["run_dir_template"].format(
            pack=pack["pack_id"], network=pack["network"]
        )
    ).resolve()
