"""Capture a reproducible linux4090 runtime snapshot for arterial experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def run(command, *, check=False):
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=check)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "output": completed.stdout.rstrip(),
    }


def atomic_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/system_profile")
    parser.add_argument("--machine", default="linux4090")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = {
        "lscpu": run(["lscpu"]),
        "memory": run(["free", "-h"]),
        "disk": run(["df", "-h", str(Path.cwd())]),
        "swap": run(["swapon", "--show"]),
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_status": run(["git", "status", "--short"]),
        "nvidia_smi": run([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total,"
            "memory.used,utilization.gpu", "--format=csv,noheader"]),
    }
    sumo_binary = shutil.which("sumo")
    commands["sumo_version"] = (
        run([sumo_binary, "--version"]) if sumo_binary else {
            "command": ["sumo", "--version"], "exit_code": 127,
            "output": "sumo executable not found; experiments use libsumo",
        })

    git_diff = run(["git", "diff", "--binary"])["output"]
    pip_freeze = run([sys.executable, "-m", "pip", "freeze"], check=True)["output"]
    atomic_write(output_dir / "git_diff.patch", git_diff + "\n")
    atomic_write(output_dir / "pip_freeze.txt", pip_freeze + "\n")

    profile = {
        "schema_version": 1,
        "machine": args.machine,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "working_directory": str(Path.cwd().resolve()),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "packages": {
            name: distribution_version(name)
            for name in ("libsumo", "traci", "sumolib", "torch", "numpy",
                         "PyYAML", "gym", "gymnasium", "pyarrow")
        },
        "environment": {
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        },
        "commands": commands,
        "files": {
            "pip_freeze.txt": sha256(output_dir / "pip_freeze.txt"),
            "git_diff.patch": sha256(output_dir / "git_diff.patch"),
        },
    }
    atomic_write(
        output_dir / "profile.json",
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    profile_hash = sha256(output_dir / "profile.json")
    atomic_write(output_dir / "profile.sha256", profile_hash + "  profile.json\n")
    print(json.dumps({
        "profile": str(output_dir / "profile.json"),
        "sha256": profile_hash,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
