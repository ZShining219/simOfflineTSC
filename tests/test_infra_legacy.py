"""历史 checkpoint 兼容：旧模块路径（sequential/plan5/arterial）解析到 infra/ 与 semi_offline/。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(code):
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip().splitlines()[-1]


def test_target_mapping():
    from infra import _legacy
    assert _legacy.target("sequential.core") == "infra.core"
    assert _legacy.target("sequential.offline") == "semi_offline.offline"
    assert _legacy.target("sequential.ha_agent") == "semi_offline.sequential.ha_agent"
    assert _legacy.target("sequential") == "semi_offline.sequential"
    assert _legacy.target("plan5.config") == "semi_offline.plan5.config"
    assert _legacy.target("arterial") == "semi_offline.arterial"
    assert _legacy.target("semi_offline.sequential") is None and _legacy.target("infra.core") is None


def test_legacy_names_alias_the_same_module_objects():
    out = _run("""
import importlib, infra
import semi_offline.plan5.config as real, infra.core as core
legacy = importlib.import_module("plan5.config")
import semi_offline.plan5.config as again
print(legacy is real, again is real, importlib.import_module("sequential.core") is core,
      real.__name__ == "semi_offline.plan5.config")
""")
    assert out == "True True True True"


def test_unpickle_object_saved_under_legacy_path():
    out = _run("""
import pickle, infra
from infra.core import ReplayMetadata
import dataclasses
assert dataclasses.is_dataclass(ReplayMetadata)
obj = ReplayMetadata(**{f.name: f"v_{f.name}" for f in dataclasses.fields(ReplayMetadata)})
legacy_bytes = pickle.dumps(obj, protocol=0).replace(b"infra.core\\n", b"sequential.core\\n")
assert b"sequential.core" in legacy_bytes
back = pickle.loads(legacy_bytes)
print(type(back).__module__, back == obj)
""")
    assert out == "infra.core True"
