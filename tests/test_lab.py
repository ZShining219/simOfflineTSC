"""lab 基础设施层：代码溯源、台账读取口径、产物定位与 CLI。"""
import json
import subprocess

import pytest

from lab import cli, evidence, ledger, provenance


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-q", "-m", "初始")
    return tmp_path


def test_code_identity_clean_then_dirty(repo):
    clean = provenance.code_identity(repo)
    assert len(clean["commit"]) == 40
    assert clean["dirty"] is False and clean["diff_sha256"] is None

    (repo / "a.py").write_text("x = 2\n")
    (repo / "new.py").write_text("")
    dirty = provenance.code_identity(repo)
    assert dirty["commit"] == clean["commit"]
    assert dirty["dirty"] is True and dirty["dirty_file_count"] == 2
    assert dirty["diff_sha256"]


def test_code_identity_outside_git_never_raises(tmp_path):
    identity = provenance.code_identity(tmp_path)
    assert identity["commit"] is None and identity["dirty"] is None


def test_machine_alias_from_env(monkeypatch):
    monkeypatch.setenv(provenance.MACHINE_ENV, "73")
    assert provenance.machine_alias() == "73"
    monkeypatch.delenv(provenance.MACHINE_ENV)
    assert provenance.machine_alias() is None


def _write_ledger(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


def test_load_normalizes_kind_and_reports_duplicates(tmp_path):
    path = _write_ledger(tmp_path / "runs.jsonl", [
        {"run_id": "a", "run_kind": "collect", "status": "DONE"},
        {"run_id": "b", "run_kind": "train", "status": "FAILED"},
        {"run_id": "b", "run_kind": "train", "status": "DONE"},
    ])
    duplicates = []
    rows = ledger.load(path, duplicates=duplicates)
    assert rows["a"]["run_kind"] == "collection"
    assert rows["b"]["status"] == "DONE" and rows["b"]["_line_no"] == 3
    assert duplicates == [("b", 2, 3)]


@pytest.mark.parametrize("row, manifest, expected", [
    ({}, {"code": {"commit": "f" * 40}}, ledger.CODE_VERIFIED),
    ({"code_commit": "cc40585"}, None, ledger.CODE_REGISTERED),
    ({"code_commit": "73d860bb"}, None, ledger.CODE_CONSTANT),
    ({}, {"baseline_commit": ledger.CONSTANT_BASELINE_COMMIT}, ledger.CODE_CONSTANT),
    ({}, None, ledger.CODE_NONE),
])
def test_code_level(row, manifest, expected):
    assert ledger.code_level(row, manifest)[0] == expected


def test_config_mismatch():
    assert ledger.config_mismatch({"config_sha": "a"}, {"config_hash": "b"})
    assert not ledger.config_mismatch({"config_sha": "a"}, {"config_hash": "a"})
    assert not ledger.config_mismatch({"config_sha": None}, {"config_hash": "b"})


def test_locate(tmp_path, monkeypatch):
    main, other = tmp_path / "main", tmp_path / "wt"
    (main / "out" / "r1").mkdir(parents=True)
    (other / "out" / "r2").mkdir(parents=True)
    monkeypatch.setattr(evidence, "REPO_ROOT", main)
    monkeypatch.setattr(evidence, "worktree_roots", lambda: (other,))
    assert evidence.locate("out/r1") == (evidence.HERE, main / "out" / "r1")
    assert evidence.locate("out/r2") == (evidence.WORKTREE, other / "out" / "r2")
    assert evidence.locate("out/r3") == (evidence.MISSING, None)
    assert evidence.locate(str(main / "out" / "r1"))[0] == evidence.ABSOLUTE
    assert evidence.locate(None) == (evidence.NO_RUN_DIR, None)


def test_cli_audit_and_show(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "out" / "r1"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps(
        {"config_hash": "h2", "code": {"commit": "e" * 40, "dirty": False}}))
    path = _write_ledger(tmp_path / "runs.jsonl", [
        {"run_id": "r1", "campaign": "c", "status": "DONE", "run_dir": "out/r1",
         "config_sha": "h1", "metrics": {"travel_time": 30.0}},
        {"run_id": "r2", "campaign": "c", "status": "DONE", "run_dir": "out/gone"},
    ])
    monkeypatch.setattr(ledger, "LEDGER_PATH", path)
    monkeypatch.setattr(evidence, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(evidence, "worktree_roots", lambda: ())

    out_json = tmp_path / "audit.json"
    cli.main(["audit", "--json", str(out_json)])
    stats = json.loads(out_json.read_text())["campaigns"]["c"]
    assert stats["n"] == 2 and stats["missing_unflagged"] == 1
    assert stats["code"] == {"verified": 1, "none": 1}
    assert stats["config_mismatch"] == 1

    capsys.readouterr()
    cli.main(["show", "r1"])
    shown = capsys.readouterr().out
    assert "可信等级=verified" in shown and "不一致" in shown


def test_data_identity_follows_sumocfg_inputs(tmp_path):
    raw = tmp_path / "data" / "raw_data" / "net1"
    raw.mkdir(parents=True)
    (raw / "a.net.xml").write_text("<net/>")
    (raw / "a.rou.xml").write_text("<routes/>")
    (raw / "a.add.xml").write_text("<additional/>")
    (raw / "a.sumocfg").write_text(
        '<configuration><input><net-file value="a.net.xml"/>'
        '<route-files value="a.rou.xml"/><additional-files value="a.add.xml"/>'
        "</input></configuration>")
    cfg = tmp_path / "sim.cfg"
    cfg.write_text(json.dumps({"dir": "data/", "combined_file": "raw_data/net1/a.sumocfg",
                               "roadnetFile": "raw_data/net1/a.net.xml",
                               "flowFile": "raw_data/net1/a.rou.xml"}))
    first = provenance.data_identity(cfg, root=tmp_path)
    assert sorted(first["files"]) == [f"data/raw_data/net1/a.{ext}"
                                      for ext in ("add.xml", "net.xml", "rou.xml", "sumocfg")]
    (raw / "a.add.xml").write_text("<additional>changed</additional>")
    assert provenance.data_identity(cfg, root=tmp_path)["fingerprint"] != first["fingerprint"]


def test_data_identity_missing_inputs(tmp_path):
    assert provenance.data_identity(tmp_path / "absent.cfg") == {"files": {}, "fingerprint": None}
    cfg = tmp_path / "sim.cfg"
    cfg.write_text(json.dumps({"dir": "", "roadnetFile": "gone.json"}))
    assert provenance.data_identity(cfg, root=tmp_path)["files"] == {"gone.json": None}
