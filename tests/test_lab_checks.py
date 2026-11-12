"""提交闸门：每条规则在临时仓库里实际制造一次违规，确认能拦住。"""
import os
import subprocess

import pytest

from lab import checks


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "lab").mkdir()
    (tmp_path / "lab" / "ok.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    return tmp_path


def _stage(repo, path, content=b"x\n"):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    _git(repo, "add", "-f", path)


def test_clean_stage_passes(repo):
    assert checks.check_staged(repo) == []


def test_symlink_blocked(repo):
    os.symlink("lab", repo / "lab" / "link")
    _git(repo, "add", "lab/link")
    assert any("符号链接" in p for p in checks.check_staged(repo))


@pytest.mark.parametrize("path, needle", [
    ("data/output_data/tsc/run/x.json", "产物区"),
    ("artifacts/q/run_queue.json", "产物区"),
    ("lab/model.pt", "产物类型"),
    ("stray_notes.md", "仓根"),
])
def test_path_rules(repo, path, needle):
    _stage(repo, path)
    assert any(needle in p for p in checks.check_staged(repo))


def test_wiki_claims_readme_allowed(repo):
    _stage(repo, "artifacts/wiki_claims/README.md")
    assert checks.check_staged(repo) == []


def test_size_limits(repo):
    _stage(repo, "lab/big.json", b"0" * (checks.MAX_BYTES + 1))
    _stage(repo, "data/raw_data/net/roadnet.json", b"0" * (checks.MAX_BYTES + 1))
    problems = checks.check_staged(repo)
    assert any("lab/big.json" in p for p in problems)
    assert not any("roadnet.json" in p for p in problems)


@pytest.mark.parametrize("line", ["host = '10.21.237.34'", "root = '/home/dev/x'",
                                  "p = '/projects/simOfflineTSC/run'"])
def test_sensitive_content_blocked(repo, line):
    _stage(repo, "lab/conf.py", (line + "\n").encode())
    assert checks.check_staged(repo)


def test_relative_paths_and_localhost_allowed(repo):
    _stage(repo, "lab/conf.py", b"p = 'data/output_data/tsc'\nurl = 'http://127.0.0.1:6009'\n")
    assert checks.check_staged(repo) == []


@pytest.mark.parametrize("text, ok", [
    ("基础设施：新增闸门\n\n细节", True),
    ("# comment\nfix stuff", False),
    ("", False),
])
def test_commit_message(text, ok):
    assert (checks.check_commit_message(text) == []) is ok


def test_renamed_file_legacy_content_not_flagged(repo):
    _stage(repo, "lab/old.py", b"x = 1\nroot = '/home/dev/legacy'\n" + b"y = 2\n" * 20)
    _git(repo, "commit", "-q", "-m", "init", "--no-verify")
    _git(repo, "mv", "lab/old.py", "lab/new.py")
    assert checks.check_staged(repo) == []
    (repo / "lab" / "new.py").write_text("x = 1\nroot = '/home/dev/legacy'\n" + "y = 2\n" * 20
                                          + "z = '/projects/added'\n")
    _git(repo, "add", "lab/new.py")
    problems = checks.check_staged(repo)
    assert len(problems) == 1 and "/projects/" in problems[0]


def test_ledger_invariants_checked_when_staged(repo):
    import json
    good = {"run_id": "a", "metrics": {"travel_time": 1.0}, "metric_source": "best_ckpt",
            "run_dir": "data/output_data/x"}
    _stage(repo, "ledger/runs.jsonl", (json.dumps(good) + "\n").encode())
    assert checks.check_ledger(repo, ["ledger/runs.jsonl"]) == []
    bad = [good, {**good, "metric_source": None}, {"run_id": "b", "run_dir": "/abs/x"}]
    _stage(repo, "ledger/runs.jsonl", "".join(json.dumps(r) + "\n" for r in bad).encode())
    problems = checks.check_ledger(repo, ["ledger/runs.jsonl"])
    assert any("重复" in p for p in problems) and any("口径" in p for p in problems)
    assert any("相对路径" in p for p in problems)
    assert checks.check_ledger(repo, ["lab/ok.py"]) == []
