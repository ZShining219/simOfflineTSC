"""架构棘轮：存量违规登记在 tests/architecture_baseline.json，只许减少不许新增。

修掉一处存量违规后，同步从基线中删去对应条目（测试会提示），基线即随之收紧。
重新生成基线：python tests/test_architecture.py --write-baseline（仅在有意收紧时使用）。
"""
import ast
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # 以脚本方式（--write-baseline）运行时也能 import lab
    sys.path.insert(0, str(ROOT))
BASELINE = Path(__file__).with_name("architecture_baseline.json")

CORE = ("agent", "world", "trainer", "task", "common", "utils", "generator", "data")
RESEARCH = ("arterial", "sequential", "scene_attention", "hitl", "plan5", "tools", "dev")
OUTPUT_LITERAL = re.compile(r"""['"](?:\./)?(?:data/output_data|analysis/|final_result/|/tmp/)""")
OUTPUT_LITERAL_EXEMPT = {"lab/paths.py", "lab/checks.py"}  # 规则定义本身
ABSOLUTE_PATH = re.compile(r"""['"]/(?:projects|home)/""")
ABSOLUTE_PATH_EXEMPT = {"lab/checks.py", "tests/test_lab_checks.py", "tests/test_architecture.py"}


def _tracked_py(*prefixes):
    # 含尚未 git add 的新文件（--others），否则新文件在提交前躲过检查。
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard",
                          "--", *[f"{p}/*.py" for p in prefixes]],
                         cwd=ROOT, capture_output=True, check=True).stdout.decode()
    return [p for p in out.split("\0") if p and (ROOT / p).exists()]


def _imported_roots(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


def core_research_imports():
    found = set()
    for path in _tracked_py(*CORE):
        for root in _imported_roots(path):
            if root in RESEARCH:
                found.add(f"{path} -> {root}")
    return sorted(found)


def lab_internal_imports():
    return sorted(f"{path} -> {root}" for path in _tracked_py("lab")
                  for root in _imported_roots(path) if root in CORE + RESEARCH)


def hardcoded_output_paths():
    counts = Counter()
    for path in _tracked_py(*CORE, *RESEARCH, "lab"):
        if path in OUTPUT_LITERAL_EXEMPT:
            continue
        hits = len(OUTPUT_LITERAL.findall((ROOT / path).read_text(encoding="utf-8")))
        if hits:
            counts[path] = hits
    return dict(sorted(counts.items()))


def absolute_paths():
    counts = Counter()
    for path in _tracked_py(*CORE, *RESEARCH, "lab", "tests"):
        if path in ABSOLUTE_PATH_EXEMPT:
            continue
        hits = len(ABSOLUTE_PATH.findall((ROOT / path).read_text(encoding="utf-8")))
        if hits:
            counts[path] = hits
    return dict(sorted(counts.items()))


def root_config_overlays():
    from lab import configs
    agents = configs.agent_names()
    return sorted(p.name for p in configs.TSC_DIR.glob("*.yml") if p.stem not in agents)


def _baseline():
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_core_does_not_import_research_lines():
    current, allowed = set(core_research_imports()), set(_baseline()["core_research_imports"])
    assert not current - allowed, (
        "核心层新增了对研究线的依赖（LibSignal 核心不得 import 研究线包）："
        f"{sorted(current - allowed)}")
    assert not allowed - current, (
        f"这些存量违规已修复，请从 {BASELINE.name} 删除以收紧基线：{sorted(allowed - current)}")


def test_lab_is_dependency_free():
    assert lab_internal_imports() == [], "lab/ 只能依赖标准库与 lab 自身"


def test_no_new_hardcoded_output_paths():
    allowed = _baseline()["hardcoded_output_paths"]
    grown = {p: n for p, n in hardcoded_output_paths().items() if n > allowed.get(p, 0)}
    assert not grown, (
        "新增了硬编码产物路径；请改用 lab.paths.output_root()/report_dir()/queue_dir()："
        f"{grown}（基线允许数：{ {p: allowed.get(p, 0) for p in grown} }）")


def test_no_new_absolute_paths():
    allowed = _baseline()["absolute_paths"]
    grown = {p: n for p, n in absolute_paths().items() if n > allowed.get(p, 0)}
    assert not grown, f"新增了本机绝对路径（公网仓库禁止，GOVERNANCE §10）：{grown}"


def test_configs_index_is_current():
    from lab import configs
    assert configs.README.read_text(encoding="utf-8") == configs.render(), (
        "configs/tsc/README.md 与实际文件不一致，运行 python -m lab configs --write")


def test_no_new_root_config_overlays():
    allowed = set(_baseline()["configs_root_overlays"])
    new = sorted(set(root_config_overlays()) - allowed)
    assert not new, f"新 overlay 须放 configs/tsc/<campaign>/，不得放根目录：{new}"


def test_dev_dirs_registered():
    index = (ROOT / "dev" / "INDEX.md").read_text(encoding="utf-8")
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "dev"],
                         cwd=ROOT, capture_output=True, check=True).stdout.decode()
    dirs = {p.split("/")[1] for p in out.split("\0") if p.count("/") >= 2}
    missing = sorted(d for d in dirs if f"dev/{d}/" not in index and not d.startswith("_"))
    assert not missing, f"dev/ 子目录须在 dev/INDEX.md 登记用途与所属 campaign：{missing}"


def test_root_entries_registered():
    from lab.checks import ROOT_ALLOWLIST
    out = subprocess.run(["git", "-c", "core.quotepath=off", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, check=True).stdout.decode()
    roots = {p.split("/", 1)[0] for p in out.split("\0") if p}
    assert roots <= ROOT_ALLOWLIST, f"仓根有未登记条目：{sorted(roots - ROOT_ALLOWLIST)}"


if __name__ == "__main__" and "--write-baseline" in sys.argv:
    BASELINE.write_text(json.dumps({
        "core_research_imports": core_research_imports(),
        "hardcoded_output_paths": hardcoded_output_paths(),
        "absolute_paths": absolute_paths(),
        "configs_root_overlays": root_config_overlays(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写出 {BASELINE}")
