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

# 三层：框架层（LibSignal）→ 实验基座（infra，lab 另为纯标准库的管理工具）→ 研究线；只许上层依赖下层。
CORE = ("agent", "world", "trainer", "task", "common", "utils", "generator", "data")
INFRA = ("infra",)
RESEARCH = ("semi_offline", "scene_attention", "hitl", "tools", "dev")
LEGACY = ("sequential", "plan5", "arterial")  # 2026-11-12 迁入 infra/ 与 semi_offline/ 的旧包名
LEGACY_MODULES = ("agent.offline_dqn", "trainer.offline_tsc_trainer", "task.offline_tsc_task",
                  "data.dataset.offline_trajectory_dataset")  # 同日迁入 semi_offline/offline/
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
            if root in RESEARCH + INFRA:
                found.add(f"{path} -> {root}")
    return sorted(found)


def infra_research_imports():
    return sorted(f"{path} -> {root}" for path in _tracked_py(*INFRA)
                  for root in _imported_roots(path) if root in RESEARCH)


def _imported_modules(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module
            yield from (f"{node.module}.{alias.name}" for alias in node.names)


LEGACY_PATH = re.compile(r"(?<![\w/.-])(?:sequential|plan5|arterial)/[\w/]*\.py\b|(?<![\w/.-])arterial_run\.py\b"
                         r"|(?<![\w/.-])(?:agent/offline_dqn|trainer/offline_tsc_trainer|task/offline_tsc_task"
                         r"|data/dataset/offline_trajectory_dataset|hitl/live/injects)\.py\b")


def legacy_path_strings():
    """代码字符串里的旧源码路径（迁移后应指向新位置；产物/配置目录不受影响）。"""
    found = set()
    for path in _tracked_py(*CORE, *INFRA, *RESEARCH, "lab") + ["run.py"]:
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
        # 文档字符串允许保留"自 X 迁入"这类历史说明
        docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                      and node.body and isinstance(node.body[0], ast.Expr)
                      and isinstance(node.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
                    and LEGACY_PATH.search(node.value)):
                found.add(f"{path}: {LEGACY_PATH.search(node.value).group(0)}")
    return sorted(found)


def legacy_imports():
    found = set()
    for path in _tracked_py(*CORE, *INFRA, *RESEARCH, "lab", "tests") + ["run.py"]:
        for name in _imported_modules(path):
            if name.split(".")[0] in LEGACY or name in LEGACY_MODULES:
                found.add(f"{path} -> {name}")
    return sorted(found)


def lab_internal_imports():
    return sorted(f"{path} -> {root}" for path in _tracked_py("lab")
                  for root in _imported_roots(path) if root in CORE + INFRA + RESEARCH)


def hardcoded_output_paths():
    counts = Counter()
    for path in _tracked_py(*CORE, *INFRA, *RESEARCH, "lab"):
        if path in OUTPUT_LITERAL_EXEMPT:
            continue
        hits = len(OUTPUT_LITERAL.findall((ROOT / path).read_text(encoding="utf-8")))
        if hits:
            counts[path] = hits
    return dict(sorted(counts.items()))


def absolute_paths():
    counts = Counter()
    for path in _tracked_py(*CORE, *INFRA, *RESEARCH, "lab", "tests"):
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


LINES = ("semi_offline", "scene_attention", "hitl")


def cross_line_imports():
    """研究线之间的依赖（文件级）。存量登记在基线，只许减少。"""
    found = set()
    for path in _tracked_py(*LINES):
        if "/_legacy/" in path:
            continue
        source = path.split("/")[0]
        for root in _imported_roots(path):
            if root in LINES and root != source:
                found.add(f"{path} -> {root}")
    return sorted(found)


def _baseline():
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_core_does_not_import_research_lines():
    current, allowed = set(core_research_imports()), set(_baseline()["core_research_imports"])
    assert not current - allowed, (
        "核心层新增了对研究线的依赖（LibSignal 核心不得 import 研究线包）："
        f"{sorted(current - allowed)}")
    assert not allowed - current, (
        f"这些存量违规已修复，请从 {BASELINE.name} 删除以收紧基线：{sorted(allowed - current)}")


def test_infra_does_not_import_research_lines():
    assert infra_research_imports() == [], "实验基座 infra/ 不得依赖研究线"


def test_no_imports_of_legacy_package_names():
    assert legacy_imports() == [], (
        "sequential/plan5/arterial 已迁入 infra/ 与 semi_offline/，旧名字仅供读取历史 checkpoint，"
        f"代码中请改用新路径：{legacy_imports()[:10]}")


def test_no_new_cross_line_imports():
    allowed = set(_baseline()["cross_line_imports"])
    new = sorted(set(cross_line_imports()) - allowed)
    assert not new, f"研究线之间新增了依赖（通用能力请下沉 infra/ 或框架层）：{new}"
    fixed = sorted(allowed - set(cross_line_imports()))
    assert not fixed, f"这些跨线依赖已消除，请从 {BASELINE.name} 删除以收紧基线：{fixed}"


def test_no_legacy_source_paths_in_strings():
    assert legacy_path_strings() == [], f"代码中仍引用已迁移的源码路径：{legacy_path_strings()[:10]}"


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
        "cross_line_imports": cross_line_imports(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写出 {BASELINE}")
