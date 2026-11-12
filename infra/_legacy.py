"""旧模块路径 → 新位置的解析，只为读取历史产物（pickle 中记录的类路径）。

    sequential.<io|core|checkpoint|transaction|state|launcher|evaluator>  → infra.<同名>
    sequential.offline                                                     → semi_offline.offline
    sequential[.<其余>]                                                    → semi_offline.sequential[.<其余>]
    plan5[.<x>] / arterial[.<x>]                                           → semi_offline.plan5 / semi_offline.arterial
    agent.offline_dqn / trainer.offline_tsc_trainer / task.offline_tsc_task /
    data.dataset.offline_trajectory_dataset                               → semi_offline.offline.<agent|trainer|task|dataset>
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

INFRA_MODULES = ("io", "core", "checkpoint", "transaction", "state", "launcher", "evaluator")
PACKAGES = {"sequential": "semi_offline.sequential", "plan5": "semi_offline.plan5",
            "arterial": "semi_offline.arterial"}
# 从框架层迁出的单个模块（其所在框架包仍存在，只按完整模块名精确匹配）
MOVED_MODULES = {"agent.offline_dqn": "semi_offline.offline.agent",
                 "trainer.offline_tsc_trainer": "semi_offline.offline.trainer",
                 "task.offline_tsc_task": "semi_offline.offline.task",
                 "data.dataset.offline_trajectory_dataset": "semi_offline.offline.dataset"}


def target(name):
    """旧模块名对应的新模块名；不是旧名字时返回 None。"""
    if name in MOVED_MODULES:
        return MOVED_MODULES[name]
    root, _, rest = name.partition(".")
    if root not in PACKAGES:
        return None
    if root == "sequential" and rest in INFRA_MODULES:
        return f"infra.{rest}"
    if root == "sequential" and rest == "offline":
        return "semi_offline.offline"
    return PACKAGES[root] + (f".{rest}" if rest else "")


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, new_name):
        self.new_name = new_name

    def create_module(self, spec):
        return importlib.import_module(self.new_name)  # 旧名字与新名字是同一个模块对象

    def exec_module(self, module):
        pass


class LegacyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target_module=None):
        new_name = target(fullname)
        if new_name is None:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(new_name))


def install():
    if not any(isinstance(finder, LegacyFinder) for finder in sys.meta_path):
        # 必须排在最前：否则旧包名解析到新包后，PathFinder 会沿新包 __path__ 把子模块再加载一份副本。
        # 只匹配已不存在的旧包名（sequential/plan5/arterial），不影响其他导入。
        sys.meta_path.insert(0, LegacyFinder())
