"""兼容层：实现已迁至 common/history_replay.py（2026-11-12 整理，消除核心层对研究线的依赖）。

`arterial.replay` 与 `common.history_replay` 是同一个模块对象，旧的 import 与属性修改行为完全不变。
新代码请直接 import common.history_replay。
"""
import sys

from common import history_replay as _module

sys.modules[__name__] = _module
