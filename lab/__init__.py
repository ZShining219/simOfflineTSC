"""实验基础设施层：台账读取、产物定位、代码溯源与核查命令。

分层约定：本包只依赖标准库，不 import 任何 LibSignal 核心模块或研究线包；
核心层（agent/world/trainer/utils/common …）与研究线可以 import 本包。
命令行入口：``python -m lab``。
"""
