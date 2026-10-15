"""``python -m hitl.probe`` — 探针面卡：列出可用探针模块。

每个探针直接以 ``python -m hitl.probe.<x> [args]`` 拉起。
"""
import pkgutil
from pathlib import Path


def main():
    probes = sorted(
        m.name for m in pkgutil.iter_modules([str(Path(__file__).resolve().parent)])
        if m.name != '__main__' and not m.name.startswith('_')
    )
    print('hitl.probe 探针面（python -m hitl.probe.<x>）：')
    for name in probes:
        print(f'  hitl.probe.{name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
