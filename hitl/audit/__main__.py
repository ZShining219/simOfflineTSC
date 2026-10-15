"""``python -m hitl.audit`` — 审计面卡：列出 B′判官审计件。

每个审计件直接以 ``python -m hitl.audit.<x> [args]`` 拉起。
"""
import pkgutil
from pathlib import Path


def main():
    audits = sorted(
        m.name for m in pkgutil.iter_modules([str(Path(__file__).resolve().parent)])
        if m.name != '__main__' and not m.name.startswith('_')
    )
    print("hitl.audit 判官审计面（python -m hitl.audit.<x>）：")
    for name in audits:
        print(f'  hitl.audit.{name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
