"""``python -m hitl.plans build --round <r> [args]`` — 正式 plan 生成器面。

round 按本目录 ``build_<round>_plan.py`` 自动枚举；新轮次新增同名文件即接入，
无需改本分发器。
"""
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

_PATTERN = re.compile(r'build_(.+)_plan')
_HERE = str(Path(__file__).resolve().parent)


def _rounds():
    rounds = {}
    for m in pkgutil.iter_modules([_HERE]):
        match = _PATTERN.fullmatch(m.name)
        if match:
            rounds[match.group(1)] = m.name
    return rounds


def main():
    rounds = _rounds()
    argv = sys.argv[1:]
    if argv[:1] != ['build'] or '--round' not in argv:
        print(__doc__)
        print('可用 round：' + (', '.join(sorted(rounds)) or '（无）'))
        return 0 if not argv else 2
    idx = argv.index('--round')
    try:
        round_name = argv[idx + 1]
    except IndexError:
        sys.stderr.write('--round 缺参数\n')
        return 2
    module = rounds.get(round_name)
    if module is None:
        sys.stderr.write(
            f'未知 round: {round_name}（可用: {", ".join(sorted(rounds))}）\n'
        )
        return 2
    forward = argv[:idx] + argv[idx + 2:]
    return subprocess.call(
        [sys.executable, '-m', f'hitl.plans.{module}', *forward[1:]]
    )


if __name__ == '__main__':
    raise SystemExit(main())
