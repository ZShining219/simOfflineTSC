"""``python -m hitl.pipeline <stage> [args]`` — 数据面 ETL 步骤分发。

stage 自动枚举本目录全部步骤模块；直通形态
``python -m hitl.pipeline.<stage> [args]`` 同样可用。
"""
import pkgutil
import subprocess
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)


def _stages():
    return sorted(
        m.name for m in pkgutil.iter_modules([_HERE])
        if m.name not in {'__main__'} and not m.name.startswith('_')
    )


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        print('可用 stage：' + ', '.join(_stages()))
        return 0
    stage = sys.argv[1]
    if stage not in _stages():
        sys.stderr.write(
            f'未知 stage: {stage}（可用: {", ".join(_stages())}）\n'
        )
        return 2
    return subprocess.call(
        [sys.executable, '-m', f'hitl.pipeline.{stage}', *sys.argv[2:]]
    )


if __name__ == '__main__':
    raise SystemExit(main())
