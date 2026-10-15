#!/usr/bin/env python3
"""兼容 shim：Plan 2 纯离线入口已迁入 sequential 包。

正式用法：`python -m sequential.offline <子命令>`。本文件仅作过渡兼容，
将在后续版本移除（见 GOVERNANCE §3.1）。
"""
import sys

from sequential.offline import main

if __name__ == '__main__':
    sys.stderr.write(
        '[deprecation] offline_run.py 已迁移为 python -m sequential.offline；'
        '本文件仅作兼容，后续将移除\n'
    )
    raise SystemExit(main())
