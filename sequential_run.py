#!/usr/bin/env python3
"""兼容 shim：sequential 入口已迁入包门面。

正式用法：`python -m sequential <子命令>`。本文件仅作过渡兼容，
将在后续版本移除（见 GOVERNANCE §3.1）。
"""
import sys

from sequential.cli import main

if __name__ == '__main__':
    sys.stderr.write(
        '[deprecation] sequential_run.py 已迁移为 python -m sequential；'
        '本文件仅作兼容，后续将移除\n'
    )
    raise SystemExit(main())
