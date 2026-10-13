"""HITL 实时服务面：``python -m hitl.serve``（等价 ``python -m hitl.live``）。"""
from .live.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
