#!/usr/bin/env bash
# P1 条款检查（GOVERNANCE §8.5）：仓内不得有符号链接条目；merge 前用 --tree <ref> 检查对方分支。
set -u
hit=$(git ls-files -s | awk '$1 == "120000"')
if [ -z "$hit" ]; then hit=$(git ls-tree -r HEAD | awk '$1 == "120000"'); fi
if [ -n "$hit" ]; then echo "FAIL: 仓内存在符号链接条目:"; echo "$hit"; exit 1; fi
if [ "${1:-}" = "--tree" ] && [ -n "${2:-}" ]; then
  hit=$(git ls-tree -r "$2" | awk '$1 == "120000"')
  if [ -n "$hit" ]; then echo "FAIL: 分支 $2 含符号链接条目（merge 会物化覆盖 ignored 目录）:"; echo "$hit"; exit 1; fi
fi
echo "OK: no symlink entries"
