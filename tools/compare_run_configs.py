#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.run_config_compare import compare_runs


def main():
    parser = argparse.ArgumentParser(
        description='Compare archived run configs using the Milestone 0 closed allowlist.'
    )
    parser.add_argument('run_paths', nargs='+')
    args = parser.parse_args()
    result = compare_runs(args.run_paths)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result['compatible'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
