import argparse
import json
import os

from .manifest import build_formal_plan, validate_formal_plan
from .parents import import_parents, validate_parent_catalog


DEFAULT_WHITELIST = (
    'data/output_data/analysis/plan1/'
    'p1_formal_20_trajectory_whitelist_20260722.csv'
)
DEFAULT_OUTPUT = 'data/output_data/sequential/plan34_engineering_20260723'


def _parser():
    parser = argparse.ArgumentParser(description='Plan 3/4 Sequential DQN runner')
    subparsers = parser.add_subparsers(dest='command', required=True)

    import_parser = subparsers.add_parser('import-parents')
    import_parser.add_argument('--whitelist', default=DEFAULT_WHITELIST)
    import_parser.add_argument('--output-dir', default=DEFAULT_OUTPUT)
    import_parser.add_argument('--config', default='configs/sequential/plan34.yml')

    validate_parent_parser = subparsers.add_parser('validate-parents')
    validate_parent_parser.add_argument(
        '--catalog', default=os.path.join(DEFAULT_OUTPUT, 'parent_catalog.json')
    )
    validate_parent_parser.add_argument('--no-source-revalidation', action='store_true')

    build_parser = subparsers.add_parser('build-plan')
    build_parser.add_argument(
        '--parent-catalog', default=os.path.join(DEFAULT_OUTPUT, 'parent_catalog.json')
    )
    build_parser.add_argument(
        '--output', default=os.path.join(DEFAULT_OUTPUT, 'formal_60_child_manifest.json')
    )
    build_parser.add_argument('--config', default='configs/sequential/plan34.yml')

    validate_parser = subparsers.add_parser('validate')
    validate_parser.add_argument(
        '--plan', default=os.path.join(DEFAULT_OUTPUT, 'formal_60_child_manifest.json')
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == 'import-parents':
        catalog, parents = import_parents(args.whitelist, args.output_dir, args.config)
        result = {'catalog': catalog, 'parent_count': len(parents), 'valid': True}
    elif args.command == 'validate-parents':
        result = validate_parent_catalog(
            args.catalog, revalidate_sources=not args.no_source_revalidation,
        )
    elif args.command == 'build-plan':
        plan = build_formal_plan(args.parent_catalog, args.output, args.config)
        result = {
            'output': os.path.abspath(args.output),
            'child_count': plan['child_count'], 'plan_digest': plan['plan_digest'],
        }
    elif args.command == 'validate':
        result = validate_formal_plan(args.plan)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
