import argparse
import json
import os

from .manifest import (
    build_formal_plan, build_hybrid_plan, build_pilot_plan,
    validate_formal_plan, validate_hybrid_plan,
)
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

    hybrid_build_parser = subparsers.add_parser('build-hybrid-plan')
    hybrid_build_parser.add_argument(
        '--parent-catalog', default=os.path.join(DEFAULT_OUTPUT, 'parent_catalog.json')
    )
    hybrid_build_parser.add_argument('--output', required=True)
    hybrid_build_parser.add_argument('--config', default='configs/sequential/plan34_b100.yml')
    hybrid_build_parser.add_argument('--condition', choices=('M0', 'M1', 'M2', 'M3'), default='M3')

    hybrid_validate_parser = subparsers.add_parser('validate-hybrid-plan')
    hybrid_validate_parser.add_argument('--plan', required=True)

    validate_parser = subparsers.add_parser('validate')
    validate_parser.add_argument(
        '--plan', default=os.path.join(DEFAULT_OUTPUT, 'formal_60_child_manifest.json')
    )
    validate_parser.add_argument('--output-root', default=None)

    analyze_parser = subparsers.add_parser('analyze')
    analyze_parser.add_argument('--manifest', required=True)
    analyze_parser.add_argument('--output-root', required=True)
    analyze_parser.add_argument(
        '--parent-catalog', default=os.path.join(DEFAULT_OUTPUT, 'parent_catalog.json')
    )
    analyze_parser.add_argument('--output', required=True)
    analyze_parser.add_argument(
        '--orders', nargs='+', default=None,
        help='Optional complete order subset (for example: O1 O2).',
    )
    analyze_parser.add_argument(
        '--validation-workers', type=int, default=1,
        help='Parallel workers for independent run validation.',
    )
    cross_parser = subparsers.add_parser('analyze-budgets')
    cross_parser.add_argument('--b100-report', required=True)
    cross_parser.add_argument('--b400-report', required=True)
    cross_parser.add_argument('--output', required=True)

    snapshot_parser = subparsers.add_parser('snapshot-parent')
    snapshot_parser.add_argument('--parent-import', required=True)
    snapshot_parser.add_argument('--output', required=True)

    evaluate_parser = subparsers.add_parser('evaluate-cell')
    evaluate_parser.add_argument('--snapshot', required=True)
    evaluate_parser.add_argument('--network', required=True)
    evaluate_parser.add_argument('--output-root', required=True)
    evaluate_parser.add_argument('--stage-index', type=int, required=True)
    evaluate_parser.add_argument('--training-network', required=True)
    evaluate_parser.add_argument('--local-episode', type=int, required=True)
    evaluate_parser.add_argument('--global-episode', type=int, required=True)
    evaluate_parser.add_argument('--interface', choices=('libsumo', 'traci'), default='libsumo')
    evaluate_parser.add_argument('--timeout-seconds', type=int, default=300)
    evaluate_parser.add_argument('--evaluation-seed', type=int, default=None)
    evaluate_parser.add_argument('--controller-id', default=None)
    evaluate_parser.add_argument('--agent-label', default='dqn')
    evaluate_parser.add_argument('--training-seed', type=int, default=None)

    launch_parser = subparsers.add_parser('launch')
    launch_parser.add_argument('--manifest', required=True)
    launch_parser.add_argument('--output-root', required=True)
    launch_parser.add_argument('--logical-run-id', action='append', default=None)
    launch_parser.add_argument('--authorize-formal', action='store_true')
    launch_parser.add_argument('--max-child', type=int, choices=(8, 6, 4), default=8)

    status_parser = subparsers.add_parser('status')
    status_parser.add_argument('--output-root', required=True)

    resume_parser = subparsers.add_parser('resume-failed')
    resume_parser.add_argument('--manifest', required=True)
    resume_parser.add_argument('--output-root', required=True)
    resume_parser.add_argument('--authorize-formal', action='store_true')
    resume_parser.add_argument('--max-child', type=int, choices=(8, 6, 4), default=8)

    child_parser = subparsers.add_parser('run-child')
    child_parser.add_argument('--manifest', required=True)
    child_parser.add_argument('--logical-run-id', required=True)
    child_parser.add_argument('--attempt-dir', required=True)
    child_parser.add_argument('--resume', default=None)
    child_parser.add_argument('--resume-state', default=None)

    pilot_parser = subparsers.add_parser('build-pilot')
    pilot_parser.add_argument(
        '--parent-catalog', default=os.path.join(DEFAULT_OUTPUT, 'parent_catalog.json')
    )
    pilot_parser.add_argument('--output', required=True)
    pilot_parser.add_argument('--later-stage-episodes', type=int, default=15)
    pilot_parser.add_argument('--config', default='configs/sequential/plan34.yml')

    reproduction_parser = subparsers.add_parser('compare-plan1-reproduction')
    reproduction_parser.add_argument('--source-run', required=True)
    reproduction_parser.add_argument('--reproduction-run', required=True)
    reproduction_parser.add_argument('--network', required=True)
    reproduction_parser.add_argument('--training-seed', type=int, default=0)
    reproduction_parser.add_argument('--output', required=True)

    fault_parser = subparsers.add_parser('pilot-fault')
    fault_parser.add_argument('--manifest', required=True)
    fault_parser.add_argument('--output-root', required=True)
    fault_parser.add_argument('--logical-run-id', required=True)
    fault_parser.add_argument('--timeout-seconds', type=int, default=3600)
    fault_parser.add_argument('--max-child', type=int, choices=(8, 6, 4), default=4)

    calibration_parser = subparsers.add_parser('compare-evaluator-calibration')
    calibration_parser.add_argument('--legacy', required=True)
    calibration_parser.add_argument('--new-alias', required=True)
    calibration_parser.add_argument('--output', required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    exit_code = 0
    if args.command == 'import-parents':
        catalog, parents = import_parents(args.whitelist, args.output_dir, args.config)
        result = {'catalog': catalog, 'parent_count': len(parents), 'valid': True}
    elif args.command == 'build-hybrid-plan':
        result = build_hybrid_plan(
            args.parent_catalog, args.output, args.config, args.condition,
        )
    elif args.command == 'validate-hybrid-plan':
        result = validate_hybrid_plan(args.plan)
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
        if args.output_root:
            from .validation import validate_experiment_outputs
            result = validate_experiment_outputs(args.plan, args.output_root)
        else:
            result = validate_formal_plan(args.plan)
    elif args.command == 'analyze':
        from .analysis import analyze_experiment
        result = analyze_experiment(
            args.manifest, args.output_root, args.parent_catalog, args.output,
            selected_orders=args.orders, validation_workers=args.validation_workers,
        )
    elif args.command == 'analyze-budgets':
        from .analysis import analyze_cross_budget
        result = analyze_cross_budget(args.b100_report, args.b400_report, args.output)
    elif args.command == 'snapshot-parent':
        import torch
        from .evaluator import save_online_state_snapshot
        from .io import read_json

        parent = read_json(args.parent_import)
        checkpoint = torch.load(parent['checkpoint_path'], map_location='cpu')
        payload = save_online_state_snapshot(
            checkpoint['agents'][0]['online_model_state_dict'], {
                'input_dim': parent['dimensions']['model_input_dim'],
                'output_dim': parent['dimensions']['action_dim'],
                'phase': True, 'one_hot': True,
            }, args.output, {
                'source_parent_import': os.path.abspath(args.parent_import),
                'stage_index': 1,
                'global_episode': int(parent['checkpoint_episode']),
            },
        )
        result = {
            'output': os.path.abspath(args.output),
            'checkpoint_digest': payload['online_parameter_digest'],
        }
    elif args.command == 'evaluate-cell':
        from .config import simulator_config_path
        from .evaluator import IndependentEvaluator

        evaluator = IndependentEvaluator(
            args.output_root, retries=3,
            timeout_seconds=args.timeout_seconds,
        )
        result = evaluator.evaluate(
            args.snapshot, args.network, {
                'simulator_config': simulator_config_path(args.network),
                'interface': args.interface,
                'steps': 3600, 'action_interval': 10,
                'sumo_seed_mode': (
                    'fixed_default' if args.evaluation_seed is None else 'explicit'
                ),
                'evaluation_seed': args.evaluation_seed,
            }, {
                'stage_index': args.stage_index,
                'training_network': args.training_network,
                'evaluation_network': args.network,
                'local_episode': args.local_episode,
                'global_episode': args.global_episode,
                'controller_id': args.controller_id,
                'agent': args.agent_label,
                'training_seed': args.training_seed,
            },
        )
    elif args.command == 'launch':
        from .launcher import SequentialLauncher
        launcher = SequentialLauncher(
            args.manifest, args.output_root,
            initial_concurrency=args.max_child,
        )
        result = {
            'runs': launcher.launch(
                logical_run_ids=args.logical_run_id,
                authorize_formal=args.authorize_formal,
            )
        }
    elif args.command == 'status':
        from .launcher import collect_status
        result = collect_status(args.output_root)
    elif args.command == 'resume-failed':
        from .launcher import (
            AttemptLineage, SequentialLauncher, collect_status,
            is_auto_recoverable,
        )
        statuses = collect_status(args.output_root)['runs']
        eligible = []
        for run in statuses:
            latest = run['attempts'][-1] if run['attempts'] else None
            if latest is None or latest['status'] not in ('failed', 'interrupted'):
                continue
            lineage = AttemptLineage(args.output_root, run['logical_run_id'])
            recovery = lineage.latest_recovery_checkpoint()
            if is_auto_recoverable(latest.get('failure_class'), recovery):
                eligible.append(run['logical_run_id'])
        launcher = SequentialLauncher(
            args.manifest, args.output_root,
            initial_concurrency=args.max_child,
        )
        result = {
            'eligible': eligible,
            'runs': launcher.launch(
                logical_run_ids=eligible,
                authorize_formal=args.authorize_formal,
            ) if eligible else [],
        }
    elif args.command == 'run-child':
        from .runtime import run_child
        result = run_child(
            args.manifest, args.logical_run_id, args.attempt_dir,
            resume_path=args.resume, resume_state_path=args.resume_state,
        )
        exit_code = int(result.get('exit_code', 0))
    elif args.command == 'build-pilot':
        plan = build_pilot_plan(
            args.parent_catalog, args.output,
            later_stage_episodes=args.later_stage_episodes,
            config_path=args.config,
        )
        result = {
            'output': os.path.abspath(args.output),
            'child_count': plan['child_count'],
            'plan_digest': plan['plan_digest'],
            'later_stage_episodes': args.later_stage_episodes,
        }
    elif args.command == 'compare-plan1-reproduction':
        from .reproduction import compare_plan1_reproduction
        report = compare_plan1_reproduction(
            args.source_run, args.reproduction_run, args.network,
            output_path=args.output, training_seed=args.training_seed,
        )
        result = {
            'valid': report['valid'], 'network': args.network,
            'output': os.path.abspath(args.output), 'checks': report['checks'],
        }
    elif args.command == 'pilot-fault':
        from .fault_harness import run_fault_recovery
        result = run_fault_recovery(
            args.manifest, args.output_root, args.logical_run_id,
            timeout_seconds=args.timeout_seconds, max_child=args.max_child,
        )
    elif args.command == 'compare-evaluator-calibration':
        from .calibration import compare_evaluator_calibration
        report = compare_evaluator_calibration(
            args.legacy, args.new_alias, args.output,
        )
        result = {
            'valid': report['valid'], 'network': report['network'],
            'output': os.path.abspath(args.output),
            'action_sequence_equal': report['action_sequence_equal'],
            'metric_checks': report['metric_checks'],
        }
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code
