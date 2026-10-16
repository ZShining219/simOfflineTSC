"""Explicit opt-in entry for paper_* reward variants and persistent monitoring.

Accepts the same CLI arguments as run.py; -a must be paper_frap,
paper_presslight or paper_colight. Frozen run.py/agents/trainer are not edited.
Example (from project root, colight environment):

    python -m tools.run_paper_baseline -w sumo -a paper_frap -n hz4x4 \
      --seed 7 --prefix paper_frap_smoke \
      --experiment-config configs/tsc/paper_monitor_smoke.yml
"""
import os
from pathlib import Path
import runpy
import signal
import sys
import argparse
import json
import shutil
from types import SimpleNamespace


def cli():
    """One permanent interface for planning, scheduling and worker execution."""
    parser = argparse.ArgumentParser(description='Reusable paper-baseline experiment lifecycle')
    sub = parser.add_subparsers(dest='operation', required=True)
    prepare = sub.add_parser('plan', help='Freeze YAML, scenario assets and all task commands; does not launch')
    prepare.add_argument('--config', required=True)
    prepare.add_argument('--output', required=True)
    extend = sub.add_parser('extend', help='Freeze a new training stage from completed resumable checkpoints')
    extend.add_argument('--config', required=True)
    extend.add_argument('--output', required=True)
    extend.add_argument('--source-manifest', action='append', required=True)
    evaluate = sub.add_parser('evaluate-frozen', help='Plan final evaluation of completed runs under an explicitly audited runtime repair')
    evaluate.add_argument('--source-manifest', required=True)
    evaluate.add_argument('--output', required=True)
    evaluate.add_argument('--role', choices=('validation', 'test'), default='test')
    evaluate.add_argument('--run-id', action='append', help='Only these completed runs (repeatable)')
    evaluate.add_argument('--checkpoint', type=int, help='Only this published checkpoint')
    inspect = sub.add_parser('config', help='Resolve/validate an experiment profile and show its budget; no simulation')
    inspect.add_argument('--config', required=True)
    preflight = sub.add_parser('preflight', help='Run development-seed physical event gate; does not train')
    preflight.add_argument('--config', required=True)
    preflight.add_argument('--output', required=True)
    preflight.add_argument('--workers', type=int, default=4)
    for name in ('launch', 'resume'):
        command = sub.add_parser(name)
        command.add_argument('--manifest', required=True)
        command.add_argument('--train-workers', type=int, default=4)
        command.add_argument('--eval-workers', type=int, default=2)
        command.add_argument('--test-workers', type=int, default=8)
        command.add_argument('--max-workers', type=int, help='Optional shared cap, e.g. 1 for serial verification')
    status = sub.add_parser('status')
    status.add_argument('--manifest', required=True)
    worker = sub.add_parser('worker', help='Internal manifest task, also useful for controlled single-task verification')
    worker.add_argument('--manifest', required=True)
    worker.add_argument('--task-id', required=True)
    worker.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    from common.paper_experiment import plan, preflight as prepare_preflight, describe_config, load_manifest, write_json, read_json
    if args.operation == 'config':
        print(json.dumps(describe_config(args.config), ensure_ascii=False, indent=2))
        return 0
    if args.operation == 'plan':
        print(plan(args.config, args.output))
        return 0
    if args.operation == 'extend':
        from common.paper_experiment import plan_extension
        print(plan_extension(args.config, args.output, args.source_manifest))
        return 0
    if args.operation == 'evaluate-frozen':
        from common.paper_experiment import plan_frozen_evaluation
        print(plan_frozen_evaluation(args.source_manifest, args.output, role=args.role,
                                    run_ids=args.run_id, checkpoint=args.checkpoint))
        return 0
    if args.operation == 'preflight':
        if args.workers < 1:
            raise ValueError('workers must be positive')
        args.manifest = str(prepare_preflight(args.config, args.output))
        args.train_workers = args.eval_workers = args.test_workers = args.max_workers = args.workers
        args.operation = 'launch'
    manifest = load_manifest(args.manifest, verify=args.operation != 'status')
    root = Path(manifest['root'])
    if args.operation == 'status':
        state = root / 'queue' / 'status.json'
        value = read_json(state) if state.exists() else {'status': 'planned'}
        value.update(protocol_hash=manifest['protocol_hash'], budget=manifest['budget'],
                     planned_tasks=len(manifest['tasks']))
        assessment = root / 'preflight_summary.json'
        if assessment.exists():
            value['preflight_admission'] = read_json(assessment)['admission']
        print(json.dumps(value, indent=2, ensure_ascii=False))
        return 0
    if args.operation == 'worker':
        task = next((t for t in manifest['tasks'] if t['run_id'] == args.task_id), None)
        if task is None:
            raise ValueError('Unknown manifest task')
        from sequential.launcher import LogicalRunLock
        with LogicalRunLock(root / 'locks' / (args.task_id + '.lock')):
            marker = Path(task['output_path']) / 'completed.json'
            if marker.exists():
                if read_json(marker).get('protocol_hash') != manifest['protocol_hash']:
                    raise ValueError('Completion belongs to a different protocol')
                return 0
            if task['kind'] == 'preflight':
                if manifest.get('stage') == 'strong_calibration':
                    from world.sumo_events.calibration import worker
                    return worker(manifest, task)
                from world.sumo_events.preflight import run_preflight
                return run_preflight(manifest, task)
            run = next(r for r in manifest['runs'] if r['run_id'] == task['run'])
            if task['kind'] == 'train':
                return train_worker(manifest, run, args.resume)
            return evaluation_worker(manifest, run, task)
    if min(args.train_workers, args.eval_workers, args.test_workers) < 1 or (
            args.max_workers is not None and args.max_workers < 1):
        raise ValueError('Worker pool sizes must be positive')
    from common.experiment_queue import QueueRunner
    from sequential.launcher import LogicalRunLock
    with LogicalRunLock(root / 'launch.lock'):
        state = root / 'queue'
        if args.operation == 'launch' and (state / 'run_manifest.json').exists():
            raise ValueError('Queue already exists; use status or resume')
        from torch.utils.tensorboard import SummaryWriter
        import time
        writer = SummaryWriter(str(state / 'tensorboard'))
        def dashboard_status(payload):
            step = int(time.time())  # monotone across queue restarts, not a reset elapsed counter
            for key in ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'BLOCKED', 'RESUMABLE', 'SKIPPED_COMPLETED'):
                writer.add_scalar('tasks/' + key.lower(), payload['counts'].get(key, 0), step)
            for key in ('cpu_percent', 'memory_percent', 'run_process_rss_bytes'):
                writer.add_scalar('resources/' + key, payload['latest_resource_sample'][key], step)
            writer.flush()
        settings = SimpleNamespace(manifest=str(Path(args.manifest).resolve()), state_dir=str(state),
            max_workers=args.max_workers or max(args.train_workers + args.eval_workers, args.test_workers), max_retries=0,
            resume_failed=args.operation == 'resume',
            pool_workers={'train': args.train_workers, 'eval': args.eval_workers, 'test': args.test_workers},
            status_callback=dashboard_status,
            stagger_seconds=.1, poll_interval=.5, monitor_interval=5., max_memory_percent=85.,
            max_swap_growth_mb=128., min_disk_free_gb=20., max_load_ratio=1.)
        try:
            code = QueueRunner(settings).run()
            if manifest.get('stage') in ('preflight', 'strong_calibration'):
                if manifest['stage'] == 'strong_calibration':
                    from world.sumo_events.calibration import summarize as summarize_preflight
                else:
                    from world.sumo_events.preflight import summarize_preflight
                summary = summarize_preflight(manifest)
                print(json.dumps({'summary': str(root / 'preflight_summary.json'),
                                  'admission': summary['admission']}, ensure_ascii=False))
                return code or (0 if summary['admission'] == 'development_checks_passed' else 2)
            return code
        finally:
            writer.close()


def train_worker(manifest, run, resume):
    from common.paper_experiment import read_json, write_json, file_digest
    output = Path(run['output_path'])
    argv = ['paper-worker', '-w', 'sumo', '-a', manifest['config']['agent'], '-n', manifest['config']['network'],
            '--seed', str(run['seed']), '--interface', 'libsumo', '--prefix', run['run_id'],
            '--experiment-config', run['overlay']]
    checkpoint = None
    if output.exists():
        if not resume:
            raise ValueError('Run output exists; use resume rather than overwriting it')
        publications = sorted((output / 'published').glob('episode_*.json'))
        if publications:
            last = read_json(publications[-1])
            checkpoint = last['resumable']
            state = read_json(output / 'run_status.json')
            if state['status'] == '已完成':
                if (last['episode'] != manifest['config']['episodes']
                        or file_digest(checkpoint) != last['resumable_sha256']):
                    raise ValueError('Completed run has no valid final checkpoint')
                write_json(output / 'completed.json', {'status': 'completed', 'protocol_hash': manifest['protocol_hash']})
                return 0
            argv += ['--resume-output', str(output), '--resume-checkpoint', checkpoint]
        else:
            # No published checkpoint means no recoverable training state.
            # Keep the failed attempt; deterministic restart begins at episode 0.
            archive = output.parent.parent / 'failed_attempts' / run['run_id']
            archive.mkdir(parents=True, exist_ok=True)
            shutil.move(str(output), str(archive / f'attempt_{len(list(archive.iterdir())):03d}'))
    sys.argv = argv
    code = main(worker_lock_held=True)
    if code == 0:
        final = output / 'published' / f'episode_{manifest["config"]["episodes"]:04d}.json'
        if not final.is_file():
            raise RuntimeError('Training exited without its final checkpoint publication')
        write_json(output / 'completed.json', {'status': 'completed', 'protocol_hash': manifest['protocol_hash'],
                                             'episodes': manifest['config']['episodes']})
    return code


def prepare_resume(output, checkpoint):
    """Validate before pruning, then roll all monitor streams back together."""
    import torch
    import yaml
    from common.paper_experiment import read_json, file_digest, write_json
    from utils.training_monitor import read_rows
    output = Path(output).resolve()
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.parent != output / 'checkpoints' / 'resumable':
        raise ValueError('Resume checkpoint must belong to this run')
    payload = torch.load(checkpoint, map_location='cpu')
    contract = read_json(output / 'experiment_contract.json')
    if payload.get('experiment_contract_hash') != contract['hash']:
        raise ValueError('Resume experiment contract mismatch')
    for source, expected in contract['implementation_sha256'].items():
        if file_digest(source) != expected:
            raise ValueError(f'Resume source mismatch: {source}')
    config = yaml.safe_load((output / 'config/resolved_config.yaml').read_text())
    from common.paper_experiment import checked_json
    settings = config['trainer']
    if not settings.get('episode_plan'):
        raise ValueError('Resume is supported for manifest-planned paper runs; legacy monitor timelines are not migrated')
    checked_json(settings['episode_plan'], settings['episode_plan_hash'])
    for key, field in [('network_sha256', 'roadnetFile'), ('route_sha256', 'flowFile')]:
        if file_digest(Path(config['world']['dir']) / config['world'][field]) != contract[key]:
            raise ValueError('Resume scenario asset mismatch')
    publication = read_json(output / 'published' / f'episode_{payload["episode"]:04d}.json')
    if publication['resumable_sha256'] != file_digest(checkpoint):
        raise ValueError('Published resumable checkpoint changed')
    episode, decision = payload['episode'], payload['global_decision_step']
    archive = output / 'monitor_orphans' / f'after_{episode:04d}'
    archive.mkdir(parents=True, exist_ok=True)
    for name in ('episodes', 'decisions', 'updates', 'environment'):
        path = output / 'monitor' / (name + '.jsonl')
        if not path.exists():
            continue
        rows = read_rows(path)
        kept = [r for r in rows if r.get('episode', 0) <= episode] if name == 'episodes' else [
            r for r in rows if r['decision'] <= decision]
        shutil.copy2(path, archive / path.name)
        path.write_text(''.join(json.dumps(r) + '\n' for r in kept))
        # Regenerate CSV rather than append to a stale post-checkpoint tail.
        csv_path = path.with_suffix('.csv')
        if csv_path.exists():
            csv_path.unlink()
        if kept:
            import csv
            with csv_path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(kept[0]))
                writer.writeheader()
                writer.writerows(kept)
    # TensorBoard uses different step axes for episodes/decisions. Rebuild from
    # canonical retained JSONL instead of a single incorrect purge_step value.
    tb = output / 'tensorboard'
    if tb.exists():
        target = archive / f'tensorboard_{len(list(archive.glob("tensorboard_*"))):03d}'
        shutil.move(str(tb), str(target))
    environment = output / 'environment/train'
    for path in environment.glob('episode_*.*'):
        if int(path.stem.split('_')[-1]) > episode:
            shutil.move(str(path), str(archive / path.name))
    return payload


def evaluation_worker(manifest, run, task):
    """Use the existing trainer/agents and isolation guard in a separate process."""
    import logging
    import torch
    import yaml
    from common.paper_experiment import read_json, write_json, file_digest
    from common.registry import Registry
    from trainer.paper_trainer import PaperTrainer
    from utils.training_monitor import TrainingMonitor
    import agent.paper_baselines
    Registry.register_trainer('tsc')(PaperTrainer)
    publication = read_json(task['publication'])
    checkpoint = publication['evaluation']
    if publication['protocol_hash'] != run.get('source_protocol_hash', manifest['protocol_hash']) or file_digest(checkpoint) != publication['evaluation_sha256']:
        raise ValueError('Frozen evaluation checkpoint identity mismatch')
    source = Path(run['output_path'])
    original_contract = read_json(source / 'experiment_contract.json')
    payload = torch.load(checkpoint, map_location='cpu')
    if payload.get('experiment_contract_hash') != original_contract['hash']:
        raise ValueError('Checkpoint does not match its training contract')
    output = Path(task['output_path'])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run').exists():
        archive = output / 'failed_attempts'
        archive.mkdir(exist_ok=True)
        shutil.move(str(output / 'run'), str(archive / str(len(list(archive.iterdir())))))
    config = yaml.safe_load(Path(run['overlay']).read_text())
    config['command']['output_path'] = str(output / 'run')
    overlay = output / 'evaluation.yml'
    overlay.write_text(yaml.safe_dump(config))
    sys.argv = ['paper-evaluate', '-w', 'sumo', '-a', manifest['config']['agent'], '-n', manifest['config']['network'],
                '--seed', str(run['seed']), '--prefix', task['run_id'], '--experiment-config', str(overlay)]
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'run.py'), run_name='paper_eval_runner')
    runner = namespace['Runner'](namespace['args'])
    logger_dir = Path(runner.output_path) / 'logger'
    logger_dir.mkdir(exist_ok=True)
    logger = logging.getLogger(task['run_id'])
    handler = logging.FileHandler(logger_dir / 'evaluation.log')
    logger.addHandler(handler)
    trainer = None
    try:
        trainer = PaperTrainer(logger)
        runner.run_state.transition('运行中')
        if trainer.reward_contract['hash'] != payload['reward_contract_hash']:
            raise ValueError('Evaluation reward/agent/signal contract mismatch')
        if run.get('evaluation_compatibility') is not None:
            from common.paper_experiment import verify_evaluation_compatibility
            verify_evaluation_compatibility(original_contract, trainer.experiment_contract,
                                           run['evaluation_compatibility'])
        elif trainer.experiment_contract['hash'] != original_contract['hash']:
            raise ValueError('Evaluation reconstruction differs from frozen training identity')
        if len(trainer.agents) != len(payload['agents']):
            raise ValueError('Evaluation checkpoint agent layout mismatch')
        for agent, saved in zip(trainer.agents, payload['agents']):
            agent.model.load_state_dict(saved['online_model_state_dict'])
        trainer.episode_events.callback = None
        # Per-case artifacts survive a worker restart; only its logging session
        # is archived. A case is reused only with matching checkpoint/plan/data.
        trainer.episode_events.root = output
        trainer.monitor.writer.add_text('protocol/evaluation', json.dumps({
            'role': task['role'], 'checkpoint_episode': task['episode'],
            'checkpoint_sha256': publication['evaluation_sha256'],
            'training_contract_hash': original_contract['hash']}, indent=2), 0)
        cases = read_json(Path(manifest['root']) / 'cases.json')[task['role']]
        results, isolation = [], []
        trainer.monitor.auto_render = False
        def evaluate_case(case, subfolder="cases"):
            from common.paper_experiment import digest
            artifact = output / subfolder / (case['case_id'] + '.json')
            if artifact.exists():
                prior = read_json(artifact)
                if (prior['checkpoint_sha256'] != publication['evaluation_sha256']
                        or prior['case_hash'] != digest(case)
                        or file_digest(prior['result']['timeline']) != prior['timeline_sha256']):
                    raise ValueError('Completed evaluation case identity changed')
                result, check = prior['result'], prior['isolation']
            else:
                # Reuse the existing evaluation loop and its isolation guard.
                trainer.episode_events.select(case, task['role'])
                trainer.train_test(task['episode'], record_type='EVALUATION')
                result = trainer.episode_events.finish()
                check = trainer.evaluation_isolation_checks[-1]
                write_json(artifact, {'checkpoint_sha256': publication['evaluation_sha256'],
                    'case_hash': digest(case), 'result': result, 'isolation': check,
                    'timeline_sha256': file_digest(result['timeline'])})
            return result, check

        for case in cases:
            result, check = evaluate_case(case)
            results.append(result)
            isolation.append(check)
            write_json(output / 'progress.json', {'completed_cases': len(results), 'total_cases': len(cases),
                'checkpoint_episode': task['episode'], 'last_case': case['case_id']})
            for key in ('system_time_per_vehicle', 'mean_queue', 'pending_due', 'arrived'):
                trainer.monitor.writer.add_scalar(f'{task["role"]}/{case["case_id"]}/{key}', result[key], task['episode'])
            trainer.monitor.writer.flush()
            trainer.monitor.writer.add_text(f'{task["role"]}/{case["case_id"]}/schedule',
                                            json.dumps(result['schedule']), task['episode'])
        controls = {r['sumo_seed']: r for r in results if r['kind'] == 'normal'}
        clearance_results = []
        from world.sumo_events.schema import Schedule
        for result in results:
            if result['kind'] != 'normal':
                normal = controls[result['sumo_seed']]
                begin = str(int(min(e['begin'] for e in result['schedule']['events'])))
                if result['pre_event_trace_sha256'] != normal['trace_prefixes_before'][begin]:
                    raise RuntimeError('Paired normal/event traces differ before the first event')
                result['paired_system_time_delta'] = result['system_time_per_vehicle'] - normal['system_time_per_vehicle']
                if 'strong_events' in manifest['config']:
                    counterfactual = normal
                    removals = result['events'].get('removed_vehicles', [])
                    if removals:
                        source_case = next(c for c in cases if c['case_id'] == result['case_id'])
                        audit = dict(source_case, case_id=source_case['case_id']+'_clearance',
                            kind='clearance_only', schedule=Schedule(()).to_dict(), clearance_replay=removals)
                        counterfactual, audit_check = evaluate_case(audit, 'clearance_cases')
                        if counterfactual['pre_event_trace_sha256'] != normal['trace_prefixes_before'][begin]:
                            raise RuntimeError('Clearance-only pre-event pairing mismatch')
                        if counterfactual['forced_removed'] != result['forced_removed']:
                            raise RuntimeError('Clearance-only vehicle count mismatch')
                        clearance_results.append(counterfactual)
                        isolation.append(audit_check)
                    result['paired_physical_J_delta'] = result['physical_system_time_per_vehicle'] - normal['physical_system_time_per_vehicle']
                    result['net_event_J_delta'] = result['physical_system_time_per_vehicle'] - counterfactual['physical_system_time_per_vehicle']
                    result['net_event_J_percent'] = 100 * result['net_event_J_delta'] / normal['physical_system_time_per_vehicle']
                    result['clearance_control_case'] = counterfactual['case_id']
        write_json(output / 'progress.json', {'completed_cases': len(results), 'total_cases': len(cases),
            'clearance_cases': len(clearance_results), 'checkpoint_episode': task['episode'],
            'stage': 'case_and_clearance_audits_completed'})
        write_json(output / 'results.json', {'protocol_hash': manifest['protocol_hash'], 'role': task['role'],
            'checkpoint_episode': task['episode'], 'checkpoint_sha256': publication['evaluation_sha256'],
            'training_contract_hash': original_contract['hash'], 'cases': results,
            'isolation_checks': isolation, 'clearance_cases': clearance_results})
        trainer.monitor.close('completed')
        runner.run_state.transition('已完成', exit_code=0)
        write_json(output / 'completed.json', {'protocol_hash': manifest['protocol_hash'], 'status': 'completed'})
        return 0
    except Exception as error:
        if runner.run_state.status['status'] in {'已创建', '运行中'}:
            runner.run_state.transition('失败', exit_code=1, error=error)
        raise
    finally:
        if trainer is not None:
            if not trainer.monitor.closed:
                trainer.monitor.close('failed')
            trainer.episode_events.close_stream()
            trainer.world.close()
        handler.close()


def main(worker_lock_held=False):
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    import torch
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    if len(sys.argv) > 1 and sys.argv[1] in {'config', 'plan', 'extend', 'evaluate-frozen', 'preflight', 'launch', 'status', 'resume', 'worker', '--help', '-h'}:
        return cli()
    import agent.paper_baselines  # explicit registration, not a legacy default
    from common.registry import Registry
    from trainer.paper_trainer import PaperTrainer
    Registry.register_trainer('tsc')(PaperTrainer)
    sys.argv[0] = str(root / 'run.py')
    namespace = runpy.run_path(sys.argv[0], run_name='paper_reward_runner')
    args = namespace['args']
    if args.agent not in {'paper_frap', 'paper_presslight', 'paper_colight'}:
        raise ValueError('Use paper_frap, paper_presslight or paper_colight; legacy entry remains run.py')
    if args.resume_output or args.resume_checkpoint:
        if not (args.resume_output and args.resume_checkpoint):
            raise ValueError('Both resume arguments are required')
        if not worker_lock_held:
            import yaml
            from sequential.launcher import LogicalRunLock
            saved = yaml.safe_load((Path(args.resume_output) / 'config/resolved_config.yaml').read_text())
            plan_path = saved['trainer'].get('episode_plan')
            if not plan_path:
                raise ValueError('Resume requires a manifest-planned run')
            plan_path = Path(plan_path)
            with LogicalRunLock(plan_path.parent.parent / 'locks' / (plan_path.stem + '.lock')):
                return main(worker_lock_held=True)
        prepare_resume(args.resume_output, args.resume_checkpoint)
    if args.agent not in Registry.mapping['model_mapping']:
        raise RuntimeError('CoLight graph dependencies missing: install requirements-colight.txt')
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    # A full-budget published checkpoint may only need completion finalization.
    # Legacy Runner rejects that case; permit it for this explicit entry only.
    class PaperRunner(namespace['Runner']):
        def _initialize_resume(self, pArgs):
            import yaml
            saved = yaml.safe_load((Path(pArgs.resume_output) / 'config/resolved_config.yaml').read_text())
            ckpt = torch.load(pArgs.resume_checkpoint, map_location='cpu')
            if ckpt['episode'] < saved['trainer']['episodes']:
                return super()._initialize_resume(pArgs)
            if ckpt['episode'] != saved['trainer']['episodes']:
                raise ValueError('Resume episode exceeds budget')
            self.output_path = str(Path(pArgs.resume_output).resolve())
            saved.pop('config_record', None)
            self.config = saved
            self.config['command'].update(output_path=self.output_path, resume_episode=ckpt['episode'],
                                           resume_checkpoint=pArgs.resume_checkpoint)
            self.duplicate_config = {}
            directory = Path(self.output_path) / 'config'
            self.config_sources = {p.name: p.read_bytes() for p in directory.iterdir()
                                   if p.name in ('base.yml', f'{args.agent}.yml', 'simulator_source.cfg', 'experiment_overlay.yml')}
            self.run_state = namespace['RunStateManager'].resume(self.output_path, pArgs.resume_checkpoint, ckpt['episode'])
            self.config_registry()
            self.config_archive_path = str(directory)
            self.resume_checkpoint = pArgs.resume_checkpoint
    runner = PaperRunner(args)
    status = 'failed'
    try:
        from common.paper_rewards import load_profile
        load_profile(runner.config['model']['reward_profile'], args.agent.removeprefix('paper_'))
        runner.run()
        status = 'completed'
    except KeyboardInterrupt as error:
        status = 'interrupted'
        if runner.run_state.status['status'] in {'已创建', '运行中'}:
            runner.run_state.transition('失败', exit_code=130, error=error)
        print('Training interrupted; exporting collected data and charts.', flush=True)
        return 130
    except Exception as error:
        if runner.run_state.status['status'] in {'已创建', '运行中'}:
            runner.run_state.transition('失败', exit_code=1, error=error)
        raise
    finally:
        trainer = getattr(runner, 'trainer', None)
        if trainer is not None:
            try:
                if hasattr(trainer, 'monitor'):
                    trainer.monitor.close(status)
            finally:
                if getattr(trainer, 'episode_events', None) is not None:
                    trainer.episode_events.close_stream()
                trainer.world.close()
        print(f'Run: {runner.output_path}\nCharts and CSV: {runner.output_path}/monitor', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
