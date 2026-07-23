"""Run this file directly to execute the frozen Plan 1 evaluator live."""

import argparse
import json
import logging
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkout', required=True)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--network', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    checkout = os.path.abspath(args.checkout)
    source_run = os.path.abspath(args.source_run)
    output = os.path.abspath(args.output)
    os.chdir(checkout)
    sys.path.insert(0, checkout)
    sys.argv = [
        'run.py', '-t', 'tsc', '-a', 'dqn', '-w', 'sumo',
        '-n', args.network, '-d', 'onfly', '--seed', '0', '--ngpu', '-1',
        '--interface', 'libsumo', '--prefix', args.prefix,
    ]
    import run as frozen_run
    from common.registry import Registry

    runner = frozen_run.Runner(frozen_run.args)
    logger = frozen_run.setup_logging(logging.INFO)
    trainer = Registry.mapping['trainer_mapping']['tsc'](logger)
    checkpoint_path = os.path.join(
        source_run, 'checkpoints', 'resumable', 'episode_0400.pt'
    )
    trainer.load_resumable_checkpoint(checkpoint_path)
    action_sequence = []
    original_record = trainer._record_actions

    def record(actions):
        import numpy as np
        action_sequence.append(np.asarray(actions).reshape(-1).astype(int).tolist())
        original_record(actions)

    trainer._record_actions = record
    runner.run_state.transition('运行中')
    close_report = None
    try:
        trainer.train_test(400, record_type='FINAL_EVALUATION')
        runner.run_state.transition('已完成', exit_code=0)
    except BaseException as error:
        runner.run_state.transition('失败', exit_code=1, error=error)
        raise
    finally:
        close_report = trainer.world.close()
    payload = {
        'schema_version': 1, 'frozen_checkout': checkout,
        'source_run': source_run, 'checkpoint_path': checkpoint_path,
        'network': args.network, 'training_seed': 0,
        'decision_count': len(action_sequence),
        'action_sequence': action_sequence,
        'metrics': trainer.last_evaluation_record,
        'world_close': close_report,
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(json.dumps({
        'output': output, 'network': args.network,
        'decision_count': len(action_sequence),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
