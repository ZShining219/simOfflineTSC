#!/usr/bin/env python3
"""Watcher: pick up newly completed stage-5 TARL runs and enqueue evals.

Loop (every --interval seconds):
  1. Re-run build_stage4_eval_manifests.py on the stage-5 queue state so
     manifests/eval queue cover every run that has SUCCEEDED so far.
  2. If the eval queue runner is not alive AND no run.py --evaluation-manifest
     process is in flight, (re)start the queue runner; it skips SUCCEEDED
     tasks and picks up PENDING ones.
  3. If the training queue state shows every task finished and the eval
     queue state shows every eval task finished, write WATCHER_DONE and exit.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / 'data/output_data/cross_algorithm/plan5_b100/environment/'
         'micromamba/envs/colight/bin/python3.10')
TRAIN_STATE = ROOT / 'artifacts/att_entity_003/run_state_stage5_tarl/run_manifest.json'
EVAL_DIR = ROOT / 'artifacts/att_entity_003/eval_stage5_tarl'
EVAL_STATE = ROOT / 'artifacts/att_entity_003/run_state_eval_stage5'
EVAL_QUEUE = EVAL_DIR / 'run_queue.json'
LOG = EVAL_STATE / 'logs' / 'watcher.log'


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a') as f:
        f.write(line + '\n')


def state_tasks(path):
    try:
        return json.loads(Path(path).read_text()).get('tasks', [])
    except Exception:
        return []


def pgrep(pattern):
    out = subprocess.run(['pgrep', '-f', pattern],
                         capture_output=True, text=True).stdout.strip()
    return [int(p) for p in out.split() if p.strip()] if out else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--interval', type=int, default=600)
    args = ap.parse_args()
    while True:
        # 1. rebuild manifests to cover newly completed runs
        r = subprocess.run(
            [PY, 'tools/build_stage4_eval_manifests.py',
             '--queue-state', str(TRAIN_STATE),
             '--out-dir', str(EVAL_DIR),
             '--checkpoints', '200', '--seeds', '400007'],
            cwd=ROOT, capture_output=True, text=True)
        tail = (r.stdout or r.stderr).strip().splitlines()
        if tail:
            log(f"builder: {tail[-1]}")
        # 2. restart eval queue when idle
        eval_procs = pgrep('evaluation-manifest')
        runner = pgrep('run_experiment_queue.py.*eval_stage5_tarl')
        if not eval_procs and not runner and EVAL_QUEUE.exists():
            # State file only reflects tasks the last runner saw; derive
            # pending as queue tasks not yet SUCCEEDED in state.
            done = {t['run_id'] for t in state_tasks(
                EVAL_STATE / 'run_manifest.json')
                if t.get('status') == 'SUCCEEDED'}
            pending = [t for t in state_tasks(EVAL_QUEUE)
                       if t.get('run_id') not in done]
            if pending:
                log(f"restarting eval queue for {len(pending)} pending tasks")
                logf = open(EVAL_STATE / 'logs' / 'queue_runner.log', 'a')
                subprocess.Popen(
                    [PY, 'tools/run_experiment_queue.py', '--queue',
                     str(EVAL_QUEUE), '--state', str(EVAL_STATE),
                     '--parallel', '3', '--poll', '60'],
                    cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                    start_new_session=True)
        # 2b. text counterfactual probes for completed text-arm runs
        for t in state_tasks(TRAIN_STATE):
            if t.get('status') != 'SUCCEEDED':
                continue
            rid = t['run_id'].split('/')[-1]
            arm = None
            for tag, a in (('gatg', 'tarl_gating'), ('att', 'tarl_attention')):
                if f'_{tag}_' in rid:
                    arm, key = a, tag
                    break
            if arm is None:
                continue
            probe_out = (ROOT / 'data/output_data/analysis/att_entity_003'
                         / f'tarl_text_cf/{rid.split("tarlp_")[-1]}_ep200')
            ckpt = (ROOT / 'data/output_data/tsc'
                    / f'sumo_{arm}/hz4x4/{rid}'
                    / 'checkpoints/resumable/episode_0200.pt')
            if probe_out.exists() or not ckpt.exists():
                continue
            if pgrep('run_tarl_text_cf'):
                continue
            probe_out.mkdir(parents=True, exist_ok=True)
            log(f"text-CF probe -> {probe_out.name}")
            logf = open(EVAL_STATE / 'logs' / 'probes.log', 'a')
            subprocess.Popen(
                [PY, 'tools/run_tarl_text_cf.py', '--checkpoint', str(ckpt),
                 '--config', 'configs/tsc/tarl_paper_cfprobe.yml',
                 '--agent', arm,
                 '--prefix', f'tarlcf_{key}_{rid.rsplit("_s", 1)[-1]}',
                 '--output', str(probe_out),
                 '--seed', rid.rsplit('_s', 1)[-1]],
                cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True)

        # 3. done condition: every training run finished AND every eval task
        # in the queue file is recorded SUCCEEDED in state.
        train = state_tasks(TRAIN_STATE)
        evals = state_tasks(EVAL_STATE / 'run_manifest.json')
        done_ids = {t['run_id'] for t in evals
                    if t.get('status') == 'SUCCEEDED'}
        queue_ids = {t['run_id'] for t in state_tasks(EVAL_QUEUE)}
        if (train and all(t.get('status') == 'SUCCEEDED' for t in train)
                and queue_ids and queue_ids <= done_ids):
            log('WATCHER_DONE: all training and eval tasks finished')
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
