#!/usr/bin/env python3
"""Minimal experiment queue runner (same manifest schema as ATT-ENTITY-002).

Reads a run_queue.json with tasks=[{run_id, command, cwd, env, priority,
completion:{path,json_field,equals}, ...}], runs at most --parallel at a
time in priority order, and tracks state in <out>/run_manifest.json so a
watcher/dashboard can poll progress.  Completion is detected via the run's
own run_status.json (never by process exit alone).

Usage:
    python3 tools/run_experiment_queue.py \
        --queue artifacts/<exp>/run_queue.json \
        --state artifacts/<exp>/run_state --parallel 4
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

POLL_S = 60


def pid_alive(pid):
    """True if a previously launched (but not owned) process still runs."""
    try:
        os.kill(int(pid), 0)
        return True
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def load_state(state_dir):
    path = Path(state_dir) / 'run_manifest.json'
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_state(state_dir, spec, tasks):
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {'schema_version': 1, 'source_manifest': str(spec),
               'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
               'tasks': tasks}
    tmp = state_dir / 'run_manifest.json.tmp'
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    tmp.replace(state_dir / 'run_manifest.json')
    with open(state_dir / 'run_manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run_id', 'status', 'pid',
                                        'start_time', 'end_time',
                                        'exit_code'])
        w.writeheader()
        for t in tasks:
            w.writerow({k: t.get(k) for k in
                        ('run_id', 'status', 'pid', 'start_time',
                         'end_time', 'exit_code')})


def completed(task):
    comp = task.get('completion') or {}
    path = Path(comp.get('path', ''))
    try:
        data = json.loads(path.read_text())
        return data.get(comp['json_field']) == comp['equals'], \
            data.get(comp['json_field'])
    except Exception:
        return False, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--queue', required=True)
    ap.add_argument('--state', required=True)
    ap.add_argument('--parallel', type=int, default=4)
    ap.add_argument('--poll', type=int, default=POLL_S)
    ap.add_argument('--stagger', type=int, default=0,
                    help='seconds to sleep between consecutive task launches')
    args = ap.parse_args()

    spec = json.loads(Path(args.queue).read_text())
    prev = load_state(args.state)
    prev_tasks = {t['run_id']: t for t in prev['tasks']} if prev else {}
    tasks = []
    for raw in spec['tasks']:
        t = dict(raw)
        old = prev_tasks.get(t['run_id'])
        if old and old.get('status') in ('SUCCEEDED', 'RUNNING'):
            t.update({k: old.get(k) for k in
                      ('status', 'pid', 'start_time', 'end_time',
                       'exit_code', 'failure_reason')})
        elif old and old.get('status') == 'FAILED':
            t['status'] = 'PENDING'
            t['retry_count'] = int(old.get('retry_count') or 0) + 1
        else:
            t.setdefault('status', 'PENDING')
        tasks.append(t)
    save_state(args.state, args.queue, tasks)

    log_dir = Path(args.state) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    running = {}  # run_id -> Popen
    while True:
        adopted = 0
        for t in tasks:
            rid = t['run_id']
            if t['status'] == 'RUNNING' and rid in running:
                proc = running[rid]
                done, status_val = completed(t)
                if done:
                    t.update(status='SUCCEEDED',
                             end_time=time.strftime('%Y-%m-%dT%H:%M:%S'),
                             exit_code=proc.poll())
                    running.pop(rid)
                elif proc.poll() is not None:
                    # Process exited without producing the success marker.
                    _, status_val = completed(t)
                    t.update(status='FAILED',
                             end_time=time.strftime('%Y-%m-%dT%H:%M:%S'),
                             exit_code=proc.returncode,
                             failure_reason=f'exit {proc.returncode}, '
                                            f'status={status_val}')
                    running.pop(rid)
            elif t['status'] == 'RUNNING' and rid not in running:
                # Adopted from a previous state file.  If the recorded pid is
                # still alive the orphan keeps running and counts toward the
                # parallel limit; only truly dead orphans are requeued.
                done, _ = completed(t)
                if done:
                    t['status'] = 'SUCCEEDED'
                elif pid_alive(t.get('pid')):
                    adopted += 1
                else:
                    t['status'] = 'PENDING'
        save_state(args.state, args.queue, tasks)

        slots = args.parallel - len(running) - adopted
        pending = sorted((t for t in tasks if t['status'] == 'PENDING'),
                         key=lambda t: t.get('priority', 0))
        for t in pending[:max(slots, 0)]:
            log_path = log_dir / (t['run_id'].replace('/', '_') + '.log')
            env = {**__import__('os').environ, **(t.get('env') or {})}
            fh = open(log_path, 'ab')
            proc = subprocess.Popen(
                t['command'], cwd=t.get('cwd'), env=env,
                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            t.update(status='RUNNING', pid=proc.pid,
                     start_time=time.strftime('%Y-%m-%dT%H:%M:%S'))
            running[t['run_id']] = proc
            if args.stagger:
                time.sleep(args.stagger)
        save_state(args.state, args.queue, tasks)

        if not running and not any(t['status'] == 'PENDING'
                                   for t in tasks):
            break
        time.sleep(args.poll)

    counts = {}
    for t in tasks:
        counts[t['status']] = counts.get(t['status'], 0) + 1
    print('queue done:', counts)


if __name__ == '__main__':
    main()
