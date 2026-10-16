"""Serve TensorBoard locally or regenerate figures for any stopped paper run.

    python -m tools.training_dashboard serve --logdir data/output_data/tsc
    python -m tools.training_dashboard serve --manifest path/to/manifest.json
    python -m tools.training_dashboard serve --queue-manifest path/to/run_queue.json
    python -m tools.training_dashboard render --run path/to/run

Use SSH port forwarding when viewing a remote machine. TensorBoard remains a
reader of persistent event files; stopping its server does not stop training.
"""
import argparse
import collections
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


TRAFFIC_METRICS = {
    'J_seconds': 'system_time_per_vehicle',
    'queue_vehicles': 'mean_queue',
    'arrived_vehicles': 'arrived',
    'removed_vehicles': 'forced_removed',
}


class ExperimentProjection:
    """Read-only projection of completed records; no simulator/worker imports."""
    def __init__(self, manifest):
        self.manifest = Path(manifest).resolve()
        self.spec = json.loads(self.manifest.read_text())
        self.root = self.manifest.parent
        if Path(self.spec['root']).resolve() != self.root:
            raise ValueError('Experiment manifest moved')
        self.names = {r['run_id']: f"{'N' if r['regime'] == 'normal' else 'E'}"
                      f"{self.spec['config']['episodes']}_seed{r['seed']}" for r in self.spec['runs']}
        self.cache = {}

    def cached(self, path, extract):
        stat = path.stat()
        identity = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        old = self.cache.get(path)
        if old is None or old[0] != identity:
            self.cache[path] = (identity, extract(path.read_text()))
        return self.cache[path][1]

    @staticmethod
    def episode_rows(text):
        # The writer can be appending the last JSONL row during this read.
        return [json.loads(line) for line in text.splitlines(keepends=True)
                if line.endswith('\n') and line.strip()]

    def points(self):
        points = {}

        def put(run, tag, episode, value):
            if isinstance(value, (int, float)) and math.isfinite(value):
                points[(self.names[run], tag, int(episode))] = float(value)

        for run in self.spec['runs']:
            folder = Path(run['output_path'])
            path = folder / 'monitor/episodes.jsonl'
            if not path.exists():
                continue
            rows = self.cached(path, self.episode_rows)
            completed = set()
            for row in rows:
                if row['record_type'] != 'TRAIN':
                    continue
                ep = row['episode']
                completed.add(ep)
                for label, field in [('reward_per_node', 'reward_per_node_mean'),
                                     ('td_loss', 'loss_mean'), ('epsilon', 'epsilon')]:
                    put(run['run_id'], 'Train/' + label, ep, row.get(field))
            for path in (folder / 'environment/train').glob('episode_*.json'):
                ep = int(path.stem.split('_')[-1])
                if ep not in completed:
                    continue
                row = self.cached(path, lambda text: {
                    k: v for k, v in json.loads(text).items() if k in TRAFFIC_METRICS.values()})
                for label, field in TRAFFIC_METRICS.items():
                    put(run['run_id'], 'Train/' + label, ep, row.get(field))

        def evaluation(text):
            value = json.loads(text)
            groups = collections.defaultdict(list)
            for row in value['cases']:
                group = row['kind'] + ('/' + row['severity'] if row.get('severity') else '')
                groups[group].append(row)
            result = {}
            for kind, rows in groups.items():
                for label, field in dict(TRAFFIC_METRICS, net_event_J_percent='net_event_J_percent',
                                         J_physical_seconds='physical_system_time_per_vehicle').items():
                    values = [row[field] for row in rows if field in row]
                    if len(values) == len(rows):
                        result[f'{kind}/{label}'] = sum(values) / len(values)
            return value['protocol_hash'], value['checkpoint_episode'], result

        for task in self.spec['tasks']:
            if task['kind'] != 'evaluate':
                continue
            path = Path(task['output_path']) / 'results.json'
            if not path.exists():
                continue  # Only complete checkpoint evaluations are charted.
            protocol, episode, values = self.cached(path, evaluation)
            # A recovery view can explicitly pin each source protocol. Never
            # accept arbitrary foreign results merely because names match.
            if protocol != task.get('source_protocol_hash', self.spec['protocol_hash']):
                raise ValueError('Evaluation belongs to another protocol')
            if episode != task['episode']:
                raise ValueError('Evaluation checkpoint mismatch')
            section = 'Validation' if task['role'] == 'validation' else 'Test'
            for tag, value in values.items():
                put(task['run'], section + '/' + tag, episode, value)
        return points


class QueueProjection(ExperimentProjection):
    """Projection for experiment_queue manifests (tasks + metrics/records.jsonl)."""

    FIELDS = [('travel_time', 'travel_time'), ('queue', 'queue'),
              ('throughput', 'throughput'), ('unfinished', 'unfinished_vehicles'),
              ('delay', 'delay'), ('reward', 'reward_mean'),
              ('epsilon', 'epsilon'), ('replay_size', 'replay_size')]
    SECTIONS = {'TRAIN': 'Train', 'EVALUATION': 'Eval'}

    def __init__(self, manifest):
        self.manifest = Path(manifest).resolve()
        self.spec = json.loads(self.manifest.read_text())
        self.root = self.manifest.parent
        self.names = {}
        for task in self.spec['tasks']:
            base = task['run_id'].rsplit('/', 1)[-1]
            match = re.match(r'(.+?)_(event|normal)_\d+_s(\d+)$', base)
            if not match:
                raise ValueError(f'Unrecognized run_id: {task["run_id"]}')
            agent, regime, seed = match.groups()
            self.names[task['run_id']] = (f"{agent.replace('_text', '')}_"
                                          f"{'E' if regime == 'event' else 'N'}_s{seed}")
        self.cache = {}

    def points(self):
        points = {}

        def put(run, tag, episode, value):
            if isinstance(value, (int, float)) and math.isfinite(value):
                points[(self.names[run], tag, int(episode))] = float(value)

        for task in self.spec['tasks']:
            path = Path(task['output_path']) / 'metrics/records.jsonl'
            if not path.exists():
                continue
            for row in self.cached(path, self.episode_rows):
                section = self.SECTIONS.get(row['record_type'])
                if section is None:
                    continue
                for label, field in self.FIELDS:
                    put(task['run_id'], f'{section}/{label}', row['episode'], row.get(field))
        return points


class ScalarMirror:
    """Idempotent derived event stream; repair reordered/revised histories."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.writers = {}
        self.previous = {}

    def sync(self, points):
        from torch.utils.tensorboard import SummaryWriter
        from tensorboard.compat.proto.event_pb2 import Event, SessionLog
        from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
        cutoffs = {}
        last_steps = {}
        for run, tag, step in self.previous:
            last_steps[(run, tag)] = max(step, last_steps.get((run, tag), -1))
        for key, old in self.previous.items():
            if key not in points or points[key] != old:
                run, _, step = key
                cutoffs[run] = min(step, cutoffs.get(run, step))
        for run, tag, step in points.keys() - self.previous.keys():
            if step < last_steps.get((run, tag), -1):
                cutoffs[run] = min(step, cutoffs.get(run, step))
        for run in {key[0] for key in points} | set(cutoffs):
            if run not in self.writers:
                self.writers[run] = SummaryWriter(str(self.directory / run), max_queue=1000, flush_secs=2)
                # The live plugin accumulator treats its first START as initial
                # session setup, not a purge. Emit it before any scalar so the
                # first later history repair is recognized as a restart too.
                self.writers[run].file_writer.add_event(Event(
                    wall_time=time.time(), step=0, session_log=SessionLog(status=SessionLog.START)))
        for run, step in cutoffs.items():
            # TensorBoard v2 session restart purges all tags >= this episode.
            self.writers[run].file_writer.add_event(Event(
                wall_time=time.time(), step=step, session_log=SessionLog(status=SessionLog.START)))
        written = 0
        for key in sorted(points, key=lambda k: (k[0], k[2], k[1])):
            run, tag, step = key
            if (run in cutoffs and step >= cutoffs[run]) or key not in self.previous or points[key] != self.previous[key]:
                description = ('X: completed training episode / source checkpoint episode. '
                    'Validation/Test: arithmetic mean of all completed cases of this scenario type. '
                    'J_seconds: horizon-truncated system time per planned vehicle, including '
                    'cleared vehicles as unfinished demand. Lower J/queue is better. '
                    'Removed vehicles are not arrivals. Train reward is the mean per intersection.')
                self.writers[run].file_writer.add_summary(Summary(value=[Summary.Value(
                    tag=tag, simple_value=points[key],
                    metadata=SummaryMetadata(summary_description=description))]), step)
                written += 1
        for writer in self.writers.values():
            writer.flush()
        self.previous = dict(points)
        return written

    def close(self):
        for writer in self.writers.values():
            writer.close()


def serve_experiment(manifest, port, refresh, host='127.0.0.1', queue=False):
    """Own only a derived view and its TensorBoard child; never signal workers."""
    projection = QueueProjection(manifest) if queue else ExperimentProjection(manifest)
    view = projection.root / 'dashboard_view'
    with (projection.root / 'dashboard_view.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = view / 'owner.json'
        identity = {'schema': 'paper-dashboard-view-v1', 'manifest': str(projection.manifest),
                    'protocol_hash': projection.spec.get(
                        'protocol_hash', projection.spec.get('experiment_id'))}
        if view.is_symlink():
            raise ValueError('Dashboard view must not be a symlink')
        if view.exists() and (not marker.is_file() or json.loads(marker.read_text()) != identity):
            raise ValueError('Refusing to overwrite an unowned dashboard directory')
        view.mkdir(exist_ok=True)
        marker.write_text(json.dumps(identity, indent=2) + '\n')
        derived = view / 'runs'
        if derived.is_symlink():
            raise ValueError('Derived run directory must not be a symlink')
        if derived.exists():
            shutil.rmtree(derived)  # Exclusively owned, reproducible presentation data.
        mirror = ScalarMirror(derived)
        server = None
        stopped = False

        def stop(signum, frame):
            nonlocal stopped
            stopped = True

        handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            while not stopped:
                started = time.monotonic()
                points = projection.points()
                written = mirror.sync(points)
                status = {'updated_at_unix': time.time(), 'runs': len(mirror.writers),
                          'points': len(points), 'new_events': written,
                          'refresh_seconds': refresh, 'sync_seconds': time.monotonic() - started,
                          'url': f'http://127.0.0.1:{port}', 'source_manifest': str(projection.manifest)}
                temporary = view / 'status.json.tmp'
                temporary.write_text(json.dumps(status, indent=2) + '\n')
                temporary.replace(view / 'status.json')
                if server is None:
                    print(f'Experiment dashboard: http://{host}:{port}', flush=True)
                    tb_args = [sys.executable, '-m', 'tensorboard.main',
                        '--logdir', str(derived), '--port', str(port), '--reload_interval', '2']
                    # TensorBoard 2.9 exposes --bind_all and rejects --host
                    # together with its default bind-all setting. Newer
                    # versions accept --host; support both for container views.
                    if host in ('0.0.0.0', '::'):
                        tb_args.append('--bind_all')
                    else:
                        tb_args.extend(['--host', host])
                    server = subprocess.Popen(tb_args)
                elif server.poll() is not None:
                    raise RuntimeError(f'TensorBoard exited: {server.returncode}')
                deadline = time.monotonic() + refresh
                while not stopped and time.monotonic() < deadline:
                    time.sleep(min(.25, max(0, deadline - time.monotonic())))
        finally:
            if server is not None and server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            mirror.close()
            for s, handler in handlers.items():
                signal.signal(s, handler)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    serve = sub.add_parser('serve')
    source = serve.add_mutually_exclusive_group()
    source.add_argument('--logdir', default='data/output_data/tsc')
    source.add_argument('--manifest', help='Read-only, continuously refreshed experiment view')
    source.add_argument('--queue-manifest',
                        help='experiment_queue manifest (tasks + metrics/records.jsonl)')
    serve.add_argument('--port', type=int, default=6006)
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--refresh', type=float, default=15, help='Experiment view refresh interval in seconds')
    render = sub.add_parser('render')
    render.add_argument('--run', required=True)
    args = parser.parse_args()
    if args.operation == 'serve':
        if args.manifest or args.queue_manifest:
            if not math.isfinite(args.refresh) or args.refresh < 1:
                parser.error('--refresh must be finite and >= 1 second')
            return serve_experiment(args.manifest or args.queue_manifest, args.port,
                                    args.refresh, args.host, queue=bool(args.queue_manifest))
        print(f'Training dashboard: http://127.0.0.1:{args.port}', flush=True)
        return subprocess.call([sys.executable, '-m', 'tensorboard.main', '--logdir', str(Path(args.logdir).resolve()),
                                '--host', args.host, '--port', str(args.port), '--reload_interval', '2'])
    from utils.training_monitor import render_run
    print(render_run(args.run))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
