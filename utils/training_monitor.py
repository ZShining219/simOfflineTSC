"""TensorBoard scalar logging plus durable CSV/PNG/SVG exports.

TensorBoard upstream: https://github.com/tensorflow/tensorboard (Apache-2.0).
No metrics are sent to an external service. CSV rows are flushed immediately;
plots are refreshed each episode and on normal completion/SIGINT/SIGTERM.
After SIGKILL/power loss, render_run() rebuilds plots from completed JSONL rows.
"""
import csv
import json
import math
from pathlib import Path
import time


def read_rows(path):
    if not Path(path).exists():
        return []
    lines = Path(path).read_text().splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        if not line.endswith('\n') and index == len(lines) - 1:
            break  # A concurrent write or interrupted final row is not complete.
        if line.strip():
            rows.append(json.loads(line))
    return rows


def render_run(output):
    """Export existing observations, including partial runs, without simulation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    directory = output / 'monitor'
    directory.mkdir(parents=True, exist_ok=True)
    episodes = read_rows(directory / 'episodes.jsonl')
    decisions = read_rows(directory / 'decisions.jsonl')
    updates = read_rows(directory / 'updates.jsonl')
    # Rebuild tabular exports from complete records, including after a hard kill.
    for name, rows in (('episodes', episodes), ('decisions', decisions), ('updates', updates)):
        if rows:
            temporary = directory / f'{name}.csv.tmp'
            with temporary.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(directory / f'{name}.csv')
    contract_path = output / 'reward_contract.json'
    contract = json.loads(contract_path.read_text()) if contract_path.exists() else {}
    identity = contract.get('profile', {}).get('profile_id', 'legacy reward')
    node_count = max(1, len(contract.get('lanes', {})))
    for row in episodes:
        if row.get('reward_mean') is not None:
            row.setdefault('reward_per_node_mean', row['reward_mean'] / node_count)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    metrics = [('reward_per_node_mean', f'Mean node reward: {identity}'), ('loss_mean', 'Training TD loss'),
               ('travel_time', 'Arrived-vehicle travel time (s)'), ('queue', 'Legacy observation queue metric'),
               ('throughput', 'Arrived vehicles'), ('epsilon', 'Exploration epsilon')]
    for ax, (key, title) in zip(axes.flat, metrics):
        for kind in ('TRAIN', 'EVALUATION', 'FINAL_EVALUATION'):
            rows = [r for r in episodes if r['record_type'] == kind
                    and isinstance(r.get(key), (float, int)) and math.isfinite(r[key])]
            if rows:
                ax.plot([r['episode'] for r in rows], [r[key] for r in rows], marker='.', label=kind)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Episode')
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, 'No completed records yet', ha='center', transform=ax.transAxes)
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(directory / f'episode_curves.{ext}', dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    for ax, key, title in zip(axes[:3], ('reward_mean', 'stopped_vehicles', 'active_reports'),
                              ('Action-interval reward (native scale)', 'Current stopped vehicles (full measured lanes)',
                               'Active public event reports')):
        if decisions:
            x, y = [r['decision'] for r in decisions], [r[key] for r in decisions]
            if key == 'active_reports':
                ax.step(x, y, where='post', linewidth=.8)
            else:
                ax.plot(x, y, linewidth=.8)
        ax.set_ylabel(title, fontsize=8)
        ax.set_xlabel('Training decision (across episodes)')
        ax.grid(alpha=.25)
    if updates:
        axes[3].plot([r['decision'] for r in updates], [r['loss_mean'] for r in updates], linewidth=.8)
    axes[3].set_ylabel('TD loss (mean across updates)', fontsize=8)
    axes[3].set_xlabel('Training decision (across episodes)')
    axes[3].grid(alpha=.25)
    boundaries = [(row['decision'], row['episode']) for prev, row in zip(decisions, decisions[1:])
                  if row['episode'] != prev['episode']]
    for decision, episode in boundaries:
        for ax in axes:
            ax.axvline(decision, color='grey', linestyle='--', alpha=.5, linewidth=.8)
        axes[0].text(decision, .96, f' reset: episode {episode}', va='top', fontsize=8,
                     transform=axes[0].get_xaxis_transform())
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(directory / f'decision_curves.{ext}', dpi=130)
    plt.close(fig)
    return directory


class TrainingMonitor:
    def __init__(self, output, contract, render_every=1):
        from torch.utils.tensorboard import SummaryWriter
        self.output = Path(output)
        self.directory = self.output / 'monitor'
        self.directory.mkdir(exist_ok=True)
        self.writer = SummaryWriter(str(self.output / 'tensorboard'), max_queue=1, flush_secs=1)
        self.contract = contract
        self.reward_tag = contract['profile']['profile_id'] + '_' + contract['hash'][:8]
        self.closed = False
        if type(render_every) is not int or render_every < 1:
            raise ValueError('render_every must be a positive integer')
        self.render_every = render_every
        self.last_step = 0
        self._status = None
        self.writer.add_text('protocol/reward', json.dumps(contract, indent=2), 0)
        self.status('running')

    def status(self, state):
        if getattr(self, '_replaying', False):
            return
        if state != self._status and not self.closed:
            self.writer.add_text('run/status', state, self.last_step)
            self.writer.flush()
            self._status = state
        path = self.directory / 'status.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'status': state, 'updated_at_unix': time.time(),
                                         'reward_contract_hash': self.contract['hash']}, indent=2) + '\n')
        temporary.replace(path)

    def _append(self, kind, record):
        if getattr(self, '_replaying', False):
            return
        with (self.directory / f'{kind}.jsonl').open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
            handle.flush()
        path = self.directory / f'{kind}.csv'
        exists = path.exists()
        with path.open('a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            if not exists:
                writer.writeheader()
            writer.writerow(record)
            handle.flush()

    def episode(self, record):
        fields = ('record_type', 'episode', 'global_decision_step', 'gradient_updates', 'reward_mean',
                  'loss_mean', 'travel_time', 'queue', 'throughput', 'epsilon', 'waiting_time',
                  'unfinished_vehicles', 'wall_time_seconds', 'replay_size')
        row = {field: (record[field].item() if hasattr(record.get(field), 'item') else record.get(field))
               for field in fields}
        # Legacy Metrics.rewards() sums nodes and averages decisions despite its
        # structured field being called reward_mean. Keep it, and label a true
        # node mean separately so interval and episode scales are not confused.
        count = max(1, len(self.contract.get('lanes', {})))
        row['reward_per_node_mean'] = None if row['reward_mean'] is None else row['reward_mean'] / count
        self._append('episodes', row)
        for key, value in row.items():
            if key not in ('episode',) and isinstance(value, (int, float)) and math.isfinite(value):
                group = f'episode/{row["record_type"]}'
                if key in ('reward_mean', 'reward_per_node_mean'):
                    aggregation = 'network_sum_mean' if key == 'reward_mean' else 'node_mean'
                    tag = f'{group}/reward/{self.reward_tag}/{aggregation}'
                else:
                    tag = f'{group}/{key}'
                self.writer.add_scalar(tag, value, row['episode'])
        self.writer.flush()
        self.status('running')
        if getattr(self, 'auto_render', True) and row['episode'] % self.render_every == 0:
            render_run(self.output)

    def restore_tensorboard(self):
        """Rebuild every axis from retained JSONL after checkpoint rollback."""
        self._replaying, self.auto_render = True, False
        try:
            for row in read_rows(self.directory / 'episodes.jsonl'):
                self.episode(row)
            for row in read_rows(self.directory / 'decisions.jsonl'):
                self.decision(row)
            for row in read_rows(self.directory / 'updates.jsonl'):
                for key in ('loss_mean', 'loss_min', 'loss_max'):
                    self.writer.add_scalar(f'optimization/{self.reward_tag}/{key}', row[key], row['decision'])
            for row in read_rows(self.directory / 'environment.jsonl'):
                data = dict(row)
                step = data.pop('decision')
                self.environment(data, step)
            for path in sorted((self.output / 'environment/train').glob('episode_*.json')):
                row = json.loads(path.read_text())
                self.traffic_episode(row, int(path.stem.split('_')[-1]))
            self.writer.flush()
        finally:
            self._replaying, self.auto_render = False, True

    def environment(self, row, step):
        self._append('environment', dict(decision=step, **row))
        for key in ('system_time_per_vehicle', 'pending_due', 'queue_vehicles', 'running', 'active_reports'):
            self.writer.add_scalar('environment/' + key, row[key], step)
        self.writer.flush()

    def traffic_episode(self, row, episode):
        for key in ('system_time_per_vehicle', 'pending_due', 'mean_queue', 'arrived', 'running'):
            self.writer.add_scalar('traffic/TRAIN/' + key, row[key], episode)
        self.writer.add_text('traffic/event_kind', row['kind'], episode)
        if 'schedule' in row:
            self.writer.add_text('traffic/event_schedule', json.dumps(row['schedule'], ensure_ascii=False), episode)
        self.writer.flush()

    def decision(self, row):
        self.last_step = row['decision']
        self._append('decisions', row)
        for key in ('reward_mean', 'reward_min', 'reward_max', 'stopped_vehicles', 'active_reports'):
            tag = f'decision/reward/{self.reward_tag}/{key}' if key.startswith('reward') else f'decision/{key}'
            self.writer.add_scalar(tag, row[key], row['decision'])
        self.writer.flush()
        self.status('running')

    def optimization(self, decision, gradient_updates, losses):
        values = [float(v) for v in losses]
        if not values or not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite or empty training loss; refusing misleading dashboard data')
        row = {'decision': int(decision), 'gradient_updates': int(gradient_updates),
               'loss_mean': sum(values) / len(values), 'loss_min': min(values), 'loss_max': max(values)}
        self._append('updates', row)
        for key in ('loss_mean', 'loss_min', 'loss_max'):
            self.writer.add_scalar(f'optimization/{self.reward_tag}/{key}', row[key], decision)
        self.writer.flush()

    def close(self, status):
        if not self.closed:
            self.writer.flush()
            try:
                render_run(self.output)
            finally:
                self.status(status)
                self.writer.close()
                self.closed = True
