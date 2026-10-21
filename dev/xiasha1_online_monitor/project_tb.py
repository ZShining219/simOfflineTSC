"""xiasha1_online_v1 只读 TensorBoard 投影器（campaign 绑定一次性件，登记 dev/INDEX.md）。

把 tsc_trainer run 目录里的 ``metrics/records.jsonl`` 与 ``evaluation/summary.json``
投影为 TensorBoard event 文件；纯只读，不与训练进程交互，可随时停起。
镜像复用 ``tools.training_dashboard.ScalarMirror``（含断点重写 session 重启语义）。

用法（conda env colight）::

    # 跟随正式队列（持续同步 + 起 TB 服务）：
    python dev/xiasha1_online_monitor/project_tb.py \
        --queue artifacts/xiasha1_online_v1/run_queue.json \
        --logdir artifacts/xiasha1_online_v1/tb \
        --serve --port 6006 --interval 30
    # 单次同步/只投影指定 run 目录（冒烟验证用）：
    python dev/xiasha1_online_monitor/project_tb.py --once \
        --run-dir data/output_data/tsc/sumo_sac/.../sac_smoke_... \
        --logdir /tmp/xi1on_tb_smoke

TB 视图：每 run 一个目录（run_id），tag 命名 ``train/<metric>`` / ``eval/<metric>`` /
``best/<metric>``，横轴均为 episode。
"""
import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.training_dashboard import ScalarMirror  # noqa: E402

TRAIN_TAGS = ['reward_mean', 'loss_mean', 'travel_time', 'throughput', 'queue', 'delay',
              'real_delay', 'waiting_time', 'unfinished_vehicles', 'epsilon', 'replay_size',
              'update_to_data_ratio', 'phase_switch_frequency', 'wall_time_seconds']
EVAL_TAGS = ['travel_time', 'throughput', 'unfinished_vehicles', 'queue', 'delay',
             'real_delay', 'reward_mean', 'phase_switches']


def read_rows(path):
    """容错读 jsonl：末尾半行跳过（与 utils.training_monitor.read_rows 同语义）。"""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines(keepends=True):
        if not line.endswith('\n'):
            break
        if line.strip():
            rows.append(json.loads(line))
    return rows


def project_run(run_dir, run_id, points):
    """把单个 run 目录投影进 points[(run_id, tag, episode)] = value。"""
    records = run_dir / 'metrics' / 'records.jsonl'
    for row in read_rows(records):
        rtype = row.get('record_type')
        ep = row.get('episode')
        if ep is None:
            continue
        tags = TRAIN_TAGS if rtype == 'TRAIN' else EVAL_TAGS if rtype in (
            'EVALUATION', 'FINAL_EVALUATION') else []
        prefix = 'train' if rtype == 'TRAIN' else 'eval'
        for tag in tags:
            value = row.get(tag)
            if isinstance(value, (int, float)) and value == value:  # 排除 None/NaN
                points[(run_id, f'{prefix}/{tag}', ep)] = float(value)
    summary = run_dir / 'evaluation' / 'summary.json'
    if summary.exists():
        try:
            s = json.loads(summary.read_text())
            for tag, key in (('best/episode', 'best_episode'),
                             ('best/travel_time', 'best_travel_time')):
                if isinstance(s.get(key), (int, float)):
                    points[(run_id, tag, 0)] = float(s[key])
        except (json.JSONDecodeError, OSError):
            pass  # summary.json 可能正在写入


def collect_points(queue_path, run_dirs):
    points, seen = {}, []
    if queue_path:
        spec = json.loads(Path(queue_path).read_text())
        for task in spec['tasks']:
            run_dir = Path(task['output_path'])
            seen.append(task['run_id'])
            project_run(run_dir, task['run_id'], points)
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        project_run(run_dir, run_dir.name, points)
        seen.append(run_dir.name)
    return points, seen


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--queue', help='run_queue.json（用其中 output_path 定位 run 目录）')
    p.add_argument('--run-dir', action='append', default=[], help='额外 run 目录（可多次，冒烟验证用）')
    p.add_argument('--logdir', required=True, help='TB event 输出目录（建议 artifacts/ 下，勿入库）')
    p.add_argument('--interval', type=float, default=30, help='watch 同步周期秒')
    p.add_argument('--once', action='store_true', help='同步一次后退出')
    p.add_argument('--serve', action='store_true', help='同步同时拉起 tensorboard.main 子进程')
    p.add_argument('--port', type=int, default=6006)
    p.add_argument('--host', default='127.0.0.1')
    args = p.parse_args()

    logdir = Path(args.logdir).resolve()
    derived = logdir / 'runs'
    derived.mkdir(parents=True, exist_ok=True)
    mirror = ScalarMirror(derived)
    server, stopped = None, False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        while not stopped:
            started = time.monotonic()
            points, seen = collect_points(args.queue, args.run_dir)
            written = mirror.sync(points)
            status = {'schema': 'xiasha1_online_tb_projection-v1', 'updated_at_unix': time.time(),
                      'runs_seen': len(seen), 'points': len(points), 'new_events': written,
                      'sync_seconds': time.monotonic() - started,
                      'url': f'http://{args.host}:{args.port}' if args.serve else None,
                      'queue': str(args.queue)}
            (logdir / 'status.json.tmp').write_text(json.dumps(status, indent=2) + '\n')
            (logdir / 'status.json.tmp').replace(logdir / 'status.json')
            print(f"[{time.strftime('%H:%M:%S')}] runs={len(seen)} points={len(points)} "
                  f"new={written} sync={status['sync_seconds']:.1f}s", flush=True)
            if args.serve and server is None:
                server = subprocess.Popen([sys.executable, '-m', 'tensorboard.main',
                                           '--logdir', str(derived), '--port', str(args.port),
                                           '--host', args.host, '--reload_interval', '5'])
                print(f'TensorBoard: http://{args.host}:{args.port}', flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        mirror.close()
        if server is not None:
            server.terminate()


if __name__ == '__main__':
    main()
