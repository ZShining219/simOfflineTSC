"""Resource snapshots and the real-SUMO Phase0 concurrency gate."""
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from sequential.io import atomic_json, read_json, sha256_file

from .config import load_config
from .validation import validate_sumo_command


def resource_snapshot(output_root):
    usage=shutil.disk_usage(output_root if os.path.exists(output_root) else os.getcwd())
    available_ram=None
    try:
        with open('/proc/meminfo',encoding='utf-8') as handle:
            for line in handle:
                if line.startswith('MemAvailable:'):
                    available_ram=int(line.split()[1])*1024; break
    except OSError: pass
    return {"disk_free_bytes":usage.free,"disk_total_bytes":usage.total,"memory_available_bytes":available_ram}


def _host_memory():
    values = {}
    with open('/proc/meminfo', encoding='utf-8') as handle:
        for line in handle:
            key, value = line.split(':', 1)
            if key in {'MemTotal', 'MemAvailable', 'SwapFree'}:
                values[key] = int(value.split()[0]) * 1024
    if set(values) != {'MemTotal', 'MemAvailable', 'SwapFree'}:
        raise RuntimeError('Cannot read host memory/swap counters')
    return values


def _run_level(level, root, baseline_seconds, *, interface, timeout_seconds,
               projected_formal_bytes):
    level_dir = root / f'concurrency_{level}'
    level_dir.mkdir(parents=True, exist_ok=False)
    config = load_config()
    seeds = [9500 + level * 10 + index for index in range(level)]
    children = []
    initial_memory = _host_memory()
    min_available = initial_memory['MemAvailable']
    min_swap_free = initial_memory['SwapFree']
    started = time.perf_counter()
    try:
        for index, seed in enumerate(seeds):
            output = level_dir / f'seed_{seed}'
            stdout_path = level_dir / f'seed_{seed}.stdout.log'
            stderr_path = level_dir / f'seed_{seed}.stderr.log'
            command = [
                sys.executable, '-m', 'plan5', 'smoke',
                '--algorithm', 'DDQN', '--output', str(output),
                '--seed', str(seed), '--episodes', '3',
                '--interface', interface, '--skip-evaluation-isolation',
            ]
            stdout_handle = open(stdout_path, 'w', encoding='utf-8')
            stderr_handle = open(stderr_path, 'w', encoding='utf-8')
            process = subprocess.Popen(
                command, cwd=os.getcwd(), stdout=stdout_handle,
                stderr=stderr_handle,
                env={
                    **os.environ, 'PYTHONNOUSERSITE': '1',
                    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                },
            )
            children.append({
                'process': process, 'seed': seed, 'output': output,
                'command': command, 'stdout_path': stdout_path,
                'stderr_path': stderr_path, 'stdout_handle': stdout_handle,
                'stderr_handle': stderr_handle,
            })
        while any(item['process'].poll() is None for item in children):
            elapsed = time.perf_counter() - started
            if elapsed > timeout_seconds:
                for item in children:
                    if item['process'].poll() is None:
                        item['process'].terminate()
                raise TimeoutError(
                    f'Plan5 concurrency {level} exceeded {timeout_seconds}s'
                )
            memory = _host_memory()
            min_available = min(min_available, memory['MemAvailable'])
            min_swap_free = min(min_swap_free, memory['SwapFree'])
            time.sleep(0.2)
        memory = _host_memory()
        min_available = min(min_available, memory['MemAvailable'])
        min_swap_free = min(min_swap_free, memory['SwapFree'])
    finally:
        for item in children:
            if item['process'].poll() is None:
                item['process'].wait(timeout=30)
            item['stdout_handle'].close()
            item['stderr_handle'].close()

    child_reports = []
    for item in children:
        report_path = item['output'] / 'smoke_report.json'
        report = read_json(report_path) if report_path.is_file() else None
        checkpoint_valid = bool(
            report and os.path.isfile(report['checkpoint_path'])
            and sha256_file(report['checkpoint_path'])
            == report['checkpoint_sha256']
        )
        counters = {} if report is None else report.get('counters', {})
        semantic_valid = bool(
            item['process'].returncode == 0 and report and report.get('valid')
            and report.get('config_sha256') == config.sha256()
            and report.get('network') == 'sumohz1x1'
            and report.get('episodes') == 3
            and report.get('decisions_per_episode') == 360
            and counters.get('global_decision_step') == 1080
            and counters.get('gradient_updates') == 80
            and counters.get('target_updates') == 8
            and checkpoint_valid
        )
        if report:
            try:
                validate_sumo_command(report['resolved_sumo_command'])
            except (KeyError, ValueError):
                semantic_valid = False
        child_reports.append({
            'seed': item['seed'], 'command': item['command'],
            'exit_code': item['process'].returncode,
            'output': str(item['output'].resolve()),
            'stdout_path': str(item['stdout_path'].resolve()),
            'stderr_path': str(item['stderr_path'].resolve()),
            'stdout_sha256': sha256_file(item['stdout_path']),
            'stderr_sha256': sha256_file(item['stderr_path']),
            'smoke_report': report, 'checkpoint_valid': checkpoint_valid,
            'semantic_valid': semantic_valid,
        })

    run_seconds = [
        float(item['smoke_report']['wall_time_seconds'])
        for item in child_reports if item['smoke_report']
    ]
    median_seconds = statistics.median(run_seconds) if run_seconds else None
    slowdown = (
        1.0 if level == 1 and median_seconds is not None
        else None if baseline_seconds in (None, 0) or median_seconds is None
        else median_seconds / baseline_seconds
    )
    disk = shutil.disk_usage(root)
    disk_after_projected = disk.free - int(projected_formal_bytes)
    peak_ram = initial_memory['MemTotal'] - min_available
    swap_growth = max(0, initial_memory['SwapFree'] - min_swap_free)
    paths = [item['output'] for item in children]
    result = {
        'level': level, 'children': child_reports,
        'all_complete': all(item['exit_code'] == 0 for item in child_reports)
        and len(child_reports) == level,
        'no_collisions': len(set(paths)) == level,
        'semantic_valid': all(
            item['semantic_valid'] for item in child_reports
        ),
        'host_memory_total_bytes': initial_memory['MemTotal'],
        'peak_host_ram_bytes': peak_ram,
        'ram_ok': peak_ram <= 0.80 * initial_memory['MemTotal'],
        'swap_growth_bytes': swap_growth,
        'swap_ok': swap_growth <= 1024 ** 3,
        'disk_free_bytes': disk.free,
        'projected_formal_output_bytes': int(projected_formal_bytes),
        'disk_free_after_projected_bytes': disk_after_projected,
        'disk_ok': disk_after_projected >= 150 * 1024 ** 3,
        'median_run_wall_time_seconds': median_seconds,
        'single_run_baseline_seconds': baseline_seconds,
        'median_slowdown': slowdown,
        'slowdown_ok': slowdown is not None and slowdown <= 2.5,
        'batch_wall_time_seconds': time.perf_counter() - started,
    }
    result['passed'] = all(result[key] for key in (
        'all_complete', 'no_collisions', 'semantic_valid', 'ram_ok',
        'swap_ok', 'disk_ok', 'slowdown_ok',
    ))
    atomic_json(level_dir / 'level_report.json', result)
    return result


def run_concurrency_gate(output_root, report_path, *, interface='traci',
                         timeout_seconds=1800,
                         projected_formal_bytes=150 * 1024 ** 3):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    results = {}
    baseline = None
    for level in (1, 4, 8):
        results[level] = _run_level(
            level, output_root, baseline, interface=interface,
            timeout_seconds=timeout_seconds,
            projected_formal_bytes=projected_formal_bytes,
        )
        if level == 1:
            baseline = results[level]['median_run_wall_time_seconds']
            results[level]['single_run_baseline_seconds'] = baseline
            atomic_json(
                output_root / 'concurrency_1' / 'level_report.json',
                results[level],
            )
    selected = select_concurrency(results)
    report = {
        'schema_version': 1, 'profile': 'DDQN S2 3-episode smoke',
        'levels': {str(key): value for key, value in results.items()},
        'selected_formal_concurrency': selected,
        'valid': selected is not None,
    }
    atomic_json(report_path, report)
    return report


def select_concurrency(results):
    """Select highest passing level, requiring all semantic/resource gates."""
    for level in (8,4,1):
        item=results.get(level)
        if item and item.get('all_complete') and item.get('no_collisions') and item.get('semantic_valid') and item.get('ram_ok') and item.get('swap_ok') and item.get('disk_ok') and item.get('slowdown_ok'):
            return level
    return None
