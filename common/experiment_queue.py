"""Shared manifest-driven process queue; legacy arterial CLI re-exports this module."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import tempfile
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import psutil


STATUSES = {
    "PENDING", "RUNNING", "SUCCEEDED", "FAILED", "INTERRUPTED",
    "RESUMABLE", "SKIPPED_COMPLETED", "INVALID",
}
TERMINAL = {"SUCCEEDED", "FAILED", "SKIPPED_COMPLETED", "INVALID", "BLOCKED", "INTERRUPTED", "RESUMABLE"}
RECOVERABLE_FAILURES = {
    "RESOURCE_FAILURE", "SIMULATION_FAILURE", "INTERRUPTED",
    "ENVIRONMENT_FAILURE",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def completion_valid(task):
    completion = task.get("completion")
    if not completion:
        return False
    path = Path(completion["path"])
    if not path.is_file():
        return False
    if "json_field" not in completion:
        return True
    try:
        value = read_json(path)
        for key in completion["json_field"].split("."):
            value = value[key]
        return value == completion.get("equals")
    except (OSError, ValueError, KeyError, TypeError):
        return False


def classify_failure(exit_code, stderr_text):
    text = stderr_text.lower()
    if exit_code in {
            -signal.SIGINT, -signal.SIGTERM,
            128 + signal.SIGINT, 128 + signal.SIGTERM}:
        return "INTERRUPTED"
    if any(token in text for token in (
            "out of memory", "cannot allocate memory", "no space left",
            "killed")):
        return "RESOURCE_FAILURE"
    if any(token in text for token in (
            "nan", "inf loss", "q-value", "numerical")):
        return "NUMERICAL_FAILURE"
    if any(token in text for token in (
            "schema", "causal", "history access", "manifest", "config")):
        return "CONFIG_FAILURE"
    if any(token in text for token in (
            "checkpoint", "pickle data was truncated", "failed finding central")):
        return "CHECKPOINT_FAILURE"
    if any(token in text for token in (
            "libsumo", "traci", "sumo", "connection closed", "teleport")):
        return "SIMULATION_FAILURE"
    if any(token in text for token in (
            "no module named", "shared object file", "importerror")):
        return "ENVIRONMENT_FAILURE"
    return "UNKNOWN_FAILURE"


class QueueRunner:
    def __init__(self, args):
        self.args = args
        self.manifest_path = Path(args.manifest).resolve()
        self.payload = read_json(self.manifest_path)
        self.root = Path(args.state_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_json = self.root / "run_manifest.json"
        self.state_csv = self.root / "run_manifest.csv"
        self.status_json = self.root / "status.json"
        self.status_md = self.root / "status.md"
        self.resource_path = self.root / "resource_samples.jsonl"
        self.stop_requested = False
        self.running = {}
        self.samples = deque(maxlen=120)
        self.initial_swap = psutil.swap_memory().used
        self.previous_disk_io = psutil.disk_io_counters()
        self.previous_io_time = time.monotonic()
        self.tasks = self._load_tasks()
        self._adopt_existing_workers()

    @staticmethod
    def _live_matching_process(task):
        """Return a surviving task process only when PID and argv both match."""
        pid = task.get("pid")
        if not pid:
            return None
        try:
            process = psutil.Process(int(pid))
            if process.status() == psutil.STATUS_ZOMBIE:
                return None
            actual = process.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None
        expected = [str(value) for value in (task.get("active_command") or task.get("command", ()))]
        if actual != expected:
            return None
        return process

    def _adopt_existing_workers(self):
        """Monitor matching workers orphaned by a previous queue process."""
        for task in self.tasks:
            if task.get("status") != "RUNNING":
                continue
            process = self._live_matching_process(task)
            if process is None:
                task["status"] = (
                    "RESUMABLE" if self._checkpoint_exists(task)
                    else "INTERRUPTED")
                continue
            task_dir = self.root / "logs" / task["run_id"]
            attempt = int(task.get("retry_count", 0)) + 1
            self.running[task["run_id"]] = {
                "process": process, "task": task, "adopted": True,
                "stdout": None, "stderr": None,
                "stderr_path": task_dir / f"attempt_{attempt}.stderr.log",
                "started_monotonic": time.monotonic(),
            }

    def _load_tasks(self):
        raw_tasks = self.payload.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError("Queue manifest must contain a non-empty tasks list")
        previous = {}
        if self.state_json.is_file():
            previous = {item["run_id"]: item for item in read_json(
                self.state_json).get("tasks", [])}
        tasks = []
        seen_ids = set()
        seen_outputs = set()
        for raw in raw_tasks:
            task = dict(raw)
            if task.get('run_id') in seen_ids:
                raise ValueError('Duplicate queue run_id')
            seen_ids.add(task.get('run_id'))
            required = {"run_id", "command", "output_path"}
            missing = sorted(required - set(task))
            if missing or not isinstance(task["command"], list):
                task.update({
                    "status": "INVALID",
                    "failure_reason": f"missing/invalid fields: {missing}",
                })
            output = str(Path(task["output_path"]).resolve())
            if output in seen_outputs:
                task.update({
                    "status": "INVALID",
                    "failure_reason": "duplicate output_path",
                })
            seen_outputs.add(output)
            task["output_path"] = output
            task.setdefault("status", "PENDING")
            task.setdefault("pid", None)
            task.setdefault('active_command', None)
            task.setdefault("start_time", None)
            task.setdefault("end_time", None)
            task.setdefault("exit_code", None)
            task.setdefault("retry_count", 0)
            task.setdefault("failure_reason", None)
            task.setdefault("checkpoint_path", None)
            task.setdefault("config_hash", None)
            task.setdefault("git_commit", None)
            if task["run_id"] in previous:
                saved = previous[task["run_id"]]
                if ('command' in saved and saved['command'] != task['command']) or (
                        'output_path' in saved and saved['output_path'] != output):
                    raise ValueError('Queue manifest changed an existing task identity')
                for field in (
                        "status", "pid", "start_time", "end_time", "exit_code",
                        "retry_count", "failure_reason", "active_command"):
                    task[field] = saved.get(field, task[field])
                if (task["status"] == "RUNNING"
                        and self._live_matching_process(task) is None):
                    task["status"] = (
                        "RESUMABLE" if self._checkpoint_exists(task)
                        else "INTERRUPTED")
            if completion_valid(task):
                task["status"] = (
                    "SUCCEEDED" if task["status"] == "SUCCEEDED"
                    else "SKIPPED_COMPLETED")
            elif task["status"] in {"SUCCEEDED", "SKIPPED_COMPLETED"}:
                task["status"] = "INVALID"
                task["failure_reason"] = "completion evidence is missing or invalid"
            if (getattr(self.args, 'resume_failed', True)
                    and task['status'] in {'FAILED', 'INTERRUPTED', 'RESUMABLE', 'BLOCKED'}):
                task['use_resume'] = self._checkpoint_exists(task)
                task['status'] = 'PENDING'
                task['failure_reason'] = None
            tasks.append(task)
        return tasks

    @staticmethod
    def _latest_checkpoint(task):
        directory = Path(task["output_path"]) / "checkpoints" / "resumable"
        candidates = sorted(directory.glob("episode_*.pt"), reverse=True)
        return str(candidates[0].resolve()) if candidates else None

    @classmethod
    def _checkpoint_exists(cls, task):
        path = cls._latest_checkpoint(task)
        if path:
            task["checkpoint_path"] = path
            return True
        return False

    @staticmethod
    def _resume_command(task):
        command = list(task["command"])
        if "--resume-output" in command or "--resume-checkpoint" in command:
            raise ValueError("Queue task command already contains resume arguments")
        return command + [
            "--resume-output", task["output_path"],
            "--resume-checkpoint", task["checkpoint_path"],
        ]

    def persist(self):
        payload = {
            "schema_version": 1,
            "source_manifest": str(self.manifest_path),
            "updated_at": utc_now(),
            "tasks": self.tasks,
        }
        atomic_text(self.state_json, json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        fields = [
            "run_id", "plan", "method", "scene", "scene_order",
            "offline_ratio", "history_mode", "epsilon_mode", "training_seed",
            "episode_budget", "config_path", "output_path", "status", "pid",
            "start_time", "end_time", "exit_code", "retry_count",
            "checkpoint_path", "config_hash", "git_commit", "failure_reason",
        ]
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", newline="", dir=self.root,
                delete=False) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for task in self.tasks:
                writer.writerow(task)
            temporary = Path(handle.name)
        temporary.replace(self.state_csv)

    def resource_sample(self):
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_usage(self.root)
        disk_io = psutil.disk_io_counters()
        io_now = time.monotonic()
        io_elapsed = max(io_now - self.previous_io_time, 1e-9)
        read_rate = max(
            0, disk_io.read_bytes - self.previous_disk_io.read_bytes) / io_elapsed
        write_rate = max(
            0, disk_io.write_bytes - self.previous_disk_io.write_bytes) / io_elapsed
        self.previous_disk_io = disk_io
        self.previous_io_time = io_now
        load = os.getloadavg()
        process_rss = 0
        process_cpu = 0.0
        process_count = 0
        for context in self.running.values():
            try:
                root = psutil.Process(context["process"].pid)
                processes = [root, *root.children(recursive=True)]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            for process in processes:
                try:
                    process_rss += process.memory_info().rss
                    process_cpu += process.cpu_percent(interval=None)
                    process_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        sample = {
            "timestamp": utc_now(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": virtual.percent,
            "memory_available_bytes": virtual.available,
            "swap_used_bytes": swap.used,
            "swap_growth_bytes": max(0, swap.used - self.initial_swap),
            "disk_free_bytes": disk.free,
            "disk_percent": disk.percent,
            "disk_read_bytes_per_second": read_rate,
            "disk_write_bytes_per_second": write_rate,
            "load_1m": load[0], "load_5m": load[1], "load_15m": load[2],
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "running_jobs": len(self.running),
            "run_process_count": process_count,
            "run_process_rss_bytes": process_rss,
            "run_process_cpu_percent": process_cpu,
        }
        self.samples.append(sample)
        with self.resource_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
        return sample

    def submission_allowed(self, sample):
        reasons = []
        if sample["memory_percent"] >= self.args.max_memory_percent:
            reasons.append("memory")
        if sample["swap_growth_bytes"] >= self.args.max_swap_growth_mb * 1024 ** 2:
            reasons.append("swap")
        if sample["disk_free_bytes"] < self.args.min_disk_free_gb * 1024 ** 3:
            reasons.append("disk")
        if sample["load_1m"] > sample["logical_cpu_count"] * self.args.max_load_ratio:
            reasons.append("load")
        return not reasons, reasons

    def write_status(self, sample, gate_reasons=()):
        counts = Counter(task["status"] for task in self.tasks)
        payload = {
            "schema_version": 1, "updated_at": utc_now(),
            "counts": dict(sorted(counts.items())),
            "max_workers": self.args.max_workers,
            "resource_gate_reasons": list(gate_reasons),
            "latest_resource_sample": sample,
            "pool_workers": getattr(self.args, 'pool_workers', {}),
        }
        atomic_text(self.status_json, json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        lines = [
            "# Experiment queue status", "",
            f"Updated: {payload['updated_at']}", "",
            f"Max workers: {self.args.max_workers}", "",
            "## Task counts", "",
        ]
        lines.extend(f"- {key}: {value}" for key, value in sorted(counts.items()))
        lines.extend(["", "## Resources", "",
                      f"- CPU: {sample['cpu_percent']:.1f}%",
                      f"- Memory: {sample['memory_percent']:.1f}%",
                      f"- Swap growth: {sample['swap_growth_bytes']} bytes",
                      f"- Disk free: {sample['disk_free_bytes']} bytes",
                      f"- Load 1m: {sample['load_1m']:.2f}"])
        if gate_reasons:
            lines.extend(["", "Submission paused: " + ", ".join(gate_reasons)])
        atomic_text(self.status_md, "\n".join(lines) + "\n")
        callback = getattr(self.args, 'status_callback', None)
        if callback is not None:
            callback(payload)

    def launch_task(self, task):
        task_dir = self.root / "logs" / task["run_id"]
        task_dir.mkdir(parents=True, exist_ok=True)
        attempt = int(task["retry_count"]) + 1
        stdout_path = task_dir / f"attempt_{attempt}.stdout.log"
        stderr_path = task_dir / f"attempt_{attempt}.stderr.log"
        stdout = stdout_path.open("ab", buffering=0)
        stderr = stderr_path.open("ab", buffering=0)
        command = task["command"]
        if task["status"] == "RESUMABLE" or task.get('use_resume'):
            command = (task.get("resume_command")
                       or self._resume_command(task))
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in task.get("env", {}).items()})
        process = subprocess.Popen(
            command, cwd=task.get("cwd"), env=env, stdout=stdout, stderr=stderr,
            start_new_session=True)
        task.update({
            "status": "RUNNING", "pid": process.pid,
            "active_command": command,
            "start_time": utc_now(), "end_time": None, "exit_code": None,
            "failure_reason": None,
        })
        self.running[task["run_id"]] = {
            "process": process, "task": task, "stdout": stdout,
            "stderr": stderr, "stderr_path": stderr_path,
            "started_monotonic": time.monotonic(),
        }
        self.persist()

    def finish_task(self, context):
        process = context["process"]
        task = context["task"]
        if context.get("adopted"):
            exit_code = 0 if completion_valid(task) else None
        else:
            context["stdout"].close()
            context["stderr"].close()
            exit_code = process.returncode
        stderr_text = (context["stderr_path"].read_text(
            encoding="utf-8", errors="replace")
            if context["stderr_path"].is_file() else "")
        task.update({
            "pid": None, "end_time": utc_now(), "exit_code": exit_code,
            "wall_time_seconds": time.monotonic() - context["started_monotonic"],
        })
        if completion_valid(task):
            task["exit_code"] = 0
            task["status"] = "SUCCEEDED"
        elif context.get("adopted"):
            task["status"] = (
                "RESUMABLE" if self._checkpoint_exists(task)
                else "INTERRUPTED")
            task["failure_reason"] = "adopted worker exited without completion"
        elif exit_code == 0:
            task["status"] = "INVALID"
            task["failure_reason"] = "process exited 0 but completion check failed"
        else:
            failure = classify_failure(exit_code, stderr_text)
            task["failure_reason"] = failure
            if self._checkpoint_exists(task):
                task["status"] = "RESUMABLE"
            else:
                task["status"] = (
                    "INTERRUPTED" if failure == "INTERRUPTED" else "FAILED")
            if (failure in RECOVERABLE_FAILURES
                    and task["retry_count"] < self.args.max_retries
                    and (task.get("retry_command")
                         or task.get("resume_command")
                         or self._checkpoint_exists(task))):
                task["retry_count"] += 1
                task["command"] = task.get("retry_command", task["command"])
                task['use_resume'] = self._checkpoint_exists(task)
                task["status"] = "PENDING"
        self.persist()

    def handle_signal(self, signum, _frame):
        self.stop_requested = True
        for context in self.running.values():
            process = context["process"]
            if process.poll() is None:
                os.killpg(process.pid, signum)

    def ready(self, task):
        """Optional pools/dependencies; plain legacy manifests remain valid."""
        by_id = {t['run_id']: t for t in self.tasks}
        parents = [by_id[key] for key in task.get('depends_on', [])]
        if any(t['status'] not in {'SUCCEEDED', 'SKIPPED_COMPLETED'} for t in parents):
            if any(t['status'] in TERMINAL for t in parents
                   if t['status'] not in {'SUCCEEDED', 'SKIPPED_COMPLETED'}):
                task['status'] = 'BLOCKED'
            return False
        if not all(completion_valid({'completion': item}) for item in task.get('requires', [])):
            producer = by_id.get(task.get('producer'))
            if producer and producer['status'] in TERMINAL:
                task['status'] = 'BLOCKED'
                task['failure_reason'] = 'Producer ended without required checkpoint publication'
            return False
        limits = getattr(self.args, 'pool_workers', {}) or {}
        pool = task.get('pool')
        used = sum(ctx['task'].get('pool') == pool for ctx in self.running.values())
        return pool not in limits or used < limits[pool]

    def run(self):
        from sequential.launcher import LogicalRunLock
        with LogicalRunLock(self.root / 'queue.lock'):
            return self._run_locked()

    def _run_locked(self):
        self.persist()
        old_handlers = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        for signum in old_handlers:
            signal.signal(signum, self.handle_signal)
        last_sample = 0.0
        sample = self.resource_sample()
        try:
            while not self.stop_requested:
                now = time.monotonic()
                if now - last_sample >= self.args.monitor_interval:
                    sample = self.resource_sample()
                    last_sample = now
                allowed, reasons = self.submission_allowed(sample)
                self.write_status(sample, reasons)
                pending = [task for task in self.tasks if task["status"] == "PENDING"]
                pending.sort(key=lambda t: t.get('priority', 0))
                while (allowed and pending
                       and len(self.running) < self.args.max_workers):
                    task = next((t for t in pending if self.ready(t)), None)
                    if task is None:
                        break
                    pending.remove(task)
                    self.launch_task(task)
                    if self.args.stagger_seconds:
                        time.sleep(self.args.stagger_seconds)
                for run_id, context in list(self.running.items()):
                    process = context["process"]
                    if context.get("adopted"):
                        try:
                            finished = (not process.is_running()
                                        or process.status() == psutil.STATUS_ZOMBIE)
                        except (psutil.NoSuchProcess, psutil.ZombieProcess):
                            finished = True
                    else:
                        finished = process.poll() is not None
                    if finished:
                        self.finish_task(context)
                        del self.running[run_id]
                remaining = [task for task in self.tasks
                             if task["status"] not in TERMINAL]
                if not remaining and not self.running:
                    break
                if remaining and not pending and not self.running:
                    break
                time.sleep(self.args.poll_interval)
            return 0 if all(
                task["status"] in {"SUCCEEDED", "SKIPPED_COMPLETED"}
                for task in self.tasks) else 1
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            if self.running:
                for context in self.running.values():
                    if context.get("adopted"):
                        continue
                    process = context["process"]
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=10)
                    self.finish_task(context)
            self.persist()
            self.write_status(self.resource_sample())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--stagger-seconds", type=float, default=2.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--monitor-interval", type=float, default=5.0)
    parser.add_argument("--max-memory-percent", type=float, default=85.0)
    parser.add_argument("--max-swap-growth-mb", type=float, default=128.0)
    parser.add_argument("--min-disk-free-gb", type=float, default=100.0)
    parser.add_argument("--max-load-ratio", type=float, default=1.25)
    args = parser.parse_args()
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    return args


def main():
    raise SystemExit(QueueRunner(parse_args()).run())


if __name__ == "__main__":
    main()
