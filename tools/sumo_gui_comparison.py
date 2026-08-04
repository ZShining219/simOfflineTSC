"""Replay three recorded SUMO controllers in synchronized GUI screenshots.

The tool consumes the existing decision-level ``decisions.jsonl`` files.  It
does not run training or infer new actions: each recorded action is applied for
the same action interval as in the source evaluation.  SUMO-GUI is started
through TraCI, screenshots are captured at common integer simulation times, and
the three screenshots are assembled into a simple side-by-side GIF.

This is intentionally separate from the normal experiment runner.  It is a
presentation/replay utility and therefore never changes a source simulator
configuration or an experiment output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

def _repo_root():
    return Path(__file__).resolve().parents[1]


def _is_native_executable(path):
    try:
        return (
            path.is_file() and os.access(path, os.X_OK)
            and path.read_bytes()[:4] == b"\x7fELF"
        )
    except OSError:
        return False


def _ensure_sumo_home():
    """Select a usable SUMO installation when an old wrapper is present."""
    current = os.environ.get("SUMO_HOME")
    if current and _is_native_executable(Path(current) / "bin" / "sumo-gui"):
        return Path(current)
    prefix = Path(sys.prefix)
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [prefix / "lib" / f"python{version}" / "site-packages" / "sumo"]
    for candidate in candidates:
        if _is_native_executable(candidate / "bin" / "sumo-gui"):
            os.environ["SUMO_HOME"] = str(candidate)
            return candidate
    raise RuntimeError(
        "No usable SUMO-GUI binary was found. Set SUMO_HOME to a SUMO installation "
        "whose bin/sumo-gui is executable."
    )


if str(_repo_root()) not in sys.path:
    sys.path.insert(0, str(_repo_root()))

from tools.traffic_visualization import METHODS, METHOD_LABELS, load_decisions


def _load_replay(path, method, scene, controller_id=None, evaluation_seed=None,
                 training_seed=None):
    _, records = load_decisions(
        path, controller_id=controller_id, evaluation_seed=evaluation_seed,
        training_seed=training_seed,
    )
    prepared = []
    for record in records:
        if str(record["network"]) != scene:
            raise ValueError(
                f"{method} record network {record['network']!r} does not match "
                f"scene {scene!r}"
            )
        if not isinstance(record.get("actions"), list) or not record["actions"]:
            raise ValueError(f"{method} record has no actions")
        interval = int(record.get("action_interval_seconds", 0))
        if interval <= 0:
            raise ValueError(f"{method} has an invalid action interval: {interval}")
        prepared.append({
            "time": float(record["simulation_time_seconds"]),
            "actions": [int(action) for action in record["actions"]],
            "interval": interval,
        })
    prepared.sort(key=lambda item: item["time"])
    if not prepared:
        raise ValueError(f"{method} has no replay records")
    intervals = {item["interval"] for item in prepared}
    if len(intervals) != 1:
        raise ValueError(f"{method} changes action interval within one replay")
    interval = intervals.pop()
    expected_time = float(interval)
    for item in prepared:
        if abs(item["time"] - expected_time) > 1e-6:
            raise ValueError(
                f"{method} replay timestamps are not contiguous from t={interval}: "
                f"expected {expected_time}, got {item['time']}"
            )
        expected_time += interval
    return prepared, interval


def _make_world_config(source_path, temp_dir, method, gui=True):
    source_path = Path(source_path).expanduser().resolve()
    with source_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"SUMO config must be a JSON object: {source_path}")
    data_dir = (_repo_root() / "data").resolve()
    config["dir"] = str(data_dir) + os.sep
    config["gui"] = bool(gui)
    config["name"] = f"traffic_gui_{method}"
    target = Path(temp_dir) / f"{method}.cfg"
    target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return target


def _configure_registry(world_config, evaluation_seed, output_dir):
    # Importing world only after SUMO_HOME is checked keeps this utility
    # importable for command-line help and unit tests.
    _ensure_sumo_home()
    from common.registry import Registry
    from common import interface  # noqa: F401 - registers runtime interfaces
    import world  # noqa: F401 - registers SUMO World

    Registry.mapping["command_mapping"]["setting"].param = {
        "interface": "traci",
        "world": "sumo",
        "sumo_seed": int(evaluation_seed),
    }
    Registry.mapping["world_mapping"]["setting"].param = json.loads(
        world_config.read_text(encoding="utf-8")
    )
    Registry.mapping["logger_mapping"]["path"].path = str(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return Registry.mapping["world_mapping"]["sumo"]


def _font(size=18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _compose_frame(frame_paths, output_path, timestamp):
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    header_height = 34
    canvas = Image.new("RGB", (width * len(images), height + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = _font(18)
    for index, (image, method) in enumerate(zip(images, METHODS)):
        x = index * width
        canvas.paste(image, (x, header_height))
        draw.text((x + 8, 7), f"{METHOD_LABELS[method]} | t={timestamp:.0f}s",
                  fill="black", font=label_font)
        image.close()
    canvas.save(output_path)
    return canvas.size


def _capture_method(method, records, interval, source_config, seed, start, end,
                    output_dir, screenshot_every):
    method_dir = Path(output_dir) / method
    frames_dir = method_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = method_dir / "runtime"
    temp_dir.mkdir(parents=True, exist_ok=True)
    world_config = _make_world_config(source_config, temp_dir, method)
    world_class = _configure_registry(world_config, seed, method_dir)
    world = None
    captured = []
    try:
        world = world_class(str(world_config), interface="traci")
        world.reset()
        view_id = getattr(world.eng.gui, "DEFAULT_VIEW", "View #0")
        current_time = 0
        for record in records:
            action = record["actions"]
            if len(action) != len(world.intersections):
                raise ValueError(
                    f"{method} action count {len(action)} does not match SUMO "
                    f"intersection count {len(world.intersections)}"
                )
            for _ in range(interval):
                # TraCI's gui.screenshot() is executed at the *next*
                # simulationStep. Queue it immediately before the step so the
                # file represents the state at next_time, rather than one
                # second later or an unmaterialized request at the end.
                next_time = int(round(float(world.get_current_time()))) + 1
                if (
                    start <= next_time <= end
                    and (next_time - start) % screenshot_every == 0
                ):
                    frame_path = frames_dir / f"frame_{next_time:06d}.png"
                    world.eng.gui.screenshot(view_id, str(frame_path))
                    captured.append({"time": next_time, "path": str(frame_path)})
                world.step(action)
                current_time = int(round(float(world.get_current_time())))
            if current_time >= end:
                break
        if not captured:
            raise ValueError(f"{method} captured no frames in [{start}, {end}]")
        return captured
    finally:
        if world is not None:
            world.close()
        # Keep screenshots and logs, but the generated configuration is only a
        # runtime detail and can be reconstructed from the command.
        try:
            world_config.unlink()
        except FileNotFoundError:
            pass


def render_gui_comparison(inputs, scene, source_config, evaluation_seed, output,
                          start, end, selectors=None, screenshot_every=1, fps=10):
    if start < 0 or end <= start or int(start) != start or int(end) != end:
        raise ValueError("start/end must be integer seconds with 0 <= start < end")
    if screenshot_every <= 0:
        raise ValueError("screenshot_every must be positive")
    selectors = selectors or {}
    replay = {}
    intervals = set()
    for method in METHODS:
        replay[method], interval = _load_replay(
            inputs[method], method, scene, **selectors.get(method, {})
        )
        intervals.add(interval)
    if len(intervals) != 1:
        raise ValueError(f"Controllers use different action intervals: {intervals}")
    common_interval = intervals.pop()
    if start % common_interval != 0:
        raise ValueError("start must align with the common action interval")

    output_dir = Path(output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    captures = {}
    for method in METHODS:
        captures[method] = _capture_method(
            method, replay[method][0], common_interval, source_config, evaluation_seed,
            int(start), int(end), output_dir, int(screenshot_every),
        )
    by_method_time = {
        method: {item["time"]: item["path"] for item in captures[method]}
        for method in METHODS
    }
    common_times = sorted(set.intersection(*(set(values) for values in by_method_time.values())))
    if len(common_times) < 2:
        raise ValueError("Fewer than two common GUI screenshot times were captured")
    panel_dir = output_dir / "comparison_frames"
    panel_dir.mkdir(exist_ok=True)
    panel_paths = []
    panel_size = None
    for time_seconds in common_times:
        paths = [by_method_time[method][time_seconds] for method in METHODS]
        panel_path = panel_dir / f"frame_{time_seconds:06d}.png"
        panel_size = _compose_frame(paths, panel_path, time_seconds)
        panel_paths.append(panel_path)
    gif_path = output_dir / "sumo_gui_comparison.gif"
    images = [Image.open(path).convert("RGB") for path in panel_paths]
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   duration=max(1, round(1000 / fps)), loop=0)
    for image in images:
        image.close()
    manifest = {
        "schema_version": 1,
        "scene": scene,
        "evaluation_seed": int(evaluation_seed),
        "source_config": str(Path(source_config).resolve()),
        "methods": list(METHODS),
        "action_interval_seconds": common_interval,
        "screenshot_every_seconds": int(screenshot_every),
        "timestamps": {"count": len(common_times), "start": common_times[0],
                       "end": common_times[-1]},
        "panel_size": panel_size,
        "raw_screenshots": captures,
        "comparison_frames": [str(path) for path in panel_paths],
        "gif": str(gif_path),
        "replay_mode": "recorded_actions_exact_interval",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--sim-config", default=None)
    parser.add_argument("--online-dqn", required=True)
    parser.add_argument("--hadhoa", required=True)
    parser.add_argument("--fixedtime", required=True)
    for method in METHODS:
        parser.add_argument(f"--{method.replace('_', '-')}-controller-id")
        parser.add_argument(f"--{method.replace('_', '-')}-training-seed", type=int)
        parser.add_argument(f"--{method.replace('_', '-')}-evaluation-seed", type=int)
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--screenshot-every", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    _ensure_sumo_home()
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError(
            "SUMO-GUI requires a desktop display; set DISPLAY/WAYLAND_DISPLAY "
            "or run this command on a machine with X/Wayland (optionally Xvfb)."
        )
    source_config = args.sim_config or str(_repo_root() / "configs" / "sim" / f"{args.scene}.cfg")
    selectors = {
        method: {
            "controller_id": getattr(args, f"{method}_controller_id"),
            "training_seed": getattr(args, f"{method}_training_seed"),
            "evaluation_seed": getattr(args, f"{method}_evaluation_seed"),
        }
        for method in METHODS
    }
    manifest = render_gui_comparison(
        {method: getattr(args, method) for method in METHODS},
        args.scene, source_config, args.evaluation_seed, args.output, args.start,
        args.end, selectors, args.screenshot_every, args.fps,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
