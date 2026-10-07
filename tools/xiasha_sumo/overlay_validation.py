"""Overlay a SUMO network on a real aerial image and report alignment evidence.

This utility is deliberately independent of the simulator runtime.  It reads a
SUMO ``.net.xml`` file, projects lane shapes into image pixels, writes a
transparent overlay and a side-by-side comparison, and records reproducible
geometry/luminance diagnostics in JSON.  The default affine transform fits the
network bounds to the image (with the SUMO y axis flipped).  For a georeferenced
or manually calibrated view, pass three or more ``--control-point`` arguments
of the form ``world_x,world_y,image_x,image_y``.

Example::

    python tools/xiasha_sumo/overlay_validation.py \
      --image a1俯视底图.png \
      --network data/raw_data/xiasha1*1/xiasha1_sumo_signal.net.xml \
      --output-dir final_result/xiasha_sumo_overlay

The generated files are presentation/evidence artefacts and never modify SUMO
source files or experiment outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _parse_shape(value: str | None) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for token in (value or "").split():
        try:
            x, y = token.split(",")[:2]
            points.append((float(x), float(y)))
        except (ValueError, IndexError):
            continue
    return points


def load_network(path: Path) -> tuple[list[dict], list[dict], dict]:
    """Read drawable lane and junction geometry from a SUMO net XML."""
    root = ET.parse(path).getroot()
    lanes: list[dict] = []
    points: list[tuple[float, float]] = []
    for edge in root.findall("edge"):
        internal = edge.get("function") == "internal" or edge.get("id", "").startswith(":")
        for lane in edge.findall("lane"):
            shape = _parse_shape(lane.get("shape"))
            if len(shape) < 2:
                continue
            item = {
                "id": lane.get("id", ""),
                "edge": edge.get("id", ""),
                "internal": bool(internal),
                "width": float(lane.get("width", 3.2)),
                "shape": shape,
            }
            lanes.append(item)
            points.extend(shape)
    junctions: list[dict] = []
    for junction in root.findall("junction"):
        try:
            x, y = float(junction.get("x")), float(junction.get("y"))
        except (TypeError, ValueError):
            continue
        junctions.append({"id": junction.get("id", ""), "type": junction.get("type", ""), "x": x, "y": y})
        points.append((x, y))
    if not points:
        raise ValueError(f"No drawable lane/junction geometry in {path}")
    xs, ys = zip(*points)
    bounds = {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)}
    return lanes, junctions, bounds


def _fit_affine(bounds: dict, size: tuple[int, int], margin: float = 0.04, flip_y: bool = True) -> np.ndarray:
    width, height = size
    x0, x1 = float(bounds["min_x"]), float(bounds["max_x"])
    y0, y1 = float(bounds["min_y"]), float(bounds["max_y"])
    span_x, span_y = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    px0, px1 = margin * width, (1.0 - margin) * width
    py0, py1 = margin * height, (1.0 - margin) * height
    sx, sy = (px1 - px0) / span_x, (py1 - py0) / span_y
    if flip_y:
        sy = -sy
        ty = py1 - sy * y0
    else:
        ty = py0 - sy * y0
    return np.array([[sx, 0.0, px0 - sx * x0], [0.0, sy, ty]], dtype=float)


def _control_affine(points: Sequence[Sequence[float]]) -> np.ndarray:
    if len(points) < 3:
        raise ValueError("At least three control points are required for affine calibration")
    a = np.array([[float(p[0]), float(p[1]), 1.0] for p in points], dtype=float)
    bx = np.array([float(p[2]) for p in points], dtype=float)
    by = np.array([float(p[3]) for p in points], dtype=float)
    mx, *_ = np.linalg.lstsq(a, bx, rcond=None)
    my, *_ = np.linalg.lstsq(a, by, rcond=None)
    predicted = a @ np.vstack((mx, my)).T
    residual = np.sqrt(np.sum((predicted - np.column_stack((bx, by))) ** 2, axis=1))
    # Keep the solver intentionally small and dependency free.  Residuals are
    # recomputed by callers when they need to audit control-point quality.
    return np.vstack((mx, my))


def _project(point: Sequence[float], transform: np.ndarray) -> tuple[float, float]:
    result = transform @ np.array([float(point[0]), float(point[1]), 1.0])
    return float(result[0]), float(result[1])


def _polyline_length(shape: Sequence[Sequence[float]]) -> float:
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(shape, shape[1:])))


def _iter_samples(shape: Sequence[Sequence[float]], spacing: float = 2.0) -> Iterable[tuple[float, float, float, float]]:
    """Yield world x/y and unit normal for evenly spaced shape samples."""
    for a, b in zip(shape, shape[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        count = max(1, int(math.ceil(length / spacing)))
        nx, ny = -dy / length, dx / length
        for i in range(count):
            t = (i + 0.5) / count
            yield a[0] + t * dx, a[1] + t * dy, nx, ny


def _sample_luma(gray: np.ndarray, x: float, y: float) -> float | None:
    ix, iy = int(round(x)), int(round(y))
    if ix < 0 or iy < 0 or iy >= gray.shape[0] or ix >= gray.shape[1]:
        return None
    return float(gray[iy, ix])


def alignment_metrics(image: Image.Image, lanes: Sequence[dict], transform: np.ndarray) -> tuple[dict, list[dict]]:
    """Compute geometry summaries and a dark-road contrast proxy per lane."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    lane_rows: list[dict] = []
    center_values: list[float] = []
    background_values: list[float] = []
    for lane in lanes:
        center: list[float] = []
        background: list[float] = []
        for wx, wy, nx, ny in _iter_samples(lane["shape"]):
            px, py = _project((wx, wy), transform)
            value = _sample_luma(gray, px, py)
            if value is None:
                continue
            # A road lane is sampled at its center; the ring is offset outside
            # the lane.  The contrast is descriptive, not a pass/fail test.
            ring = [_sample_luma(gray, px + nx * offset, py + ny * offset) for offset in (10.0, 18.0)]
            ring = [v for v in ring if v is not None]
            center.append(value)
            background.extend(ring)
        cmean = float(np.mean(center)) if center else None
        bmean = float(np.mean(background)) if background else None
        row = {
            "lane_id": lane["id"],
            "edge_id": lane["edge"],
            "internal": lane["internal"],
            "length_m": round(_polyline_length(lane["shape"]), 3),
            "samples": len(center),
            "center_luma": None if cmean is None else round(cmean, 3),
            "ring_luma": None if bmean is None else round(bmean, 3),
            "dark_road_contrast": None if cmean is None or bmean is None else round(bmean - cmean, 3),
        }
        lane_rows.append(row)
        center_values.extend(center)
        background_values.extend(background)
    metrics = {
        "lane_count": len(lanes),
        "internal_lane_count": sum(1 for lane in lanes if lane["internal"]),
        "total_lane_length_m": round(sum(_polyline_length(lane["shape"]) for lane in lanes), 3),
        "sample_count": len(center_values),
        "center_luma_mean": None if not center_values else round(float(np.mean(center_values)), 3),
        "ring_luma_mean": None if not background_values else round(float(np.mean(background_values)), 3),
        "dark_road_contrast_mean": None if not center_values or not background_values else round(float(np.mean(background_values) - np.mean(center_values)), 3),
        "contrast_interpretation": "positive values indicate darker network centerlines than nearby image pixels",
    }
    return metrics, lane_rows


def _font(size: int = 24):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_overlay(image: Image.Image, lanes: Sequence[dict], junctions: Sequence[dict], transform: np.ndarray, output: Path, network_only: Path | None = None) -> None:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for lane in lanes:
        points = [_project(p, transform) for p in lane["shape"]]
        if len(points) < 2:
            continue
        xy = [(round(x), round(y)) for x, y in points]
        color = (255, 94, 0, 185) if lane["internal"] else (0, 174, 255, 185)
        width = max(2, int(round(lane["width"] * max(abs(transform[0, 0]), abs(transform[1, 1])))))
        draw.line(xy, fill=(20, 20, 20, 100), width=width + 3, joint="curve")
        draw.line(xy, fill=color, width=2, joint="curve")
    for junction in junctions:
        x, y = _project((junction["x"], junction["y"]), transform)
        r = 8
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 255, 220), outline=(255, 255, 255, 255), width=2)
        draw.text((x + 10, y - 12), str(junction["id"]), fill=(255, 255, 255, 255), font=_font(18), stroke_width=2, stroke_fill=(0, 0, 0, 220))
    result = Image.alpha_composite(base, layer)
    result.convert("RGB").save(output)
    if network_only is not None:
        layer.save(network_only)


def _write_side_by_side(image: Image.Image, overlay_path: Path, output: Path) -> None:
    over = Image.open(overlay_path).convert("RGB")
    h = max(image.height, over.height)
    canvas = Image.new("RGB", (image.width + over.width, h + 42), "white")
    canvas.paste(image.convert("RGB"), (0, 42))
    canvas.paste(over, (image.width, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "Aerial reference", fill="black", font=_font(22))
    draw.text((image.width + 12, 12), "SUMO overlay", fill="black", font=_font(22))
    canvas.save(output)


def _parse_control(value: str) -> tuple[float, float, float, float]:
    values = [float(item.strip()) for item in value.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("control point must be world_x,world_y,image_x,image_y")
    return tuple(values)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-point", action="append", type=_parse_control, default=[], help="world_x,world_y,image_x,image_y (repeat at least three times)")
    parser.add_argument("--margin", type=float, default=0.04, help="fractional image margin used by automatic fitting")
    parser.add_argument("--no-flip-y", action="store_true", help="do not invert SUMO's upward y axis for image coordinates")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image = Image.open(args.image).convert("RGB")
    lanes, junctions, bounds = load_network(args.network)
    transform = _control_affine(args.control_point) if args.control_point else _fit_affine(bounds, image.size, args.margin, not args.no_flip_y)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = args.output_dir / "network_overlay.png"
    network_path = args.output_dir / "network_geometry.png"
    side_path = args.output_dir / "aerial_vs_sumo_overlay.png"
    metrics_path = args.output_dir / "alignment_metrics.json"
    lanes_path = args.output_dir / "lane_alignment.csv"
    render_overlay(image, lanes, junctions, transform, overlay_path, network_path)
    _write_side_by_side(image, overlay_path, side_path)
    metrics, lane_rows = alignment_metrics(image, lanes, transform)
    calibration_residual = None
    if args.control_point:
        residuals = []
        for wx, wy, px, py in args.control_point:
            qx, qy = _project((wx, wy), transform)
            residuals.append(math.hypot(qx - px, qy - py))
        calibration_residual = {
            "count": len(residuals),
            "rmse_px": round(float(np.sqrt(np.mean(np.square(residuals)))), 4),
            "max_px": round(float(max(residuals)), 4),
            "values_px": [round(float(value), 4) for value in residuals],
        }
    payload = {
        "image": str(args.image.resolve()),
        "network": str(args.network.resolve()),
        "image_size_px": {"width": image.width, "height": image.height},
        "network_bounds_m": bounds,
        "transform_world_to_image": [[round(float(v), 9) for v in row] for row in transform],
        "calibration": "control_points" if args.control_point else "automatic_bounds_fit",
        "control_points": [list(point) for point in args.control_point],
        "calibration_residual": calibration_residual,
        "metrics": metrics,
        "outputs": {"overlay": str(overlay_path), "network_geometry": str(network_path), "side_by_side": str(side_path), "lane_csv": str(lanes_path)},
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with lanes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(lane_rows[0]) if lane_rows else ["lane_id"])
        writer.writeheader()
        writer.writerows(lane_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
