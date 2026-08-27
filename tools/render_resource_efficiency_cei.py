#!/usr/bin/env python3
"""Render CEI improvement against the frozen continuous Online-DQN baseline.

This is a read-only presentation utility.  It derives the congestion-efficiency
index (CEI) from the existing final-stage scene summary and does not train an
agent or start SUMO.

CEI is the equal-weight reduction in normalized queue burden and real delay:

    CEI = 0.5 * queue_auc_reduction + 0.5 * real_delay_reduction

The input summary already contains the two scene-level reductions relative to
the matched CONT-FIFO frozen DQN condition.  The project summary is an equal-
scene mean, so a high-improvement scene cannot replace the other scenes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/output_data/resource_efficiency_ha_vs_continuous/resource_efficiency_scenario_summary.csv"
DEFAULT_OUTPUT = ROOT / "data/output_data/resource_efficiency_ha_vs_continuous"

SCENES = ("sumohz1x1_config2", "sumohz1x1", "sumohz1x1_config4", "sumohz1x1_config3")
SCENE_LABELS = {
    "sumohz1x1_config2": "S1",
    "sumohz1x1": "S2",
    "sumohz1x1_config4": "S3",
    "sumohz1x1_config3": "S4",
}
METHODS = ("P1C-DHOA-R25", "P1C-DHOA-R50")
COLORS = {"P1C-DHOA-R25": "#4c78a8", "P1C-DHOA-R50": "#f58518"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value in (None, ""):
        raise ValueError(f"Missing {field} in row: {row}")
    return float(value)


def derive(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    selected = [
        row
        for row in rows
        if row.get("scope") == "final_stage_scene"
        and row.get("method") in METHODS
        and row.get("scenario") in SCENES
        and row.get("scene") in SCENE_LABELS.values()
    ]
    expected = {(method, scene) for method in METHODS for scene in SCENES}
    found = {(row["method"], row["scenario"]) for row in selected}
    missing = sorted(expected - found)
    if missing:
        raise ValueError(f"Missing final-stage scene rows: {missing}")

    output: list[dict[str, object]] = []
    for row in sorted(selected, key=lambda item: (METHODS.index(item["method"]), SCENES.index(item["scenario"]))):
        queue = number(row, "queue_auc_reduction_pct_mean")
        delay = number(row, "real_delay_reduction_pct_mean")
        output.append(
            {
                "scope": "scene",
                "method": row["method"],
                "scene": row["scene"],
                "source_scene": row["scenario"],
                "n_pairs": int(row["n_pairs"]),
                "coverage_orders": row.get("orders", ""),
                "queue_auc_reduction_pct": queue,
                "real_delay_reduction_pct": delay,
                "cei_improvement_pct": 0.5 * (queue + delay),
            }
        )

    for method in METHODS:
        method_rows = [row for row in output if row["method"] == method]
        output.append(
            {
                "scope": "scene_equal_macro",
                "method": method,
                "scene": "Equal-scene summary",
                "source_scene": "S1;S2;S3;S4",
                "n_pairs": sum(int(row["n_pairs"]) for row in method_rows),
                "coverage_orders": ";".join(sorted({order for row in method_rows for order in str(row["coverage_orders"]).split(";") if order})),
                "queue_auc_reduction_pct": sum(float(row["queue_auc_reduction_pct"]) for row in method_rows) / len(method_rows),
                "real_delay_reduction_pct": sum(float(row["real_delay_reduction_pct"]) for row in method_rows) / len(method_rows),
                "cei_improvement_pct": sum(float(row["cei_improvement_pct"]) for row in method_rows) / len(method_rows),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "scope",
        "method",
        "scene",
        "source_scene",
        "n_pairs",
        "coverage_orders",
        "queue_auc_reduction_pct",
        "real_delay_reduction_pct",
        "cei_improvement_pct",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def annotate(ax, bars, values) -> None:
    for bar, value in zip(bars, values):
        y = float(value) + (1.2 if float(value) >= 0 else -1.2)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{float(value):.1f}%",
            ha="center",
            va="bottom" if float(value) >= 0 else "top",
            fontsize=8,
        )


def render(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    method = "P1C-DHOA-R25"
    scenes = ["S1", "S2", "S3", "S4"]
    macro = next(
        row
        for row in rows
        if row["scope"] == "scene_equal_macro" and row["method"] == method
    )
    scene_values = [
        next(
            row
            for row in rows
            if row["scope"] == "scene"
            and row["method"] == method
            and row["scene"] == scene
        )["cei_improvement_pct"]
        for scene in scenes
    ]
    values = [*scene_values, macro["cei_improvement_pct"]]
    labels = [*scenes, "4 Orders\nequal-scene"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    bars = ax.bar(
        x,
        values,
        width=0.62,
        color=[COLORS[method]] * len(scenes) + ["#254f7d"],
    )
    annotate(ax, bars, values)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(10, color="crimson", linestyle="--", linewidth=1.5, label="10% target")
    ax.grid(axis="y", alpha=0.22)
    ax.set_ylabel("CEI improvement (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 20)
    ax.set_title("P1C-DHOA-R25: scene CEI and four-order equal-scene summary")
    ax.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        "CEI improvement relative to frozen continuous Online DQN\n"
        "CEI = 0.5 × queue-AUC reduction + 0.5 × real-delay reduction"
    )
    fig.tight_layout()
    output = output_dir / "resource_efficiency_cei_vs_dqn.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = derive(read_rows(args.input_csv))
    csv_path = args.output_dir / "resource_efficiency_cei_vs_dqn.csv"
    write_csv(csv_path, rows)
    image_path = render(args.output_dir, rows)
    macro = [row for row in rows if row["scope"] == "scene_equal_macro"]
    print(f"input={args.input_csv}")
    print(f"csv={csv_path}")
    print(f"image={image_path}")
    for row in macro:
        print(f"{row['method']}: CEI={float(row['cei_improvement_pct']):.2f}%")


if __name__ == "__main__":
    main()
