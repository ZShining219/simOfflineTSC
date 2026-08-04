#!/usr/bin/env python3
"""Render a frozen-data throughput and allocation-accuracy evidence dashboard."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENES = ("sumohz1x1_config2", "sumohz1x1", "sumohz1x1_config4", "sumohz1x1_config3")
LABEL = dict(zip(SCENES, ("S1", "S2", "S3", "S4")))
METHODS = ("P1C-DHOA-R25", "P1C-DHOA-R50")
COLORS = {"P1C-DHOA-R25": "#4c78a8", "P1C-DHOA-R50": "#f58518"}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    efficiency = ROOT / "data/output_data/resource_efficiency_p1c_dhoa_r25_r50"
    p.add_argument("--throughput-detail", type=Path, default=efficiency / "resource_efficiency_throughput_detail.csv")
    p.add_argument("--efficiency-episodes", type=Path, default=efficiency / "resource_efficiency_episode_level.csv")
    p.add_argument("--output", type=Path, default=efficiency / "resource_throughput_accuracy_evidence_dashboard.png")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def number(row, field):
    value = row.get(field, "")
    return float(value) if value not in (None, "") else None


def one(items, **conditions):
    matches = [r for r in items if all(r.get(k) == v for k, v in conditions.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {conditions}, found {len(matches)}")
    return matches[0]


def plot_equal_weight_index(ax, detail):
    macro = {method: one(detail, scope="macro", method=method) for method in METHODS}
    improvements = [0.0] + [number(macro[m], "improvement_pct_vs_fixed") for m in METHODS]
    labels = ["FixedTime", "P1C-DHOA-R25", "P1C-DHOA-R50"]
    x = np.arange(3)
    ax.bar(x, [100, 100, 100], color="#858585", width=.62, label="FixedTime normalized base = 100")
    ax.bar(x[1:], improvements[1:], bottom=100, color=[COLORS[m] for m in METHODS], width=.62)
    ax.text(x[0], 102, "Index 100.00\nFixedTime\nbaseline", ha="center", va="bottom", fontweight="bold", fontsize=9)
    for pos, value in zip(x[1:], improvements[1:]):
        ax.text(pos, 100 + value + 1.5, f"Index {100+value:.2f}\n+{value:.2f}%\nvs FixedTime", ha="center", va="bottom", fontweight="bold", fontsize=9)
        ax.annotate("", xy=(pos+.34, 100+value), xytext=(pos+.34, 100), arrowprops={"arrowstyle":"<->", "color":"#333333", "lw":1.2})
    ax.set_xticks(x, labels); ax.set_ylim(0, 126); ax.set_ylabel("Normalized throughput index")
    ax.set_title("A. S1–S4 equal-weight throughput evaluation\nFixedTime = 100; colored segment = improvement")
    ax.grid(axis="y", alpha=.2)


def plot_scene_variation(ax, detail, episodes):
    rng = np.random.default_rng(20260803)
    x = np.arange(len(SCENES)); offsets = {"FixedTime": -.27, METHODS[0]: 0, METHODS[1]: .27}
    fixed = {s: number(one(detail, scope="scenario", scenario=s, method=METHODS[0]), "fixedtime_throughput") for s in SCENES}
    ax.scatter(x + offsets["FixedTime"], np.full(len(x), 100.0), marker="D", s=48, color="#666666", label="FixedTime = 100", zorder=5)
    for method in METHODS:
        for i, scenario in enumerate(SCENES):
            values = [number(r, "throughput") / fixed[scenario] * 100 for r in episodes if r["scenario"] == scenario and r["method"] == method]
            jitter = rng.uniform(-.045, .045, len(values)); center = x[i] + offsets[method]
            ax.scatter(center + jitter, values, s=15, alpha=.38, color=COLORS[method], edgecolors="none")
            row = one(detail, scope="scenario", scenario=scenario, method=method)
            mean_index = 100 + number(row, "improvement_pct_vs_fixed")
            low_index = 100 + number(row, "improvement_pct_ci95_low"); high_index = 100 + number(row, "improvement_pct_ci95_high")
            ax.errorbar(center, mean_index, yerr=[[mean_index-low_index], [high_index-mean_index]], fmt="o", markersize=6, color=COLORS[method], capsize=4, zorder=6)
            absolute = number(row, "method_throughput_mean")
            lift = 1.2 if method == METHODS[0] else .2
            ax.text(center, mean_index + lift, f"{absolute:.1f}\n(+{mean_index-100:.1f}%)", ha="center", fontsize=7)
    ax.axhline(100, color="#666666", linestyle="--", linewidth=1)
    ax.set_xticks(x, [LABEL[s] for s in SCENES]); ax.set_ylim(98, 124.8); ax.set_ylabel("Scene-normalized throughput index")
    ax.set_title("B. Frozen-episode throughput variation within each scene\nDots = episodes; marker/error bar = mean and 95% bootstrap interval")
    ax.legend(handles=[plt.Line2D([], [], marker="D", linestyle="", color="#666666", label="FixedTime = 100"), plt.Line2D([], [], marker="o", linestyle="", color=COLORS[METHODS[0]], label="R25 episodes/mean"), plt.Line2D([], [], marker="o", linestyle="", color=COLORS[METHODS[1]], label="R50 episodes/mean")], fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=.2)


def main():
    args = arguments()
    detail = rows(args.throughput_detail); episodes = rows(args.efficiency_episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_index, ax_scene) = plt.subplots(1, 2, figsize=(16, 6.3), gridspec_kw={"wspace": .22})
    plot_equal_weight_index(ax_index, detail)
    plot_scene_variation(ax_scene, detail, episodes)
    fig.suptitle("Frozen-evaluation throughput evidence", fontsize=18, fontweight="bold", y=.985)
    fig.text(.5, .015, "All throughput values use existing formal frozen evaluations; S1–S4 aggregation is an equal-weight average of scene-level relative improvements.", ha="center", fontsize=10)
    fig.subplots_adjust(left=.055, right=.985, bottom=.14, top=.79, wspace=.22)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
