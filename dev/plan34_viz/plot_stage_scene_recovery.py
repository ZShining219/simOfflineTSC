#!/usr/bin/env python3
"""Plan 3/4 持续学习证据图（fifo 口径）— v2 强调偏移可见性。

图1 学习趋势：frozen_eval_timeseries 当前场景逐 ep 冻结评估，三个 stage
按 local_episode 对齐叠加；y=ln(norm TT) 对数轴，1.0=0 基准线；
mean±95%CI（5 seeds）。每个新场景冲高→收敛的形状直接可见。

图2a 历史场景被改写：Δnorm = 该边界 norm − 本场景引入边界 norm；
仅 k>i 段；每场景 5 条 seed 细线 + mean 粗线（均值藏散布，故画逐 seed）。

图2b 退化幅度分布：所有 k>i 格 ΔTT 秒的 strip+box，x=场景、点按 order
着色；同格跨 seed 跨度 0.66s~126s 的散布即"非固定幅度"证据。

口径：norm TT = TT/同场景 Plan-1 400ep 参考；fifo（=E4 CONT-FIFO 对照）；
历史场景仅 stage 边界有测评；Δ 全部同 run 内配对。
"""
import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = pathlib.Path(
    "data/output_data/analysis/plan34/plan34_b100_formal_60_plots_20260725"
)
BUNDLE = pathlib.Path(
    "data/output_data/analysis/plan_ha_analysis_bundle_v1/plan34_temporal"
)
CELLS_CSV = BASE / "tables" / "stage_scene_evaluations.csv"
FROZEN_TS = BUNDLE / "plan34_frozen_eval_timeseries.csv"
OUT = pathlib.Path(
    "data/output_data/analysis/plan34/stage_scene_recovery_view_20261015"
)
POLICY = "fifo"
ORDERS = ["O1", "O2", "O3", "O4"]
SCENES = ["S1", "S2", "S3", "S4"]
SCENE_COLOR = {"S1": "#1f77b4", "S2": "#ff7f0e", "S3": "#2ca02c", "S4": "#d62728"}
ORDER_COLOR = {"O1": "#1f77b4", "O2": "#ff7f0e", "O3": "#2ca02c", "O4": "#9467bd"}


def load_cells():
    df = pd.read_csv(CELLS_CSV)
    df = df[df["policy"] == POLICY].copy()
    df["norm"] = df["travel_time"] / df["plan1_reference_travel_time"]
    df["stage"] = df["stage_index"].astype(int)
    df["intro"] = df["evaluation_scene_introduced_stage"].astype(int)
    df["is_current"] = df["stage"] == df["intro"]
    df["seed"] = df["training_seed"].astype(int)
    ref = df.groupby("evaluation_network")["plan1_reference_travel_time"].first()
    # 同 run 内引入 stage 基准
    base = (df[df["is_current"]]
            .set_index(["logical_run_id", "evaluation_network"])
            [["norm", "travel_time"]]
            .rename(columns={"norm": "norm0", "travel_time": "tt0"}))
    df = df.join(base, on=["logical_run_id", "evaluation_network"])
    df["dnorm"] = df["norm"] - df["norm0"]
    df["dtt"] = df["travel_time"] - df["tt0"]
    return df, ref


def fig1_learning_trend(ref, outdir):
    ts = pd.read_csv(FROZEN_TS)
    ts = ts[ts["logical_run_id"].str.startswith("plan34_b100_")
            & (ts["policy"] == POLICY)].copy()
    ts["norm"] = ts["travel_time"] / ts["evaluation_network"].map(ref)
    cur = ts[(ts["evaluation_network"] == ts["current_network"])
             & (ts["stage"].isin([2, 3, 4]))]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True, sharey=True)
    for ax, order in zip(axes.flat, ORDERS):
        s = cur[cur["order_id"] == order]
        for stage in (2, 3, 4):
            d = s[s["stage"] == stage]
            sc = d["current_scene"].iloc[0]
            g = (d.groupby("training_episode")["norm"]
                 .agg(["mean", "std", "count"]).reset_index())
            g["lo"] = g["mean"] - 1.96 * g["std"] / np.sqrt(g["count"])
            g["hi"] = g["mean"] + 1.96 * g["std"] / np.sqrt(g["count"])
            for col in ("mean", "lo", "hi"):
                g[col] = np.log(g[col])
            ax.fill_between(g["training_episode"], g["lo"], g["hi"],
                            color=SCENE_COLOR[sc], alpha=0.18)
            ax.plot(g["training_episode"], g["mean"], color=SCENE_COLOR[sc], lw=1.5,
                    label=f"stage{stage}: {sc}")
        ax.axhline(0.0, ls="--", lw=0.9, color="#555")
        ax.set_title(order, fontsize=11)
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Episode within stage (0 = first eval on new scene)")
        ax.legend(fontsize=8, loc="upper right")
    yt = [0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]
    axes[0][0].set_yticks(np.log(yt))
    axes[0][0].set_yticklabels([f"{v:g}" for v in yt])
    axes[0][0].set_ylabel("Frozen-eval TT / Plan 1 ref (log)")
    axes[1][0].set_ylabel("Frozen-eval TT / Plan 1 ref (log)")
    fig.suptitle(
        "Sequential learning trend (FIFO replay): frozen eval on the scene "
        "currently being trained, stages aligned at switch\n"
        "mean ± 95% seed CI (n=5); 0 = Plan 1 single-scene parity — every "
        "new scene spikes at the switch, then converges back",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig1_learning_trend_{POLICY}.{ext}", dpi=200)
    plt.close(fig)


def fig2a_delta_lines(df, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, order in zip(axes.flat, ORDERS):
        s = df[(df["order_id"] == order) & (df["stage"] > df["intro"])]
        for scene in SCENES:
            sc = s[s["scene"] == scene]
            if sc.empty:
                continue
            for seed, d in sc.groupby("seed"):
                ax.plot(d["stage"], d["dnorm"], "o-", ms=3, lw=0.8,
                        color=SCENE_COLOR[scene], alpha=0.35)
            m = sc.groupby("stage")["dnorm"].mean()
            ax.plot(m.index, m.values, "o-", ms=6, lw=1.8,
                    color=SCENE_COLOR[scene], mec="black", mew=0.7,
                    label=scene)
        ax.axhline(0.0, ls="--", lw=0.9, color="#555")
        ax.set_title(order, fontsize=11)
        ax.set_xticks([1, 2, 3, 4])
        ax.grid(alpha=0.25)
        ax.set_xlabel("Training stage completed")
    axes[0][0].set_ylabel("Δ norm TT vs own introduction stage")
    axes[1][0].set_ylabel("Δ norm TT vs own introduction stage")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, title="Evaluated scene", ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        "How continued training rewrites earlier scenes (FIFO replay)\n"
        "thin lines = individual seeds; thick = seed mean; 0 = performance "
        "at the stage the scene was trained — drift is heterogeneous, "
        "non-monotone, occasionally improving",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig2a_delta_lines_{POLICY}.{ext}", dpi=200)
    plt.close(fig)


def fig3a_path(df, outdir):
    """Fig.3a 式单路径双数据链：O2/seed0/fifo。

    蓝实线 = adaptation_curves 当前场景逐 ep（norm-1）%，x=100(k-2)+e，
    三个阶段分别连接（x=100/200 处不跨段连）；橙虚线 = stage_scene_evaluations
    历史场景 stage 末冻结回测，x=100(k-1)，圆/三角/方标记区分场景。
    """
    order, seed = "O2", 0
    ac = pd.read_csv(BASE / "tables" / "adaptation_curves.csv")
    ac = ac[(ac["order_id"] == order) & (ac["training_seed"] == seed)
            & (ac["policy"] == POLICY)]
    s = df[(df["order_id"] == order) & (df["seed"] == seed)]
    scene2stage = s.drop_duplicates("scene").set_index("scene")["intro"]

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    # 橙虚线：历史场景（先画在底层）
    markers = {"S2": "o", "S3": "^", "S1": "s"}
    for sc, mk in markers.items():
        i = scene2stage[sc]
        d = s[(s["scene"] == sc) & (s["stage"] >= i)].sort_values("stage")
        x = (d["stage"] - 1) * 100
        y = (d["norm"] - 1) * 100
        ax.plot(x, y, "--", marker=mk, ms=6, lw=1.4, color="#e07b00",
                mec="#e07b00", label=f"{sc} (stage-{i} demand)", zorder=2)
    # 蓝实线：当前场景逐 ep，三段分接
    for stage in (2, 3, 4):
        d = ac[ac["stage_index"] == stage].sort_values("local_episode")
        x = 100 * (stage - 2) + d["local_episode"]
        y = (d["normalized_travel_time"] - 1) * 100
        ax.plot(x, y, "-", color="#1f77b4", lw=1.6, zorder=3)
    ax.plot([], [], "-", color="#1f77b4", lw=1.6,
            label="current demand (per-episode, in training)")
    ax.axhline(0, ls="--", lw=0.9, color="#555")
    for xb in (100, 200):
        ax.axvline(xb, ls=":", lw=0.9, color="#888")
    for k, sc in [(2, "S3"), (3, "S1"), (4, "S4")]:
        ax.text(100 * (k - 2) + 50, 0.03, f"train {sc}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9,
                color="#1f77b4")
    ax.set_xlim(0, 300)
    ax.set_xlabel("Sequential training progress (episodes since first switch)")
    ax.set_ylabel("Travel time excess vs Plan-1 reference (%)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title(
        "Fig.3a reproduction · O2/seed0/FIFO — current-demand learning curve "
        "vs frozen re-evaluations of previously learned demands",
        fontsize=10,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig3a_path_{order}_seed{seed}_{POLICY}.{ext}",
                    dpi=200)
    plt.close(fig)


def fig4_all_paths(df, outdir):
    """全 20 路径剖面图：x=6 个 (eval stage, intro stage) 槽位，
    y=ΔT% vs 自身引入 stage 末 TT（同 run 配对）。

    灰细线=全部 20 run；蓝粗=O2/seed0（representative）；橙粗=O2/seed2
    （worst，stage4 三格合计 +357%、6 格均值 +59% 双口径垫底）；
    深灰粗=逐槽位中位数。symlog 轴同时容纳主体云与重尾。
    """
    post = df[df["stage"] > df["intro"]].copy()
    post["dtt_pct"] = (post["travel_time"] - post["tt0"]) / post["tt0"] * 100
    slots = [(2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)]
    xs = list(range(len(slots)))

    fig, ax = plt.subplots(figsize=(9.5, 6))
    for run, g in post.groupby("logical_run_id"):
        m = g.set_index(["stage", "intro"])["dtt_pct"]
        y = [m.get((k, i), np.nan) for k, i in slots]
        ax.plot(xs, y, "-o", ms=2.5, lw=0.7, color="#888", alpha=0.35,
                zorder=1)
    med = [post[(post["stage"] == k) & (post["intro"] == i)]["dtt_pct"].median()
           for k, i in slots]
    ax.plot(xs, med, "-D", lw=2.2, ms=6, color="#333333", zorder=4,
            label="Median over 20 runs")
    for run, color, mk, lab in [
        ("plan34_b100_O2_seed0_fifo", "#1f77b4", "o",
         "Representative run (O2·seed0)"),
        ("plan34_b100_O2_seed2_fifo", "#e07b00", "s",
         "Worst observed run (O2·seed2)"),
    ]:
        g = post[post["logical_run_id"] == run]
        m = g.set_index(["stage", "intro"])["dtt_pct"]
        y = [m.get((k, i), np.nan) for k, i in slots]
        ax.plot(xs, y, "-", marker=mk, ms=7, lw=2.0, color=color,
                mec="black", mew=0.6, label=lab, zorder=5)
    ax.axhline(0, ls="--", lw=0.9, color="#555")
    ax.set_yscale("symlog", linthresh=5)
    ax.set_ylim(-8, 400)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"({k},{i})" for k, i in slots])
    ax.set_xlabel("(evaluation stage k, demand introduction stage i)")
    ax.set_ylabel("Δ travel time vs introduction stage (%, symlog)")
    ax.grid(alpha=0.25)
    # 按评价 stage 分组：底色带 + 底部标签
    for lo, hi, lab in [(-0.4, 0.5, "after stage 2"), (0.5, 2.5, "after stage 3"),
                        (2.5, 5.4, "after stage 4")]:
        ax.axvspan(lo, hi, color="#eeeeee", alpha=0.5, zorder=0)
        ax.text((lo + hi) / 2, 0.03, lab, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color="#777")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(
        "All-path forgetting profiles (FIFO replay, 20 runs = 4 orders × 5 "
        "seeds)", fontsize=11,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig4_all_path_profiles_{POLICY}.{ext}", dpi=200)
    plt.close(fig)


def fig5_drift_trajectories(df, outdir):
    """按引入 stage 分面：每 run 的第 i 个需求在引入后各 stage 末的 ΔT%
    —— 每条线是同 run 同场景的真实轨迹（修正规：不再有跨场景连线）。
    order 着色 + 黑色中位线。symlog 容纳重尾。"""
    post = df[df["stage"] > df["intro"]].copy()
    post["dtt_pct"] = (post["travel_time"] - post["tt0"]) / post["tt0"] * 100
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for ax, intro in zip(axes, [1, 2, 3]):
        s = post[post["intro"] == intro]
        for run, g in s.groupby("logical_run_id"):
            order = g["order_id"].iloc[0]
            g = g.sort_values("stage")
            ax.plot(g["stage"], g["dtt_pct"], "-o", ms=3, lw=0.8,
                    color=ORDER_COLOR[order], alpha=0.4, zorder=2)
        med = s.groupby("stage")["dtt_pct"].median()
        ax.plot(med.index, med.values, "-D", lw=2.2, ms=6, color="#222",
                zorder=4, label="median" if intro == 1 else None)
        ax.axhline(0, ls="--", lw=0.9, color="#555")
        ax.set_yscale("symlog", linthresh=5)
        ax.set_xticks(range(intro + 1, 5))
        ax.set_title(f"demand introduced at stage {intro}", fontsize=10)
        ax.set_xlabel("evaluation stage k")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Δ travel time vs introduction stage (%, symlog)")
    hs = [plt.Line2D([], [], marker="o", ls="", color=ORDER_COLOR[o], label=o)
          for o in ORDERS]
    fig.legend(handles=hs, title="Order", ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.suptitle(
        "Drift of each learned demand after its training stage ends (FIFO)\n"
        "each line = one run × one real scene trajectory across stage "
        "boundaries — degradation direction and size vary strongly by run "
        "and order", fontsize=10)
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig5_drift_trajectories_{POLICY}.{ext}", dpi=200)
    plt.close(fig)


def fig6_gap_scatter(df, outdir):
    """x=距引入几个 stage（gap）；全部 k>i 格散点+box+中位数标注。
    回答：学习之后隔多久，旧需求差多少——散布在各级 gap 都大。"""
    post = df[df["stage"] > df["intro"]].copy()
    post["dtt_pct"] = (post["travel_time"] - post["tt0"]) / post["tt0"] * 100
    post["gap"] = post["stage"] - post["intro"]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    gaps = [1, 2, 3]
    boxdata = []
    for j, gap in enumerate(gaps):
        cell = post[post["gap"] == gap]
        boxdata.append(cell["dtt_pct"].values)
        for k, order in enumerate(ORDERS):
            y = cell[cell["order_id"] == order]["dtt_pct"]
            ax.scatter(rng.normal(j + 1 + (k - 1.5) * 0.10, 0.03, len(y)), y,
                       s=13, color=ORDER_COLOR[order], alpha=0.55, zorder=3)
        med = np.median(cell["dtt_pct"])
        ax.text(j + 1.34, med, f"med {med:+.1f}%", va="center", fontsize=8,
                color="#333")
    bp = ax.boxplot(boxdata, positions=np.array(gaps) - 0.18, widths=0.22,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.6),
                    whiskerprops=dict(color="#555"),
                    capprops=dict(color="#555"),
                    boxprops=dict(edgecolor="#555"))
    for patch in bp["boxes"]:
        patch.set(facecolor="#dddddd", alpha=0.6)
    ax.axhline(0, ls="--", lw=0.9, color="#555")
    ax.set_yscale("symlog", linthresh=5)
    ax.set_xticks(gaps)
    ax.set_xticklabels([f"{g} stage{'s' if g > 1 else ''} later" for g in gaps])
    ax.set_xlabel("Stages elapsed since the demand was trained (gap k−i)")
    ax.set_ylabel("Δ travel time vs introduction stage (%, symlog)")
    ax.grid(axis="y", alpha=0.25)
    hs = [plt.Line2D([], [], marker="o", ls="", color=ORDER_COLOR[o], label=o)
          for o in ORDERS]
    ax.legend(handles=hs, title="Order", fontsize=9, loc="upper left")
    ax.set_title(
        "Forgetting size vs time since the demand was trained (FIFO)\n"
        "spread stays huge at every gap (points: run×scene×boundary cells) — "
        "degradation is event-driven at the switch, not proportional to "
        "elapsed time", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig6_gap_scatter_{POLICY}.{ext}", dpi=200)
    plt.close(fig)


def fig2b_delta_distribution(df, outdir):
    post = df[df["stage"] > df["intro"]]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    boxdata = []
    for j, scene in enumerate(SCENES):
        cell = post[post["scene"] == scene]
        boxdata.append(cell["dtt"].values)
        for k, order in enumerate(ORDERS):
            y = cell[cell["order_id"] == order]["dtt"]
            ax.scatter(rng.normal(j + 1 + (k - 1.5) * 0.09, 0.02, len(y)),
                       y, s=14, color=ORDER_COLOR[order], alpha=0.6,
                       zorder=3)
    bp = ax.boxplot(boxdata, positions=range(1, 5), widths=0.45,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.5))
    for patch in bp["boxes"]:
        patch.set(facecolor="#dddddd", alpha=0.6)
    ax.axhline(0, ls="--", lw=0.9, color="#555")
    ax.set_yscale("symlog", linthresh=5)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(SCENES)
    ax.set_xlabel("Evaluated scene")
    ax.set_ylabel("Δ travel time vs introduction stage (s, symlog)")
    ax.grid(axis="y", alpha=0.25)
    hs = [plt.Line2D([], [], marker="o", ls="", color=ORDER_COLOR[o],
                     label=o) for o in ORDERS]
    ax.legend(handles=hs, title="Order", fontsize=9)
    ax.set_title(
        "Post-introduction forgetting cells (FIFO replay): ΔTT seconds per "
        "run×stage-boundary cell\nmedian ~2.6 s but spread −2 to +181 s — "
        "degradation is not a fixed amount (same cell ranges 0.7–126 s "
        "across seeds)",
        fontsize=10,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig2b_delta_dist_{POLICY}.{ext}", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    outdir = pathlib.Path(args.out)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    df, ref = load_cells()
    df.to_csv(outdir / "tables" / "stage_scene_cells_fifo.csv", index=False)

    fig1_learning_trend(ref, outdir / "figures")
    fig2a_delta_lines(df, outdir / "figures")
    fig2b_delta_distribution(df, outdir / "figures")
    fig3a_path(df, outdir / "figures")
    fig4_all_paths(df, outdir / "figures")
    fig5_drift_trajectories(df, outdir / "figures")
    fig6_gap_scatter(df, outdir / "figures")

    (outdir / "manifest.json").write_text(json.dumps({
        "policy": POLICY,
        "sources": {"stage_scene_cells": str(CELLS_CSV),
                    "frozen_eval_timeseries": str(FROZEN_TS)},
        "normalization": "travel_time / same-scene plan1_reference_travel_time",
        "delta": "cell - own introduction-stage cell, paired within run",
        "note": "historical scenes evaluated only at stage boundaries",
        "version": "v2: log-scale trend, delta views, per-seed lines",
    }, indent=2, ensure_ascii=False))
    print(f"wrote {outdir}")


if __name__ == "__main__":
    sys.exit(main())
