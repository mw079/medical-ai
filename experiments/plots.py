# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9,<4"]
# ///
"""Figures for results/<name>/report.json.

designs: each arm's accuracy change against the baseline, pooled over the test splits, one panel
per budget mode (point = mean paired difference, line = 95% bootstrap interval). Writes
designs.png next to the report.

cost: accuracy against average medical cost for each arm, one figure per split. Each arm is one
line through its budget modes; error bars span the lowest and highest run; dashed lines mark the
budget caps. Writes <split>.png next to the report.

    uv run experiments/plots.py designs results/multi-agent-test/report.json
    uv run experiments/plots.py cost results/evidence-memory-test/report.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DESIGN_COLORS = {"single": "#2a78d6", "independent": "#eb6834", "decentralized": "#1baf7a", "centralized": "#e87ba4"}
MODE_TITLES = {"budget_1000": "$1,000 budget", "budget_4000": "$4,000 budget"}
CAPS = {"budget_1000": 1000, "budget_4000": 4000}
SPLIT_TITLES = {"same_source_test": "Stanford test", "external_test": "MedR-Bench test",
                "validation": "Validation"}
# Categorical slots 1 and 2 of the reference palette, validated for color-vision deficiency.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MODEL_NAMES = {"gpt-6-luna": "Luna", "gpt-6.1-terra": "Terra"}


def label(info):
    if info["agent"] == "single":
        return "Single + roles" if info["harness"] else "Single"
    return f"{info['agent'].capitalize()}, {info['agents']} agents"


def plot_designs(report, output):
    modes = [m for m in MODE_TITLES if m in report["results"]]
    arms = [a for a in report["arms"] if a != "baseline"]
    fig, axes = plt.subplots(1, len(modes), figsize=(10, 0.42 * len(arms) + 1.4), sharey=True, facecolor=SURFACE)
    for ax, mode in zip(axes, modes):
        pooled = report["results"][mode]["pooled"]
        ax.set_facecolor(SURFACE)
        ax.axvline(0, color=MUTED, lw=1)
        for row, arm in enumerate(arms):
            info, result = report["arms"][arm], pooled[arm]
            low, high = result["ci95_points"]
            color = DESIGN_COLORS[info["agent"]]
            ax.plot([low, high], [row, row], color=color, lw=2, solid_capstyle="round", zorder=2)
            ax.scatter(result["delta_points"], row, s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.set_xlim(-15, 15)
        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_title(MODE_TITLES[mode], loc="left", fontsize=12, fontweight="bold", color=INK)
        ax.set_xlabel("Accuracy change vs single agent (points)", color=MUTED)
        ax.grid(axis="x", color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0, pad=6)
        ax.tick_params(axis="y", colors=INK)
    axes[0].set_yticks(range(len(arms)), [label(report["arms"][arm]) for arm in arms])
    axes[0].invert_yaxis()  # first arm on top; the axis is shared, so invert once
    fig.subplots_adjust(left=0.26, right=0.98, top=0.88, bottom=0.16, wspace=0.08)
    fig.savefig(output, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def arm_label(report, arm):
    info = report["arms"][arm]
    name = MODEL_NAMES.get(info["model"], info["model"])
    if info.get("agent", "single") != "single":
        return f"{name} + harness"
    return name if not info["harness"] else f"{name} + {Path(info['harness']['path']).stem}"


def plot_split(report, split, output):
    modes = [mode for mode in CAPS if split in report["results"].get(mode, {})]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for mode in modes:
        ax.axvline(CAPS[mode], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(CAPS[mode], 1.02, f"${CAPS[mode] // 1000}k cap", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, color=MUTED)
    # Within a mode, label the highest arm above its point and the others below, so labels never collide.
    order = {mode: sorted(report["arms"], key=lambda a: report["results"][mode][split][a]["mean_accuracy"])
             for mode in modes}
    for color, arm in zip(SERIES, report["arms"]):
        rows = [report["results"][mode][split][arm] for mode in modes]
        x = [r["avg_medical_cost_usd"] for r in rows]
        y = [r["mean_accuracy"] for r in rows]
        low = [r["mean_accuracy"] - r["min_accuracy"] for r in rows]
        high = [r["max_accuracy"] - r["mean_accuracy"] for r in rows]
        ax.plot(x, y, color=color, lw=2, zorder=3, label=arm_label(report, arm))
        ax.errorbar(x, y, yerr=[low, high], fmt="none", ecolor=color, elinewidth=1.5, capsize=0, zorder=3)
        ax.scatter(x, y, s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        for mode, xi, yi in zip(modes, x, y):
            top = order[mode][-1] == arm
            # A lower label goes to the left only when another arm's point sits just to its right.
            crowded = any(0 < report["results"][mode][split][other]["avg_medical_cost_usd"] - xi < 300
                          for other in report["arms"] if other != arm)
            offset, align = ((9, 9), "left") if top else (((-9, -11), "right") if crowded else ((9, -11), "left"))
            ax.annotate(f"{yi:.0%}", (xi, yi), xytext=offset, textcoords="offset points",
                        fontsize=9, color=INK, va="center", ha=align, zorder=5,
                        bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none"})
    ax.set_xlim(0, max(CAPS[m] for m in modes) * 1.12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_xlabel("Medical cost per case", color=MUTED)
    ax.set_ylabel("Accuracy", color=MUTED)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, pad=6)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title(SPLIT_TITLES.get(split, split), loc="left", fontsize=13, fontweight="bold", color=INK, pad=22)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.85, bottom=0.13)
    fig.savefig(output, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", choices=("designs", "cost"))
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if args.figure == "designs":
        outputs = [args.report.with_name("designs.png")]
        plot_designs(report, outputs[0])
    else:
        splits = sorted({split for mode in report["results"].values() for split in mode if split != "pooled"})
        outputs = [args.report.with_name(f"{split}.png") for split in splits]
        for split, output in zip(splits, outputs):
            plot_split(report, split, output)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
