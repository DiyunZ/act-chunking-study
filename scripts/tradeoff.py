"""Headline trade-off figure and the temporal-ensembling coefficient table.

Every execution mode is one point: success under 400 ms of observation latency (x) against success
after a 6 cm cube move (y), one panel per checkpoint. Temporal ensembling is drawn as a path over its
weight coefficient c, from ACT's default (0.01, favors old predictions) to strongly recency-weighted
values. Intervals are left out of the figure to keep the labels readable; the caption states their
largest half-width, and te_coefficients.md and summary.md list them.

Outputs in --out:
  tradeoff.png         the figure
  te_coefficients.md   success per coefficient and condition, paired against the default coefficient
  points.json          every plotted value

Usage:
    uv run --no-project --with numpy --with matplotlib python scripts/tradeoff.py \
        --results outputs/hub/results --out results/tradeoff
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from summarize import AXIS, GRID, MODE_STYLE, MUTED, SURFACE, TEXT, TEXT_2, cell, load_run, paired

MODELS = (("official", "Official checkpoint"), ("ours100k", "Our checkpoint (100k steps)"))
CONDITIONS = (("nominal", "d00_m0"), ("latency", "d20_m0"), ("displacement", "d00_m6"))
TE_DEFAULT = 0.01
TE_COLOR = MODE_STYLE["te"][1]
# Coefficient labels go right of their point, except where the path runs just right of it.
LABEL_LEFT = {("ours100k", 0.0)}


def te_name(c: float) -> str:
    return "te" if c == TE_DEFAULT else f"te{c:g}"


def c_label(c: float) -> str:
    return f"c = {c:g}".replace("-", "−")


def load(results: Path, model: str) -> dict[str, list[dict]]:
    by_condition: dict[str, list[dict]] = {}
    for run in (f"core_{model}", f"te_weights_{model}"):
        if (results / run).exists():
            for e in load_run(results / run):
                by_condition.setdefault(e["condition"], []).append(e)
    return by_condition


def coefficients(by_condition: dict[str, list[dict]]) -> list[float]:
    found = {TE_DEFAULT}
    for name in by_condition:
        head = name.split("_")[0]
        if head.startswith("te") and head != "te":
            found.add(float(head[2:]))
    # Complete coefficients only (all three conditions present), from the default toward recency weighting.
    return sorted((c for c in found if all(f"{te_name(c)}_{s}" in by_condition for _, s in CONDITIONS)), reverse=True)


def point(by_condition: dict[str, list[dict]], name: str) -> dict:
    return {key: cell(by_condition[f"{name}_{suffix}"]) for key, suffix in CONDITIONS}


def xy(p: dict) -> tuple[float, float]:
    return 100 * p["latency"]["rate"], 100 * p["displacement"]["rate"]


def pct_range(values: list[float]) -> str:
    lo, hi = min(values), max(values)
    return f"{lo:g}%" if lo == hi else f"{lo:g}–{hi:g}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="folder with the runs' result folders")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(10.4, 5.6), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    points: dict = {}
    half_widths: list[float] = []
    table = ["# Temporal ensembling: weight coefficient", "",
             "ACT weights the predictions for the current step by exp(−c·i), i = 0 the oldest; c = 0.01 is the default.",
             "Success rate with a 95% Wilson interval, 200 paired seeds; p is an exact McNemar test against c = 0.01 "
             "on the same seeds.", ""]
    for ax, (model, title) in zip(axes, MODELS):
        by_condition = load(args.results, model)
        points[model] = {"modes": {}, "te": {}}
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=TEXT_2, labelsize=9, length=0)

        te_points = [(c, point(by_condition, te_name(c))) for c in coefficients(by_condition)]
        path = np.array([xy(p) for _, p in te_points])
        ax.plot(path[:, 0], path[:, 1], color=TE_COLOR, linewidth=1.5, linestyle=(0, (4, 3)), alpha=0.8, zorder=3)
        for c, p in te_points:
            points[model]["te"][f"{c:g}"] = p
            if c == TE_DEFAULT:
                continue
            x, y = xy(p)
            ax.plot([x], [y], marker="o", markersize=7, color=TE_COLOR, markerfacecolor=SURFACE,
                    markeredgecolor=TE_COLOR, markeredgewidth=1.5, linestyle="none", zorder=4)
            near_axis = y < 4  # keep the label clear of the x axis
            left = (model, c) in LABEL_LEFT
            ax.annotate(c_label(c), (x, y), textcoords="offset points",
                        xytext=(7, 4) if near_axis else (-8 if left else 8, 0), ha="right" if left else "left",
                        va="bottom" if near_axis else "center", fontsize=9, color=TE_COLOR)

        for mode, (_, color, marker) in MODE_STYLE.items():
            p = point(by_condition, mode)
            points[model]["modes"][mode] = p
            x, y = xy(p)
            ax.plot([x], [y], marker=marker, markersize=8, color=color, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, linestyle="none", zorder=5)

        for p in [p for _, p in te_points] + list(points[model]["modes"].values()):
            for key in ("latency", "displacement"):
                half_widths.append(100 * max(p[key]["rate"] - p[key]["lo"], p[key]["hi"] - p[key]["rate"]))

        ax.set_title(title, color=TEXT, fontsize=11, loc="left")
        ax.set_xlabel("success with 400 ms observation latency (%)", color=TEXT_2)
        ax.text(0.98, 0.98, "better ↗", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=MUTED)

        table += [f"## {title}", "", "| c | nominal | 400 ms latency | 6 cm displacement |", "|---|---|---|---|"]
        for c, p in te_points:
            row = []
            for key, suffix in CONDITIONS:
                entry = f"{100 * p[key]['rate']:.1f}% [{100 * p[key]['lo']:.0f}–{100 * p[key]['hi']:.0f}]"
                if c != TE_DEFAULT:
                    t = paired(by_condition[f"te_{suffix}"], by_condition[f"{te_name(c)}_{suffix}"])
                    entry += f", p = {t['p']:.2g}"
                row.append(entry)
            table.append(f"| {c:g}{' (default)' if c == TE_DEFAULT else ''} | " + " | ".join(row) + " |")
        table.append("")

    axes[0].set_ylabel("success after a 6 cm cube move (%)", color=TEXT_2)
    for ax in axes:
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_yticks(np.arange(0, 101, 20))

    handles = [Line2D([0], [0], marker=MODE_STYLE[m][2], color=MODE_STYLE[m][1], markersize=8, markeredgecolor=SURFACE,
                      markeredgewidth=1.5, linestyle="none",
                      label="temporal ensembling, default c = 0.01" if m == "te" else MODE_STYLE[m][0])
               for m in MODE_STYLE]
    handles.append(Line2D([0], [0], marker="o", color=TE_COLOR, markersize=7, markerfacecolor=SURFACE,
                          markeredgewidth=1.5, linewidth=1.5, linestyle=(0, (4, 3)),
                          label="temporal ensembling, other c"))
    leg = fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.95))
    for text in leg.get_texts():
        text.set_color(TEXT_2)

    collapsed = {c: [100 * points[m]["te"][f"{c:g}"]["nominal"]["rate"] for m, _ in MODELS]
                 for c in (-0.1, -0.2) if all(f"{c:g}" in points[m]["te"] for m, _ in MODELS)}
    caption = ("Dashed line: temporal ensembling as its weight coefficient c goes from ACT's default 0.01 (favors older "
               "predictions) to negative values (favor newer ones).")
    if collapsed:
        caption += "\nWithout latency, " + " and ".join(c_label(c) for c in collapsed) + " succeed in only " + \
                   " and ".join(pct_range(v) for v in collapsed.values()) + " of episodes. "
    else:
        caption += "\n"
    caption += f"200 paired seeds per point; 95% intervals are at most ±{max(half_widths):.0f} points."
    fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=8.5, color=TEXT_2, linespacing=1.5)
    fig.suptitle("Latency robustness vs reactivity to a moved cube", color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(args.out / "tradeoff.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    (args.out / "te_coefficients.md").write_text("\n".join(table))
    (args.out / "points.json").write_text(json.dumps(points, indent=1))
    print(f"wrote {sorted(p.name for p in args.out.iterdir())}")


if __name__ == "__main__":
    main()
