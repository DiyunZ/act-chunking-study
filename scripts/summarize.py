"""Summarize chunk_eval.py results into tables and figures.

For every checkpoint and condition, the tables report:
  * success rate with a 95% Wilson interval;
  * failure stage, i.e. the highest reward reached (0 no contact, 1 touched, 2 lifted,
    3 handover attempted);
  * a paired exact McNemar test against full-chunk execution on the same seeds.

Figures (light surface). Series colors come from the dataviz reference palette, and
scripts/validate_palette.js passes it for these five adjacent slots. Three slots sit below 3:1
contrast, so every series also gets its own marker shape and summary.md carries the numbers.
  delay.png         success vs observation latency, one panel per checkpoint
  displacement.png  success vs cube displacement, one panel per checkpoint
  checkpoints.png   training loss, and full-chunk success at each checkpoint (two panels, one axis each)

Usage (needs only numpy and matplotlib):
    uv run --no-project --with numpy --with matplotlib python scripts/summarize.py \
        --runs RUN_DIR [RUN_DIR ...] --out results/summary [--checkpoint-runs DIR ...] [--training-log CSV]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, TEXT, TEXT_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Fixed slot per mode (color follows the entity, never its rank); legend order follows the replanning interval.
MODE_STYLE = {
    "full": ("full chunk", "#eb6834", "s"),
    "replan50": ("replan every 50", "#e87ba4", "v"),
    "replan25": ("replan every 25", "#1baf7a", "^"),
    "replan10": ("replan every 10", "#eda100", "D"),
    "te": ("temporal ensembling", "#2a78d6", "o"),
}
STEP_MS = 20  # 50 Hz control


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def mode_of(e: dict) -> str:
    return f"replan{e['k']}" if e["mode"] == "replan" else e["mode"]


def load_run(run_dir: Path) -> list[dict]:
    episodes = []
    for path in sorted(run_dir.glob("*/seeds_*.json")):
        result = json.loads(path.read_text())
        for e in result["episodes"]:
            episodes.append({**e, "checkpoint": result["checkpoint"], "mode_name": mode_of(e)})
    return episodes


def cell(episodes: list[dict]) -> dict:
    n = len(episodes)
    k = sum(e["success"] for e in episodes)
    lo, hi = wilson(k, n)
    stages = {s: sum(1 for e in episodes if not e["success"] and int(e["max_reward"]) == s) for s in range(4)}
    return {"n": n, "k": k, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi, "failure_stages": stages}


def paired(a: list[dict], b: list[dict]) -> dict:
    """McNemar of b against a on shared seeds (a is the reference)."""
    sa, sb = {e["seed"]: e["success"] for e in a}, {e["seed"]: e["success"] for e in b}
    seeds = sa.keys() & sb.keys()
    only_b = sum(1 for s in seeds if sb[s] and not sa[s])
    only_a = sum(1 for s in seeds if sa[s] and not sb[s])
    return {"n": len(seeds), "only_mode": only_b, "only_full": only_a, "p": mcnemar_p(only_b, only_a)}


def sweep(episodes: list[dict], kind: str) -> dict:
    """{mode: {level: [episodes]}} for the delay sweep (no displacement) or the displacement sweep (no delay)."""
    table: dict = defaultdict(lambda: defaultdict(list))
    for e in episodes:
        if kind == "delay" and e["displacement_cm"] == 0:
            table[e["mode_name"]][e["delay"]].append(e)
        elif kind == "displacement" and e["delay"] == 0:
            table[e["mode_name"]][e["displacement_cm"]].append(e)
    return table


def fmt(c: dict) -> str:
    return f"{100 * c['rate']:.1f}% [{100 * c['lo']:.0f}–{100 * c['hi']:.0f}]"


def markdown_tables(by_ckpt: dict[str, list[dict]]) -> tuple[str, dict]:
    lines, data = ["# Results summary", ""], {}
    for ckpt, episodes in by_ckpt.items():
        data[ckpt] = {}
        for kind, unit, scale in (("delay", "ms", STEP_MS), ("displacement", "cm", 1)):
            table = sweep(episodes, kind)
            if not table:
                continue
            levels = sorted({lvl for m in table.values() for lvl in m})
            lines += [f"## {ckpt}: success rate vs {kind} (95% Wilson CI)", ""]
            header = "| mode | " + " | ".join(f"{lvl * scale:g} {unit}" for lvl in levels) + " |"
            lines += [header, "|" + "---|" * (len(levels) + 1)]
            data[ckpt][kind] = {}
            for mode in MODE_STYLE:
                if mode not in table:
                    continue
                cells = {lvl: cell(table[mode][lvl]) for lvl in levels if table[mode].get(lvl)}
                data[ckpt][kind][mode] = {str(lvl): c for lvl, c in cells.items()}
                row = [fmt(cells[lvl]) if lvl in cells else "–" for lvl in levels]
                lines.append(f"| {MODE_STYLE[mode][0]} | " + " | ".join(row) + " |")
            lines.append("")
            if "full" in table:
                lines += [f"Paired against full chunk on the same seeds (exact McNemar p; wins = only this mode succeeds):", ""]
                lines += ["| mode | level | n | wins | losses | p |", "|---|---|---|---|---|---|"]
                for mode in MODE_STYLE:
                    if mode in ("full",) or mode not in table:
                        continue
                    for lvl in levels:
                        if table[mode].get(lvl) and table["full"].get(lvl):
                            t = paired(table["full"][lvl], table[mode][lvl])
                            lines.append(f"| {MODE_STYLE[mode][0]} | {lvl * scale:g} {unit} | {t['n']} | "
                                         f"{t['only_mode']} | {t['only_full']} | {t['p']:.3g} |")
                lines.append("")
            lines += [f"Failure stages ({kind}): episodes that failed, by the highest reward reached "
                      "(0 no contact · 1 touched · 2 lifted · 3 handover attempted)", ""]
            lines += ["| mode | level | 0 | 1 | 2 | 3 |", "|---|---|---|---|---|---|"]
            for mode in MODE_STYLE:
                for lvl in levels:
                    if mode in table and table[mode].get(lvl):
                        st = cell(table[mode][lvl])["failure_stages"]
                        lines.append(f"| {MODE_STYLE[mode][0]} | {lvl * scale:g} {unit} | "
                                     + " | ".join(str(st[s]) for s in range(4)) + " |")
            lines.append("")
    return "\n".join(lines), data


def panel_title(ckpt: str) -> str:
    if ckpt.startswith("ours_"):
        return f"Our checkpoint ({int(ckpt.rsplit('_', 1)[-1]) // 1000}k steps)"
    return "Official checkpoint" if ckpt == "official" else ckpt


def legend_handles(modes: list[str]) -> list:
    from matplotlib.lines import Line2D

    return [Line2D([0], [0], color=MODE_STYLE[m][1], linewidth=2, marker=MODE_STYLE[m][2], markersize=7,
                   markeredgecolor=SURFACE, markeredgewidth=1.5, label=MODE_STYLE[m][0]) for m in modes]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.xaxis.label.set_color(TEXT_2)
    ax.yaxis.label.set_color(TEXT_2)


def sweep_figure(by_ckpt: dict[str, list[dict]], kind: str, path: Path) -> None:
    unit, scale, xlabel = ("ms", STEP_MS, "observation latency (ms)") if kind == "delay" else ("cm", 1, "cube displacement (cm)")
    panels = [(ckpt, sweep(eps, kind)) for ckpt, eps in by_ckpt.items() if sweep(eps, kind)]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.9), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    present = [m for m in MODE_STYLE if any(m in t for _, t in panels)]
    offsets = {m: (i - (len(present) - 1) / 2) for i, m in enumerate(present)}
    for ax, (ckpt, table) in zip(axes[0], panels):
        style_axes(ax)
        levels = sorted({lvl for m in table.values() for lvl in m})
        span = (max(levels) - min(levels)) * scale or 1
        for mode in present:
            if mode not in table:
                continue
            label, color, marker = MODE_STYLE[mode]
            pts = [(lvl, cell(table[mode][lvl])) for lvl in levels if table[mode].get(lvl)]
            x = np.array([lvl * scale for lvl, _ in pts]) + offsets[mode] * span * 0.012  # dodge the CI bars
            y = np.array([100 * c["rate"] for _, c in pts])
            err = np.array([[100 * (c["rate"] - c["lo"]) for _, c in pts], [100 * (c["hi"] - c["rate"]) for _, c in pts]])
            ax.errorbar(x, y, yerr=err, color=color, linewidth=2, elinewidth=1, capsize=0, marker=marker,
                        markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label,
                        solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ax.set_xticks([lvl * scale for lvl in levels])
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 100)
        ax.set_title(panel_title(ckpt), color=TEXT, fontsize=11, loc="left")
        pad = 0.06 * span
        ax.set_xlim(min(levels) * scale - pad, max(levels) * scale + pad)
    axes[0][0].set_ylabel("success rate (%)")
    ncol = len(present) if len(panels) > 1 else min(3, len(present))
    leg = fig.legend(handles=legend_handles(present), loc="upper center", ncol=ncol, frameon=False, fontsize=9,
                     bbox_to_anchor=(0.5, 1.0))
    for text in leg.get_texts():
        text.set_color(TEXT_2)
    top = 0.93 if ncol == len(present) else 0.86
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def checkpoint_figure(ckpt_runs: list[Path], training_log: Path | None, path: Path) -> dict:
    import re

    runs = {}
    for run in ckpt_runs:
        eps = [e for e in load_run(run) if e["mode_name"] == "full" and e["delay"] == 0 and e["displacement_cm"] == 0]
        if eps:
            runs[eps[0]["checkpoint"]] = eps
    ours_runs = {int(m.group(1)): eps for name, eps in runs.items() if (m := re.search(r"ours_(\d+)", name))}
    ours = sorted((step, cell(eps)) for step, eps in ours_runs.items())
    ours_seeds = {e["seed"] for eps in ours_runs.values() for e in eps}
    # Reference checkpoints are scored on the same seeds as our checkpoints so the comparison is paired.
    reference = []
    for name, eps in runs.items():
        if re.search(r"ours_\d+", name):
            continue
        shared = [e for e in eps if e["seed"] in ours_seeds] or eps
        reference.append((name.replace("ckpt_", ""), cell(shared)))
    n_panels = 1 + (training_log is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 3.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    col = 0
    if training_log is not None:
        rows = training_log.read_text().strip().splitlines()[1:]
        loss = np.array([float(r.split(",")[2]) for r in rows])
        steps = 200 * np.arange(1, len(loss) + 1)  # logged every 200 steps; the log abbreviates steps (e.g. "12K")
        ax = axes[0][col]
        style_axes(ax)
        ax.plot(steps / 1000, loss, color="#2a78d6", linewidth=2, solid_capstyle="round")
        ax.set_yscale("log")
        ax.set_xlabel("training step (thousands)")
        ax.set_ylabel("training loss (log scale)")
        ax.set_title("Training loss, our run (L1 + 10·KL)", color=TEXT, fontsize=11, loc="left")
        col += 1
    ax = axes[0][col]
    style_axes(ax)
    if ours:
        x = np.array([s / 1000 for s, _ in ours])
        y = np.array([100 * c["rate"] for _, c in ours])
        err = np.array([[100 * (c["rate"] - c["lo"]) for _, c in ours], [100 * (c["hi"] - c["rate"]) for _, c in ours]])
        ax.errorbar(x, y, yerr=err, color="#2a78d6", linewidth=2, elinewidth=1, capsize=0, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        for i, ((s, c), xi, yi) in enumerate(zip(ours, x, y)):  # beside the point, clear of the CI bar
            last = i == len(ours) - 1
            ax.annotate(f"{100 * c['rate']:.1f}%", (xi, yi), textcoords="offset points", xytext=(-8 if last else 8, -2),
                        ha="right" if last else "left", va="top", fontsize=9, color=TEXT_2)
    for name, c in reference:
        ax.axhline(100 * c["rate"], color=MUTED, linewidth=1)
        ax.text(x.min() if ours else 0, 100 * c["rate"] - 2, f"{name} checkpoint, same seeds: {100 * c['rate']:.1f}%",
                fontsize=9, color=TEXT_2, va="top")
    ax.set_ylim(0, 100)
    ax.set_xlabel("training step (thousands)")
    ax.set_ylabel("success rate (%)")
    ax.set_title("Full-chunk success, seeds 1000–1199", color=TEXT, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return {"ours": {str(s): c for s, c in ours}, "reference": {n: c for n, c in reference}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", type=Path, default=[], help="chunk_eval.py output folders (one per checkpoint)")
    parser.add_argument("--checkpoint-runs", nargs="*", type=Path, default=[], help="full-chunk runs per training checkpoint")
    parser.add_argument("--training-log", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    by_ckpt: dict[str, list[dict]] = {}
    for run in args.runs:
        for e in load_run(run):
            by_ckpt.setdefault(e["checkpoint"], []).append(e)
    summary: dict = {}
    if by_ckpt:
        md, summary["sweeps"] = markdown_tables(by_ckpt)
        (args.out / "summary.md").write_text(md)
        sweep_figure(by_ckpt, "delay", args.out / "delay.png")
        sweep_figure(by_ckpt, "displacement", args.out / "displacement.png")
    if args.checkpoint_runs or args.training_log:
        summary["checkpoints"] = checkpoint_figure(args.checkpoint_runs, args.training_log, args.out / "checkpoints.png")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"wrote {sorted(p.name for p in args.out.iterdir())}")


if __name__ == "__main__":
    main()
