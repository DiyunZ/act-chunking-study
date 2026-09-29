"""Multiple-comparison check for every paired test the writeup reports.

Each claim in WRITEUP.md rests on one exact McNemar test per checkpoint. This script recomputes all of
them from the per-episode results, treats them as one family, and reports:
  * the raw p-value, the Holm-adjusted p-value, and whether it clears the Bonferroni threshold 0.05 / m;
  * the paired difference in success rate with a 95% interval. It is a Clopper-Pearson interval on the
    split of the discordant seeds, scaled by their share of all seeds, so it excludes zero exactly when
    the McNemar test gives p < 0.05. For "no difference" claims it shows how large an effect the data
    still allow: a non-significant test is not evidence of equivalence.

Usage (per-episode results downloaded from the dataset repo into outputs/hub):
    uv run --no-project --with numpy --with matplotlib python scripts/multiple_comparisons.py \
        --results outputs/hub/results --out results/stats
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from summarize import load_run, mcnemar_p

ALPHA = 0.05
MODELS = {"official": "official", "ours": "ours100k"}

# (claim id, claim, kind, (run, condition) A, (run, condition) B). Runs: core, te (the ensembling-weight
# grid) or ckpt. "diff" claims need a significant difference; "null" claims report no difference.
TESTS = [
    ("baseline", "Retrained vs official checkpoint, full chunk, nominal", "null",
     ("ckpt", "official"), ("ckpt", "ours")),
    ("F1a", "Temporal ensembling (TE) is insensitive to latency: 0 vs 400 ms", "null",
     ("core", "te_d00_m0"), ("core", "te_d20_m0")),
    ("F1b", "TE is no better than full chunk after a 6 cm move", "null",
     ("core", "full_d00_m6"), ("core", "te_d00_m6")),
    ("F2a", "Replan 25 beats full chunk after a 6 cm move", "diff",
     ("core", "full_d00_m6"), ("core", "replan25_d00_m6")),
    ("F2b", "Replan 25 loses success with latency: 0 vs 400 ms", "diff",
     ("core", "replan25_d00_m0"), ("core", "replan25_d20_m0")),
    ("F2c", "Replan 25 is worse than full chunk at 400 ms", "diff",
     ("core", "full_d20_m0"), ("core", "replan25_d20_m0")),
    ("F3", "Replan 10 improves with latency: 0 vs 200 ms", "diff",
     ("core", "replan10_d00_m0"), ("core", "replan10_d10_m0")),
    ("C1", "Full chunk loses success with latency: 0 vs 400 ms (checkpoint-dependent)", "diff",
     ("core", "full_d00_m0"), ("core", "full_d20_m0")),
    ("C2", "TE vs full chunk, nominal (checkpoint-dependent)", "diff",
     ("core", "full_d00_m0"), ("core", "te_d00_m0")),
    ("F4a", "TE at c = -0.05 beats the default after a 6 cm move", "diff",
     ("core", "te_d00_m6"), ("te", "te-0.05_d00_m6")),
    ("F4b", "TE at c = -0.05 keeps the latency robustness: default vs -0.05 at 400 ms", "null",
     ("core", "te_d20_m0"), ("te", "te-0.05_d20_m0")),
    ("F4c", "TE at c = -0.05 costs nominal success", "diff",
     ("core", "te_d00_m0"), ("te", "te-0.05_d00_m0")),
    ("F4d", "TE at c = -0.05 beats replan 25 at 400 ms", "diff",
     ("core", "replan25_d20_m0"), ("te", "te-0.05_d20_m0")),
    ("F4e", "TE at c = -0.05 vs replan 25 after a 6 cm move", "diff",
     ("core", "replan25_d00_m6"), ("te", "te-0.05_d00_m6")),
    ("F4f", "TE at c = -0.05 vs replan 25, nominal", "diff",
     ("core", "replan25_d00_m0"), ("te", "te-0.05_d00_m0")),
    ("F5a", "TE at c = -0.1 loses nominal success", "diff",
     ("core", "te_d00_m0"), ("te", "te-0.1_d00_m0")),
    ("F5b", "Reactivity peaks between c = 0 and -0.1: -0.05 vs -0.1 after a 6 cm move", "diff",
     ("te", "te-0.05_d00_m6"), ("te", "te-0.1_d00_m6")),
    ("F5c", "Latency revives TE at c = -0.2: 0 vs 400 ms", "diff",
     ("te", "te-0.2_d00_m0"), ("te", "te-0.2_d20_m0")),
]


def outcomes(results: Path, run: str, key: str, model: str) -> dict[int, bool]:
    if run == "ckpt":  # the baseline test compares the two checkpoints, so `key` names the checkpoint
        folder = "ckpt_official" if key == "official" else "ckpt_ours_100000"
        cond = "full_d00_m0"
    else:
        folder, cond = f"{'core' if run == 'core' else 'te_weights'}_{MODELS[model]}", key
    return {e["seed"]: e["success"] for e in load_run(results / folder) if e["condition"] == cond}


def binom_tail(k: int, n: int, p: float, upper: bool) -> float:
    """P(X >= k) if upper else P(X <= k), X ~ Binomial(n, p)."""
    ks = range(k, n + 1) if upper else range(0, k + 1)
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in ks)


def clopper_pearson(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    def solve(f, target: float) -> float:  # f increases in p; bisection for f(p) = target
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: binom_tail(k, n, p, upper=True), alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: 1 - binom_tail(k, n, p, upper=False), 1 - alpha / 2)
    return lower, upper


def paired_test(a: dict[int, bool], b: dict[int, bool]) -> dict:
    seeds = sorted(a.keys() & b.keys())
    n = len(seeds)
    only_a = sum(1 for s in seeds if a[s] and not b[s])
    only_b = sum(1 for s in seeds if b[s] and not a[s])
    discordant = only_a + only_b
    lo, hi = clopper_pearson(only_b, discordant) if discordant else (0.5, 0.5)
    return {"n": n, "success_a": sum(a[s] for s in seeds), "success_b": sum(b[s] for s in seeds),
            "only_a": only_a, "only_b": only_b, "p": mcnemar_p(only_a, only_b),
            "diff": (only_b - only_a) / n,  # B minus A
            "diff_lo": discordant / n * (2 * lo - 1), "diff_hi": discordant / n * (2 * hi - 1)}


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=ps.__getitem__)
    adjusted, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="folder with the runs' result folders")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for tid, claim, kind, (run_a, key_a), (run_b, key_b) in TESTS:
        models = ["pair"] if run_a == "ckpt" else list(MODELS)
        for model in models:
            a = outcomes(args.results, run_a, key_a, model)
            b = outcomes(args.results, run_b, key_b, model)
            if not a or not b:
                raise SystemExit(f"missing results for {tid} ({model}): {key_a} or {key_b}")
            rows.append({"id": tid, "claim": claim, "kind": kind, "model": model, "a": key_a, "b": key_b,
                         **paired_test(a, b)})
    m = len(rows)
    for row, p_holm in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = p_holm
        row["bonferroni"] = row["p"] <= ALPHA / m

    lines = [
        "# Multiple-comparison check",
        "",
        f"All {m} paired tests behind the writeup's claims form one family. Bonferroni threshold: "
        f"{ALPHA} / {m} = {ALPHA / m:.2g}. Holm-adjusted p-values control the same family-wise error rate "
        "and are never larger than Bonferroni's.",
        "",
        "A and B are conditions on the same seeds; `only A` / `only B` count the seeds where only one of them "
        "succeeds; the difference is B − A in success rate, with a 95% interval that is consistent with the "
        "exact test (Clopper–Pearson on the discordant seeds).",
        "",
        "| id | claim | model | A | B | successes A / B | only A / only B | p | Holm p | Bonferroni | B − A [95% CI] |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        verdict = ("yes" if r["bonferroni"] else "no") if r["kind"] == "diff" else "n/a (no-difference claim)"
        lines.append(
            f"| {r['id']} | {r['claim']} | {r['model']} | `{r['a']}` | `{r['b']}` | {r['success_a']} / {r['success_b']} | "
            f"{r['only_a']} / {r['only_b']} | {r['p']:.2g} | {r['p_holm']:.2g} | {verdict} | "
            f"{100 * r['diff']:+.1f} [{100 * r['diff_lo']:+.1f}, {100 * r['diff_hi']:+.1f}] |")
    lines += ["", "## Verdict per claim", ""]
    for tid in dict.fromkeys(r["id"] for r in rows):
        group = [r for r in rows if r["id"] == tid]
        if group[0]["kind"] == "diff":
            survived = [r["model"] for r in group if r["bonferroni"]]
            status = ("survives on both checkpoints" if len(survived) == len(group)
                      else f"survives only on {', '.join(survived)}" if survived else "does not survive")
        else:
            ci = "; ".join(f"{r['model']} {100 * r['diff']:+.1f} [{100 * r['diff_lo']:+.1f}, {100 * r['diff_hi']:+.1f}] points"
                           for r in group)
            status = f"no-difference claim; paired difference {ci}"
        lines.append(f"- **{tid}** {group[0]['claim']}: {status}.")
    (args.out / "multiple_comparisons.md").write_text("\n".join(lines) + "\n")
    (args.out / "multiple_comparisons.json").write_text(json.dumps({"alpha": ALPHA, "m": m, "tests": rows}, indent=1))
    print(f"{m} tests; wrote {args.out / 'multiple_comparisons.md'}")


if __name__ == "__main__":
    main()
