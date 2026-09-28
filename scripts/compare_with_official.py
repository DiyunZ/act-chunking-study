"""Paired comparison of a LeRobot eval run against the official eval_info.json.

lerobot-eval seeds episode i with start_seed + i (see rollout seeding in
lerobot/scripts/lerobot_eval.py), so with --seed=1000 our episode i uses the same
initial state as official episode i (seed 1000 + i).

Usage:
    python scripts/compare_with_official.py outputs/eval/official_500ep/eval_info.json
"""

import argparse
import json
import math
from pathlib import Path

OFFICIAL = Path(__file__).resolve().parents[1] / "results/day01/official_eval_info_hub.json"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ours", type=Path, help="eval_info.json written by lerobot-eval")
    parser.add_argument("--official", type=Path, default=OFFICIAL)
    parser.add_argument("--start-seed", type=int, default=1000)
    args = parser.parse_args()

    ours = json.loads(args.ours.read_text())["per_task"][0]["metrics"]["successes"]
    official = {e["seed"]: e["success"] for e in json.loads(args.official.read_text())["per_episode"]}
    seeds = [args.start_seed + i for i in range(len(ours))]
    missing = [s for s in seeds if s not in official]
    if missing:
        raise SystemExit(f"official eval has no episodes for seeds {missing[:5]}...")
    theirs = [official[s] for s in seeds]

    n = len(ours)
    k_ours, k_theirs = sum(ours), sum(theirs)
    both = sum(a and b for a, b in zip(ours, theirs))
    only_ours = sum(a and not b for a, b in zip(ours, theirs))
    only_theirs = sum(b and not a for a, b in zip(ours, theirs))
    agree = n - only_ours - only_theirs

    lo, hi = wilson(k_ours, n)
    lo_t, hi_t = wilson(k_theirs, n)
    print(f"seeds {seeds[0]}-{seeds[-1]} (n={n})")
    print(f"ours     : {k_ours}/{n} = {100 * k_ours / n:.1f}%  (95% Wilson CI {100 * lo:.1f}-{100 * hi:.1f}%)")
    print(f"official : {k_theirs}/{n} = {100 * k_theirs / n:.1f}%  (95% Wilson CI {100 * lo_t:.1f}-{100 * hi_t:.1f}%)")
    print(f"same outcome on {agree}/{n} seeds; both succeed {both}, only ours {only_ours}, only official {only_theirs}")
    discordant = only_ours + only_theirs
    if discordant:
        # Exact two-sided McNemar test on the discordant pairs (binomial with p = 0.5).
        k = min(only_ours, only_theirs)
        p = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(k + 1)) / 2**discordant)
        print(f"exact McNemar p = {p:.3f} ({discordant} discordant seeds)")


if __name__ == "__main__":
    main()
