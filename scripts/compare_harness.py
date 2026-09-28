"""Check chunk_eval.py against lerobot-eval: same checkpoint, same seeds, zero delay, no displacement.

For each execution mode, lerobot-eval runs ACT with the matching n_action_steps /
temporal_ensemble_coeff override, while chunk_eval.py runs the same mode through ChunkExecutor.
Matching per-seed outcomes and summed rewards show that the harness adds no behavior of its own.

Usage:
    python scripts/compare_harness.py --lerobot OUT/lerobot --harness OUT/harness --out summary.json
"""

import argparse
import json
from pathlib import Path

PAIRS = {"full": "full_d00_m0", "replan25": "replan25_d00_m0", "replan10": "replan10_d00_m0", "te": "te_d00_m0"}


def lerobot_episodes(eval_info: Path, start_seed: int) -> dict[int, tuple[bool, float]]:
    metrics = json.loads(eval_info.read_text())["per_task"][0]["metrics"]
    pairs = zip(metrics["successes"], metrics["sum_rewards"], strict=True)
    return {start_seed + i: (bool(s), float(r)) for i, (s, r) in enumerate(pairs)}


def harness_episodes(condition_dir: Path) -> dict[int, tuple[bool, float]]:
    episodes = {}
    for path in sorted(condition_dir.glob("seeds_*.json")):
        for e in json.loads(path.read_text())["episodes"]:
            episodes[e["seed"]] = (bool(e["success"]), float(e["sum_reward"]))
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lerobot", type=Path, required=True, help="folder with <mode>/eval_info.json")
    parser.add_argument("--harness", type=Path, required=True, help="chunk_eval.py --out folder")
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for mode, condition in PAIRS.items():
        ref = lerobot_episodes(args.lerobot / mode / "eval_info.json", args.start_seed)
        ours = harness_episodes(args.harness / condition)
        seeds = sorted(ref.keys() & ours.keys())
        row = {
            "mode": mode,
            "n": len(seeds),
            "lerobot_successes": sum(ref[s][0] for s in seeds),
            "harness_successes": sum(ours[s][0] for s in seeds),
            "same_outcome": sum(ref[s][0] == ours[s][0] for s in seeds),
            "same_sum_reward": sum(abs(ref[s][1] - ours[s][1]) < 1e-6 for s in seeds),
        }
        rows.append(row)
        print(row, flush=True)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
