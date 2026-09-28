"""Is the arm slow, or does it dither? Closed-loop right-arm kinematics per execution mode.

Reads episodes logged with chunk_eval.py --log-trajectory (rows: step, 6 right-arm joint angles,
gripper-cube distance) and summarizes the approach phase (steps 0-150):
  speed        joint-space path length per step (rad/step)
  efficiency   net joint displacement / path length (1 = straight path, low = back and forth)
  reversals    share of steps where a joint's velocity changes sign (joints moving > 1e-4 rad/step)
  dist@100/150 gripper-cube distance (m) at steps 100 and 150

Usage:
    python scripts/analyze_trajectories.py outputs/hub/results/diag_official outputs/hub/results/diag_ours100k \
        --out results/analysis/trajectories.json
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

T_END = 150
ORDER = ["full_d00_m0", "full_d10_m0", "replan25_d00_m0", "replan10_d00_m0", "replan10_d10_m0"]


def episode_metrics(traj: list[list[float]]) -> dict | None:
    rows = np.array([r for r in traj if r[0] <= T_END])
    if len(rows) < T_END:  # ended early (success before step 150) or not logged
        return None
    q, dist = rows[:, 1:7], rows[:, 7]
    dq = np.diff(q, axis=0)
    step = np.linalg.norm(dq, axis=1)
    path = step.sum()
    moving = np.abs(dq[1:]) > 1e-4
    flips = (np.sign(dq[1:]) != np.sign(dq[:-1])) & moving & (np.abs(dq[:-1]) > 1e-4)
    return {
        "speed": float(step.mean()),
        "efficiency": float(np.linalg.norm(q[-1] - q[0]) / path) if path > 0 else float("nan"),
        "reversals": float(flips.sum() / max(moving.sum(), 1)),
        "dist100": float(dist[100]),
        "dist150": float(dist[T_END]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = {}
    for run in args.runs:
        summary[run.name] = {}
        print(f"== {run.name}   (median over episodes; n = episodes still running at step {T_END})")
        print(f"   {'condition':17s} {'n':>3s} {'speed':>7s} {'effic.':>7s} {'revers.':>8s} {'d@100':>6s} {'d@150':>6s} {'success':>8s}")
        for cond in ORDER:
            eps = [e for f in glob.glob(str(run / cond / "seeds_*.json")) for e in json.load(open(f))["episodes"]]
            if not eps:
                continue
            ms = [m for e in eps if (m := episode_metrics(e.get("trajectory", [])))]
            med = {k: float(np.median([m[k] for m in ms])) for k in ms[0]} if ms else {}
            med["n"] = len(ms)
            med["success_rate"] = sum(e["success"] for e in eps) / len(eps)
            summary[run.name][cond] = med
            print(f"   {cond:17s} {med['n']:3d} {med.get('speed', float('nan')):7.4f} {med.get('efficiency', float('nan')):7.2f} "
                  f"{med.get('reversals', float('nan')):8.3f} {med.get('dist100', float('nan')):6.3f} {med.get('dist150', float('nan')):6.3f} "
                  f"{100 * med['success_rate']:7.0f}%")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
