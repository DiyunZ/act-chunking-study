"""How fast does an ACT chunk move, as a function of the index inside the chunk?

The policy predicts a chunk for demonstration frames of the approach phase. For every chunk
index i we measure the right-arm step size ||a[i+1] - a[i]|| (joint targets, radians per control
step) and average it over frames. For comparison we also measure the demonstrator's own step size
at the same frames.

This tests the replan-10 hypothesis: if the first actions of each chunk are slow, a policy that
only ever executes indices 0-9 moves slowly. Under latency d the executed slice starts at index d,
which would explain why latency helps replan 10.

Usage (small; no simulator):
    python scripts/chunk_profile.py --checkpoint PATH --label NAME --out results/analysis/profile_NAME.json
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RIGHT_ARM = slice(7, 13)  # ALOHA action/state layout: left arm 0-5, left gripper 6, right arm 7-12, right gripper 13
DATASET = "lerobot/aloha_sim_transfer_cube_human"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
    parser.add_argument("--frames-per-episode", type=int, default=6, help="frames sampled from steps 0-150 of each episode")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.envs.configs import AlohaEnv
    from lerobot.policies import make_policy, make_pre_post_processors

    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    policy_cfg.pretrained_path = Path(args.checkpoint)
    policy_cfg.device = args.device
    policy = make_policy(cfg=policy_cfg, env_cfg=AlohaEnv(task="AlohaTransferCube-v0"), rename_map={})
    policy.eval()
    pre, post = make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=args.checkpoint,
                                         preprocessor_overrides={"device_processor": {"device": args.device},
                                                                 "rename_observations_processor": {"rename_map": {}}})

    ds = LeRobotDataset(DATASET, video_backend="pyav")
    # Demonstration actions per episode, straight from the parquet files.
    root = Path(ds.root)
    actions = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)))
    demo = {ep: np.stack(g.sort_values("frame_index")["action"].to_numpy()) for ep, g in actions.groupby("episode_index")}
    starts = {ep: int(g["index"].min()) for ep, g in actions.groupby("episode_index")}

    rng = np.random.default_rng(0)
    chunk_steps, demo_steps, first_offsets = [], [], []
    for ep in sorted(demo):
        for t in sorted(rng.choice(np.arange(0, 150), size=args.frames_per_episode, replace=False)):
            item = ds[starts[ep] + int(t)]
            batch = {"observation.images.top": item["observation.images.top"].unsqueeze(0),
                     "observation.state": item["observation.state"].unsqueeze(0), "task": [""]}
            with torch.no_grad():
                chunk = policy.predict_action_chunk(pre(batch))  # (1, H, 14), normalized
                chunk = post(chunk).cpu().numpy()[0]  # joint targets
            arm = chunk[:, RIGHT_ARM]
            chunk_steps.append(np.linalg.norm(np.diff(arm, axis=0), axis=1))  # (H-1,)
            first_offsets.append(float(np.linalg.norm(arm[0] - item["observation.state"].numpy()[RIGHT_ARM])))
            demo_arm = demo[ep][int(t):int(t) + 12, RIGHT_ARM]
            demo_steps.append(float(np.linalg.norm(np.diff(demo_arm, axis=0), axis=1).mean()))

    profile = np.mean(chunk_steps, axis=0)
    result = {
        "label": args.label,
        "n_frames": len(chunk_steps),
        "step_size_by_index": profile.round(6).tolist(),
        "mean_step_first_10": float(profile[:10].mean()),
        "mean_step_10_to_40": float(profile[10:40].mean()),
        "demo_mean_step_next_10": float(np.mean(demo_steps)),
        "mean_offset_first_action_to_state": float(np.mean(first_offsets)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print({k: v for k, v in result.items() if k != "step_size_by_index"})
    print("step size at indices 0, 2, 5, 10, 20, 40, 80:", [round(float(profile[i]), 4) for i in (0, 2, 5, 10, 20, 40, 80)])


if __name__ == "__main__":
    main()
