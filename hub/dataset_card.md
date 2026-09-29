---
license: mit
pretty_name: ACT action chunking study, per-episode results
tags:
- robotics
- lerobot
- act
- aloha
- evaluation
size_categories:
- 10K<n<100K
---

# ACT action chunking under delay and disturbance: per-episode results

This repository holds the raw outputs of the study
[ACT action chunking under delay and disturbance](https://github.com/DiyunZ/act-chunking-study).
Every simulated episode behind the study's tables is stored as one JSON record. The repository also
holds the scripts that the Hugging Face Jobs ran.

- `code/scripts/` holds the scripts as uploaded for the jobs. The GitHub repository is the maintained copy.
- `results/<run>/meta.json` holds the settings of a run: grid, seeds, deterministic kernels, displacement window, checkpoint, versions and GPU.
- `results/<run>/<condition>/seeds_<first>-<last>.json` holds one block of 25 seeds for one condition, under the key `episodes`.

The task is ALOHA transfer cube (gym-aloha), 400 steps at 50 Hz. The checkpoints are the official
[lerobot/act_aloha_sim_transfer_cube_human](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human)
("official") and a retrained [FTG64/act_aloha_transfer_cube](https://huggingface.co/FTG64/act_aloha_transfer_cube)
("ours").

## Runs

| Run | Checkpoint | Contents |
| --- | --- | --- |
| `validation_official` | official | Harness vs `lerobot-eval` on seeds 1000–1049 (`harness/`, `lerobot/`), a displacement check, and videos |
| `ckpt_ours_025000` to `ckpt_ours_100000` | ours at 25k to 100k steps | Full chunk, no delay, seeds 1000–1199 |
| `ckpt_official` | official | Full chunk, no delay, seeds 1000–1499 (the seeds of its published evaluation) |
| `noise_floor` | official | Repeated runs that measure GPU nondeterminism |
| `core_official`, `core_ours100k` | official, ours at 100k | 5 execution modes × (5 latencies + 3 displacements), seeds 1000–1199 |
| `diag_official`, `diag_ours100k` | official, ours at 100k | Arm trajectories for the first 200 steps (field `trajectory`), seeds 1000–1049 |
| `diag_te_official`, `diag_te_ours100k` | official, ours at 100k | The same trajectories for temporal ensembling: c = 0.01 without latency, c = −0.2 at 0 and 400 ms |
| `te_weights_official`, `te_weights_ours100k` | official, ours at 100k | Temporal ensembling with coefficients 0, −0.05, −0.1 and −0.2, seeds 1000–1199 |

## Condition names

Condition names have the form `<mode>_d<delay>_m<displacement>`, for example `replan25_d00_m6`.

- The mode is `full` (the whole chunk), `replan50`, `replan25` or `replan10` (a new chunk every k
  steps), `te` (temporal ensembling with ACT's default coefficient 0.01), or `te<c>` (coefficient c,
  for example `te-0.05`).
- The delay is the observation latency in 20 ms control steps: 0, 2, 5, 10 or 20.
- The displacement is the distance in cm that the cube is moved, at a random step inside the run's
  displacement window (in `meta.json`; 50–75 for the core, diagnostic and ensembling grids).

## Episode fields

| Field | Meaning |
| --- | --- |
| `seed` | Environment seed. The displacement direction uses seed + 10000. |
| `success` | The cube was transferred to the left gripper (reward 4). |
| `max_reward` | Highest reward stage reached: 1 right gripper touches the cube, 2 lifted, 3 left gripper touches, 4 success. |
| `sum_reward` | Sum of the per-step rewards. |
| `first_step_at_reward` | First step at which each reward stage was reached. |
| `cube_start_xy` | Initial cube position in m. |
| `t_move`, `delta_m`, `moved_m` | Step of the displacement, its vector, and its length in m. |
| `reward_before_move` | Reward stage when the cube was moved (0 means not yet touched). |
| `arms_unchanged_by_move` | The move left the arm joints unchanged. |
| `mode`, `k`, `delay`, `displacement_cm` | The condition's settings. Later runs also record `te_coeff`. |

## Load

```python
import glob, json

episodes = [e for f in glob.glob("results/core_ours100k/*/seeds_*.json") for e in json.load(open(f))["episodes"]]
```
