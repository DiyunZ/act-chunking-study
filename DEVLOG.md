# Dev log

Project: how an ACT policy's action-chunk execution holds up under inference delay and
mid-episode disturbance (LeRobot, ALOHA sim transfer cube).

Timebox: Day 1 = 2026-09-27, with a hard freeze at the end of Day 14 (2026-10-10). Diyun
(2026-09-27): stopping early is fine once the results are in, and the Mac can run around the clock.

Every entry records what was done, the evidence, and each technical decision as
*chose / rejected / why*.

---

## Day 1 — 2026-09-27

**Goal:** run the official pipeline on the Mac and produce the first external evidence:
the official checkpoint's success rate plus commands that reproduce it.

### Done

| Item | Evidence |
|---|---|
| Environment: LeRobot commit `e595b79` (2026-09-25), Python 3.12.14, torch 2.11.0 with MPS, gym-aloha | commands below |
| Official checkpoint `lerobot/act_aloha_sim_transfer_cube_human` converted to the current processor format | `checkpoints/` (local only) |
| Smoke eval, seeds 1000–1009: 7/10 success, **same outcome as the official eval on 10/10 seeds** | `results/day01/official_smoke_10ep_compare.txt` |
| 300-step training speed test on MPS: **0.288 s/step** at batch 8, loss 15.9 → 3.3 | `results/day01/speedtest_mps.txt` |
| Full 500-episode eval of the official checkpoint (seeds 1000–1499) | running |

Reference numbers from the Hub: the official checkpoint scored **83.0%** over 500 episodes
(seeds 1000–1499) and was trained for 100k steps at batch 8 (AdamW, lr 1e-5, weight decay
1e-4, chunk size 100, n_action_steps 100, no temporal ensembling, ResNet-18, VAE on).

### Decisions

1. **Environment manager.** Chose uv with `uv sync --locked` from the LeRobot source tree.
   The AGENT_GUIDE recommends it, and the lockfile makes the install reproducible.
   Rejected conda/miniforge, the installation guide's default, to avoid a second Python
   distribution on the machine.
2. **Video decoding.** Chose PyAV (`--dataset.video_backend=pyav`). TorchCodec 0.11.1 only
   loads FFmpeg 4–8, and Homebrew ships FFmpeg 9.0.2. Rejected installing an older FFmpeg
   and pointing the loader at it: it adds an environment hack, and the backend only affects
   loading speed (measured data time 0.003 s/step, negligible next to the 0.285 s update).
3. **Official checkpoint format.** The Hub checkpoint predates LeRobot's processor pipeline,
   so it was converted locally with LeRobot's own
   `src/lerobot/processor/migrate_policy_normalization.py`; all weights matched. Nothing
   was pushed.
4. **Eval seeds.** Start at 1000, as the official `eval_info.json` does. `lerobot-eval`
   seeds episode *i* with `start_seed + i`, so every episode can be compared with the
   official run seed by seed (`scripts/compare_with_official.py`).
5. **Task flag.** `--env.task=AlohaTransferCube-v0` must be passed explicitly; the aloha env
   defaults to the insertion task.
6. **Baseline training recipe (decided by Diyun: option C).** Use the official recipe
   (100k steps, batch 8, seed 1000), save a checkpoint every 25k steps, and evaluate each
   checkpoint on seeds 1000–1199. A single run then answers two questions: whether the official
   result reproduces on this machine, and whether the AGENT_GUIDE's 5–10 epochs (12.5k–25k
   steps here) would have been enough. It also gives a steps-vs-success curve.
   - Rejected A (official recipe with no intermediate evals) because it answers only the first
     question.
   - Rejected B (25k steps only) because a lower success rate could not be attributed to the
     shorter run rather than to something else.
   - The current `ACTConfig` defaults were diffed against the official `train_config.json`. The
     policy and optimizer are identical, neither run uses an LR scheduler, and the official run
     also decoded video with PyAV.
   - One deliberate difference: the official run evaluated in simulation during training
     (`eval_freq=20000`). Here training has no env attached and checkpoints are evaluated
     afterwards on fixed seeds. This changes only the random-number stream, not the recipe.
7. **Perturbation timing input.** In the 50 human demos, the right gripper is at its widest
   opening at step 68 (median). It closes on the cube at step 153 (median; earliest 105).
   The left gripper closes for the handover at step 298 (median). A mid-episode cube
   displacement therefore belongs in roughly steps 70–100: the arm is approaching and no demo
   has started grasping. This still has to be checked against the policy's own timing.

### Commands

```bash
# Setup (macOS, Apple silicon)
brew install uv ffmpeg
git clone https://github.com/huggingface/lerobot.git lerobot
cd lerobot && git checkout e595b7902714ba51f91e47523f66f89c5181b649
uv python install 3.12
uv sync --locked --python 3.12 --extra training --extra aloha
source .venv/bin/activate

# Convert the official checkpoint to the current format
python src/lerobot/processor/migrate_policy_normalization.py \
  --pretrained-path lerobot/act_aloha_sim_transfer_cube_human \
  --output-dir ../checkpoints/official_act_transfer_cube

# Evaluate it (10-episode smoke test; the full run uses --eval.n_episodes=500)
lerobot-eval --policy.path=../checkpoints/official_act_transfer_cube \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=10 --eval.batch_size=1 --policy.device=mps --seed=1000 \
  --output_dir=../outputs/eval/official_smoke_10ep
python ../scripts/compare_with_official.py ../outputs/eval/official_smoke_10ep/eval_info.json

# Training speed test
lerobot-train --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.video_backend=pyav --policy.type=act --policy.device=mps \
  --policy.push_to_hub=false --batch_size=8 --steps=300 --log_freq=50 \
  --save_freq=100000 --wandb.enable=false --seed=1000 \
  --output_dir=../outputs/train/speedtest_mps --job_name=speedtest_mps
```

```bash
# Baseline training, official recipe (started 2026-09-27 21:53, about 8 h on MPS)
caffeinate -is lerobot-train --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.video_backend=pyav --policy.type=act --policy.device=mps \
  --policy.push_to_hub=false --batch_size=8 --steps=100000 --save_freq=25000 \
  --log_freq=200 --num_workers=4 --wandb.enable=false --seed=1000 \
  --output_dir=../outputs/train/act_tc_official_recipe --job_name=act_tc_official_recipe
```

Environment note: importing both OpenCV and PyAV prints macOS `objc` warnings about duplicate
`AVF*` classes, because both bundle libavdevice. Nothing has crashed; keep an eye on it.

### Open decisions (Diyun)

- Design of the core experiment (Days 4–7): the delay model, the execution modes, the
  perturbation, the episode counts, and whether to also run the official checkpoint. The
  proposal is in the Day 1 chat.

### Next session (Day 2)

1. Evaluate each checkpoint on seeds 1000–1199 as soon as it lands (25k, 50k, 75k, 100k).
2. Once the design is approved, build the harness and validate it at zero delay against
   `lerobot-eval` on the same seeds.
