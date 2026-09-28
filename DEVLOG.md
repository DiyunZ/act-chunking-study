# Dev log

Project: how an ACT policy's action-chunk execution holds up under inference delay and
mid-episode disturbance (LeRobot, ALOHA sim transfer cube).

Timebox: Day 1 = 2026-09-27. Freeze at the end of Day 14 = 2026-10-10 (dates to be confirmed by Diyun).

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

### Open decisions (Diyun)

- Training recipe for our own baseline (options and trade-offs in the Day 1 chat summary).
- Daily time budget.
- Confirm the timebox dates above.

### Next session (Day 2)

Start the chosen training run overnight. Then evaluate its checkpoints on seeds 1000–1199
(200 episodes each) and compare them with the official run using
`scripts/compare_with_official.py`.
