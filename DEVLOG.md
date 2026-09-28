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

### Day 1, late: compute moves off the Mac

At Diyun's request, heavy compute no longer runs on the Mac. The local training run was
stopped at step 1k and the 500-episode official eval at episode 253 of 500. Neither left a
checkpoint or a result; both will be rerun in the cloud.

The core-experiment design was approved as proposed, with two additions:

- **Execution modes.** Four modes: full chunk (100 − d steps), replan every 25 steps, replan
  every 10 steps, and temporal ensembling (coefficient 0.01).
- **Delay.** Time-aligned observation latency d ∈ {0, 2, 5, 10, 20} steps (0–400 ms).
- **Perturbation.** Cube displacement of 2, 4 or 6 cm at a random step in 70–100. The new
  position is clipped to the training region, and the same perturbation schedule is used for
  every mode.
- **Sample size.** 200 episodes per condition on seeds 1000–1199, with Wilson CIs and paired
  (McNemar) comparisons. Every failure is labeled with the reward stage it reached.
- **Harness validation.** The harness must reproduce `lerobot-eval` at zero delay before any
  experiment runs.
- **Models.** Run the full grid on both our model and the official checkpoint.

Cloud option under consideration: Hugging Face Jobs. It is LeRobot's own remote path
(`lerobot-train --job.target=<flavor>`). The official image `huggingface/lerobot-gpu:latest`
is rebuilt daily (last on 2026-09-28) and ships gym-aloha with `MUJOCO_GL=egl`. Billing is
per second from prepaid credits, for example `l4x1` (8 vCPU, L4 24 GB) at $0.80/h.

### Day 1, night: Hugging Face Jobs

Cloud provider: Hugging Face Jobs. Diyun logged in on the Mac, and the token stays in the local
HF cache. The account is `FTG64`. PRO is not needed: Jobs only require a positive credit balance.

| Item | Evidence |
|---|---|
| Smoke test 1 failed. Root cause: `AlohaEnv.create_envs` builds `AsyncVectorEnv(..., context="forkserver")`, and forkserver workers never import `gym_aloha`, so `gym.make` raises `NamespaceNotFound`. This is a LeRobot bug that affects any `--eval.batch_size > 1` run with async envs. | `outputs/cloud/smoke_test.log` (not committed) |
| Smoke test 2 passed with `--eval.use_async_envs=false`. Hardware: L4 23 GB, 8 vCPU. Image: LeRobot 0.6.2, the same version as the local pin. Headless EGL render at 6.1 ms/step. Official checkpoint on seeds 1000–1009: **10/10 outcomes identical to the official eval**. | job `6ab9de6b52d0dbd7f1d9f7b2`, `scripts/cloud_smoke_test.sh` |
| Baseline training on `l4x1` at **8.6 steps/s**, so 100k steps take about 3.2 h. Checkpoints go to the private model repo `FTG64/act_aloha_transfer_cube` under `checkpoints/<step>`, tagged by step. | job `6ab9df5152d0dbd7f1d9f7ea` |
| Harness `scripts/chunk_eval.py` passes **20/20 time-alignment checks**: 4 modes × 5 delays, verifying the executed-step index, the observation used by each prediction, and the prediction counts. | `scripts/test_chunk_executor.py` |

Submit command for the training run (the job reuses this exact config):

```bash
lerobot-train --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --dataset.video_backend=pyav \
  --policy.type=act --policy.repo_id=FTG64/act_aloha_transfer_cube --batch_size=8 --steps=100000 \
  --save_freq=25000 --log_freq=200 --num_workers=4 --save_checkpoint_to_hub=true \
  --wandb.enable=false --seed=1000 --output_dir=outputs/train/act_aloha_transfer_cube \
  --job_name=act_aloha_transfer_cube --job.target=l4x1 --job.timeout=8h --job.detach=true
```

Decisions:

1. **Harness built on LeRobot's own loop.** Chose to call `eval_policy` (the function behind
   `lerobot-eval`) and to build the env, policy and processors exactly as `eval_main` does.
   Only two layers are new:
   - `ChunkExecutor` decides which predicted actions are executed. It reuses
     `ACTPolicy.predict_action_chunk` and `ACTTemporalEnsembler`.
   - `CubeDisplacement` is a gym wrapper.

   Rejected writing a custom rollout loop: it would have to re-derive seeding, autoreset and
   success accounting, and any difference there would contaminate the comparison.
2. **Start-of-episode latency.** During the first d steps no frame older than o_0 exists, so the
   policy uses o_0 with offset t (the effective delay ramps from 0 to d). Rejected holding the arms
   still for d steps: that would need a hand-built "hold" action, which is expressed in a
   different normalization space from the state.
3. **Moving the cube.** Chose to edit `qpos`/`qvel` of the cube's free joint in place and then
   call `physics.forward()`. Rejected `physics.reset_context()`: the dm_control source shows it
   calls `reset()` on entry, which would snap both arms back to their initial pose mid-episode.
   Every move now records `arms_unchanged_by_move` and the distance actually moved.
4. **Parallelism.** Chose independent worker processes (spawn), each with in-process envs and one
   batched forward pass per step. Rejected LeRobot's AsyncVectorEnv because of the forkserver bug.
5. **Results storage.** A private dataset repo `FTG64/act-chunking-study` holds `code/` (the
   exact scripts each job ran) and `results/`. Jobs sync results as they go and skip tasks whose
   results already exist, so a crashed job can be resumed.

Costs so far: the two smoke tests took about 2.5 min of `l4x1`, roughly $0.03.

Diyun approved short local debug runs (1–2 episodes). Training and experiments stay in the cloud.

**Local 2-episode checks on MPS (Day 1, night)**

- The first cloud validation job was canceled before it started, so it cost nothing. A local run
  had shown that it would fail: `eval_policy` rejects anything that is not a `PreTrainedPolicy`.
  `ChunkExecutor` now subclasses `PreTrainedPolicy`, with `config_class = ACTConfig` and training
  methods passed through, and the unit test still passes.
- Zero-delay reproduction: the harness gives the same outcome and summed reward as `lerobot-eval`
  on the same machine. Seed 1000 scored 69.0 and failed; seed 1001 scored 167.0 and succeeded.
- Displacement mechanics: every cube moved exactly 0.040 m and `arms_unchanged_by_move` was true.
  The frames just before and after the move (`results/day01/seed1001_move_before_after.png`)
  show the cube jump about 15 px while both arms stay put.
- **Issue: the displacement window is too late.** On seed 1000 the policy first touched the
  cube at step 95, inside the planned window of 70–100 (the demos' earliest grasp is step 105). A
  move after contact would test something other than reaction to a moved target. The validation
  job reports each mode's first-contact steps over 50 seeds. The window will be adjusted from that
  data, subject to Diyun's approval.

**Cloud validation of the harness** (job `6ab9e5346b030d633f69b769`; evidence in `results/day01/validation_official/`)

An earlier attempt (job `6ab9e30b6b030d633f69b705`, $0.10) was OOM-killed. It ran four
`lerobot-eval` processes and four harness workers at once, and each `lerobot-eval` process holds
the frames of its 10 rendered episodes in RAM (about 7 GB per process). The rerun kept at most
two `lerobot-eval` processes at a time (peak 13.7 GB of 30.9 GB). The harness phase peaked at
4.9 GB with 4 workers × 5 envs, which leaves room for 7–8 workers.

Official checkpoint, seeds 1000–1049, zero delay, no displacement:

| Mode | lerobot-eval successes | harness successes | same outcome |
|---|---|---|---|
| full chunk (`n_action_steps=100`) | 35/50 | 35/50 | 50/50 |
| replan every 25 | 38/50 | 39/50 | 43/50 |
| replan every 10 | **4/50** | **4/50** | 50/50 |
| temporal ensembling (0.01) | 35/50 | 35/50 | 50/50 |

- **The harness reproduces the official evaluator's success rate in every mode.** Summed rewards
  are rarely identical on CUDA: the median gap is 6 and only 3 of 50 match in full mode. On MPS
  they were bit-identical. Suspected cause: nondeterministic GPU kernels (`lerobot-eval` sets
  `cudnn.benchmark = True`), with small numerical differences amplified by contact dynamics.
  This still needs to be confirmed (see the open decisions).
- **Finding: replanning every 10 steps nearly breaks ACT even with no delay.** The official
  evaluator shows it too, at 4/50. Only 13 of 50 episodes ever touch the cube, and the median
  first touch comes at step 239, against 145 in full-chunk mode. Replanning every 25 steps is
  fine (38/50). Working hypothesis for the failure analysis: the arm moves too slowly when only
  the first 10 actions of each chunk are ever executed.
- **Contact timing.** First right-gripper contact by mode (minimum / 5th percentile / median
  step): full 84 / 92 / 145, replan25 118 / 125 / 184, TE 117 / 127 / 166, and replan10 152
  (only 13/50 episodes touch). In full mode, 5 of 50 episodes touch before step 100 (steps 84,
  86, 90, 95 and 98), so the planned 70–100 window is too late.
- **Displacement mechanics.** All 10 moves were exactly 0.040 m, with `arms_unchanged_by_move`
  true every time. Two of the 10 moves came after the gripper had already made contact (seeds
  1000 and 1009), which confirms the window problem.

### Open decisions (Diyun)

- New displacement window (proposal: steps 50–75).
- Whether to add a replan-every-50 mode, so that the replanning interval becomes a sweep of 100 / 50 / 25 / 10.
- Whether to run the harness with deterministic GPU kernels for run-to-run reproducibility.

### Next session (Day 2)

1. Evaluate each checkpoint on seeds 1000–1199 as soon as it lands (25k, 50k, 75k, 100k).
2. Once the design is approved, build the harness and validate it at zero delay against
   `lerobot-eval` on the same seeds.

---

## Day 2 — 2026-09-28

**Goal:** finish the baseline milestone, i.e. a training curve, checkpoint success rates and a
reproduction of the official checkpoint.

### Done

| Item | Evidence |
|---|---|
| Baseline training finished on `l4x1`: 100k steps in 3 h 18 min (8.5 steps/s), final loss 0.047 (L1 0.046, KL about 0), about $2.64. Checkpoints `025000`/`050000`/`075000`/`100000` and the final model are in the private repo `FTG64/act_aloha_transfer_cube`. | job `6ab9df5152d0dbd7f1d9f7ea`; `results/day02/training_log_ours.csv` (500 rows, one every 200 steps) |
| `scripts/summarize.py` added. It produces tables (Wilson CIs, paired McNemar against full chunk, failure stages) and the figures `delay.png`, `displacement.png` and `checkpoints.png`. Series colors come from the dataviz reference palette and pass `validate_palette.js` (5 adjacent slots, light mode). Three slots are below 3:1 contrast, which is offset by marker shapes and the Markdown table. | `scripts/summarize.py` |
| `--deterministic` option added to the harness: cuDNN benchmark off, deterministic kernels, `CUBLAS_WORKSPACE_CONFIG`. It is opt-in until Diyun decides. | `scripts/chunk_eval.py` |

Observation: the KL term went to about 0 by the end of training, so the CVAE latent is barely used
(posterior collapse, common for ACT with `kl_weight=10`). At inference ACT feeds a zero latent
either way, so this does not change evaluation.

### Decisions (Diyun, 2026-09-28: all three approved as proposed)

1. **Displacement window: steps 50–75** (1.0–1.5 s), replacing 70–100. In 50 official-checkpoint
   episodes the earliest first contact came at step 84 (full chunk). The replanning modes make
   first contact later (118 and up), so full chunk is the binding case. Every episode records
   `reward_before_move` so that any early contact can be reported.
2. **Replanning-interval sweep: full (100 − d) / 50 / 25 / 10, plus temporal ensembling.** Adding
   `replan50` makes the interval a proper sweep. Keeping `replan10`, whose zero-delay collapse
   (4/50) is itself a finding, means the core grid grows from 32 to 40 conditions per model.
3. **Deterministic GPU kernels for all experiments.** This is now the harness default;
   `--no-deterministic` reproduces the `lerobot-eval` backend settings used in the validation.
   Reruns of a condition on the same hardware then give identical episodes. Each model's grid runs
   in a single job, so all of its conditions share one GPU type (recorded in `meta.json`).

Local 2-episode check of the new defaults (`replan50_d00_m4`): the moves happened at steps 71
and 63, each moved 0.040 m with the arms unchanged, and nothing had touched the cube before the move.
