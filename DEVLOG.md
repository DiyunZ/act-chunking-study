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

### Baseline milestone (Days 1–3): met on Day 2

Checkpoint evaluation (job `6aba653a52d0dbd7f1da1c3b`, t4-medium, deterministic harness): full-chunk
execution, zero delay, no displacement.

| Checkpoint | Seeds | Success | 95% Wilson CI |
|---|---|---|---|
| ours 25k | 1000–1199 | 107/200 = 53.5% | 47–60% |
| ours 50k | 1000–1199 | 138/200 = 69.0% | 62–75% |
| ours 75k | 1000–1199 | 144/200 = 72.0% | 65–78% |
| ours 100k | 1000–1199 | 149/200 = 74.5% | 68–80% |
| official | 1000–1499 | 417/500 = 83.4% | 80–86% |

- **Official result reproduced:** 417/500 against the published 415/500 on the same seeds. Outcomes
  agree on 472/500 seeds; the discordant split is 15/13 (McNemar p = 0.85).
- **Our retrained model vs the official checkpoint** on seeds 1000–1199: 149 vs 163. Only ours
  succeeds on 16 seeds, only the official one on 30 (exact McNemar p = 0.054). That is
  borderline, not significant at 0.05. With a single training run we cannot separate
  training-seed variance from anything else, and the writeup will say so.
- **The AGENT_GUIDE's 5–10 epochs are not enough here.** 25k steps (10 epochs at batch 8) reach
  53.5%; performance keeps improving until 100k steps (40 epochs, the official recipe).
- **Failure stages** (ours at 100k): 7 failures never touched the cube, 14 touched it but failed
  to lift, 30 lifted it but no handover was attempted, and 0 attempted a handover and dropped the
  cube. Every checkpoint fails before the left gripper makes contact.
- **Contact timing** (min / 5th percentile): ours 81 / 93, official 83 / 95. No episode touched
  the cube at or before step 75, so the 50–75 window holds for both models.
- Figure: `results/day02/figures/checkpoints.png` (training loss and success per checkpoint; the
  official line is scored on the same 200 seeds).

### Incidents

1. **No l4x1 hardware.** The first checkpoint-eval job waited more than 9 minutes for an L4 and
   was canceled at no cost. Jobs now run on `t4-medium` (8 vCPU, 30 GB, T4, $0.60/h). All
   conditions of a model run in one job, so each model's results come from a single GPU type.
2. **Hung job: my bug.** In `eval_checkpoints.sh` a bare `wait` also waited for the background
   memory monitor, which never exits. The job sat idle for about 20 minutes after its real work
   was done; total job cost about $0.55, of which about $0.20 was idle. The results of steps 1–2
   were already on the Hub. The noise-floor and determinism checks moved to
   `scripts/jobs/noise_floor.sh`, which waits on specific PIDs.
3. **Local network blip.** A DNS failure on the Mac killed the local log-streaming commands; the
   cloud jobs were unaffected. Jobs are now submitted with `--detach` and tracked by pollers that
   retry on network errors.

### Core experiments launched

- `core_official`: job `6aba653a6b030d633f69ccee`, t4-medium, 320 tasks (40 conditions × 8 seed blocks)
- `core_ours100k`: job `6aba73e46b030d633f69cea2`, t4-medium, same grid, checkpoint `100000`
- Noise floor: job `6aba741b52d0dbd7f1da1f88`

### GPU noise floor (job `6aba741b52d0dbd7f1da1f88`, t4-medium; `results/day02/noise_floor/summary.json`)

Official checkpoint, full chunk, seeds 1000–1024. The same command was run twice on one machine.

| Runner | Same outcome | Same summed reward |
|---|---|---|
| `lerobot-eval` (default: `cudnn.benchmark`) | 25/25 | 3/25 |
| harness, `--deterministic` | 25/25 | 7/25 |

- `lerobot-eval` is itself not reproducible at trajectory level on GPU. That explains the
  harness-vs-`lerobot-eval` reward gaps in the Day 1 validation: they are noise, not a harness bug.
- **The deterministic mode did not deliver bit-identical reruns.** The logs show no PyTorch
  "no deterministic implementation" warning, so the remaining variation comes from outside
  PyTorch's determinism checks. Candidates are cuBLAS, EGL rendering and MuJoCo on this
  hardware; none is confirmed.
- What is reproducible is the **outcome level**: 25/25 in both runners, and 199/200 between two
  deterministic harness runs on different T4 machines (full chunk, zero delay: 163 vs 162
  successes). All reported statistics are outcome-based, so conclusions are unaffected. The
  writeup will state trajectory-level nondeterminism as a limitation.
- Decision (Diyun, 2026-09-28): **option A**. Accept outcome-level reproducibility and document
  the trajectory-level nondeterminism as a limitation. No further diagnosis (option B, a
  step-by-step divergence check, was declined).

### Core results: official checkpoint (job `6aba653a6b030d633f69ccee`; `results/core/official/`)

The job ran on t4-medium with the deterministic harness, window 50–75 and 5 modes. It reached its
4 h timeout after all 320 tasks had synced, at about $2.40. Data checks: 40 conditions × 200
episodes on seeds 1000–1199. Every displacement moved exactly the nominal distance with the arms
unchanged, and no episode had touched the cube before the move.

Success rate (%), n = 200 per cell:

| Mode | 0 ms | 40 | 100 | 200 | 400 ms | | 0 cm | 2 | 4 | 6 cm |
|---|---|---|---|---|---|---|---|---|---|---|
| full chunk | 81.0 | 79.5 | 80.0 | 76.0 | 69.0 | | 81.0 | 65.5 | 41.5 | 20.0 |
| replan 50 | 74.0 | 73.0 | 74.5 | 69.0 | 66.5 | | 74.0 | 70.5 | 51.0 | 35.5 |
| replan 25 | 75.0 | 72.0 | 69.5 | 65.0 | 58.0 | | 75.0 | 73.5 | 76.5 | 68.0 |
| replan 10 | 6.5 | 7.0 | 12.5 | 20.5 | 15.0 | | 6.5 | 8.0 | 7.5 | 6.5 |
| temporal ensembling | 71.0 | 71.0 | 70.0 | 70.0 | 68.0 | | 71.0 | 68.5 | 51.5 | 26.5 |

Paired exact McNemar tests on shared seeds. These are the findings as of the official checkpoint
only; they still have to be checked against our model.

1. **No mode wins everywhere: latency robustness and disturbance rejection pull in opposite
   directions.**
   - Temporal ensembling barely notices latency (71 → 68% from 0 to 400 ms, p = 0.21). It is
     worse than full chunk at zero latency (p = 0.011) and level with it at 400 ms (p = 0.89).
   - Replanning every 25 steps is the most latency-sensitive: 75 → 58% (p < 0.001), and worse
     than full chunk at 400 ms (p = 0.018).
2. **Replanning every 25 steps rejects cube displacement.** At 6 cm it keeps 68% (against 75% at
   0 cm, p = 0.15). Full chunk falls to 20%; the paired split is 106 wins for replan 25 against
   10 for full chunk (p = 2e-21).
3. **Temporal ensembling reacts to displacement no better than full-chunk execution**, even though
   it queries the policy every step: 71 → 26.5% at 6 cm (p = 3e-18), and not different from full
   chunk at 6 cm (p = 0.098). Hypothesis: with coefficient 0.01 the exponential weights favor
   older predictions, so the executed action stays dominated by plans made before the move.
4. **Latency *improves* replan 10** (6.5% at 0 ms against 20.5% at 200 ms, p < 0.001). This
   supports the Day 1 hypothesis that replan 10 fails because it only ever executes the start of
   each chunk. Time alignment under latency skips those first actions.

Figures: `results/core/official/delay.png` and `displacement.png`. Tables: `summary.md`.

`core_ours100k` (job `6aba73e46b030d633f69cea2`) had 272/320 tasks synced at 12:30 CDT and will hit
its 4 h timeout. It will be resubmitted with the same `RUN`, and the harness skips finished tasks.

### Core results: both models (`results/core/both/`)

`core_ours100k` (job `6aba73e46b030d633f69cea2`) also reached its 4 h timeout after all 320 tasks
had synced, so no resubmission was needed. It passed the same data checks: 40 × 200 episodes, 0
bad moves, 0 contacts before a move.

Success rate (%), ours at 100k:

| Mode | 0 ms | 40 | 100 | 200 | 400 ms | | 0 cm | 2 | 4 | 6 cm |
|---|---|---|---|---|---|---|---|---|---|---|
| full chunk | 75.0 | 76.0 | 73.5 | 72.5 | 71.5 | | 75.0 | 65.0 | 42.5 | 18.5 |
| replan 50 | 80.5 | 83.0 | 85.5 | 84.0 | 74.0 | | 80.5 | 70.0 | 47.0 | 26.5 |
| replan 25 | 73.5 | 75.0 | 73.0 | 59.0 | 54.0 | | 73.5 | 78.0 | 73.0 | 61.5 |
| replan 10 | 6.0 | 6.0 | 11.0 | 26.5 | 12.0 | | 6.0 | 3.0 | 3.0 | 3.5 |
| temporal ensembling | 74.5 | 75.0 | 76.0 | 76.0 | 77.0 | | 74.5 | 64.5 | 47.0 | 21.0 |

Official-checkpoint findings re-tested on our model (paired exact McNemar tests; official | ours):

| Finding | Official | Ours | Holds in both? |
|---|---|---|---|
| TE unaffected by latency (0 → 400 ms) | 142 → 136, p = 0.21 | 149 → 154, p = 0.38 | yes |
| Replan 25 hurt most by latency (0 → 400 ms) | 150 → 116, p = 8e-5 | 147 → 108, p = 5e-6 | yes |
| Replan 25 below full chunk at 400 ms | 29/51, p = 0.018 | 20/55, p = 7e-5 | yes |
| Full chunk collapses at 6 cm | 162 → 40 | 150 → 37 | yes |
| TE collapses at 6 cm | 142 → 53 | 149 → 42 | yes |
| TE no better than full chunk at 6 cm | 33/20, p = 0.098 | 23/18, p = 0.53 | yes |
| Replan 25 far better than full chunk at 6 cm | 106/10, p = 2e-21 | 97/11, p = 2e-18 | yes |
| Replan 25 little affected by 6 cm | 150 → 136, p = 0.15 | 147 → 123, p = 0.016 | partly: −7 vs −12 points, against −53 to −61 for full chunk and TE |
| Latency improves replan 10 (0 → 200 ms) | 13 → 41, p = 4e-5 | 12 → 53, p = 3e-8 | yes |
| Full chunk hurt by latency | 162 → 138, p = 4e-4 | 150 → 143, p = 0.34 | **no**, model-dependent |
| TE below full chunk at 0 ms | 18/38, p = 0.011 | 34/35, p = 1 | **no**, model-dependent |

**Robust conclusions (replicated on two independently trained checkpoints):**

- Temporal ensembling makes ACT insensitive to up to 400 ms of observation latency, but it
  rejects a moved target no better than full-chunk execution.
- Replanning every 25 steps is the only mode that keeps succeeding after the cube moves 6 cm
  (61–68% against 18–27% for the others), and it pays for this with the largest latency penalty
  (−17 to −20 points at 400 ms).
- Replanning every 10 steps collapses to about 6%, and latency makes it better.

What depends on the checkpoint: whether full-chunk execution degrades with latency, the ranking
of modes at zero latency, and replan 50's advantage on our model.

Cloud cost so far: about $8.8, from the job runtimes (including about $0.55 for the canceled T4
checkpoint-eval job, whose runtime the job list does not show).

### Next (Days 8–14 scope, started early)

1. Failure analysis:
   - where each mode fails, by reward stage;
   - the replan-10 speed hypothesis (measure arm speed and time to first contact);
   - the TE weighting hypothesis (rerun TE with a coefficient that favors recent predictions).
2. Side-by-side demo video, one-page writeup, README.
