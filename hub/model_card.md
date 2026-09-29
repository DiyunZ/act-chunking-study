---
datasets: lerobot/aloha_sim_transfer_cube_human
library_name: lerobot
license: apache-2.0
model_name: act
pipeline_tag: robotics
tags:
- robotics
- act
- lerobot
- aloha
- simulation
---

# ACT for ALOHA sim transfer cube (retrained)

An [ACT](https://huggingface.co/papers/2304.13705) policy trained with LeRobot's official recipe on
[lerobot/aloha_sim_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human).
It is the retrained model of the study
[ACT action chunking under delay and disturbance](https://github.com/DiyunZ/act-chunking-study).
It runs in simulation (gym-aloha) only. It has not been tested on a real robot.

## Files

- The root of `main` holds the final 100k-step policy. It is identical to `checkpoints/100000/pretrained_model`.
- `checkpoints/<step>/pretrained_model` holds the checkpoints at 25k, 50k, 75k and 100k steps, and
  `checkpoints/<step>/training_state` holds the optimizer state. Each step has a tag (`025000`, `050000`, `075000`, `100000`).

## Inputs and outputs

| Feature | Type | Shape |
| --- | --- | --- |
| `observation.images.top` | VISUAL | `(3, 480, 640)` |
| `observation.state` | STATE | `(14,)` |
| `action` (joint position targets) | ACTION | `(14,)` |

ACT predicts 100 actions (2 s at 50 Hz) at a time. By default, all 100 are executed before the next
prediction (`n_action_steps=100`).

## Training

LeRobot 0.6.2 (commit `e595b79`), 100k steps, batch size 8, AdamW with learning rate 1e-5, seed 1000,
about 3.3 h on one L4 GPU (Hugging Face Jobs). The exact command is in the study's
[README](https://github.com/DiyunZ/act-chunking-study#reproduce).

## Evaluation

Success rate with the default full-chunk execution and no delay, on seeds 1000–1199 (95% Wilson intervals):

| Checkpoint | Success |
| --- | --- |
| 25k steps | 53.5% [46.6, 60.3] |
| 50k steps | 69.0% [62.3, 75.0] |
| 75k steps | 72.0% [65.4, 77.8] |
| 100k steps (this model) | 74.5% [68.0, 80.0] |
| Official `lerobot/act_aloha_sim_transfer_cube_human`, same seeds | 81.5% [75.5, 86.3] |

The difference from the official checkpoint is not significant on these seeds (exact McNemar p = 0.054).

The study evaluates five ways of executing the chunks under observation latency and a mid-episode
cube displacement. Results for this model, 200 paired seeds per cell:

| Execution | Nominal | 400 ms latency | Cube moved 6 cm |
| --- | --- | --- | --- |
| Full chunk (default) | 75.0% | 71.5% | 18.5% |
| Replan every 50 steps | 80.5% | 74.0% | 26.5% |
| Replan every 25 steps | 73.5% | 54.0% | 61.5% |
| Replan every 10 steps | 6.0% | 12.0% | 3.5% |
| Temporal ensembling, coefficient 0.01 | 74.5% | 77.0% | 21.0% |
| Temporal ensembling, coefficient −0.05 | 66.5% | 80.0% | 68.0% |

Methods, statistics and failure analysis are in the study's
[writeup](https://github.com/DiyunZ/act-chunking-study/blob/main/WRITEUP.md).

## Run it in simulation

With LeRobot and its `aloha` extra installed:

```bash
lerobot-eval --policy.path=FTG64/act_aloha_transfer_cube --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=50 --eval.batch_size=5 --eval.use_async_envs=false --policy.device=cuda --seed=1000
```

`--eval.use_async_envs=false` works around a LeRobot 0.6.2 bug: asynchronous environment workers
do not import `gym_aloha`. Add `--policy.n_action_steps=25` to replan every 25 steps, or
`--policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01` for temporal ensembling.

## Citation

Please cite ACT and LeRobot:

```bibtex
@inproceedings{zhao2023learning,
    title = {Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
    author = {Zhao, Tony Z. and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
    booktitle = {Robotics: Science and Systems},
    year = {2023}
}

@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```
