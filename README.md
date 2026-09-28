# ACT action chunking under delay and disturbance

A two-week imitation-learning study built on [Hugging Face LeRobot](https://github.com/huggingface/lerobot).
An ACT policy is trained on the ALOHA bimanual transfer-cube task in simulation: the right
arm picks up a cube and hands it to the left arm. The study then measures how three ways
of executing the policy's 2-second action chunks hold up when actions arrive late and when
the cube is moved mid-episode:

- run the whole chunk open loop
- replan every *k* steps
- temporal ensembling

**Status:** Day 1 of 14. The pipeline is validated on Apple silicon. See [DEVLOG.md](DEVLOG.md).

## Setup

```bash
brew install uv ffmpeg
git clone https://github.com/huggingface/lerobot.git lerobot
cd lerobot && git checkout e595b7902714ba51f91e47523f66f89c5181b649
uv python install 3.12
uv sync --locked --python 3.12 --extra training --extra aloha
source .venv/bin/activate
```

Every command used so far is listed in [DEVLOG.md](DEVLOG.md) under the day it was run.

## Results

Results are added here as each milestone is completed.
