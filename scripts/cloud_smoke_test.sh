#!/usr/bin/env bash
# Smoke test for Hugging Face Jobs with the official LeRobot GPU image.
# Submit from the repo root:
#   hf jobs run --flavor l4x1 --timeout 20m --secrets HF_TOKEN --name act-smoke \
#     huggingface/lerobot-gpu:latest bash -c "$(cat scripts/cloud_smoke_test.sh)"
set -euo pipefail
cd /lerobot

echo "== hardware"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "cpu cores: $(nproc)"

echo "== versions"
python - <<'EOF'
import platform, torch, lerobot
print("python", platform.python_version(), "| torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0))
print("lerobot", lerobot.__version__)
EOF
git -C /lerobot log -1 --format='lerobot commit %H %cd' 2>/dev/null || echo "lerobot commit: no .git in image"

echo "== hub auth"
hf auth whoami | head -1

echo "== headless sim rendering"
python - <<'EOF'
import time, gymnasium as gym, gym_aloha  # noqa: F401
env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
obs, _ = env.reset(seed=1000)
t = time.time()
for _ in range(100):
    obs, *_ = env.step(env.action_space.sample())
print("render ok:", obs["pixels"]["top"].shape, f"{(time.time() - t) * 10:.1f} ms/step")
EOF

echo "== official checkpoint, seeds 1000-1009"
python src/lerobot/processor/migrate_policy_normalization.py \
  --pretrained-path lerobot/act_aloha_sim_transfer_cube_human --output-dir /tmp/official > /tmp/migrate.log 2>&1 \
  && echo "migration ok" || { tail -20 /tmp/migrate.log; exit 1; }
start=$(date +%s)
# use_async_envs=false: LeRobot starts AsyncVectorEnv workers with context="forkserver", and those
# workers never import gym_aloha, so gym.make fails with "Namespace gym_aloha not found".
lerobot-eval --policy.path=/tmp/official --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=10 --eval.batch_size=10 --eval.use_async_envs=false --policy.device=cuda --seed=1000 \
  --output_dir=/tmp/eval_smoke > /tmp/eval.log 2>&1 || { tail -40 /tmp/eval.log; exit 1; }
echo "eval wall time: $(( $(date +%s) - start )) s for 10 episodes (10 envs in one process)"
python - <<'EOF'
import json
m = json.load(open("/tmp/eval_smoke/eval_info.json"))["per_task"][0]["metrics"]
official = [False, True, True, True, True, False, True, True, True, False]  # official eval_info.json, seeds 1000-1009
print("successes:", m["successes"])
print("sum_rewards:", m["sum_rewards"])
print(f"same outcome as the official run on {sum(a == b for a, b in zip(m['successes'], official))}/10 seeds")
EOF
