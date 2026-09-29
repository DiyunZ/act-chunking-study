#!/usr/bin/env bash
# HF Jobs script for the Day 2 milestone and a GPU-noise check.
#
# 1. Our checkpoints at 25k/50k/75k/100k steps: full chunk, zero delay, seeds 1000-1199.
# 2. The official checkpoint on seeds 1000-1499, the same 500 seeds as its published eval_info.json.
# 3. Noise floor: the same lerobot-eval command twice (seeds 1000-1024). If the summed rewards
#    differ between the two runs, GPU nondeterminism is confirmed.
# 4. The same harness command twice with --deterministic. This checks whether fixed kernels make
#    reruns identical.
#
# Upload scripts/ to the dataset repo as code/scripts/, then submit from the repo root:
#   hf jobs run --flavor l4x1 --timeout 90m --secrets HF_TOKEN --name act-ckpt-eval \
#     huggingface/lerobot-gpu:latest bash -c "$(cat scripts/jobs/eval_checkpoints.sh)"
# NOTE (2026-09-28): the first run of this script hung in step 3. A bare `wait` also waits for the
# background memory monitor, which never exits. Steps 3-4 now live in scripts/jobs/noise_floor.sh,
# which waits on specific PIDs only.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /lerobot
REPO=${REPO:-FTG64/act-chunking-study}  # dataset repo holding code/scripts and results; override with -e REPO=...
MODEL=${MODEL:-FTG64/act_aloha_transfer_cube}  # model repo with our checkpoints; override with -e MODEL=...
STUDY=/tmp/study
WORKERS=7

(while true; do echo "[mem $(date +%H:%M:%S)] $(free -m | awk '/Mem:/ {print $3 " MB used of " $2 " MB"}')"; sleep 60; done) &
MEM_PID=$!
trap 'kill $MEM_PID 2>/dev/null || true' EXIT

hf download "$REPO" --repo-type dataset --include "code/*" --local-dir "$STUDY" > /dev/null
CODE="$STUDY/code/scripts"
full_mode() {  # label policy_dir seeds [extra harness args...]
  local label=$1 policy=$2 seeds=$3; shift 3
  python "$CODE/chunk_eval.py" --checkpoint "$policy" --checkpoint-label "$label" --grid validation \
    --only full_d00_m0 --seeds "$seeds" --block 25 --n-envs 5 --workers "$WORKERS" \
    --out "/tmp/$label" --hub-repo "$REPO" "$@" | grep -E "tasks to run|finished"
}

for step in 025000 050000 075000 100000; do
  echo "== ours, checkpoint $step"
  hf download "$MODEL" --revision "$step" --include "checkpoints/$step/pretrained_model/*" --local-dir /tmp/ours > /dev/null
  full_mode "ckpt_ours_$step" "/tmp/ours/checkpoints/$step/pretrained_model" 1000:1200
done

echo "== official checkpoint, seeds 1000-1499"
python src/lerobot/processor/migrate_policy_normalization.py \
  --pretrained-path lerobot/act_aloha_sim_transfer_cube_human --output-dir /tmp/official > /tmp/migrate.log 2>&1
full_mode ckpt_official /tmp/official 1000:1500

echo "== done"
