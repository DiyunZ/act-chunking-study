#!/usr/bin/env bash
# HF Jobs script: check scripts/chunk_eval.py against lerobot-eval on the official checkpoint.
#
# 1. lerobot-eval runs once per execution mode (n_action_steps 100 / 25 / 10, and temporal
#    ensembling) on seeds 1000-1049, and chunk_eval.py runs the same modes at zero delay. The
#    per-seed outcomes must match.
# 2. A short 4 cm displacement run confirms that the cube moves, that the arms are untouched, and
#    that no contact happens before the move. It also saves two videos.
#
# Upload scripts/ to the dataset repo as code/scripts/ first, then submit from the repo root:
#   hf jobs run --flavor l4x1 --timeout 45m --secrets HF_TOKEN --name act-validate \
#     huggingface/lerobot-gpu:latest bash -c "$(cat scripts/jobs/validate_harness.sh)"
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /lerobot
REPO=FTG64/act-chunking-study
STUDY=/tmp/study
OUT=/tmp/validation_official
mkdir -p "$OUT/lerobot"
hf download "$REPO" --repo-type dataset --include "code/*" --local-dir "$STUDY" > /dev/null
CODE="$STUDY/code/scripts"

python src/lerobot/processor/migrate_policy_normalization.py \
  --pretrained-path lerobot/act_aloha_sim_transfer_cube_human --output-dir /tmp/official > /tmp/migrate.log 2>&1

# Print memory use every 20 s so the job log shows headroom (the first attempt was OOM-killed:
# lerobot-eval keeps every frame of its 10 rendered episodes in RAM, about 1.8 GB per 5-env batch).
(while true; do echo "[mem $(date +%H:%M:%S)] $(free -m | awk '/Mem:/ {print $3 " MB used of " $2 " MB"}')"; sleep 20; done) &
MEM_PID=$!
trap 'kill $MEM_PID 2>/dev/null || true' EXIT

# In-process envs for lerobot-eval as well; see the note at the top of chunk_eval.py.
common=(--policy.path=/tmp/official --env.type=aloha --env.task=AlohaTransferCube-v0
        --eval.n_episodes=50 --eval.batch_size=5 --eval.use_async_envs=false --policy.device=cuda --seed=1000)
run_pair() {  # two lerobot-eval runs at a time
  local pids=() fail=0
  for spec in "$@"; do
    local name=${spec%%:*} extra=${spec#*:}
    # shellcheck disable=SC2086
    lerobot-eval "${common[@]}" $extra --output_dir="$OUT/lerobot/$name" > "$OUT/lerobot_$name.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" -eq 0 ] || { tail -n 25 "$OUT"/lerobot_*.log; exit 1; }
}
echo "== lerobot-eval: full and replan25"
run_pair "full:" "replan25:--policy.n_action_steps=25"
echo "== lerobot-eval: replan10 and te"
run_pair "replan10:--policy.n_action_steps=10" "te:--policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01"
echo "== harness: 4 modes at zero delay"
python "$CODE/chunk_eval.py" --checkpoint /tmp/official --checkpoint-label official --grid validation \
  --seeds 1000:1050 --block 25 --n-envs 5 --workers 4 --out "$OUT/harness" 2>&1 | tee "$OUT/harness.log" | grep -E "tasks|\[|finished"
echo "== harness: displacement check"
python "$CODE/chunk_eval.py" --checkpoint /tmp/official --checkpoint-label official --grid core \
  --only full_d00_m4 --seeds 1000:1010 --block 10 --n-envs 5 --workers 1 --videos-per-condition 2 \
  --out "$OUT/displacement_check" 2>&1 | tee "$OUT/displacement_check.log" | grep -E "tasks|\[|finished"

echo "== harness vs lerobot-eval (seeds 1000-1049)"
python "$CODE/compare_harness.py" --lerobot "$OUT/lerobot" --harness "$OUT/harness" --out "$OUT/validation_summary.json"
echo "== timing and displacement checks"
python - "$OUT" <<'EOF'
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
def episodes(folder):
    return [e for f in sorted(folder.glob("seeds_*.json")) for e in json.loads(f.read_text())["episodes"]]

for cond in ("full_d00_m0", "replan25_d00_m0", "replan10_d00_m0", "te_d00_m0"):
    eps = episodes(out / "harness" / cond)
    touch = sorted(e["first_step_at_reward"]["1"] for e in eps if "1" in e["first_step_at_reward"])
    print(f"first right-gripper contact, {cond}: {len(touch)}/{len(eps)} episodes touch; min {touch[0]}, "
          f"p5 {np.percentile(touch, 5):.0f}, p10 {np.percentile(touch, 10):.0f}, median {np.median(touch):.0f}; "
          f"before step 70: {sum(t < 70 for t in touch)}, before step 100: {sum(t < 100 for t in touch)}")
for e in episodes(out / "displacement_check" / "full_d00_m4"):
    print(f"seed {e['seed']}: move at step {e['t_move']}, moved {e['moved_m']} m, arms unchanged "
          f"{e['arms_unchanged_by_move']}, reward before move {e['reward_before_move']}, success {e['success']}")
EOF
hf upload "$REPO" "$OUT" results/validation_official --repo-type dataset > /dev/null
echo "uploaded to $REPO: results/validation_official"
