#!/usr/bin/env bash
# HF Jobs script: how reproducible is an evaluation run on the GPU?
#
# 1. The same lerobot-eval command twice (official checkpoint, full chunk, seeds 1000-1024).
# 2. The same chunk_eval.py command twice with deterministic kernels (the harness default).
# Each run pair is compared episode by episode (outcome and summed reward).
#
# Upload scripts/ to the dataset repo as code/scripts/, then submit from the repo root:
#   hf jobs run --flavor t4-medium --timeout 40m --secrets HF_TOKEN --name act-noise-floor \
#     huggingface/lerobot-gpu:latest bash -c "$(cat scripts/jobs/noise_floor.sh)"
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /lerobot
REPO=${REPO:-FTG64/act-chunking-study}  # dataset repo holding code/scripts and results; override with -e REPO=...
STUDY=/tmp/study
OUT=/tmp/noise_floor
mkdir -p "$OUT"
hf download "$REPO" --repo-type dataset --include "code/*" --local-dir "$STUDY" > /dev/null
CODE="$STUDY/code/scripts"
python src/lerobot/processor/migrate_policy_normalization.py \
  --pretrained-path lerobot/act_aloha_sim_transfer_cube_human --output-dir /tmp/official > /tmp/migrate.log 2>&1

wait_all() {  # wait for the given PIDs only; a bare `wait` would also wait for unrelated background jobs
  local fail=0
  for p in "$@"; do wait "$p" || fail=1; done
  return $fail
}

echo "== two identical lerobot-eval runs"
pids=()
for i in 1 2; do
  lerobot-eval --policy.path=/tmp/official --env.type=aloha --env.task=AlohaTransferCube-v0 \
    --eval.n_episodes=25 --eval.batch_size=5 --eval.use_async_envs=false --policy.device=cuda --seed=1000 \
    --output_dir="$OUT/lerobot_run$i" > "$OUT/lerobot_run$i.log" 2>&1 & pids+=($!)
done
wait_all "${pids[@]}" || { tail -n 30 "$OUT"/lerobot_run*.log; exit 1; }

echo "== two identical harness runs (deterministic)"
pids=()
for i in 1 2; do
  python "$CODE/chunk_eval.py" --checkpoint /tmp/official --checkpoint-label official --grid validation \
    --only full_d00_m0 --seeds 1000:1025 --block 25 --n-envs 5 --workers 1 \
    --out "$OUT/harness_det_run$i" > "$OUT/harness_det_run$i.log" 2>&1 & pids+=($!)
done
wait_all "${pids[@]}" || { tail -n 30 "$OUT"/harness_det_run*.log; exit 1; }

python - <<'EOF'
import json
from pathlib import Path

root = Path("/tmp/noise_floor")
def lerobot(i):
    m = json.loads((root / f"lerobot_run{i}" / "eval_info.json").read_text())["per_task"][0]["metrics"]
    return list(zip(m["successes"], m["sum_rewards"]))
def harness(i):
    eps = json.loads(next((root / f"harness_det_run{i}" / "full_d00_m0").glob("*.json")).read_text())["episodes"]
    return [(e["success"], e["sum_reward"]) for e in sorted(eps, key=lambda e: e["seed"])]
summary = {}
for name, load in (("lerobot_eval_default", lerobot), ("harness_deterministic", harness)):
    a, b = load(1), load(2)
    summary[name] = {"n": len(a), "same_outcome": sum(x[0] == y[0] for x, y in zip(a, b)),
                     "same_sum_reward": sum(x[1] == y[1] for x, y in zip(a, b))}
    print(name, summary[name], flush=True)
(root / "summary.json").write_text(json.dumps(summary, indent=1))
EOF
hf upload "$REPO" "$OUT" results/noise_floor --repo-type dataset --exclude "*.mp4" > /dev/null
echo "== done"
