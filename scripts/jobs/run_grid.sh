#!/usr/bin/env bash
# HF Jobs script: run scripts/chunk_eval.py for one checkpoint and sync results to the dataset repo.
#
# Settings come from environment variables passed with `hf jobs run -e KEY=VALUE`:
#   CKPT     "official" (the official Hub checkpoint, converted in the job) or
#            "hub:<repo>@<revision>:<subfolder>" (e.g. hub:FTG64/act_aloha_transfer_cube@025000:checkpoints/025000/pretrained_model)
#   LABEL    short name recorded in the results and used for the output folder (results/<RUN>)
#   RUN      output folder name inside results/ of the dataset repo
#   GRID     validation | core            ONLY   optional space-separated condition names
#   SEEDS    start:stop                   WORKERS / NENVS / BLOCK   parallelism settings
#   VIDEOS   videos per condition (default 0)
#
# Submit from the repo root after uploading scripts/ to the dataset repo as code/scripts/, e.g.:
#   hf jobs run --flavor l4x1 --timeout 3h --secrets HF_TOKEN --name act-grid \
#     -e CKPT=official -e LABEL=official -e RUN=core_official -e GRID=core -e SEEDS=1000:1200 \
#     huggingface/lerobot-gpu:latest bash -c "$(cat scripts/jobs/run_grid.sh)"
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /lerobot
REPO=FTG64/act-chunking-study
STUDY=/tmp/study
: "${CKPT:?}" "${LABEL:?}" "${RUN:?}" "${GRID:?}" "${SEEDS:?}"
WORKERS=${WORKERS:-4}; NENVS=${NENVS:-5}; BLOCK=${BLOCK:-25}; VIDEOS=${VIDEOS:-0}; ONLY=${ONLY:-}

(while true; do echo "[mem $(date +%H:%M:%S)] $(free -m | awk '/Mem:/ {print $3 " MB used of " $2 " MB"}')"; sleep 60; done) &
MEM_PID=$!
trap 'kill $MEM_PID 2>/dev/null || true' EXIT

hf download "$REPO" --repo-type dataset --include "code/*" --local-dir "$STUDY" > /dev/null
if [ "$CKPT" = "official" ]; then
  python src/lerobot/processor/migrate_policy_normalization.py \
    --pretrained-path lerobot/act_aloha_sim_transfer_cube_human --output-dir /tmp/policy > /tmp/migrate.log 2>&1
  POLICY=/tmp/policy
else
  spec=${CKPT#hub:}; repo=${spec%%@*}; rest=${spec#*@}; rev=${rest%%:*}; sub=${rest#*:}
  hf download "$repo" --revision "$rev" --include "$sub/*" --local-dir /tmp/policy_dl > /dev/null
  POLICY=/tmp/policy_dl/$sub
fi
echo "== policy ready: $LABEL ($POLICY)"

only_args=()
if [ -n "$ONLY" ]; then only_args=(--only $ONLY); fi  # word-splitting intended
python "$STUDY/code/scripts/chunk_eval.py" --checkpoint "$POLICY" --checkpoint-label "$LABEL" --grid "$GRID" \
  "${only_args[@]}" --seeds "$SEEDS" --block "$BLOCK" --n-envs "$NENVS" --workers "$WORKERS" \
  --videos-per-condition "$VIDEOS" --out "/tmp/$RUN" --hub-repo "$REPO"
echo "== done: results/$RUN in $REPO"
