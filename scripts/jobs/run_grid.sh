#!/usr/bin/env bash
# HF Jobs script: run scripts/chunk_eval.py for one checkpoint and sync results to the dataset repo.
#
# Settings come from environment variables passed with `hf jobs run -e KEY=VALUE`:
#   CKPT     "official" (the official Hub checkpoint, converted in the job) or
#            "hub:<repo>@<revision>:<subfolder>" (e.g. hub:FTG64/act_aloha_transfer_cube@025000:checkpoints/025000/pretrained_model)
#   LABEL    short name recorded in the results and used for the output folder (results/<RUN>)
#   RUN      output folder name inside results/ of the dataset repo
#   GRID     validation | core | te_weights   ONLY   optional space-separated condition names
#   SEEDS    start:stop                   WORKERS / NENVS / BLOCK   parallelism settings
#   VIDEOS   videos per condition (default 0)
#   MODES    optional space-separated modes (default: chunk_eval.py's MODES), e.g. "full replan50 replan25 replan10 te"
#   WINDOW   displacement step window start:end (default: chunk_eval.py's)
#   DETERMINISTIC  1 (default) fixes GPU kernels; 0 matches lerobot-eval's nondeterministic kernels
#   LOGTRAJ  log arm trajectories for the first N steps    TECOEFFS  ensembling coefficients for GRID=te_weights
#   REPO     dataset repo that holds code/scripts/ and receives results/ (default FTG64/act-chunking-study)
#
# Upload scripts/ to the dataset repo as code/scripts/, then submit from the repo root, e.g.:
#   hf upload <you>/act-chunking-study scripts code/scripts --repo-type dataset
#   hf jobs run --detach --flavor t4-medium --timeout 4h --secrets HF_TOKEN --name act-grid \
#     -e REPO=<you>/act-chunking-study -e CKPT=official -e LABEL=official -e RUN=core_official \
#     -e GRID=core -e SEEDS=1000:1200 \
#     huggingface/lerobot-gpu:latest bash -c "$(cat scripts/jobs/run_grid.sh)"
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /lerobot
REPO=${REPO:-FTG64/act-chunking-study}  # dataset repo holding code/scripts and results; override with -e REPO=...
STUDY=/tmp/study
: "${CKPT:?}" "${LABEL:?}" "${RUN:?}" "${GRID:?}" "${SEEDS:?}"
WORKERS=${WORKERS:-7}; NENVS=${NENVS:-5}; BLOCK=${BLOCK:-25}; VIDEOS=${VIDEOS:-0}; ONLY=${ONLY:-}
MODES=${MODES:-}; WINDOW=${WINDOW:-}; DETERMINISTIC=${DETERMINISTIC:-1}; LOGTRAJ=${LOGTRAJ:-0}; TECOEFFS=${TECOEFFS:-}

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

extra=()  # word-splitting of $ONLY and $MODES is intended
if [ -n "$ONLY" ]; then extra+=(--only $ONLY); fi
if [ -n "$MODES" ]; then extra+=(--modes $MODES); fi
if [ -n "$WINDOW" ]; then extra+=(--window "$WINDOW"); fi
if [ "$DETERMINISTIC" = "0" ]; then extra+=(--no-deterministic); fi
if [ "$LOGTRAJ" != "0" ]; then extra+=(--log-trajectory "$LOGTRAJ"); fi
if [ -n "$TECOEFFS" ]; then extra+=(--te-coeffs $TECOEFFS); fi
python "$STUDY/code/scripts/chunk_eval.py" --checkpoint "$POLICY" --checkpoint-label "$LABEL" --grid "$GRID" \
  "${extra[@]}" --seeds "$SEEDS" --block "$BLOCK" --n-envs "$NENVS" --workers "$WORKERS" \
  --videos-per-condition "$VIDEOS" --out "/tmp/$RUN" --hub-repo "$REPO"
echo "== done: results/$RUN in $REPO"
