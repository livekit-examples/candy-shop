#!/usr/bin/env bash
# Score every finished arm on the pre-registered holdout, incrementally.
#
# Polls rather than waiting on a single completion signal: the arms finish on different
# clusters at different times, and coordinating across them is more fragile than just
# checking what exists. Each arm is scored once, when its run has stopped producing
# checkpoints, and the result is written next to the others for comparison.
#
#   ssh pi0 'nohup bash /tmp/score_all.sh > /tmp/score_all.log 2>&1 &'
#
# Runs on the small box so it never contends with training.
set -uo pipefail

cd "$HOME/sky_workdir"
export HF_HOME="$HOME/.cache/huggingface"
RESULTS=/outputs/holdout_scores
mkdir -p "$RESULTS"

ARMS="arm-a-stage1 arm-b-curated arm-c-random arm-e-backbone"

log() { echo "[score $(date -u +%H:%M:%S)] $*"; }

# A run is finished when its newest checkpoint has been still for longer than the gap
# between checkpoints. The first version used 900s, but at ~4 s/step with save_freq=250
# the gap between saves is ~17 min, so it fired mid-training and scored arm A at step
# 2250 while it was still running. 2700s clears that with margin; the expected count
# check is the real guard, with the timer only as a backstop for runs stopped early.
declare -A EXPECTED=( [arm-a-stage1]=12 [arm-b-curated]=3 [arm-c-random]=3 [arm-e-backbone]=12 )
settled() {
  local run="$1" newest have
  newest=$(ls -dt /outputs/"$run"/checkpoints/*/ 2>/dev/null | head -1)
  [ -n "$newest" ] || return 1
  have=$(ls -d /outputs/"$run"/checkpoints/*/ 2>/dev/null | wc -l)
  [ "$have" -ge "${EXPECTED[$run]:-999}" ] && return 0
  [ $(( $(date +%s) - $(stat -c %Y "$newest") )) -gt 2700 ]
}

# Best by held-out flow-matching loss, read from whichever training log carries this
# run's name. That metric only ranks within a run; the holdout score below is what
# compares across them.
best_ckpt() {
  local run="$1"
  python3 - "$run" <<'PY'
import re, sys, pathlib, glob
run = sys.argv[1]
best = None
for f in glob.glob("/outputs/%s/checkpoints/*/pretrained_model" % run):
    pass
steps = sorted(int(p.split("/")[-2]) for p in glob.glob("/outputs/%s/checkpoints/*/pretrained_model" % run))
# Without the training log we cannot read eval_loss, so prefer the log if present.
logs = sorted(glob.glob("/outputs/%s/*.log" % run)) + sorted(glob.glob("/outputs/%s/train.log" % run))
losses = {}
for lg in logs:
    for s, v in re.findall(r"step (\d+): eval_loss=([0-9.]+)", pathlib.Path(lg).read_text(errors="ignore")):
        losses[int(s)] = float(v)
cand = [s for s in steps if s in losses]
step = min(cand, key=lambda s: losses[s]) if cand else (steps[-1] if steps else None)
print(f"/outputs/{run}/checkpoints/{step:06d}/pretrained_model" if step is not None else "")
PY
}

while true; do
  pending=0
  for arm in $ARMS; do
    out="$RESULTS/$arm.json"
    [ -f "$out" ] && continue
    if settled "$arm"; then
      ckpt=$(best_ckpt "$arm")
      if [ -n "$ckpt" ] && [ -d "$ckpt" ]; then
        log "scoring $arm -> $(basename $(dirname $ckpt))"
        .venv/bin/python -m operators.policy.eval_holdout \
          --checkpoint "$ckpt" \
          --holdout-root /outputs/datasets/candy-holdout-rel \
          --holdout-repo-id binhpham/candy-shop-holdout \
          --json-out "$out" 2>&1 | tail -20
        log "$arm scored"
      else
        log "$arm: no checkpoint yet"; pending=1
      fi
    else
      pending=1
    fi
  done
  if [ "$pending" = "0" ]; then
    log "all arms scored"
    echo "=== HOLDOUT SCORES (mean absolute error, joint units) ==="
    for f in "$RESULTS"/*.json; do echo "--- $(basename $f .json)"; cat "$f"; done
    break
  fi
  sleep 900
done
