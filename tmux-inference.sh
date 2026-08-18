#!/usr/bin/env bash
# Bring up the three inference-side operators in a tmux session named `ops`.
#
#   bash tmux-inference.sh
#   CKPT=~/candy-shop/outputs/dit-lr1e4/checkpoints/007500/pretrained_model bash tmux-inference.sh
#   TASK="pick up a kitkat" bash tmux-inference.sh
#
# The robot itself is not started here -- it runs on its own machine and joins the room.
set -euo pipefail

REPO="${REPO:-$HOME/candy-shop}"
CKPT="${CKPT:-$REPO/outputs/dit-orig/checkpoints/007500/pretrained_model}"
TASK="${TASK:-pick up a twix}"
SESSION="${SESSION:-ops}"

# Diffusion needs its full 100 denoising steps; dropping to 10 moved holdout MAE from
# 5.66 to 6.27 on this checkpoint. Set NUM_STEPS to override (0 = the checkpoint's own).
NUM_STEPS="${NUM_STEPS:-0}"
STEPS_ARG=""
[ "$NUM_STEPS" -gt 0 ] && STEPS_ARG="--num-steps $NUM_STEPS"

[ -f "$CKPT/model.safetensors" ] || {
  echo "no checkpoint at $CKPT" >&2
  echo "checkpoints live in the bucket; pull one with:" >&2
  echo "  aws s3 sync s3://candy-shop/<run>/checkpoints/ $REPO/outputs/<run>/checkpoints/ --exclude '*/training_state/*'" >&2
  exit 1
}

cd "$REPO"
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n reward  -c "$REPO"
tmux new-window  -t "$SESSION"    -n move-to -c "$REPO"
tmux new-window  -t "$SESSION"    -n policy  -c "$REPO"

tmux send-keys -t "$SESSION":reward  "uv run reward --task '$TASK'" C-m
tmux send-keys -t "$SESSION":move-to "uv run move-to" C-m
tmux send-keys -t "$SESSION":policy  "uv run policy --checkpoint $CKPT $STEPS_ARG --task '$TASK'" C-m

echo "started '$SESSION': reward / move-to / policy"
echo "  checkpoint : $CKPT"
echo "  task       : $TASK"
echo "  read a pane: tmux capture-pane -t $SESSION:policy -p -S -200 | grep -v video-overflow"

# Only attach from a terminal, so this stays usable over ssh in a script or from CI.
[ -t 1 ] && exec tmux attach -t "$SESSION"
