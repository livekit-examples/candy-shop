#!/usr/bin/env bash
# Build the curated dataset for the stage-2 fine-tune, and everything it needs.
#
# The folding blog's largest single gain came from this step, not from any algorithmic
# change: across 11 experiments their algorithm work moved success rate 5-20 points,
# while fine-tuning the same architecture on a curated dataset one fifth the size moved
# it 50. Their curation was manual (discard episodes whose final frame is not a finished
# fold, drop length outliers, keep consistent technique); ours is the SARM progress
# signal doing the same job -- see operators/policy/curate.py.
#
# Runs on the cheap 1-GPU box while the stage-1 run holds the 8xH100, so none of this
# sits on the expensive critical path.
#
#   bash operators/policy/stage2_prep.sh
#
# Everything is keyed off KEEP_FRACTION; rerunning with a different value rebuilds from
# the untouched ~/data/candy-shop, so it is safe to try more than one.
set -euo pipefail

REPO_ID="${REPO_ID:-binhpham/candy-shop}"
HQ_ID="${HQ_ID:-binhpham/candy-shop-hq}"
SRC_ROOT="${SRC_ROOT:-$HOME/data/candy-shop}"          # absolute stats, never mutated
HQ_ROOT="${HQ_ROOT:-$HOME/data/candy-shop-hq}"
SARM="${SARM:-$HOME/models/sarm}"
RABC_FULL="${RABC_FULL:-/outputs/rabc/candy-shop-rel-sarmv2-12000-s4.parquet}"
KEEP_FRACTION="${KEEP_FRACTION:-0.6}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
STRIDE="${STRIDE:-4}"
PY="${PY:-$HOME/sky_workdir/.venv/bin/python}"
EDIT="${EDIT:-$HOME/sky_workdir/.venv/bin/lerobot-edit-dataset}"

cd "$(dirname "$0")/../.."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

[ -f "$RABC_FULL" ] || { echo "stage2: $RABC_FULL not there yet; stage 1 has not saved it"; exit 1; }

echo "stage2: ranking episodes"
DROP=$("$PY" operators/policy/curate.py \
  --progress-parquet "$RABC_FULL" \
  --keep-fraction "$KEEP_FRACTION" \
  --out-json "$HOME/data/episode_scores.json" \
  | tee /dev/stderr | grep -oE '^--operation.episode_indices .*' | cut -d' ' -f2-)

echo "stage2: building curated dataset"
rm -rf "$HQ_ROOT"
# delete_episodes renumbers episodes 0..N-1 and rebuilds the frame index, which is why
# both the stats and the RABC parquet below have to be recomputed rather than reused.
"$EDIT" \
  --repo_id "$REPO_ID" --root "$SRC_ROOT" \
  --new_repo_id "$HQ_ID" --new_root "$HQ_ROOT" \
  --operation.type delete_episodes \
  --operation.episode_indices "$DROP"

echo "stage2: recomputing relative-action stats over the kept episodes"
rm -rf "${HQ_ROOT}-rel"
"$EDIT" \
  --repo_id "$HQ_ID" --root "$HQ_ROOT" \
  --new_repo_id "${HQ_ID}-rel" --new_root "${HQ_ROOT}-rel" \
  --operation.type recompute_stats \
  --operation.relative_action true \
  --operation.chunk_size "$CHUNK_SIZE"

# Rescore rather than reuse: delete_episodes renumbered the frame index the weights are
# keyed on. Single process -- scoring is video-decode bound at ~8 episodes/min, so the
# curated set costs ~40 min. Sharding it by passing `episodes=` to LeRobotDataset does
# NOT work: compute_sarm_progress indexes the dataset by global frame index while a
# subset re-indexes positionally, so it dies with IndexError out of bounds.
echo "stage2: rescoring SARM progress on the curated dataset"
"$PY" -c "
from lerobot.rewards.sarm.compute_rabc_weights import compute_sarm_progress
import os
compute_sarm_progress(
    dataset_repo_id=os.path.expanduser('${HQ_ROOT}-rel'),
    reward_model_path=os.path.expanduser('$SARM'),
    output_path=os.path.expanduser('$HOME/data/rabc_progress_hq.parquet'),
    head_mode='sparse', device='cuda', num_visualizations=0, stride=$STRIDE,
)
"

echo "stage2: publishing to the bucket"
mkdir -p /outputs/datasets /outputs/rabc
rm -rf /outputs/datasets/candy-shop-hq-rel
cp -rL "${HQ_ROOT}-rel" /outputs/datasets/candy-shop-hq-rel
cp "$HOME/data/rabc_progress_hq.parquet" /outputs/rabc/candy-shop-hq-rel-sarmv2-12000-s4.parquet
echo "stage2: DONE"
