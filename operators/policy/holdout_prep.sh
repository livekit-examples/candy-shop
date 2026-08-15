#!/usr/bin/env bash
# Split candy-shop into a pre-registered holdout and a training set, then prepare both.
#
# Why this exists: the first round of runs each held out their own 5% *after* curation
# had already reshuffled the episode set, so no two models were scored on the same
# episodes and stage-1's eval episodes leaked into stage-2's training data. Only 14 of
# 500 episodes ended up unseen by every candidate, and those 14 are biased low (mean
# SARM final progress 0.750 against the dataset's 0.810) precisely because curation
# rejected them. That cannot rank models.
#
# So: carve the holdout FIRST, off the raw dataset, and exclude it from everything
# downstream. Stratified 10 per task rather than the tail, so it spans all five tasks
# and is not selected on quality.
#
#   bash operators/policy/holdout_prep.sh
set -euo pipefail

REPO_ID="${REPO_ID:-binhpham/candy-shop}"
SRC_ROOT="${SRC_ROOT:-$HOME/data/candy-shop}"
PER_TASK="${PER_TASK:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
STRIDE="${STRIDE:-4}"
SARM="${SARM:-$HOME/models/sarm}"
PY="${PY:-$HOME/sky_workdir/.venv/bin/python}"
EDIT="${EDIT:-$HOME/sky_workdir/.venv/bin/lerobot-edit-dataset}"

cd "$(dirname "$0")/../.."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "holdout: choosing episodes"
# Deterministic and stratified: every (len/PER_TASK)-th episode within each task, so the
# holdout is spread across each task's recording order rather than concentrated in the
# late episodes, which are systematically better (operators improve with practice).
"$PY" - "$SRC_ROOT" "$REPO_ID" "$PER_TASK" > "$HOME/data/holdout.json" << 'EOF'
import json, sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
root, repo_id, per_task = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = LeRobotDataset(repo_id, root=root)
tasks = d.meta.episodes["tasks"]
by = {}
for ep in range(d.num_episodes):
    by.setdefault(tasks[ep][0] if tasks[ep] else "", []).append(ep)
hold = []
for eps in by.values():
    step = max(1, len(eps) // per_task)
    hold += eps[::step][:per_task]
print(json.dumps(sorted(hold)))
EOF
HOLD=$(cat "$HOME/data/holdout.json")
echo "holdout episodes: $HOLD"

"$PY" - "$SRC_ROOT" "$REPO_ID" "$HOME/data/holdout.json" > "$HOME/data/trainset.json" << 'EOF'
import json, sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
root, repo_id, hold_path = sys.argv[1], sys.argv[2], sys.argv[3]
d = LeRobotDataset(repo_id, root=root)
hold = set(json.load(open(hold_path)))
print(json.dumps(sorted(set(range(d.num_episodes)) - hold)))
EOF
REST=$(cat "$HOME/data/trainset.json")

echo "holdout: building the two datasets"
rm -rf "$HOME/data/candy-holdout" "$HOME/data/candy-train"
"$EDIT" --repo_id "$REPO_ID" --root "$SRC_ROOT" \
  --new_repo_id "${REPO_ID}-holdout" --new_root "$HOME/data/candy-holdout" \
  --operation.type delete_episodes --operation.episode_indices "$REST"
"$EDIT" --repo_id "$REPO_ID" --root "$SRC_ROOT" \
  --new_repo_id "${REPO_ID}-train" --new_root "$HOME/data/candy-train" \
  --operation.type delete_episodes --operation.episode_indices "$HOLD"

echo "holdout: relative-action stats for the training set"
rm -rf "$HOME/data/candy-train-rel"
"$EDIT" --repo_id "${REPO_ID}-train" --root "$HOME/data/candy-train" \
  --new_repo_id "${REPO_ID}-train-rel" --new_root "$HOME/data/candy-train-rel" \
  --operation.type recompute_stats --operation.relative_action true \
  --operation.chunk_size "$CHUNK_SIZE"

# The holdout is scored with the *training set's* stats, never its own: a benchmark that
# renormalises to itself would measure something different for every model tested on it.
echo "holdout: relative-action stats for the holdout (copied from train, not recomputed)"
rm -rf "$HOME/data/candy-holdout-rel"
cp -rL "$HOME/data/candy-holdout" "$HOME/data/candy-holdout-rel"
cp "$HOME/data/candy-train-rel/meta/stats.json" "$HOME/data/candy-holdout-rel/meta/stats.json" 2>/dev/null || \
  cp "$HOME/data/candy-train-rel/meta/"*stats* "$HOME/data/candy-holdout-rel/meta/" 2>/dev/null || true

echo "holdout: scoring SARM progress on the training set"
"$PY" -c "
from lerobot.rewards.sarm.compute_rabc_weights import compute_sarm_progress
import os
compute_sarm_progress(
    dataset_repo_id=os.path.expanduser('$HOME/data/candy-train-rel'),
    reward_model_path=os.path.expanduser('$SARM'),
    output_path=os.path.expanduser('$HOME/data/rabc_train.parquet'),
    head_mode='sparse', device='cuda', num_visualizations=0, stride=$STRIDE,
)
"

echo "holdout: publishing"
mkdir -p /outputs/datasets /outputs/rabc
rm -rf /outputs/datasets/candy-train-rel /outputs/datasets/candy-holdout-rel
cp -rL "$HOME/data/candy-train-rel" /outputs/datasets/candy-train-rel
cp -rL "$HOME/data/candy-holdout-rel" /outputs/datasets/candy-holdout-rel
cp "$HOME/data/rabc_train.parquet" /outputs/rabc/candy-train-rel-sarmv2-12000-s4.parquet
cp "$HOME/data/holdout.json" /outputs/datasets/holdout_episodes.json
echo "holdout: DONE"
