#!/usr/bin/env bash
# Run the remaining experiment arms back to back on an already-provisioned cluster.
#
# Deliberately not `sky launch`: the Nebius credential expires about every 8 hours and
# died twice during the day, which would strand anything queued overnight. Everything
# here runs over SSH against a cluster that is already up, touching no cloud API. The
# bucket mount is likewise already in place.
#
#   ssh pi05 'nohup bash /tmp/overnight_driver.sh > /tmp/overnight.log 2>&1 &'
#
# Arms, in order:
#   B  curated 270   from arm A's best checkpoint   -- does SARM curation help?
#   C  random 270    from arm A's best checkpoint   -- ... or would any 270 do?
#
# Arm E (backbone trainable) runs on its own cluster instead; it would otherwise write
# to the same /outputs/arm-e-backbone and interleave checkpoints with that run.
set -uo pipefail

cd "$HOME/sky_workdir"
export HF_HOME="$HOME/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_XET=1
RENAME_MAP='{"observation.images.overhead_camera": "observation.images.base_0_rgb", "observation.images.arm_camera": "observation.images.left_wrist_0_rgb"}'

log() { echo "[driver $(date -u +%H:%M:%S)] $*"; }

log "waiting for arm A to finish"
while pgrep -f "operators.policy.train_fast" >/dev/null; do sleep 60; done
log "arm A finished"

# Pick A's best checkpoint by held-out loss rather than taking the last one. Eval runs
# every 250 steps and checkpoints land every 250, so the argmin step maps straight onto
# a saved directory.
best_ckpt() {
  local run="$1" logfile
  logfile=$(ls -t "$HOME"/sky_logs/*/tasks/*.log 2>/dev/null | head -1)
  python3 - "$logfile" "$run" <<'PY'
import re, sys, pathlib
log, run = sys.argv[1], sys.argv[2]
pairs = re.findall(r"step (\d+): eval_loss=([0-9.]+)", pathlib.Path(log).read_text(errors="ignore"))
if not pairs:
    print(""); raise SystemExit
step, loss = min(((int(s), float(v)) for s, v in pairs), key=lambda t: t[1])
d = pathlib.Path(f"/outputs/{run}/checkpoints/{step:06d}/pretrained_model")
# Fall back to the newest saved checkpoint if the argmin step was not itself saved.
if not d.is_dir():
    saved = sorted(pathlib.Path(f"/outputs/{run}/checkpoints").glob("*/pretrained_model"))
    d = saved[-1] if saved else ""
print(d)
print(f"# best step {step} eval_loss {loss}", file=sys.stderr)
PY
}

A_BEST=$(best_ckpt arm-a-stage1)
log "arm A best checkpoint: ${A_BEST:-NONE}"
[ -n "$A_BEST" ] || { log "no checkpoint found, aborting"; exit 1; }

run_arm() {
  local name="$1" base="$2" data="$3" repo="$4" rabc="$5" freeze="$6" steps="$7"
  log "=== $name: starting ($steps steps) ==="
  rm -rf "$HOME/data/run-$name" "$HOME/models/base-$name"
  cp -r "$data" "$HOME/data/run-$name"
  cp -r "$base" "$HOME/models/base-$name"
  uv run --only-group train-pi0 accelerate launch \
    --multi_gpu --num_processes 8 \
    -m operators.policy.train_fast \
    --policy.path "$HOME/models/base-$name" \
    --policy.dtype bfloat16 \
    --policy.push_to_hub false \
    --policy.use_relative_actions true \
    --policy.gradient_checkpointing true \
    --policy.chunk_size 50 \
    $freeze \
    --dataset.repo_id "$repo" \
    --dataset.root "$HOME/data/run-$name" \
    --dataset.video_backend torchcodec \
    --sample_weighting.type rabc \
    --sample_weighting.progress_path "$rabc" \
    --sample_weighting.kappa 0.0215 \
    --rename_map "$RENAME_MAP" \
    --output_dir "/outputs/$name" \
    --job_name "$name" \
    --batch_size 32 \
    --steps "$steps" \
    --save_freq 250 \
    --num_workers 12 \
    --dataset.eval_split 0.05 \
    --eval_steps 250 \
    --wandb.enable false
  log "=== $name: exit $? ==="
}

run_arm arm-b-curated "$A_BEST" \
  /outputs/datasets/candy-cur60-rel binhpham/candy-shop-cur60-rel \
  /outputs/rabc/candy-cur60-rel-sarmv2-12000-s4.parquet \
  "--policy.train_expert_only true" 800

run_arm arm-c-random "$A_BEST" \
  /outputs/datasets/candy-shop-rand60-rel binhpham/candy-shop-rand60-rel \
  /outputs/rabc/candy-shop-rand60-rel-sarmv2-12000-s4.parquet \
  "--policy.train_expert_only true" 800

log "all arms complete"
