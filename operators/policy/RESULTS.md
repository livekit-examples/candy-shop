# pi0.5 folding-recipe runs, 2026-08-15

> **Round 2 (below, "Controlled comparison") supersedes the round-1 conclusions.**
> Round 1 could not compare models: each run held out its own 5% *after* curation had
> reshuffled the episodes, so no two were scored on the same data. Round 2 carves a
> stratified 50-episode holdout off the raw dataset first, excludes it everywhere, and
> scores every arm on it in joint units.

## Controlled comparison (the one to trust)

450 training episodes / 50 held out, stratified 10 per task. Mean absolute error between
the predicted action chunk and what the operator actually did, in joint units, over 382
samples spanning all 50 episodes.

| Arm | What | h1 | h10 | h25 | h50 |
|:---|:---|---:|---:|---:|---:|
| C | stage 2 on a **random** 270 | 3.517 | 4.906 | 6.736 | **8.716** |
| A | **stage 1 only**, all 450 | 3.559 | 5.004 | 6.839 | **8.827** |
| B | stage 2 on the **SARM-curated** 270 | 3.732 | 5.169 | 7.019 | **9.136** |
| E | stage 1, **backbone trainable** | 4.009 | 5.485 | 7.318 | **9.369** |

**SARM curation does not help; it slightly hurts.** B is worse than the random control at
every horizon, and worse than doing no stage 2 at all. The blog curated 5,688 raw
episodes containing real failures and several competing strategies, and cutting to 21%
removed genuine garbage. Every episode here already completes the task, so ranking them
by SARM progress selects on something other than quality, and discarding 180 of 450 costs
more than the ranking returns. Their largest single lever does not transfer.

**The two-stage recipe does not transfer either.** Stage 1 alone beats the curated
fine-tune and ties the random one.

**Freezing the VLM is confirmed**, now on a matched comparison: E is worst here, and its
own curve rises monotonically from its first eval (0.0719 at step 250 to 0.1333 at 3000)
while A falls to 0.0664. Identical data, steps, schedule and LR; only the trainable
parameter set differs.

**Ship arm A**: `/outputs/arm-a-stage1/checkpoints/002250`, pi0.5, `train_expert_only`,
relative actions, RABC, no curation. Simplest pipeline and among the best measured.

**What this does not establish.** A, B and C sit within 1-5% of each other with no seed
replicates, and a replicate in round 1 showed differences that size can be noise; only
E's 6% gap is comfortably outside it. So "curation is not earning its cost" and "backbone
training is harmful" are supported; "A beats C" is not. And all of it is open-loop
imitation error, which rewards matching the demonstration rather than succeeding at the
task -- the arm is still the only thing that settles that.

---

## Round 1 (superseded, kept for the reasoning)

Applying the LeRobot folding recipe (https://huggingface.co/spaces/lerobot/robot-folding)
to candy-shop: 500 episodes, 92,941 frames, 30 fps, 2 cameras. Everything below ran on
8xH100 at effective batch 256, the blog's own figure, via
`operators/policy/skypilot_pi05.yaml`.

The headline caveat first: **`eval_loss` here is flow-matching loss on held-out
episodes, not task success.** The blog was emphatic that the two diverge and that none
of their conclusions survived without rollouts. Everything here ranks candidates; the
arm decides.

## Runs

| # | Config | Data | Best eval_loss | @ step |
|:--|:--|:--|:--:|:--:|
| 1 | pi0.5, `freeze_vision_encoder` (blog-faithful) | 500 eps | 0.0707 | 500 |
| 2 | pi0.5, `train_expert_only` | 500 eps | **0.0645** | 2500 |
| 3 | stage 2 from #2, 60% curated | 300 eps | 0.0548 | 100 |
| 4 | stage 2 from #2, 40% curated | 200 eps | 0.0600 | 600 |
| 5 | #3 repeated with seed 2000 | 300 eps | 0.0538 | 150 |

All share: relative actions, RABC (kappa 0.0215) on SARM progress, chunk_size 50,
5% held-out eval split.

## What holds up

**Freeze the VLM.** Run 2 beat run 1 by 9% on an identical eval set, and run 1 was
overfitting from step 500 (0.0707 -> 0.0722 -> 0.0770) while run 2 kept improving to
2500. This is the only strictly like-for-like comparison here, and it contradicts the
blog, which trains the Gemma backbone. The reason is data scale: they fine-tuned on
3.2M frames, we have 92,941. A 2B backbone memorises 93k frames at batch 256.

**Aggressive curation does not pay at this scale.** Run 4 scored worse than run 3
(0.0600 vs 0.0548) *despite* its eval episodes being drawn from a higher-ranked pool,
i.e. an easier target. The blog kept ~21% of 5,688 episodes; at 500 episodes, cutting
to 40% costs more than it returns. Part of that cost is mechanical: each curated set
gets its relative-action stats recomputed over only its own episodes, so the harder the
cut, the further those statistics drift from the base checkpoint's, and the more of the
fine-tune is spent re-adapting. Run 4 started at eval_loss 0.131 against run 3's 0.059
from the *same* checkpoint, purely from that shift.

**SARM curation finds real structure.** Kept vs dropped episodes differ on final
progress (0.856 vs 0.742) and monotonicity (0.809 vs 0.686), across a 0.46-0.98 range.
Kept episodes are also *shorter* (177 vs 200 frames), independently reproducing the
blog's observation that speed and quality move together because both follow from a
policy having one unambiguous strategy.

**Stage-2 fine-tuning converges fast.** Both seeds reach ~0.054-0.058 within 50-150
steps (well under one epoch on 53k frames) and then stop improving. Short fine-tunes
are sufficient.

## What does not hold up

**"Stage 2 overfits after step 100" was one seed's noise.** Seed 1000 bottomed at step
100 and appeared to rise steadily after; seed 2000 bottomed at 150 and 250 with no
trend. Within-run variation is +/-0.003, the same size as the differences being read as
signal. Checkpoints between steps 100 and 250 are indistinguishable on this metric.

**Cross-stage loss comparisons are invalid.** Stage 2's 0.0548 cannot be read against
stage 1's 0.0645: the eval sets differ, and stage 2's is drawn from curated (cleaner,
easier) episodes. There is no clean shared holdout after the fact, because the curated
sets were built by ranking all 500 episodes, so stage 1's eval episodes leak into stage
2's training data. Next time, carve a fixed holdout off the raw dataset *before* any
curation and exclude it everywhere.

## Candidates for rollout, in order

1. `/outputs/pi05-stage2-hq60/checkpoints/000100` and `.../000150` — plus
   `/outputs/pi05-stage2-hq60-seed2/checkpoints/000100`. Treat as a tie; rollouts break it.
2. `/outputs/pi05-candy-expertonly/checkpoints/002500` — the stage-1 model, no curation.
   Worth testing: it is the simplest pipeline, and stage 2's advantage is unproven on
   the metric that matters.
3. `/outputs/pi05-stage2-hq40/checkpoints/000600` — only if 1 and 2 disappoint.

## Serving: a correctness bug, found and fixed

`run.py` now loads whichever architecture the checkpoint's `config.json` names, so pi0
and pi0.5 both serve. The substantive fix was that **its inference loop was wrong for
relative actions**, silently.

`RelativeActionsProcessorStep` caches the state it last saw; the paired
`AbsoluteActionsProcessorStep` in the output pipeline adds
`relative_step.get_cached_state()` back. `_infer` currently calls `_pre` and `_post`
on *every* tick, while `select_action` only runs the model when its internal 50-step
queue drains:

    tick 1  _pre(obs@t1) caches state@t1 -> model runs -> chunk relative to t1
            _post(action[0]) adds state@t1                        correct
    tick 2  _pre(obs@t2) caches state@t2 -> queue pops action[1]
            _post(action[1]) adds state@t2                        WRONG, should be t1

Every action after the first in a chunk is referenced to the wrong pose. Nothing
raises; the arm just tracks a distorted trajectory, which would be read as the policy
being bad rather than the plumbing being wrong.

The fix, now in `_infer`, postprocesses a whole chunk at prediction time while the cache
still holds the right state, and serves absolute actions from a local queue;
`_replan_pending` reads that queue instead of `policy._action_queue`.
`to_absolute_actions` already accepts `(B, T, action_dim)` and broadcasts state across
time, so this is the supported shape, not a workaround.

Verified three ways, without hardware:

1. **Processor invariant.** A chunk of zero relative actions postprocesses to exactly
   the reference pose, identically on every row -- i.e. one reference state broadcast
   across all 50 steps.
2. **Real checkpoint end to end.** `pi05-stage2-hq60/000100` loads, predicts a `(50, 7)`
   chunk, all finite, first action adjacent to the reference pose as a relative model
   should produce.
3. **Bug quantified.** Postprocessing the same chunk after the preprocessor has seen a
   moved state -- exactly what the old loop did every tick -- shifts targets by
   `[3.0, 5.0, 6.0, 4.0, 7.0, 0.0]` against a state drift of `[3, 5, 6, 4, 7, 6]`. The
   error equals the drift on every relative dimension and is zero on the gripper, which
   `relative_exclude_joints` skips. Up to 7 units of pose error per tick, compounding as
   the arm moves.

4. **No regression on pi0.** Against `pi0-candy/010000` (absolute actions), the chunk
   path and the old `select_action` path return bit-identical actions, max abs
   difference 0.0 -- as expected, since unnormalize is elementwise and the absolute
   step is disabled there.

Still worth a stationary-arm smoke test before a live rollout: the loop was exercised
with synthetic (black) images, so timing and camera wiring are unproven.

## Operational notes

- **Nebius has no route to the AWS ranges behind `us.aws.cdn.hf.co` (15.236.0.0/15) or
  `cas-server.xethub.hf.co` (54.211.x) from public IPs in `195.242.28.0/22`**; from
  `89.169.x` it works. Same subnet, security group and route table — all verified
  permissive — so this is upstream, not configuration. 1 of 4 allocations drew a bad
  prefix. `pi05_base` is staged at `s3://candy-shop/base_models/pi05_base` so no run
  depends on the draw.
- **lerobot trains from random weights when a fetch fails**, printing a warning and
  exiting 0 (`modeling_pi05.py:811`, `modeling_pi0.py:846`).
  `shared.lerobot_patches.require_pretrained_weights` now refuses to start instead.
- **SARM scoring is video-decode bound**, ~8 episodes/min single-process with the GPU at
  4%. `stride=4` is safe because RABC consumes progress as a delta across a 50-frame
  chunk. Sharding by `episodes=` does not work: the scorer indexes by global frame index
  while a subset re-indexes positionally.
- **Checkpoints are ~11 GB each even under `train_expert_only`** (the whole model is
  serialized, not just the trained expert). `SAVE_FREQ=50` over 800 steps wrote 173 GB
  and left an upload backlog that blocked the next job on that cluster.
- `compute_rabc_weights`'s CLI cannot disable `--push-to-hub` (`store_true` with
  `default=True`) and uploads with no error handling; call `compute_sarm_progress`
  directly.
