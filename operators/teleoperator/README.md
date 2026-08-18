# teleoperator

Fly the leslider from a physical **SO-101 leader arm** and record the session to
a `LeRobotDataset`, with an imgui **review UI** for watching the cameras, driving
the session, and curating the corpus. The human-demonstration and HITL-correction
half of the data pipeline; `policy` and `move_to` are the autonomous halves.

It is also the room's **control desk**: it finds the other operators, runs them over
their own RPCs, preempts them to take the arm, and hands the arm back — and `mimic`
drives the leader from the follower's pose while one of them is driving, so taking over
mid-pick is a handover instead of a jump. Recording is unaffected by any of it: rows
come from the *executed* action stream, so an episode spans a policy run, your
intervention, and the policy resuming, in one continuous trajectory.

The six leader joints mirror to the follower's arm `.pos`; the leader's arrow keys
drive `slider.vel` (raw ticks/s — velocity mode, see [`portal.yaml`](../../portal.yaml)).
Recording is driven by the executed action stream (`action_subscription`), so it
captures this leader's actions *and* a remote policy's — tagged by `Action.sender`
— into one corpus.

## Run

The recorder and UI are separate OS processes so the window can't stall a
recording. The recorder spawns the window as a child, so one command starts both:

```bash
uv run teleoperator            # opens the review window (default)
uv run teleoperator --no-ui    # headless; records without a window
uv run teleoperator-ui         # reattach a window to a running recorder
```

Pick the serial port, dataset, and task in the window; a fully-specified
environment (below) skips the setup screen for unattended runs. If the leader
arm's stored calibration disagrees with its file, lerobot calibrates **in the
recorder's terminal**, so watch it the first time a new arm connects — the
terminal hotkeys below are suspended until it finishes, since calibration needs
stdin for its own ENTER prompts.

## Hotkeys

Terminal (recorder must have focus) and the review window share the same
**rebindable** bindings, so a USB foot pedal works in either place.

| action | default key(s) | note |
|--------|----------------|------|
| record toggle | `r` / space / `'` | `'` is what a common USB foot pedal sends |
| discard episode | `[` / backspace | |
| take / release arm | `c` | one key both ways; taking it stops every operator first |
| mimic toggle | `m` | only the leader moves, so it keeps a letter |
| free the leader | `f` | torque off, mimic off — hold the arm first, it drops |
| reconnect the leader | *(unbound)* | re-opens the serial bus; a dropped link already retries itself |
| resume | *(unbound)* | restarts what taking the arm interrupted — it moves the rig |
| stop everything | *(unbound)* | preempts every operator, then folds the arm |
| set task | `t` (terminal only) | refused while recording |
| quit | `x` (terminal only) | |

Rebind in the window's Settings, or per-action via `TELEOPERATOR_KEYS_<ACTION>`
(`RECORD`, `DISCARD`, `CLAIM`, `RELEASE`, `RESUME`, `MIMIC`, `RELAX`, `RECONNECT`,
`STOP_ALL`) —
comma-separated imgui key names, e.g. `TELEOPERATOR_KEYS_RECORD=r,space,apostrophe`.
`resume` and `stop_all` ship unbound on purpose: a stray keystroke that restarts a
policy or folds the arm is worse than a mouse trip to the window.

Leader keys (its own pynput hook, work regardless of focus): ←/→ drive the
slider, ↑/↓ trim cruise speed, Space stops the slider.

## Driving the other operators

The window's **operators** rail is one card per operator: everything
[`shared/operators.py`](../../shared/operators.py) declares, plus any live peer it
doesn't (presence only — an undeclared peer advertises no RPCs to drive). Each card
carries that operator's one argument, its presets, and its two RPCs; a preset is a
command, so clicking one sends it. Discovery is presence + Portal's operator list, so
an operator that starts late simply appears.

Two rules the rail encodes, both load-bearing:

* **A stop travels orchestrators-first** (`reward` → `policy` → `move-to`). Preempting
  the policy while the reward operator still holds its retry loop just starts attempt
  two. The rail also refuses to drive `policy` directly while `reward` is driving it.
* **Taking the arm outlives the RPCs it preempted.** Every operator clears the robot's
  active-operator pointer in a `finally`, so an unwind landing after the claim would
  drop the arm; the teleoperator therefore re-asserts the pointer while it holds it.

**Take arm** stops everything, remembers what *it* had running, and points the robot
here. **Resume** re-issues exactly those runs with their original payloads and stops
asserting the claim; **Release** hands the pointer back without restarting anything.
Runs the *voice agent* started are point-to-point RPCs this seat never sees, so they
can be stopped but not resumed — the active operator is the only evidence they exist.

## Mimic, and intervening mid-pick

`Mimic` drives the leader's six joints from the follower's observed pose (torque on,
`Goal_Position` each tick) for as long as somebody else has the arm. The leader tracks
the pick, so the pose you would hand the robot on taking over is already the pose it is
in — which is what makes a mid-policy takeover safe. Without it, claiming slews the arm
to wherever the leader happens to be lying.

**Push the leader to take the arm.** With torque on, forcing a joint opens a gap between
where the leader is and where it was told to be; a gap past `TELEOP_MIMIC_INTERVENE_DEG`
held for `TELEOP_MIMIC_HOLD_S` is read as a hand, and the teleoperator stops everything
and claims. That path frees the leader outright, because the push *is* the proof a hand
is on it.

Not every gap is a hand, and the two are separated on purpose. A servo chasing a moving
goal runs behind it, so the gap is measured against the goal from `TELEOP_MIMIC_LAG_S`
ago — the one the leader has had time to reach — and a joint travelling *towards* its
goal is not counted at all, however far behind it is. Without that, a policy motion
faster than the leader can follow reads as a push and takes the arm mid-pick. If a hand
resting on the trigger still trips it as the gripper closes, `TELEOP_MIMIC_INTERVENE_GRIPPER=0`
takes the gripper out of push detection and leaves the five arm joints doing it.

**Free the leader** (`f`, or the window's button) drops torque and switches mimic off in
one move. The toggle does that too, but not in the case that needs it: an engage that
failed halfway leaves the arm stiff with mimic already reporting `error`, and there the
toggle has nothing to switch off. Hold the leader before pressing it.

Every other way of taking the arm **holds** the leader instead: torque stays on and the
goal stops following the robot, which parks the arm at that pose. That is deliberate —
an SO-101 leader nobody is holding falls under gravity, and a claimed teleoperator sends
the follower down with it. Hold the leader, then switch mimic off to fly.

`TELEOP_MIMIC_TORQUE_LIMIT` (per mille) is how hard the leader holds itself: the default
500 is enough to carry its own weight and still yield to a firm hand, so a push registers
without a fight. It also caps how fast the leader can follow — raise it if tracking a
brisk policy visibly falls behind, lower it if the arm fights your hand.

## When the leader drops out

The leader is one USB cable, and a bus that stops answering used to end the process and
the in-flight episode with it. Now it is a reconnect: the handle is dropped, mimic
forgets everything it believed about it (no writes to a port that isn't there), and a
worker re-opens the port — 1s, 2s, 4s, 8s, then every 15s for as long as it takes.
Everything else stays up, including the open corpus and any episode you were recording.

The window shows this as a banner plus a `LINK` row in arm control, and **Reconnect**
jumps the backoff once the cable is back in. Mimic comes back **off**: re-arming it
torques the leader, and that is your call once the arm is in your hand again.

The same path covers a leader that is up but misbehaving — Reconnect cycles the bus —
and it is what recovers a failed torque write, which is a dead link by another name.

## Configuration

`LIVEKIT_*` creds come from the repo-root `.env`. Everything below is optional —
unset values are chosen in the window (or, for unattended runs, set a port plus a
dataset to open immediately):

| var | default | meaning |
|-----|---------|---------|
| `SO101_LEADER_PORT` | *(ask in window)* | serial port of the leader arm |
| `SO101_LEADER_ID` | `so101_leader` | calibration id |
| `TELEOP_CRUISE_VELOCITY` | `1500` | slider ticks/s while an arrow is held |
| `TELEOP_MAX_VELOCITY` | `3000` | ceiling for the ↑-arrow speed trim |
| `TELEOP_MIMIC_ALIGN_S` | `1.5` | worst-case ease onto the arm's pose when torque comes on |
| `TELEOP_MIMIC_INTERVENE_DEG` | `10` | leader-vs-goal gap that counts as a push; `0` disables it |
| `TELEOP_MIMIC_INTERVENE_GRIPPER` | `20` | the same threshold on the gripper's 0-100 travel |
| `TELEOP_MIMIC_HOLD_S` | `0.2` | how long that gap must hold before the arm changes hands |
| `TELEOP_MIMIC_LAG_S` | `0.25` | how stale a goal a push is judged against — the leader's own follow lag |
| `TELEOP_MIMIC_SMOOTH_S` | `0.08` | low-pass on the leader's goal; raise if tracking looks jittery |
| `TELEOP_MIMIC_TORQUE_LIMIT` | `500` | leader holding torque, per mille; `0` leaves the motor's own |
| `DATASET_REPO_ID` | `binhpham/candy-shop` | corpus id (`org/name`) |
| `DATASET_ROOT` | `$HF_LEROBOT_HOME/<repo-id>` | where the corpus is written |
| `DATASET_TASK` | `pick up the candy` | initial task label |
| `PORTAL_FPS` | `30` | tick / record rate |
| `LIVEKIT_ROOM` | `candy-shop` | room to join |
| `TELEOPERATOR_IDENTITY` | `teleoperator` | this peer's identity |

UI-side: `TELEOPERATOR_UI_IDENTITY`, `TELEOPERATOR_UI_TARGET` (pin a specific
recorder when two share a room), `TELEOPERATOR_UI_POLL_HZ` (default 4). Window
geometry and key bindings live per-user under `candy-shop-teleoperator/` in your
OS config dir. The default root is where a bare `--dataset.repo_id` resolves at
training time, so what you record trains with no extra flags.

## Files

| file | role |
|------|------|
| [run.py](run.py) | recorder process: leader tick, RPC surface, spawns the window |
| [recorder.py](recorder.py) | the HITL recorder (writer thread, action↔obs pairing) |
| [library.py](library.py) | episode index + relabel/delete corpus rewrites |
| [dataset_repair.py](dataset_repair.py) | crash-safe parquet footer rebuild |
| [peers.py](peers.py) | the other operators: discovery, run/stop, claim + resume |
| [mimic.py](mimic.py) | leader-follows-follower, and the push that takes the arm |
| [session.py](session.py) | setup enumeration (ports, corpora) for the window |
| [protocol.py](protocol.py) | the RPC contract shared by both processes |
| [shortcuts.py](shortcuts.py) | rebindable key bindings |
| [common.py](common.py) | env / token / pacer / config-path helpers |
| [ui/app.py](ui/app.py) | the imgui review window |
| [ui/client.py](ui/client.py) | the window's room viewer + RPC client |
| [ui/player.py](ui/player.py) | episode video decode for review |
| [ui/theme.py](ui/theme.py) | LiveKit design tokens + brand fonts |
