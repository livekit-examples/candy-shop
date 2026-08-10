# teleoperator

Fly the leslider from a physical **SO-101 leader arm** and record the session to
a `LeRobotDataset`, with an imgui **review UI** for watching the cameras, driving
the session, and curating the corpus. This is the human-demonstration and
HITL-correction half of the data pipeline; `policy` and `move_to` are the
autonomous halves.

The six leader joints mirror to the follower's arm `.pos`; the leader's **arrow
keys drive `slider.vel`** (raw ticks/s — the slider runs in velocity mode, see
[`portal.yaml`](../../portal.yaml)). One action per tick is streamed on the wire,
and each recorded row's state *and* action carry `slider.vel` alongside the six
arm `.pos`.

Recording is driven by the **executed action stream** (`action_subscription` in
the wire contract), so it captures this leader's actions *and* a remote policy's
— tagged by `Action.sender` — into one corpus.

## Two processes

```
teleoperator (this)         teleoperator-ui (child window)
  owns the leader arm  ◀──── RPC ────  holds no hardware, no dataset
  owns the dataset            reads a status snapshot, posts commands
  joins as a Portal operator  joins as a plain LiveKit participant (h264 viewer)
```

They are separate OS processes so nothing the window does — repainting, crashing
— can stall a recording. The recorder spawns the window as a child, so **one
command starts both**:

```bash
uv run teleoperator            # opens the review window (default)
uv run teleoperator --no-ui    # headless; records without a window
uv run teleoperator-ui         # reattach a window to a running recorder
```

The window is also where you pick the **serial port**, the **dataset**, and the
**task** — the recorder joins the room with nothing open and waits. A
fully-specified environment (see below) skips the setup screen and opens
immediately, for unattended runs.

If the leader arm's stored calibration disagrees with its file, lerobot runs its
calibration routine **in the recorder's terminal** (not the window), so keep an
eye on the terminal the first time a new arm connects.

## Hotkeys

Terminal (recorder process; this window must have focus) and the review window
share the same **rebindable** bindings, so a USB foot pedal works in either
place. Defaults:

| action | default key(s) | note |
|--------|----------------|------|
| record toggle | `r` / space / `'` | `'` is what a common USB foot pedal sends |
| discard episode | `[` / backspace | |
| cycle operator | `c` | human ↔ policy handoff; claims/releases control |
| set task | `t` (terminal only) | refused while recording |
| quit | `x` (terminal only) | |

Rebind in the window's Settings, or per-action via `TELEOPERATOR_KEYS_RECORD`,
`TELEOPERATOR_KEYS_DISCARD`, `TELEOPERATOR_KEYS_CLAIM` (comma-separated imgui key
names, e.g. `TELEOPERATOR_KEYS_RECORD=r,space,apostrophe`).

Leader keys (the leader's own pynput hook, so they work regardless of focus):
←/→ drive the slider, ↑/↓ trim cruise speed, Space stops the slider.

## Configuration

`LIVEKIT_*` creds come from the repo-root `.env`. Everything below is optional —
unset values are *chosen in the window* (or, for unattended runs, set a port plus
a dataset to open immediately):

| var | default | meaning |
|-----|---------|---------|
| `SO101_LEADER_PORT` | *(ask in window)* | serial port of the leader arm |
| `SO101_LEADER_ID` | `so101_leader` | calibration id |
| `TELEOP_CRUISE_VELOCITY` | `1500` | slider ticks/s while an arrow is held |
| `TELEOP_MAX_VELOCITY` | `3000` | ceiling for the ↑-arrow speed trim |
| `DATASET_REPO_ID` | `binhpham/candy-shop` | corpus id (`org/name`) |
| `DATASET_ROOT` | `$HF_LEROBOT_HOME/<repo-id>` | where the corpus is written |
| `DATASET_TASK` | `pick up the candy` | initial task label |
| `PORTAL_FPS` | `30` | tick / record rate |
| `MAX_OBS_AGE_MS` | `100` | drop a row if the paired observation is older |
| `LIVEKIT_ROOM` | `candy-shop` | room to join |
| `TELEOPERATOR_IDENTITY` | `teleoperator` | this peer's identity |

UI-side (the window process): `TELEOPERATOR_UI_IDENTITY`,
`TELEOPERATOR_UI_TARGET` (pin a specific recorder when two share a room),
`TELEOPERATOR_UI_POLL_HZ` (default 4). Window geometry and key bindings live
per-user under `candy-shop-teleoperator/` in your OS config dir, never in the repo.

The default root (`$HF_LEROBOT_HOME/<repo-id>`) is where a bare
`--dataset.repo_id` resolves at training time, so what you record trains with no
extra flags.

## What the window gives you

- **Live cameras** — the robot's h264 tracks, subscribed as a plain participant.
- **Session control** — record / stop / discard / set-task / operator handoff,
  plus Portal metrics (rtt, sync, per-track jitter/evictions).
- **Episode review** — scrub any saved episode's footage (decoded straight from
  the shared mp4s, which read fine even while being appended to).
- **Curation** — relabel a task across episodes, or delete episodes. These
  rewrite the whole corpus (v3.0 packs many episodes per file), so they run as
  background **jobs**: the recorder suspends the dataset, rewrites, and resumes;
  the window watches `busy`/`error`/`revision`.

## Design notes

- **One writer thread owns every dataset mutation.** The Portal callback pairs an
  action with an observation (a sub-millisecond ring scan) and enqueues; the
  writer thread does the decode / `add_frame` / `save_episode`. Blocking the
  callback thread starves Portal's video-receive worker of the GIL and makes
  observation delivery bursty — see [`recorder.py`](recorder.py).
- **Crash-safe.** lerobot only writes parquet footers at `finalize()`. A hard
  kill leaves footerless files that pyarrow refuses to open;
  [`dataset_repair.py`](dataset_repair.py) rebuilds the footers from the intact
  pages at resume time, so at most the un-saved in-flight episode is lost.
- Dropped rows are reported (terminal) and shown (window) **by cause** (`stale` /
  `unpaired` / `error` / `backlog`), each with a distinct fix.
- The UI joins as a **plain LiveKit participant, not a Portal peer** — Portal has
  only robot/operator roles, and a viewer in the operator list could have control
  cycled onto it. It discovers the recorder by a participant attribute
  (`operators.teleoperator.role`), so renaming the recorder doesn't break it.
- The two processes must not share an interpreter: Portal's FFI and the `livekit`
  rtc SDK each statically link libwebrtc, so loading both registers every `RTC*`
  class twice. The window reads the wire contract as YAML rather than importing
  `livekit.portal` for exactly this reason (`common.contract_camera_names`).

## Files

| file | role |
|------|------|
| [run.py](run.py) | recorder process: leader tick, RPC surface, spawns the window |
| [recorder.py](recorder.py) | the HITL recorder (writer thread, action↔obs pairing) |
| [library.py](library.py) | episode index + relabel/delete corpus rewrites |
| [dataset_repair.py](dataset_repair.py) | crash-safe parquet footer rebuild |
| [session.py](session.py) | setup enumeration (ports, corpora) for the window |
| [protocol.py](protocol.py) | the RPC contract shared by both processes |
| [shortcuts.py](shortcuts.py) | rebindable key bindings |
| [common.py](common.py) | env / token / pacer / config-path helpers |
| [ui/app.py](ui/app.py) | the imgui review window |
| [ui/client.py](ui/client.py) | the window's room viewer + RPC client |
| [ui/player.py](ui/player.py) | episode video decode for review |
| [ui/theme.py](ui/theme.py) | LiveKit design tokens + brand fonts |
