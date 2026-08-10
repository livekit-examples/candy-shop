# teleoperator

Fly the leslider from a physical **SO-101 leader arm** and record the session to
a `LeRobotDataset`, with an imgui **review UI** for watching the cameras, driving
the session, and curating the corpus. The human-demonstration and HITL-correction
half of the data pipeline; `policy` and `move_to` are the autonomous halves.

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
recorder's terminal**, so watch it the first time a new arm connects.

## Hotkeys

Terminal (recorder must have focus) and the review window share the same
**rebindable** bindings, so a USB foot pedal works in either place.

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

Leader keys (its own pynput hook, work regardless of focus): ←/→ drive the
slider, ↑/↓ trim cruise speed, Space stops the slider.

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
| `DATASET_REPO_ID` | `binhpham/candy-shop` | corpus id (`org/name`) |
| `DATASET_ROOT` | `$HF_LEROBOT_HOME/<repo-id>` | where the corpus is written |
| `DATASET_TASK` | `pick up the candy` | initial task label |
| `PORTAL_FPS` | `30` | tick / record rate |
| `MAX_OBS_AGE_MS` | `100` | drop a row if the paired observation is older |
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
| [session.py](session.py) | setup enumeration (ports, corpora) for the window |
| [protocol.py](protocol.py) | the RPC contract shared by both processes |
| [shortcuts.py](shortcuts.py) | rebindable key bindings |
| [common.py](common.py) | env / token / pacer / config-path helpers |
| [ui/app.py](ui/app.py) | the imgui review window |
| [ui/client.py](ui/client.py) | the window's room viewer + RPC client |
| [ui/player.py](ui/player.py) | episode video decode for review |
| [ui/theme.py](ui/theme.py) | LiveKit design tokens + brand fonts |
