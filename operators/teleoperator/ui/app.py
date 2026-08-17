"""The review UI: watch the cameras, run the session, curate the corpus.

A separate process from the teleoperator; it reads a status snapshot over RPC and
posts commands back, so closing or crashing it leaves recording untouched.
"""
from __future__ import annotations

import logging
import math
import os
import pathlib
import time
from typing import Any, Optional

from imgui_bundle import ImVec2, ImVec4, hello_imgui, imgui, immvision

from shared import operators as roster

from operators.teleoperator import protocol, shortcuts
from operators.teleoperator.common import (
    contract_camera_names, env, load_env, mint_token, portal_config_path,
)
from operators.teleoperator.ui import theme
from operators.teleoperator.ui.client import RecorderClient
from operators.teleoperator.ui.player import EpisodePlayer
from operators.teleoperator.ui.theme import font

logger = logging.getLogger(__name__)

PACKAGE_DIR = pathlib.Path(__file__).resolve().parents[1]
SIDEBAR_WIDTH = 340.0
LABEL_WIDTH = 104.0
METRIC_LABEL_WIDTH = 152.0
HEADER_WIDTH = 660.0
# The bottom band is the control plane: who drives the arm, and the operators it can be
# handed to. Full width — the two belong side by side, because taking the arm and
# starting an operator are the same decision from opposite ends.
BAND_HEIGHT_FRACTION = 0.36
OPERATOR_CARD_HEIGHT = 232.0
OPERATOR_CARD_MIN_WIDTH = 240.0
ARM_PANEL_WIDTH = 330.0


class AppState:
    """Everything that has to survive between frames — ImGui redraws from scratch."""

    def __init__(self, client: RecorderClient, cameras: tuple[str, ...]) -> None:
        self.client = client
        self.cameras = cameras

        self.task_draft: Optional[str] = None       # None = mirror the teleoperator
        self.label_drafts: dict[int, str] = {}      # episode index -> edited task
        self.selected: set[int] = set()             # ticked for deletion
        self.confirm_delete = False
        self.drawn_seq: dict[str, int] = {}         # last frame drawn per camera
        self.last_revision = -1
        self.tab = "Teleop"
        # Setup screen state, until the teleoperator reports `configured`.
        self.setup_asked = False
        self.setup_port = ""
        self.setup_new = False            # False = continue a listed corpus
        self.setup_pick = 0
        self.setup_repo_id = ""
        self.setup_local = False          # new corpus under ./data instead of lerobot's home
        self.setup_task = ""
        self.player = EpisodePlayer()
        self.viewing: Optional[int] = None    # episode the player should show
        self.last_frame_at = 0.0              # wall clock, for playback pacing
        self.keys = shortcuts.load()
        # identity -> what's typed in that operator's card, before it is sent.
        self.peer_drafts: dict[str, str] = {}
        self.binding: Optional[str] = None           # action awaiting a keypress

    def sync(self, status: dict[str, Any]) -> None:
        """Drop index-keyed intent when the corpus changes: a delete reassigns episode
        indices, so episode 4 is no longer the one you were editing or viewing."""
        revision = int(status.get("revision", -1))
        if revision != self.last_revision:
            self.last_revision = revision
            self.label_drafts.clear()
            self.selected.clear()
            self.confirm_delete = False
            self.viewing = None
            self.player.close()


# --- small widgets ----------------------------------------------------------

def text(color, value: str) -> None:
    """Coloured text, never format-interpreted: ImGui's `text_colored` takes a printf
    format, so a task label containing `%` would render garbage."""
    imgui.push_style_color(imgui.Col_.text, color)
    imgui.text_unformatted(value)
    imgui.pop_style_color()


def heading(label: str, action: str = "", *, disabled: bool = False) -> bool:
    """Section label, optionally with a right-aligned button. Returns whether it was clicked."""
    width = imgui.get_content_region_avail().x
    with font(theme.FONTS.small):
        text(theme.FG4, label.upper())
    clicked = False
    if action:
        # Measured, not guessed: these labels carry a key hint that changes with the
        # bindings, and a fixed offset clipped the long ones.
        action_width = imgui.calc_text_size(action).x + imgui.get_style().frame_padding.x * 2
        imgui.same_line(max(width - action_width, 0.0))
        imgui.begin_disabled(disabled)
        clicked = imgui.small_button(action)
        imgui.end_disabled()
    imgui.spacing()
    return clicked


def field(label: str, value: str, color=theme.FG1, *, width: float = LABEL_WIDTH) -> None:
    """A label/value pair on a fixed grid, so a column of these lines up regardless of label length."""
    with font(theme.FONTS.small):
        text(theme.FG4, label.upper())
    imgui.same_line(width)
    with font(theme.FONTS.mono_small):
        text(color, value)


def meta_cell(label: str, value: str, color=theme.FG1) -> None:
    """One label/value pair inside a header table cell."""
    with font(theme.FONTS.small):
        text(theme.FG4, label.upper())
    imgui.same_line()
    with font(theme.FONTS.mono_small):
        text(color, value)


def _first_key(state: AppState, action: str) -> str:
    """The label shown on a button: the first bound key, or nothing if unbound."""
    keys = state.keys.get(action) or ()
    return shortcuts.describe({action: keys[:1]}, action) if keys else ""


def button_with_key(label: str, key: str, size: ImVec2, kind: str = "plain") -> bool:
    """A button with a dimmed shortcut hint drawn over it (ImGui has no rich text)."""
    pos = imgui.get_cursor_screen_pos()
    width = size.x if size.x > 0 else imgui.get_content_region_avail().x
    if width < 0:
        width = imgui.get_content_region_avail().x + width
    clicked = {"accent": accent_button, "danger": danger_button}.get(
        kind, lambda l, s: imgui.button(l, s))(label, size)
    if key:
        height = imgui.get_item_rect_size().y
        with font(theme.FONTS.mono_small):
            text_w = imgui.calc_text_size(key).x
            imgui.get_window_draw_list().add_text(
                ImVec2(pos.x + width - text_w - 12,
                       pos.y + (height - imgui.get_font_size()) * 0.5),
                imgui.get_color_u32(theme.FG4), key)
    return clicked


def accent_button(label: str, size: ImVec2 = ImVec2(0, 0)) -> bool:
    """The one loud button on screen."""
    imgui.push_style_color(imgui.Col_.button, theme.ACCENT2)
    imgui.push_style_color(imgui.Col_.button_hovered, theme.ACCENT1)
    imgui.push_style_color(imgui.Col_.button_active, theme.ACCENT1)
    imgui.push_style_color(imgui.Col_.text, theme.BG0)
    clicked = imgui.button(label, size)
    imgui.pop_style_color(4)
    return clicked


def danger_button(label: str, size: ImVec2 = ImVec2(0, 0)) -> bool:
    imgui.push_style_color(imgui.Col_.button, theme.BG_SERIOUS)
    imgui.push_style_color(imgui.Col_.button_hovered, theme.SEPARATOR_SERIOUS)
    imgui.push_style_color(imgui.Col_.button_active, theme.SEPARATOR_SERIOUS)
    imgui.push_style_color(imgui.Col_.text, theme.SERIOUS)
    clicked = imgui.button(label, size)
    imgui.pop_style_color(4)
    return clicked


BADGE_PAD = ImVec2(7.0, 2.0)


def badge(label: str, fg, bg, *, dot: bool = False) -> None:
    """A pill, drawn straight onto the draw list — ImGui has no chip widget. `dot`
    breathes, the web app's cue for something live rather than merely true."""
    with font(theme.FONTS.small):
        size = imgui.calc_text_size(label)
        height = size.y + BADGE_PAD.y * 2
        width = size.x + BADGE_PAD.x * 2 + (11.0 if dot else 0.0)
        pos = imgui.get_cursor_screen_pos()
        draw = imgui.get_window_draw_list()
        draw.add_rect_filled(pos, ImVec2(pos.x + width, pos.y + height),
                             imgui.get_color_u32(bg), height / 2)
        x = pos.x + BADGE_PAD.x
        if dot:
            # 0.45..1.0 at ~0.7 Hz: visible as motion, never a strobe.
            alpha = 0.725 + 0.275 * math.sin(imgui.get_time() * 4.4)
            draw.add_circle_filled(ImVec2(x + 3.0, pos.y + height / 2), 3.0,
                                   imgui.get_color_u32(ImVec4(fg.x, fg.y, fg.z, alpha)))
            x += 11.0
        draw.add_text(ImVec2(x, pos.y + BADGE_PAD.y), imgui.get_color_u32(fg), label)
        imgui.dummy(ImVec2(width, height))


def chip_width(label: str) -> float:
    """What `chip` will occupy, for laying a row of them out before drawing."""
    style = imgui.get_style()
    with font(theme.FONTS.small):
        return imgui.calc_text_size(label).x + style.frame_padding.x + style.item_spacing.x


def chip(label: str) -> bool:
    """A preset: quiet, one step above the card it sits on."""
    imgui.push_style_color(imgui.Col_.button, theme.BG3)
    imgui.push_style_color(imgui.Col_.button_hovered, theme.BG_ACCENT2)
    imgui.push_style_color(imgui.Col_.text, theme.FG2)
    with font(theme.FONTS.small):
        clicked = imgui.small_button(label)
    imgui.pop_style_color(3)
    return clicked


def toggle_button(label: str, on: bool, size: ImVec2 = ImVec2(0, 0)) -> bool:
    """A button that shows its own state: accent-on-dark while engaged."""
    imgui.push_style_color(imgui.Col_.button, theme.BG_ACCENT2 if on else theme.BG3)
    imgui.push_style_color(imgui.Col_.button_hovered,
                           theme.BG_ACCENT2 if on else theme.BG3_HOVER)
    imgui.push_style_color(imgui.Col_.text, theme.ACCENT1 if on else theme.FG2)
    clicked = imgui.button(label, size)
    imgui.pop_style_color(3)
    return clicked


# --- regions ----------------------------------------------------------------

def draw_header(state: AppState) -> None:
    client, status = state.client, state.client.status
    full_width = imgui.get_content_region_avail().x

    connection = client.connection
    if connection == "connected" and status:
        link, link_colour = (client.target or "teleoperator"), theme.SUCCESS
    elif connection == "connected":
        link, link_colour = "handshaking", theme.MODERATE
    elif connection == "no teleoperator":
        link, link_colour = "no teleoperator in room", theme.SERIOUS
    else:
        link, link_colour = connection, theme.MODERATE

    obs_fps = status.get("obs_fps")
    rate = f"{status.get('fps', '?')} fps"
    if obs_fps is not None:
        # Measured rate beside nominal: a gap is the first symptom of most recording problems.
        rate += f" / obs {obs_fps}"

    columns = (
        ("teleoperator", link, link_colour),
        ("rate", rate if status else "--", theme.FG1 if status else theme.FG4),
        ("driving", _driving_label(status), _driving_colour(status)),
    )
    # Stretch over a bounded width: `sizing_fixed_fit` collapses these cells (drawn
    # with same_line, so the table can't measure them) and full-window stretch parks
    # the second cell mid-screen.
    if imgui.begin_table("##meta", len(columns), imgui.TableFlags_.sizing_stretch_same,
                         ImVec2(HEADER_WIDTH, 0.0)):
        for label, value, colour in columns:
            imgui.table_next_column()
            meta_cell(label, value, colour)
        imgui.end_table()

    if status:
        with font(theme.FONTS.mono_small):
            text(theme.FG4, f"{status.get('repo_id', '?')}   {status.get('root', '?')}")
        # What is worth noticing mid-flight, so the Metrics tab is somewhere you go to
        # read *why* rather than to notice at all.
        imgui.same_line()
        _draw_link_chips(state)

    # Measured from the header's own width, not what's left of the current row: the link
    # chips leave the cursor mid-line, and `content_region_avail` there is the remainder.
    # new_line() below closes the row explicitly — a cursor left mid-line put the tab bar
    # off the right edge where it drew nothing.
    imgui.same_line(max(full_width - 96.0, 0.0))
    if imgui.button("settings"):
        imgui.open_popup("Settings")
    draw_settings(state)
    imgui.new_line()


def _driving_label(status: dict[str, Any]) -> str:
    if not status:
        return "--"
    active = status.get("active_operator")
    if active is None:
        return "nobody"
    return "this leader" if active == status.get("identity") else active


def _driving_colour(status: dict[str, Any]):
    if not status or status.get("active_operator") is None:
        return theme.FG4
    return (theme.ACCENT1 if status.get("active_operator") == status.get("identity")
            else theme.FG1)


def _chip_bg(colour):
    """The ground a status colour sits on: its own dark field for the two that mean
    something, the plain card surface for everything else."""
    if colour is theme.SERIOUS:
        return theme.BG_SERIOUS
    if colour is theme.ACCENT1:
        return theme.BG_ACCENT
    return theme.BG2


def _draw_link_chips(state: AppState) -> None:
    """The link's state as chips: round trip, what sync is waiting on, rows lost."""
    metrics = state.client.metrics
    status = state.client.status
    rtt = (metrics.get("rtt_ms") or {}).get("last")
    blocker = str((metrics.get("sync") or {}).get("blocker") or "")
    dropped = int(status.get("dropped", 0) or 0)

    if rtt is not None:
        colour = _rtt_colour(rtt)
        badge(f"rtt {rtt:g} ms", colour, _chip_bg(colour))
        imgui.same_line()
    if blocker:
        badge(f"sync waiting on {blocker}", theme.MODERATE, _chip_bg(theme.MODERATE))
        imgui.same_line()
    if dropped:
        badge(f"{dropped} dropped", theme.SERIOUS, _chip_bg(theme.SERIOUS))
        imgui.same_line()


def draw_banner(state: AppState) -> None:
    """Whatever the operator most needs to know right now, or nothing."""
    client, status = state.client, state.client.status
    busy = str(status.get("busy") or "")
    error = str(status.get("error") or "")
    notice = client.notice

    if busy:
        _banner(theme.BG_ACCENT2, theme.ACCENT1, f"{busy} — the corpus is being rewritten")
    elif error:
        _banner(theme.BG_SERIOUS, theme.SERIOUS, error)
    elif notice:
        _banner(theme.BG_SERIOUS, theme.SERIOUS, notice)
        imgui.same_line()
        if imgui.small_button("dismiss"):
            client.clear_notice()


def _banner(bg, fg, message: str) -> None:
    imgui.push_style_color(imgui.Col_.child_bg, bg)
    imgui.begin_child("##banner", ImVec2(0, imgui.get_frame_height() + 10),
                      imgui.ChildFlags_.borders)
    text(fg, message)
    imgui.end_child()
    imgui.pop_style_color()
    imgui.spacing()


def draw_cameras(state: AppState, size: ImVec2) -> None:
    imgui.begin_child("##cameras", size, imgui.ChildFlags_.borders)
    heading("cameras")

    cameras = state.cameras
    if not cameras:
        text(theme.FG4, "the wire contract declares no video tracks")
        imgui.end_child()
        return

    # Two-up at typical 640x480 fills the pane without thumbnailing.
    columns = 1 if len(cameras) == 1 else 2
    rows = (len(cameras) + columns - 1) // columns
    spacing = imgui.get_style().item_spacing
    avail = imgui.get_content_region_avail()
    label_height = imgui.get_text_line_height_with_spacing()
    cell_w = (avail.x - spacing.x * (columns - 1)) / columns
    cell_h = (avail.y - spacing.y * (rows - 1)) / rows - label_height

    for i, camera in enumerate(cameras):
        if i % columns:
            imgui.same_line()
        imgui.begin_group()
        with font(theme.FONTS.mono_small):
            text(theme.FG3, camera)
        latest = state.client.frame(camera)
        # Letterbox to avoid distortion; a frameless camera is sized 4:3 so the panel
        # doesn't jump when the track connects.
        src_h, src_w = latest[1].shape[:2] if latest is not None else (3, 4)
        width = min(cell_w, cell_h * src_w / src_h)
        height = width * src_h / src_w
        if latest is None:
            imgui.push_style_color(imgui.Col_.child_bg, theme.BG0)
            imgui.begin_child(f"##nofeed_{camera}", ImVec2(width, height),
                              imgui.ChildFlags_.borders)
            text(theme.FG4, "waiting for track ...")
            imgui.end_child()
            imgui.pop_style_color()
        else:
            seq, image = latest
            # Refresh only on a new frame; don't re-upload unchanged pixels.
            changed = state.drawn_seq.get(camera) != seq
            state.drawn_seq[camera] = seq
            immvision.image_display(f"##feed_{camera}", image, (int(width), 0), changed)
        imgui.end_group()

    imgui.end_child()


def draw_session(state: AppState, size: ImVec2) -> None:
    client, status = state.client, state.client.status
    imgui.begin_child("##session", size, imgui.ChildFlags_.borders)

    busy = bool(status.get("busy"))
    ready = bool(status.get("ready"))
    recording = bool(status.get("recording"))
    saving = bool(status.get("saving"))

    # --- task ---
    if state.task_draft is None:
        state.task_draft = str(status.get("task", ""))
    dirty = state.task_draft != str(status.get("task", ""))
    # `set` rides the section label rather than taking a row of its own: Enter in the
    # field does the same thing, so a full-width button was mostly padding.
    apply_task = heading("task for new episodes", "set",
                         disabled=not dirty or recording or busy)
    imgui.set_next_item_width(-1)
    imgui.begin_disabled(recording or busy)
    submitted, state.task_draft = imgui.input_text(
        "##task", state.task_draft, imgui.InputTextFlags_.enter_returns_true
    )
    imgui.end_disabled()
    if (submitted or apply_task) and state.task_draft.strip():
        client.call(protocol.METHOD_SET_TASK, {"task": state.task_draft})
    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    # --- record ---
    heading("recording")
    record_key = _first_key(state, "record")
    discard_key = _first_key(state, "discard")
    button_size = ImVec2(-1, imgui.get_frame_height() * 1.4)
    imgui.begin_disabled(not ready or busy or saving)
    if recording:
        if button_with_key("STOP  &  SAVE", record_key, button_size, "danger"):
            client.call(protocol.METHOD_STOP)
    else:
        if button_with_key("START  RECORDING", record_key, button_size, "accent"):
            client.call(protocol.METHOD_START)
    imgui.end_disabled()
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
        imgui.set_tooltip(shortcuts.describe(state.keys, "record"))

    imgui.begin_disabled(not recording)
    if button_with_key("Discard episode", discard_key, ImVec2(-1, 0)):
        client.call(protocol.METHOD_DISCARD)
    imgui.end_disabled()
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
        imgui.set_tooltip(f'{shortcuts.describe(state.keys, "discard")} — throws away '
                          f"the in-flight take, no confirmation")

    imgui.spacing()
    if not ready:
        field("state", "waiting for first observation", theme.MODERATE)
    elif recording:
        rows, fps = int(status.get("rows", 0)), int(status.get("fps", 30)) or 30
        field("rows", f"{rows}", theme.ACCENT1)
        field("length", f"{rows / fps:.1f} s", theme.ACCENT1)
    elif saving:
        field("state", "encoding last episode", theme.MODERATE)
    else:
        field("state", "idle", theme.FG4)
    if (depth := status.get("queue_depth")):
        field("queue", str(depth), theme.MODERATE)

    dropped = int(status.get("dropped", 0))
    if dropped:
        causes = status.get("drop_causes") or {}
        worst = max(causes, key=lambda k: causes[k], default="")
        why = {
            "error": "add_frame failing — schema/disk, not timing",
            "unpaired": "nothing to pair with — slow video or clock skew",
            "backlog": "writer can't keep up — disk or encoder bound",
        }.get(worst, "")
        # Wrapped: these run longer than the sidebar and ImGui doesn't wrap by default.
        imgui.push_text_wrap_pos(0.0)
        with font(theme.FONTS.small):
            text(theme.SERIOUS if not int(status.get("rows", 0)) else theme.MODERATE,
                 f"{dropped} dropped — {why}")
            if any(causes.values()):
                text(theme.FG4, "  ".join(f"{k}={v}" for k, v in causes.items() if v))
        imgui.pop_text_wrap_pos()
    if (resized := int(status.get("resized", 0))):
        imgui.push_text_wrap_pos(0.0)
        with font(theme.FONTS.small):
            text(theme.MODERATE, f"{resized} frames rescaled — encoder shedding "
                                 f"resolution; rows kept, detail upscaled")
        imgui.pop_text_wrap_pos()
    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    imgui.end_child()


def draw_arm(state: AppState, size: ImVec2) -> None:
    """Who the robot is obeying, and the moves that change it: take it, give it back,
    restart what taking it interrupted, and mirror it onto the leader.

    Its own panel under the session, at a fixed height: these are the controls you reach
    for while the rig is moving, so they must not be what scrolls out of view.
    """
    client, status = state.client, state.client.status
    imgui.begin_child("##arm", size, imgui.ChildFlags_.borders)
    active = status.get("active_operator")
    me = status.get("identity")
    claiming = bool(status.get("claiming"))
    holds = claiming or (active is not None and active == me)
    suspended = list(status.get("suspended") or [])
    mimic = status.get("mimic") or {}

    heading("arm control")
    if not status:
        text(theme.FG4, "waiting for the teleoperator ...")
        imgui.end_child()
        return
    if active is None:
        field("driving", "nobody", theme.FG4)
    else:
        field("driving", "this leader" if active == me else active,
              theme.ACCENT1 if active == me else theme.FG1)
    if claiming and active != me:
        # The gap between claiming and holding: a preempted operator clears the pointer
        # on its way out, and this teleoperator writes it straight back.
        field("pointer", "re-claiming ...", theme.MODERATE)
    imgui.spacing()

    style = imgui.get_style()
    half = ImVec2((imgui.get_content_region_avail().x - style.item_spacing.x) / 2, 0)
    imgui.begin_disabled(holds)
    if button_with_key("Take arm", _first_key(state, "claim"), half, "accent"):
        client.call(protocol.METHOD_CLAIM)
    imgui.end_disabled()
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
        imgui.set_tooltip("Stop every operator, then point the robot at this leader.\n"
                          "With mimic on, the leader is held at the arm's pose — hold it\n"
                          "and switch mimic off to fly.")
    imgui.same_line()
    imgui.begin_disabled(active is None and not claiming)
    if button_with_key("Release", _first_key(state, "release"), half):
        client.call(protocol.METHOD_RELEASE)
    imgui.end_disabled()
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
        imgui.set_tooltip("Clear the pointer — the arm ignores everyone until someone "
                          "claims it.\nRestarts nothing; that's Resume.")

    imgui.begin_disabled(not suspended)
    label = (f"Resume {', '.join(roster.title_for(i) for i in suspended)}"
             if suspended else "Resume")
    if button_with_key(label, _first_key(state, "resume"), ImVec2(-1, 0)):
        client.call(protocol.METHOD_RESUME)
    imgui.end_disabled()
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
        imgui.set_tooltip("Hand the arm back: re-run whatever taking it interrupted,\n"
                          "with the same prompt. Only runs started here can be resumed —\n"
                          "the agent's own RPCs are invisible from this seat.")

    imgui.spacing()
    _draw_mimic(state, mimic)
    imgui.end_child()


# What each mimic state means, and how loudly to say it.
MIMIC_LABELS = {
    "off": ("off", theme.FG4),
    "waiting": ("waiting", theme.MODERATE),
    "aligning": ("aligning", theme.ACCENT1),
    "tracking": ("tracking", theme.ACCENT1),
    "holding": ("holding", theme.MODERATE),
    "error": ("error", theme.SERIOUS),
}


def _draw_mimic(state: AppState, mimic: dict[str, Any]) -> None:
    """The mimic toggle: the leader mirrors the arm, and a push on it takes over."""
    enabled = bool(mimic.get("enabled"))
    mimic_state = str(mimic.get("state") or "off")

    if toggle_button("Mimic  —  leader follows the arm", enabled, ImVec2(-1, 0)):
        state.client.call(protocol.METHOD_MIMIC, {"enabled": not enabled})
    if imgui.is_item_hovered():
        imgui.set_tooltip("Drive the leader from the follower's pose while another "
                          "operator has\nthe arm, so you feel the policy and can take "
                          "over without a jump.\nPush the leader to take the arm.")

    label, colour = MIMIC_LABELS.get(mimic_state, (mimic_state, theme.FG1))
    with font(theme.FONTS.small):
        text(theme.FG4, "STATE")
    imgui.same_line(LABEL_WIDTH)
    badge(label, colour, _chip_bg(colour), dot=mimic_state in ("aligning", "tracking"))
    if mimic_state == "tracking":
        # How close the leader is to the push that would take the arm: the one number
        # worth watching while a policy runs.
        gap = float(mimic.get("error_deg") or 0.0)
        limit = float(mimic.get("intervene_deg") or 0.0)
        imgui.same_line()
        with font(theme.FONTS.mono_small):
            text(theme.ACCENT1 if limit and gap > limit * 0.5 else theme.FG3,
                 f"push {gap:.1f} / {limit:.0f} deg" if limit else f"push {gap:.1f} deg")

    # Only where it says something to act on: while tracking, the number above is the
    # whole story and a permanent hint would just be furniture.
    if mimic_state != "tracking" and (detail := str(mimic.get("detail") or "")):
        imgui.push_text_wrap_pos(0.0)
        with font(theme.FONTS.small):
            text(theme.SERIOUS if mimic_state == "error" else theme.FG4, detail)
        imgui.pop_text_wrap_pos()


def draw_operators(state: AppState, size: ImVec2) -> None:
    """The rail: which operators are in the room, which one the robot is obeying, and a
    form per operator that drives it over its own RPC.

    Shaped after the Playground's own rail (`OperatorDiagnostics.tsx`) — same cards, same
    badges, same commit-on-release slider — because it is the same set of operators and
    the same four RPCs.
    """
    status = state.client.status
    peers = list(status.get("peers") or [])

    imgui.begin_child("##operators", size, imgui.ChildFlags_.borders)
    # The rail's own panic button, where the Playground puts it: always on screen, and
    # in the neutral palette rather than the serious one, because a permanently red bar
    # under a diagnostics rail stops being read at all.
    if heading(f"operators  ({sum(1 for p in peers if p.get('online'))} live)",
               f"stop everything  {_first_key(state, 'stop_all')}".strip()):
        state.client.call(protocol.METHOD_PEER_STOP)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Preempt every operator, then fold the arm and stop the "
                          "slider.\nMotion only — the voice agent's next step can drive "
                          "it again.")
    if not peers:
        text(theme.FG4, "no operators in the room")
        imgui.end_child()
        return

    avail = imgui.get_content_region_avail()
    columns = max(1, min(len(peers), int(avail.x // OPERATOR_CARD_MIN_WIDTH)))
    # Cell padding trimmed from the style's 10x7: the cards carry their own padding, and
    # the default put one row of them just past the panel's height.
    imgui.push_style_var(imgui.StyleVar_.cell_padding, ImVec2(6.0, 4.0))
    if imgui.begin_table("##rail", columns, imgui.TableFlags_.sizing_stretch_same):
        for peer in peers:
            imgui.table_next_column()
            _draw_operator_card(state, peer)
        imgui.end_table()
    imgui.pop_style_var()
    imgui.end_child()


def _draw_operator_card(state: AppState, peer: dict[str, Any]) -> None:
    identity = str(peer.get("identity", ""))
    spec = roster.BY_IDENTITY.get(identity)
    online = bool(peer.get("online"))
    active = bool(peer.get("active"))
    running = bool(peer.get("running"))
    error = str(peer.get("error") or "")

    imgui.push_id(identity)
    # A card is one step above the rail it sits on, and the active one wears the accent
    # border rather than a different fill — a colour swap would make the row jump.
    imgui.push_style_color(imgui.Col_.child_bg, theme.BG2)
    imgui.push_style_color(imgui.Col_.border,
                           theme.ACCENT2 if active else theme.SEPARATOR1)
    imgui.begin_child(f"##card_{identity}", ImVec2(0, OPERATOR_CARD_HEIGHT),
                      imgui.ChildFlags_.borders, imgui.WindowFlags_.no_scrollbar)
    card_width = imgui.get_content_region_avail().x

    with font(theme.FONTS.body):
        text(theme.FG0 if online else theme.FG3, roster.title_for(identity))
    label, colour, dot = _peer_badge(online, running, active, error)
    imgui.same_line(max(card_width - badge_width(label, dot=dot), 0.0))
    badge(label, colour, _chip_bg(colour), dot=dot)
    with font(theme.FONTS.mono_small):
        text(theme.FG4, identity)

    # One line of history, clipped by the card rather than wrapped: the whole reason
    # ran to the recorder's log, and a card that grows would break the row.
    with font(theme.FONTS.small):
        if spec is None:
            text(theme.FG4, "no descriptor — presence only")
        elif running:
            text(theme.ACCENT1, f"{float(peer.get('elapsed_s') or 0.0):.0f}s  "
                                f"{peer.get('payload') or ''}")
        elif error:
            text(theme.SERIOUS, error)
        elif result := str(peer.get("result") or ""):
            text(theme.FG4, result)
        else:
            text(theme.FG4, "not run from here")

    if spec is None:
        _end_card()
        return

    imgui.spacing()
    _draw_operator_form(state, spec, peer)
    _end_card()


def badge_width(label: str, *, dot: bool) -> float:
    """Measured in the badge's own face, so right-aligning one lands where it draws."""
    with font(theme.FONTS.small):
        return imgui.calc_text_size(label).x + BADGE_PAD.x * 2 + (11.0 if dot else 0.0)


def _end_card() -> None:
    imgui.end_child()
    imgui.pop_style_color(2)
    imgui.pop_id()


def _peer_badge(online: bool, running: bool, active: bool,
                error: str) -> tuple[str, Any, bool]:
    """(label, colour, pulse). Live states pulse; the rest are stated flatly."""
    if not online:
        return ("offline", theme.FG4, False)
    if running:
        return ("running", theme.ACCENT1, True)
    if active:
        return ("active", theme.ACCENT1, True)
    if error:
        return ("error", theme.SERIOUS, False)
    return ("idle", theme.FG3, False)


def _draw_operator_form(state: AppState, spec: roster.OperatorSpec,
                        peer: dict[str, Any]) -> None:
    """The one argument, its presets, and the two RPCs. Enter or a preset sends."""
    client = state.client
    identity = spec.identity
    online = bool(peer.get("online"))
    running = bool(peer.get("running"))
    argument = spec.argument

    def run(value: str) -> None:
        client.call(protocol.METHOD_PEER_RUN, {"identity": identity, "payload": value})

    draft = state.peer_drafts.get(identity)
    if draft is None:
        draft = argument.default if argument else ""
        state.peer_drafts[identity] = draft

    if argument is not None and argument.kind == "number":
        imgui.set_next_item_width(-1)
        imgui.begin_disabled(not online or running)
        _, value = imgui.slider_float(
            "##arg", _as_float(draft, argument.minimum), argument.minimum,
            argument.maximum, "%.0f")
        state.peer_drafts[identity] = f"{value:.0f}"
        # Commit on release, not on every frame of the drag: one RPC per gesture, and
        # the arm isn't asked to chase the cursor.
        if imgui.is_item_deactivated_after_edit():
            run(state.peer_drafts[identity])
        imgui.end_disabled()
    elif argument is not None:
        imgui.set_next_item_width(-1)
        imgui.begin_disabled(not online or running)
        submitted, draft = imgui.input_text(
            "##arg", draft, imgui.InputTextFlags_.enter_returns_true)
        imgui.end_disabled()
        state.peer_drafts[identity] = draft
        if submitted and draft.strip():
            run(draft.strip())

    if argument is not None and argument.presets:
        imgui.begin_disabled(not online or running)
        right = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
        for index, (label, value) in enumerate(argument.presets):
            if index:
                # Wrap by hand: ImGui has no flow layout, and a row of chips that runs
                # off a 270 px card is how the last preset became unreachable.
                imgui.same_line()
                if (imgui.get_cursor_pos_x() + chip_width(label)) > right:
                    imgui.new_line()
            # A preset is a command, not a suggestion: clicking one sends it.
            if chip(label):
                state.peer_drafts[identity] = value
                run(value)
        imgui.end_disabled()

    style = imgui.get_style()
    # Buttons on the card's bottom edge, so a row of cards lines up whatever their forms
    # did above. Measured from the child's own height rather than the constant, which
    # ignores its border and padding.
    imgui.set_cursor_pos_y(max(imgui.get_cursor_pos_y(),
                               imgui.get_window_height() - imgui.get_frame_height()
                               - style.window_padding.y))
    half = ImVec2((imgui.get_content_region_avail().x - style.item_spacing.x) / 2, 0)
    imgui.begin_disabled(not online or running)
    if imgui.button("Run", half):
        run(state.peer_drafts.get(identity, "").strip())
    imgui.end_disabled()
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
        imgui.set_tooltip(f"{spec.run_rpc} — {spec.summary}")
    imgui.same_line()
    imgui.begin_disabled(not online)
    if imgui.button("Stop", half):
        client.call(protocol.METHOD_PEER_STOP, {"identity": identity})
    imgui.end_disabled()
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
        imgui.set_tooltip(f"{spec.stop_rpc} — safe to send while idle.")


def _as_float(value: str, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def metric(label: str, value: str, color=theme.FG1) -> None:
    """A metrics field on a wider grid; these labels are longer than the sidebar's."""
    field(label, value, color, width=METRIC_LABEL_WIDTH)


def draw_metrics(state: AppState, size: ImVec2) -> None:
    """Portal's own counters: link speed, sync blockers, evictions."""
    metrics = state.client.metrics
    imgui.begin_child("##metrics", size, imgui.ChildFlags_.borders)
    heading("metrics")
    if not metrics:
        text(theme.FG4, "waiting for the teleoperator ...")
        imgui.end_child()
        return

    rtt, sync, wire = metrics.get("rtt_ms", {}), metrics.get("sync", {}), metrics.get("wire", {})
    video = metrics.get("video", {})

    def ms(value) -> str:
        return "--" if value is None else f"{value:g} ms"

    if imgui.begin_table("##metricgrid", 3, imgui.TableFlags_.sizing_stretch_same):
        imgui.table_next_column()
        with font(theme.FONTS.small):
            text(theme.FG3, "LINK")
        imgui.spacing()
        metric("rtt", ms(rtt.get("last")), _rtt_colour(rtt.get("last")))
        metric("rtt mean", ms(rtt.get("mean")))
        metric("rtt p95", ms(rtt.get("p95")), _rtt_colour(rtt.get("p95")))
        pings = metrics.get("pings", {})
        metric("pings", f"{pings.get('pongs', 0)}/{pings.get('sent', 0)}")

        imgui.table_next_column()
        with font(theme.FONTS.small):
            text(theme.FG3, "SYNC")
        imgui.spacing()
        metric("observations", str(sync.get("observations", 0)))
        stale = int(sync.get("stale_reused", 0) or 0)
        # Non-zero isn't a fault (reuse_stale_frames at work), but a large share means
        # frames are missing their state window.
        metric("stale reused", str(stale), theme.MODERATE if stale else theme.FG1)
        metric("match p50", ms(sync.get("match_p50_ms")))
        metric("match p95", ms(sync.get("match_p95_ms")))
        blocker = str(sync.get("blocker") or "")
        metric("blocker", blocker or "none", theme.MODERATE if blocker else theme.FG4)

        imgui.table_next_column()
        with font(theme.FONTS.small):
            text(theme.FG3, "WIRE")
        imgui.spacing()
        metric("states", str(wire.get("states", 0)))
        metric("actions", str(wire.get("actions", 0)))
        metric("state jitter", ms(wire.get("state_jitter_ms")))
        metric("action jitter", ms(wire.get("action_jitter_ms")))
        imgui.end_table()

    if video:
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h
        if imgui.begin_table("##videometrics", 6, flags):
            for name, width in (("track", 220.0), ("frames", 90.0), ("jitter", 90.0),
                                ("evicted", 90.0), ("buffer", 80.0), ("received", 0.0)):
                imgui.table_setup_column(
                    name,
                    imgui.TableColumnFlags_.width_fixed if width else
                    imgui.TableColumnFlags_.width_stretch,
                    width)
            imgui.table_next_row(imgui.TableRowFlags_.headers)
            for column, title in enumerate(("TRACK", "FRAMES", "JITTER", "EVICTED",
                                            "BUFFER", "RECEIVED")):
                imgui.table_set_column_index(column)
                with font(theme.FONTS.small):
                    text(theme.FG4, title)
            for track, stats in video.items():
                imgui.table_next_row()
                for column, (value, colour) in enumerate((
                    (track, theme.FG2),
                    (str(stats.get("received", 0)), theme.FG1),
                    (ms(stats.get("jitter_ms")), _jitter_colour(stats.get("jitter_ms"))),
                    (str(stats.get("evicted", 0)),
                     theme.MODERATE if stats.get("evicted") else theme.FG1),
                    (str(stats.get("fill", 0)), theme.FG1),
                    (f"{stats.get('mbytes', 0)} MB", theme.FG2),
                )):
                    imgui.table_set_column_index(column)
                    with font(theme.FONTS.mono_small):
                        text(colour, value)
            imgui.end_table()
    imgui.end_child()


def _rtt_colour(value):
    if value is None:
        return theme.FG4
    return theme.FG1 if value < 60 else theme.MODERATE if value < 150 else theme.SERIOUS


def _jitter_colour(value):
    if value is None:
        return theme.FG4
    return theme.FG1 if value < 20 else theme.MODERATE if value < 60 else theme.SERIOUS


def draw_corpus(state: AppState, size: ImVec2) -> None:
    client, status = state.client, state.client.status
    episodes = client.episodes
    busy = bool(status.get("busy"))

    imgui.begin_child("##corpus", size, imgui.ChildFlags_.borders)

    heading(f"episodes  ({len(episodes)})")
    pending = {
        i: text for i, text in state.label_drafts.items()
        if text.strip() and text != _task_of(episodes, i)
    }
    imgui.begin_disabled(busy or not pending)
    if imgui.button(f"Apply {len(pending)} label change(s)"):
        client.call(protocol.METHOD_RELABEL, {"episodes": {str(k): v for k, v in pending.items()}})
    imgui.end_disabled()

    imgui.same_line()
    imgui.begin_disabled(busy or not state.selected)
    if danger_button(f"Delete {len(state.selected)} episode(s)"):
        state.confirm_delete = True
    imgui.end_disabled()

    if state.selected:
        imgui.same_line()
        if imgui.small_button("clear selection"):
            state.selected.clear()

    imgui.spacing()
    _draw_episode_table(state, episodes, busy)
    _draw_delete_modal(state)
    imgui.end_child()


def _task_of(episodes: list[dict], index: int) -> str:
    for episode in episodes:
        if episode["index"] == index:
            return str(episode.get("task", ""))
    return ""


def _draw_episode_table(state: AppState, episodes: list[dict], busy: bool) -> None:
    if not episodes:
        text(theme.FG4, "no episodes recorded yet")
        return

    flags = (
        imgui.TableFlags_.row_bg
        | imgui.TableFlags_.borders_inner_h
        | imgui.TableFlags_.scroll_y
        | imgui.TableFlags_.sizing_fixed_fit
    )
    if not imgui.begin_table("##episodes", 6, flags, ImVec2(0, -1)):
        return

    imgui.table_setup_column("", imgui.TableColumnFlags_.width_fixed, 28)
    imgui.table_setup_column("#", imgui.TableColumnFlags_.width_fixed, 48)
    imgui.table_setup_column("frames", imgui.TableColumnFlags_.width_fixed, 68)
    imgui.table_setup_column("length", imgui.TableColumnFlags_.width_fixed, 72)
    imgui.table_setup_column("view", imgui.TableColumnFlags_.width_fixed, 62)
    imgui.table_setup_column("task", imgui.TableColumnFlags_.width_stretch)
    imgui.table_setup_scroll_freeze(0, 1)

    imgui.table_next_row(imgui.TableRowFlags_.headers)
    for column, title in enumerate(("", "#", "frames", "length", "", "task  (edit, Enter to apply)")):
        imgui.table_set_column_index(column)
        with font(theme.FONTS.small):
            text(theme.FG4, title.upper())

    for episode in episodes:
        index = int(episode["index"])
        imgui.table_next_row()
        imgui.push_id(index)

        imgui.table_set_column_index(0)
        imgui.begin_disabled(busy)
        changed, ticked = imgui.checkbox("##sel", index in state.selected)
        imgui.end_disabled()
        if changed:
            state.selected.add(index) if ticked else state.selected.discard(index)

        imgui.table_set_column_index(1)
        with font(theme.FONTS.mono_small):
            text(theme.FG3, str(index))

        imgui.table_set_column_index(2)
        with font(theme.FONTS.mono_small):
            text(theme.FG2, str(episode.get("length", 0)))

        imgui.table_set_column_index(3)
        with font(theme.FONTS.mono_small):
            text(theme.FG2, f"{float(episode.get('seconds', 0.0)):.1f}s")

        imgui.table_set_column_index(4)
        viewing = state.viewing == index
        if viewing:
            imgui.push_style_color(imgui.Col_.button, theme.BG_ACCENT2)
            imgui.push_style_color(imgui.Col_.text, theme.ACCENT1)
        if imgui.small_button("shown" if viewing else "view"):
            state.viewing = index
            state.player.close()
            # Offsets come from the teleoperator, not disk: mid-session the metadata
            # parquet has no footer to read.
            state.client.request_episode_video(index)
        if viewing:
            imgui.pop_style_color(2)

        imgui.table_set_column_index(5)
        draft = state.label_drafts.get(index, str(episode.get("task", "")))
        imgui.set_next_item_width(-1)
        imgui.begin_disabled(busy)
        submitted, draft = imgui.input_text(
            "##task", draft, imgui.InputTextFlags_.enter_returns_true
        )
        imgui.end_disabled()
        if draft != str(episode.get("task", "")):
            state.label_drafts[index] = draft
        else:
            state.label_drafts.pop(index, None)
        if submitted and draft.strip() and draft != str(episode.get("task", "")):
            # Enter applies just this row, bypassing the batch button.
            state.client.call(protocol.METHOD_RELABEL, {"episodes": {str(index): draft}})

        imgui.pop_id()

    imgui.end_table()


def _draw_delete_modal(state: AppState) -> None:
    if state.confirm_delete:
        imgui.open_popup("Delete episodes")
        state.confirm_delete = False

    if not imgui.begin_popup_modal(
        "Delete episodes", None, imgui.WindowFlags_.always_auto_resize
    )[0]:
        return

    doomed = sorted(state.selected)
    text(theme.FG1, f"Delete episode(s) {', '.join(str(i) for i in doomed)}?")
    imgui.spacing()
    with font(theme.FONTS.small):
        text(
            theme.MODERATE,
            "This cannot be undone. Remaining episodes are renumbered from 0,\n"
            "and the whole corpus is rewritten (video segments spanning a deleted\n"
            "episode get re-encoded), so it can take a while on a large dataset.",
        )
    imgui.spacing()
    if danger_button("Delete"):
        state.client.call(protocol.METHOD_DELETE, {"episodes": doomed})
        state.selected.clear()
        imgui.close_current_popup()
    imgui.same_line()
    if imgui.button("Cancel"):
        imgui.close_current_popup()
    imgui.end_popup()


def draw_settings(state: AppState) -> None:
    """Key bindings. Rebind by *pressing the key*: a foot pedal sends whatever key its vendor chose."""
    if not imgui.begin_popup_modal(
        "Settings", None, imgui.WindowFlags_.always_auto_resize
    )[0]:
        state.binding = None
        return

    heading("key bindings")
    with font(theme.FONTS.small):
        text(theme.FG4,
             "Bindings apply to this window immediately. The teleoperator's terminal\n"
             "hotkeys pick them up on its next start, and only for keys a terminal\n"
             "can send (an F-key works here but not there).")
    imgui.spacing()

    if state.binding and imgui.is_key_pressed(imgui.Key.escape, False):
        state.binding = None  # cancel; without this it listens forever
    captured = _captured_key() if state.binding else None
    imgui.begin_table("##bindings", 3, imgui.TableFlags_.sizing_fixed_fit)
    imgui.table_setup_column("", imgui.TableColumnFlags_.width_fixed, 96)
    imgui.table_setup_column("", imgui.TableColumnFlags_.width_fixed, 190)
    imgui.table_setup_column("", imgui.TableColumnFlags_.width_fixed, 110)
    for action in shortcuts.ACTIONS:
        imgui.table_next_row()
        imgui.push_id(action)
        if captured and state.binding == action:
            # Append, not replace, so a pedal can sit alongside the keyboard binding.
            if captured not in state.keys[action]:
                state.keys[action].append(captured)
            state.binding = None
            shortcuts.save(state.keys)

        imgui.table_next_column()
        with font(theme.FONTS.small):
            text(theme.FG4, action.upper())
        imgui.table_next_column()
        listening = state.binding == action
        label = "press a key ..." if listening else shortcuts.describe(state.keys, action)
        with font(theme.FONTS.mono_small):
            text(theme.ACCENT1 if listening else theme.FG1, label)
        imgui.table_next_column()
        if imgui.small_button("bind"):
            state.binding = action
        imgui.same_line()
        imgui.begin_disabled(not state.keys.get(action))
        if imgui.small_button("clear"):
            state.keys[action] = []
            state.binding = None
            shortcuts.save(state.keys)
        imgui.end_disabled()
        imgui.pop_id()

    imgui.end_table()
    imgui.spacing()
    imgui.separator()
    imgui.spacing()
    if imgui.button("Reset to defaults"):
        state.keys = {a: list(k) for a, k in shortcuts.DEFAULTS.items()}
        state.binding = None
        shortcuts.save(state.keys)
    imgui.same_line()
    if imgui.button("Done"):
        state.binding = None
        imgui.close_current_popup()
    with font(theme.FONTS.small):
        text(theme.FG4, f"saved to {shortcuts.config_path()}")
    imgui.end_popup()


# Never bindable: modifiers carry no keypress of their own, and Escape/Enter/mouse
# are how you work the dialog itself.
_UNBINDABLE = frozenset({
    "escape", "enter", "keypad_enter", "left_ctrl", "right_ctrl", "left_shift",
    "right_shift", "left_alt", "right_alt", "left_super", "right_super",
    "mod_ctrl", "mod_shift", "mod_alt", "mod_super", "mouse_left", "mouse_right",
    "mouse_middle", "mouse_x1", "mouse_x2", "mouse_wheel_x", "mouse_wheel_y",
    "reserved_for_mod_ctrl", "reserved_for_mod_shift", "reserved_for_mod_alt",
    "reserved_for_mod_super",
})


# Tab shares its value (512) with the range sentinel, and `dir()` surfaces only
# the sentinel. `tab` is the name that round-trips: TERMINAL_CHARS knows it and
# `getattr(imgui.Key, "tab")` resolves it back.
_KEY_ALIASES = {"named_key_begin": "tab"}


def _captured_key() -> Optional[str]:
    """Name of whatever key was pressed this frame, or None. Scans the named-key range
    so an unusual pedal (F13-F24, a media key) binds like anything else."""
    if imgui.is_key_pressed(imgui.Key.escape, False):
        return None
    # `is_key_pressed` asserts outside this range, and the enum also carries the
    # range sentinels (`none`, `named_key_end`) and the mod bitmasks — scanning
    # those is what took the window down the moment a capture started.
    named = range(int(imgui.Key.named_key_begin), int(imgui.Key.named_key_end))
    for name in dir(imgui.Key):
        if name.startswith("__") or name in _UNBINDABLE:
            continue
        key = getattr(imgui.Key, name)
        if not isinstance(key, imgui.Key) or int(key) not in named:
            continue
        if imgui.is_key_pressed(key, False):
            return _KEY_ALIASES.get(name, name)
    return None


# --- frame ------------------------------------------------------------------

def draw(state: AppState) -> None:
    status = state.client.status
    state.sync(status)

    draw_header(state)
    draw_banner(state)

    # Nothing to record with until a port and corpus are chosen.
    if status and not status.get("configured"):
        draw_setup(state)
        return

    draw_view_switch(state)
    if state.tab == "Dataset":
        draw_dataset_tab(state)
    elif state.tab == "Metrics":
        draw_metrics_tab(state)
    else:
        draw_teleop_tab(state)

    handle_shortcuts(state)


def draw_setup(state: AppState) -> None:
    """Choose the leader port and corpus. The teleoperator enumerates both; this picks from what it reports."""
    client, status = state.client, state.client.status
    options = client.setup_options
    if not state.setup_asked:
        client.request_setup_options()
        state.setup_asked = True

    imgui.begin_child("##setup", imgui.get_content_region_avail(),
                      imgui.ChildFlags_.borders)
    heading("set up this session")

    opening = str(status.get("opening") or "")
    if opening:
        text(theme.ACCENT1, f"{opening} ...")
        with font(theme.FONTS.small):
            text(theme.FG4, "If this hangs, check the teleoperator's terminal — lerobot "
                            "may be asking you to calibrate the leader arm.")
        imgui.end_child()
        return

    if error := str(status.get("open_error") or ""):
        imgui.push_text_wrap_pos(0.0)
        text(theme.SERIOUS, error)
        imgui.pop_text_wrap_pos()
        imgui.spacing()

    if not options:
        text(theme.FG4, "asking the teleoperator what it can see ...")
        imgui.end_child()
        return

    defaults = options.get("defaults") or {}
    ports = options.get("ports") or []
    datasets = options.get("datasets") or []
    if not state.setup_port:
        # Only a likely-arm port is preselected; otherwise Open stays disabled until you choose.
        state.setup_port = str(defaults.get("port") or "")
    if not state.setup_repo_id:
        state.setup_repo_id = str(defaults.get("repo_id") or "")
    if not state.setup_task:
        state.setup_task = str(defaults.get("task") or "")
    if not datasets:
        state.setup_new = True

    # --- port ---
    heading("leader arm")
    if not ports:
        text(theme.MODERATE, "no serial ports detected")
    for entry in ports:
        device = str(entry.get("device", ""))
        label = f"{device}    {entry.get('description') or ''}".rstrip()
        if not entry.get("likely"):
            label += "    (not a USB serial port)"
        if imgui.radio_button(label, state.setup_port == device):
            state.setup_port = device
    imgui.spacing()
    imgui.set_next_item_width(420)
    _, state.setup_port = imgui.input_text("port", state.setup_port)
    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    # --- dataset ---
    heading("dataset")
    for i, entry in enumerate(datasets):
        selected = not state.setup_new and state.setup_pick == i
        label = (f"{entry.get('repo_id')}   ({entry.get('episodes', 0)} episodes)"
                 f"   {entry.get('root')}")
        if imgui.radio_button(label, selected):
            state.setup_new, state.setup_pick = False, i
    if imgui.radio_button("new dataset", state.setup_new):
        state.setup_new = True

    if state.setup_new:
        imgui.spacing()
        imgui.set_next_item_width(420)
        _, state.setup_repo_id = imgui.input_text("repo id", state.setup_repo_id)
        if not session_valid(state.setup_repo_id):
            with font(theme.FONTS.small):
                text(theme.MODERATE, "use the form org/name, e.g. binhpham/candy-shop")
        home = str(defaults.get("lerobot_home") or "")
        local = str(defaults.get("local_root") or "")
        if imgui.radio_button(f"{home}/{state.setup_repo_id}"
                              "    (lerobot's home — trains with no extra flags)",
                              not state.setup_local):
            state.setup_local = False
        if imgui.radio_button(f"{local}/{state.setup_repo_id}"
                              "    (local to this project)", state.setup_local):
            state.setup_local = True
    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    # --- task ---
    heading("task for new episodes")
    imgui.set_next_item_width(-1)
    _, state.setup_task = imgui.input_text("##setuptask", state.setup_task)
    imgui.spacing()

    repo_id, root = _setup_target(state, datasets, defaults)
    ready = bool(state.setup_port) and session_valid(repo_id) and bool(root)
    imgui.begin_disabled(not ready)
    if accent_button("Open", ImVec2(220, imgui.get_frame_height() * 1.4)):
        client.call(protocol.METHOD_OPEN, {
            "port": state.setup_port, "repo_id": repo_id,
            "root": root, "task": state.setup_task,
        })
    imgui.end_disabled()
    with font(theme.FONTS.mono_small):
        text(theme.FG4, f"{repo_id or '?'}   {root or '?'}")
    imgui.end_child()


def session_valid(repo_id: str) -> bool:
    """`org/name`, matching the teleoperator's own check so Open can't offer what the RPC will refuse."""
    return repo_id.count("/") == 1 and all(p.strip() for p in repo_id.split("/"))


def _setup_target(state: AppState, datasets: list, defaults: dict) -> tuple[str, str]:
    """(repo_id, root) for whatever is currently selected."""
    if not state.setup_new and datasets:
        pick = datasets[min(state.setup_pick, len(datasets) - 1)]
        return str(pick.get("repo_id", "")), str(pick.get("root", ""))
    base = str(defaults.get("local_root" if state.setup_local else "lerobot_home") or "")
    return state.setup_repo_id, f"{base}/{state.setup_repo_id}" if base else ""


def draw_view_switch(state: AppState) -> None:
    """A segmented control, not ImGui's tab bar (which collapsed to a 2 px sliver under this style)."""
    for name in ("Teleop", "Dataset", "Metrics"):
        selected = state.tab == name
        imgui.push_style_color(imgui.Col_.button,
                               theme.BG_ACCENT2 if selected else theme.BG2)
        imgui.push_style_color(imgui.Col_.button_hovered,
                               theme.BG_ACCENT2 if selected else theme.BG3)
        imgui.push_style_color(imgui.Col_.text,
                               theme.ACCENT1 if selected else theme.FG3)
        if imgui.button(name, ImVec2(130, 0)):
            state.tab = name
        imgui.pop_style_color(3)
        imgui.same_line()
    imgui.new_line()
    imgui.spacing()


def draw_teleop_tab(state: AppState) -> None:
    """Cameras and the session up top; the arm and its operators in a band below."""
    avail = imgui.get_content_region_avail()
    spacing = imgui.get_style().item_spacing
    band_h = max(avail.y * BAND_HEIGHT_FRACTION, OPERATOR_CARD_HEIGHT + 74.0)
    top_h = avail.y - band_h - spacing.y
    sidebar_w = min(SIDEBAR_WIDTH, avail.x * 0.32)
    arm_w = min(ARM_PANEL_WIDTH, avail.x * 0.3)

    draw_cameras(state, ImVec2(avail.x - sidebar_w - spacing.x, top_h))
    imgui.same_line()
    draw_session(state, ImVec2(sidebar_w, top_h))
    draw_arm(state, ImVec2(arm_w, band_h))
    imgui.same_line()
    draw_operators(state, ImVec2(avail.x - arm_w - spacing.x, band_h))


def draw_metrics_tab(state: AppState) -> None:
    """Portal's counters, on their own tab.

    They used to share the teleop tab, which cost the session panel the height it needed
    for its own counters. What matters mid-flight is a link that has gone bad, and the
    header's chips say that in four numbers — this is where you come to read why.
    """
    avail = imgui.get_content_region_avail()
    # Bounded, not full-bleed: these are label/value pairs, and stretching the grid over
    # 1500 px parks every value half a screen from its label.
    draw_metrics(state, ImVec2(min(1080.0, avail.x), min(avail.y, 430.0)))


def draw_dataset_tab(state: AppState) -> None:
    avail = imgui.get_content_region_avail()
    spacing = imgui.get_style().item_spacing
    table_h = max(avail.y * 0.42, 180.0)
    draw_corpus(state, ImVec2(avail.x, table_h))
    draw_player(state, ImVec2(avail.x, avail.y - table_h - spacing.y))


def draw_player(state: AppState, size: ImVec2) -> None:
    """Play back one episode, both cameras on a single shared scrub so they can't drift."""
    client, player = state.client, state.player
    imgui.begin_child("##player", size, imgui.ChildFlags_.borders)

    if state.viewing is None:
        heading("episode")
        text(theme.FG4, "pick an episode above to review its video")
        imgui.end_child()
        return

    # `request_episode_video` is answered asynchronously; open once the reply for the
    # episode we're actually looking at arrives.
    info = client.episode_video
    if player.episode != state.viewing and info.get("episode") == state.viewing:
        root = pathlib.Path(str(client.status.get("root") or ""))
        player.open(root, state.viewing, info.get("videos") or {},
                    int(info.get("length") or 0))

    heading(f"episode {state.viewing}")
    if not player.is_open:
        text(theme.MODERATE, player.error or "loading ...")
        imgui.end_child()
        return

    fps = int(client.status.get("fps", 30)) or 30
    now = time.monotonic()
    if state.last_frame_at:
        player.advance(now - state.last_frame_at, fps)
    state.last_frame_at = now

    # --- transport ---
    label = "pause" if player.playing else "play"
    if imgui.button(label, ImVec2(90, 0)):
        player.playing = not player.playing
    imgui.same_line()
    if imgui.button("|<", ImVec2(48, 0)):
        player.seek(0)
        player.playing = False
    imgui.same_line()
    if imgui.button("-1", ImVec2(48, 0)):
        player.seek(player.position - 1)
        player.playing = False
    imgui.same_line()
    if imgui.button("+1", ImVec2(48, 0)):
        player.seek(player.position + 1)
        player.playing = False
    imgui.same_line()
    with font(theme.FONTS.mono_small):
        text(theme.FG2, f"{player.position + 1} / {player.frames}   "
                        f"{player.position / fps:.2f}s")
    imgui.same_line()
    if imgui.button("close", ImVec2(80, 0)):
        state.viewing = None
        player.close()
        imgui.end_child()
        return

    # --- the one scrub that drives every camera ---
    imgui.set_next_item_width(-1)
    changed, position = imgui.slider_int("##scrub", player.position, 0,
                                         max(player.frames - 1, 0))
    if changed:
        player.seek(position)
        player.playing = False
    imgui.spacing()

    cameras = player.cameras()
    if not cameras:
        text(theme.FG4, "no cameras in this episode")
        imgui.end_child()
        return

    columns = min(len(cameras), 2)
    rows = (len(cameras) + columns - 1) // columns
    spacing = imgui.get_style().item_spacing
    avail = imgui.get_content_region_avail()
    label_h = imgui.get_text_line_height_with_spacing()
    cell_w = (avail.x - spacing.x * (columns - 1)) / columns
    cell_h = (avail.y - spacing.y * (rows - 1)) / rows - label_h

    for i, camera in enumerate(cameras):
        if i % columns:
            imgui.same_line()
        imgui.begin_group()
        with font(theme.FONTS.mono_small):
            text(theme.FG3, camera)
        image = player.frame(camera)
        if image is None:
            imgui.dummy(ImVec2(cell_w, max(cell_h, 1.0)))
        else:
            src_h, src_w = image.shape[:2]
            width = min(cell_w, cell_h * src_w / src_h)
            # Every scrub step is a new frame, so this always re-uploads.
            immvision.image_display(f"##play_{camera}", image, (int(width), 0), True)
        imgui.end_group()
    imgui.end_child()


def key_pressed(state: AppState, action: str) -> bool:
    """Whether any key bound to `action` was pressed this frame."""
    for name in state.keys.get(action, ()):
        key = getattr(imgui.Key, name, None)
        if key is not None and imgui.is_key_pressed(key, False):
            return True
    return False


def handle_shortcuts(state: AppState) -> None:
    """Keyboard control. Ignored while a text field has the keyboard (else typing a task
    would start/stop episodes) and while a corpus job holds the dataset."""
    if imgui.get_io().want_capture_keyboard or state.binding is not None:
        return
    status = state.client.status
    if status.get("busy"):
        return

    recording = bool(status.get("recording"))
    if key_pressed(state, "record"):
        if recording:
            state.client.call(protocol.METHOD_STOP)
        elif status.get("ready") and not status.get("saving"):
            state.client.call(protocol.METHOD_START)
    elif key_pressed(state, "discard"):
        # No confirmation: a dialog would defeat the point.
        if recording:
            state.client.call(protocol.METHOD_DISCARD)
    elif key_pressed(state, "claim"):
        # One key, both directions, matching the recorder's terminal binding.
        if status.get("claiming"):
            state.client.call(protocol.METHOD_RELEASE)
        elif status:
            state.client.call(protocol.METHOD_CLAIM)
    elif key_pressed(state, "release"):
        if status and (status.get("active_operator") is not None or status.get("claiming")):
            state.client.call(protocol.METHOD_RELEASE)
    elif key_pressed(state, "resume"):
        if status.get("suspended"):
            state.client.call(protocol.METHOD_RESUME)
    elif key_pressed(state, "mimic"):
        enabled = bool((status.get("mimic") or {}).get("enabled"))
        state.client.call(protocol.METHOD_MIMIC, {"enabled": not enabled})
    elif key_pressed(state, "stop_all"):
        state.client.call(protocol.METHOD_PEER_STOP)


def cli() -> None:
    """Console-script entry point (`uv run teleoperator-ui`)."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_env(PACKAGE_DIR)
    url = env("LIVEKIT_URL", required=True)
    room = env("LIVEKIT_ROOM", "candy-shop")
    identity = env("TELEOPERATOR_UI_IDENTITY", "teleoperator-ui")

    # Same wire contract the teleoperator loads, so video renders before one is found.
    # Read as YAML, never through livekit.portal (see contract_camera_names for why).
    cameras = contract_camera_names(portal_config_path(PACKAGE_DIR))

    client = RecorderClient(
        url=url,
        token=mint_token(identity, room, name="Candy Shop Teleoperator UI"),
        room=room,
        cameras=cameras,
        target=os.environ.get("TELEOPERATOR_UI_TARGET") or None,
        poll_hz=float(env("TELEOPERATOR_UI_POLL_HZ", "4")),
    )
    state = AppState(client, cameras)
    client.start()

    immvision.use_rgb_color_order()  # Portal hands us RGB; immvision assumes BGR

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "LiveKit — Candy Shop Teleoperator"
    params.app_window_params.window_geometry.size = (1500, 940)
    params.app_window_params.restore_previous_geometry = True
    # Per-user machine state; otherwise HelloImGui litters the repo with its ini.
    params.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
    params.ini_filename = "candy-shop-teleoperator/ui.ini"
    params.imgui_window_params.show_menu_bar = False
    params.imgui_window_params.show_status_bar = False
    params.imgui_window_params.background_color = theme.BG0
    params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_window
    )
    # Never idle down: dropping fps when the mouse stops looks like a frozen feed.
    params.fps_idling.enable_idling = False
    params.callbacks.load_additional_fonts = theme.load_fonts
    params.callbacks.setup_imgui_style = theme.apply_style
    params.callbacks.show_gui = lambda: draw(state)

    try:
        hello_imgui.run(params)
    except KeyboardInterrupt:
        # Under `teleoperator --ui`, Ctrl-C reaches this process too; swallow it so we
        # don't print a traceback over the teleoperator's shutdown output.
        pass
    finally:
        client.stop()


if __name__ == "__main__":
    cli()
