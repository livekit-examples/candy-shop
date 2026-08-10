"""LiveKit design language, ported to Dear ImGui.

Colour tokens are the same *values* `portal-playground/app/globals.css` defines,
so this desktop tool and the web console read as one product. Dark only —
`globals.css` has no light variant either.

`assets/fonts/*.ttf` are the real brand faces (Everett, Commit Mono) re-wrapped
from the repo's woff2, which ImGui's stb_truetype can't parse; Commit Mono, being
variable, is pinned to its regular instance. Mono is used for anything numeric so
the layout doesn't jitter as digits change.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Optional

from imgui_bundle import ImVec2, ImVec4, hello_imgui, imgui

ASSETS_DIR = pathlib.Path(__file__).resolve().parent / "assets"


def rgb(hex_code: str, alpha: float = 1.0) -> ImVec4:
    """`#rrggbb` -> ImVec4, so the tokens below stay copy-pasteable from CSS."""
    h = hex_code.lstrip("#")
    return ImVec4(int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, alpha)


# --- tokens (values mirror portal-playground/app/globals.css) -----------------

BG0 = rgb("#000000")
BG1 = rgb("#070707")
BG2 = rgb("#131313")
BG3 = rgb("#1f1f1f")

FG0 = rgb("#ffffff")
FG1 = rgb("#cccccc")
FG2 = rgb("#b2b2b2")
FG3 = rgb("#999999")
FG4 = rgb("#666666")

ACCENT1 = rgb("#1fd5f9")  # cyan-400
ACCENT2 = rgb("#15889f")  # cyan-600
ACCENT_SECONDARY = rgb("#dc85ff")  # purple-300
BG_ACCENT = rgb("#051518")  # cyan-900
BG_ACCENT2 = rgb("#012a32")  # cyan-800

SEPARATOR1 = rgb("#202020")
SEPARATOR2 = rgb("#30302f")
SEPARATOR_SERIOUS = rgb("#421510")  # red-800

SUCCESS = rgb("#23de6b")
MODERATE = rgb("#ffb752")
SERIOUS = rgb("#ff7566")
BG_SERIOUS = rgb("#1f0e0b")  # red-900


@dataclass
class Fonts:
    """The loaded faces. `None` if the brand fonts are missing — the app then
    falls back to ImGui's built-in font rather than refusing to start."""
    body: Optional[imgui.ImFont] = None
    small: Optional[imgui.ImFont] = None
    heading: Optional[imgui.ImFont] = None
    mono: Optional[imgui.ImFont] = None
    mono_small: Optional[imgui.ImFont] = None


FONTS = Fonts()

BODY_SIZE = 16.0
SMALL_SIZE = 13.0
HEADING_SIZE = 21.0


def load_fonts() -> None:
    """Load the brand faces. Call from HelloImGui's `load_additional_fonts`
    callback — fonts can only be added while the backend owns the atlas."""
    hello_imgui.set_assets_folder(str(ASSETS_DIR))
    everett = "fonts/Everett-Regular.ttf"
    medium = "fonts/Everett-Medium.ttf"
    mono = "fonts/CommitMono-Regular.ttf"
    if not (ASSETS_DIR / everett).exists():
        return  # keep ImGui's default font; every draw call guards on None
    FONTS.body = hello_imgui.load_font(everett, BODY_SIZE)
    FONTS.small = hello_imgui.load_font(everett, SMALL_SIZE)
    FONTS.heading = hello_imgui.load_font(medium, HEADING_SIZE)
    FONTS.mono = hello_imgui.load_font(mono, BODY_SIZE)
    FONTS.mono_small = hello_imgui.load_font(mono, SMALL_SIZE)


class font:
    """`with font(FONTS.mono):` — a no-op when that face didn't load.

    Pushes the loaded size explicitly: ImGui 1.92's atlas is dynamic, so
    `push_font(f, 0.0)` means *keep the current size* and would leave text
    body-sized when swapping to `FONTS.small`."""

    def __init__(self, face: Optional[imgui.ImFont]) -> None:
        self._face = face

    def __enter__(self) -> None:
        if self._face is not None:
            imgui.push_font(self._face, self._face.legacy_size)

    def __exit__(self, *exc) -> None:
        if self._face is not None:
            imgui.pop_font()


def apply_style() -> None:
    """Paint ImGui in the tokens above. Call from `setup_imgui_style`."""
    style = imgui.get_style()

    # Nearly square. Borders and alignment do the separating work; ImGui has no
    # drop shadows, and a large radius reads as decoration in a dense tool.
    style.window_rounding = 0.0
    style.child_rounding = 2.0
    style.frame_rounding = 2.0
    style.popup_rounding = 2.0
    style.grab_rounding = 2.0
    style.scrollbar_rounding = 2.0
    style.tab_rounding = 2.0

    style.window_border_size = 0.0
    style.child_border_size = 1.0
    style.frame_border_size = 1.0
    style.popup_border_size = 1.0

    style.window_padding = ImVec2(16, 16)
    style.frame_padding = ImVec2(12, 8)
    style.cell_padding = ImVec2(10, 7)
    style.item_spacing = ImVec2(10, 10)
    style.item_inner_spacing = ImVec2(8, 6)
    style.scrollbar_size = 10.0
    style.grab_min_size = 12.0

    c = style.set_color_
    C = imgui.Col_

    c(C.window_bg, BG1)
    c(C.child_bg, BG2)
    c(C.popup_bg, BG2)
    c(C.menu_bar_bg, BG1)
    c(C.border, SEPARATOR1)
    c(C.border_shadow, ImVec4(0, 0, 0, 0))

    c(C.text, FG1)
    c(C.text_disabled, FG4)
    c(C.text_selected_bg, BG_ACCENT2)
    c(C.text_link, ACCENT1)

    # Inputs and checkboxes sit a step ABOVE the card rather than as black wells:
    # pure black on a near-black card reads as a hole, and at these sizes a
    # checkbox became invisible until hovered.
    c(C.frame_bg, BG3)
    c(C.frame_bg_hovered, rgb("#2a2a2a"))
    c(C.frame_bg_active, BG_ACCENT2)

    # Quiet by default; the accent is reserved for the primary action.
    c(C.button, BG3)
    c(C.button_hovered, rgb("#2a2a2a"))
    c(C.button_active, SEPARATOR2)

    # Tabs. Unthemed these fall back to ImGui's blue and the labels vanish
    # against a near-black window.
    c(C.tab, BG1)
    c(C.tab_hovered, BG3)
    c(C.tab_selected, BG2)
    c(C.tab_selected_overline, ACCENT1)   # the one cyan accent that marks "here"
    c(C.tab_dimmed, BG1)
    c(C.tab_dimmed_selected, BG2)
    c(C.tab_dimmed_selected_overline, SEPARATOR2)

    c(C.header, BG_ACCENT)
    c(C.header_hovered, BG_ACCENT2)
    c(C.header_active, BG_ACCENT2)

    c(C.separator, SEPARATOR1)
    c(C.separator_hovered, SEPARATOR2)
    c(C.separator_active, ACCENT2)

    c(C.check_mark, ACCENT1)
    c(C.slider_grab, ACCENT2)
    c(C.slider_grab_active, ACCENT1)

    c(C.scrollbar_bg, ImVec4(0, 0, 0, 0))
    c(C.scrollbar_grab, SEPARATOR2)
    c(C.scrollbar_grab_hovered, FG4)
    c(C.scrollbar_grab_active, ACCENT2)

    c(C.table_header_bg, BG3)
    c(C.table_border_strong, SEPARATOR2)
    c(C.table_border_light, SEPARATOR1)
    c(C.table_row_bg, ImVec4(0, 0, 0, 0))
    c(C.table_row_bg_alt, rgb("#0d0d0d"))

    c(C.title_bg, BG1)
    c(C.title_bg_active, BG1)
    c(C.title_bg_collapsed, BG1)
    c(C.resize_grip, SEPARATOR1)
    c(C.resize_grip_hovered, SEPARATOR2)
    c(C.resize_grip_active, ACCENT2)
    c(C.nav_cursor, ACCENT2)
