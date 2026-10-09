"""Textual TUI for AudioAgent Live featuring real-time audio-reactive 3D ASCII Sphere."""

from __future__ import annotations

import math
import random
import time
from typing import Optional

import numpy as np
from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import RichLog, Static

from audio_agent.audio.live_session import (
    BANDS,
    EDGES,
    LiveConversationSession,
    SessionState,
)

TAU = math.tau

RAMPS = {
    "ramp": " .'`,:;-~=+*^!?oOxX#%@",
    "classic": " .,:-~=+*^!|/\\()[]{}<>?abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$%&@",
    "blocks": " ░▒▓█",
    "dots": " ·∙•●",
}

THEMES = {
    "aurora": [(70, 244, 255), (40, 153, 255), (74, 75, 255), (70, 244, 255)],
    "spectrum": [
        (44, 235, 224),
        (93, 230, 255),
        (255, 207, 93),
        (255, 115, 125),
        (222, 112, 255),
        (105, 123, 255),
        (44, 235, 224),
    ],
    "ember": [(255, 80, 40), (255, 150, 40), (255, 214, 102), (255, 90, 90), (255, 80, 40)],
    "matrix": [(20, 200, 90), (120, 255, 160), (10, 140, 70), (20, 200, 90)],
}
THEME_NAMES = list(THEMES)
RING_GLYPHS = "·∙:+*"
SPARK = " ▂▃▄▅▆▇█"


def mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def grad(stops, pos):
    f = (pos % 1.0) * (len(stops) - 1)
    i = int(f)
    return mix(stops[i], stops[i + 1], f - i)


class SphereCanvas(Static):
    """Rotating, shaded ASCII sphere with halo, shockwave rings and audio reactivity."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ay, self.ax_off = 0.0, 0.0
        self.spin, self.running = 0.7, True
        self.theme, self.ramp = "aurora", "ramp"
        self.level, self.bands = 0.0, [0.0] * BANDS
        self.rings: list[list[float]] = []
        self.fps = 30.0
        self._tstart = self._tprev = time.monotonic()
        self._extaudio = (0.0, [0.0] * BANDS)
        self._dragpos = None
        self._grid = (0, 0)
        self._starmap: dict = {}
        self._scache: dict = {}

    def on_mount(self) -> None:
        self.set_interval(1 / 30, self.tick)

    def push_audio(self, level: float, bands: list[float] | None = None) -> None:
        self._extaudio = (level, bands or [level] * BANDS)

    def trigger_shockwave(self) -> None:
        """Emits an expanding ripple shockwave on interruption."""
        if len(self.rings) < 4:
            self.rings.append([1.0, 0.9])

    def on_mouse_down(self, e: events.MouseDown) -> None:
        self._dragpos = (e.x, e.y)
        self.capture_mouse()

    def on_mouse_move(self, e: events.MouseMove) -> None:
        if self._dragpos:
            self.ay += (e.x - self._dragpos[0]) * 0.09
            self.ax_off = max(-1.0, min(1.0, self.ax_off + (e.y - self._dragpos[1]) * 0.07))
            self._dragpos = (e.x, e.y)

    def on_mouse_up(self, e: events.MouseUp) -> None:
        self._dragpos = None
        self.release_mouse()

    def tick(self) -> None:
        now = time.monotonic()
        dt, self._tprev = min(0.1, now - self._tprev), now
        self.fps += (1 / max(dt, 1e-3) - self.fps) * 0.1
        t = now - self._tstart

        lv, bs = self._extaudio
        ease = lambda cur, tgt: cur + (tgt - cur) * (0.55 if tgt > cur else 0.15)
        if lv - self.level > 0.20 and len(self.rings) < 4:
            self.rings.append([1.0, 0.8])
        self.level = ease(self.level, lv)
        self.bands = [ease(c, b) for c, b in zip(self.bands, bs)]
        self.rings = [[r + dt * 0.9, a * 0.93] for r, a in self.rings if a > 0.04 and r < 1.7]
        if self.running and not self._dragpos:
            self.ay += self.spin * dt * (1 + self.level * 2.5)
        self.refresh()

    def _cstyle(self, r: float, g: float, b: float) -> Style:
        k = (min(31, int(r) >> 3), min(31, int(g) >> 3), min(31, int(b) >> 3))
        s = self._scache.get(k)
        if s is None:
            s = self._scache[k] = Style(
                color=f"#{(k[0] << 3) | 4:02x}{(k[1] << 3) | 4:02x}{(k[2] << 3) | 4:02x}"
            )
        return s

    def render(self) -> Text:
        w, h = max(24, self.size.width), max(10, self.size.height)
        if (w, h) != self._grid:
            self._grid = (w, h)
            self._starmap = {
                (random.randrange(h), random.randrange(w)): (random.random() * TAU, random.choice(".·+"))
                for _ in range(max(8, w * h // 50))
            }
        t, level, bands, rings = time.monotonic() - self._tstart, self.level, self.bands, self.rings
        stops, ramp = THEMES[self.theme], RAMPS[self.ramp]
        nr = len(ramp) - 1
        cx, cy = (w - 1) / 2, (h - 1) / 2
        radius = min(w * 0.38, h * 0.85) * (1 + 0.06 * level + 0.012 * math.sin(t * 1.3))
        inv, aspect = 1 / max(radius, 1), 0.48
        ay, ax = self.ay, 0.22 + self.ax_off + math.sin(t * 0.27) * 0.16
        cyaw, syaw, cp, sp = math.cos(ay), math.sin(ay), math.cos(ax), math.sin(ax)
        LX, LY, LZ = -0.42, -0.5, 0.755
        drift, sin, cos, sqrt, atan2, acos, exp = (
            t * 0.03,
            math.sin,
            math.cos,
            math.sqrt,
            math.atan2,
            math.acos,
            math.exp,
        )
        out = Text(no_wrap=True)

        for row in range(h):
            ys = (row - cy) * inv / aspect
            ys2 = ys * ys
            chars: list[str] = []
            styles: list[Style | None] = []
            for col in range(w):
                xs = (col - cx) * inv
                r2 = xs * xs + ys2
                if r2 <= 1.0:
                    z = sqrt(1.0 - r2)
                    x2, z2 = xs * cyaw - z * syaw, xs * syaw + z * cyaw
                    y2, z3 = ys * cp - z2 * sp, ys * sp + z2 * cp
                    lon = (atan2(x2, z3) / TAU + 0.5) % 1.0
                    lat = acos(max(-1.0, min(1.0, y2))) / math.pi
                    e = bands[min(BANDS - 1, int(lat * BANDS))]
                    diff = max(0.0, xs * LX + ys * LY + z * LZ)
                    spec = diff**20
                    shade = (0.16 + 0.62 * diff + 0.30 * (1 - z) ** 2.2 + 0.55 * spec) * (0.8 + 0.45 * e)
                    base = grad(stops, lon + drift)
                    lum = 0.25 + 0.75 * min(1.0, shade)
                    white = min(1.0, spec * 0.9 + level * 0.18 * diff + e * 0.25)
                    r, g, b = (c * lum + (255 - c * lum) * white for c in base)
                    idx = int(max(0.0, min(1.0, shade * 0.95 + e * 0.2)) * nr)
                    wave = sin((lon * 9 + lat * 5) * TAU + ay * 2)
                    idx = min(nr, idx + 2) if wave > 0.68 else max(1, idx - 1) if wave < -0.68 else max(1, idx)
                    chars.append(ramp[idx])
                    styles.append(self._cstyle(r, g, b))
                    continue

                glow, r = 0.0, 0.0
                if r2 < 2.5:
                    r = sqrt(r2)
                    if level > 0.03 and r < 1.3:
                        glow = level * 0.5 * (1 - (r - 1) / 0.3) ** 2
                    for ring_r, ring_a in rings:
                        d = (r - ring_r) / 0.05
                        if d * d < 9:
                            glow += ring_a * exp(-d * d)
                if glow > 0.05:
                    c = grad(stops, atan2(ys, xs) / TAU + 0.5 + drift)
                    k = min(1.0, glow)
                    chars.append(RING_GLYPHS[min(4, int(k * 5))])
                    styles.append(self._cstyle(c[0] * k + 30, c[1] * k + 30, c[2] * k + 40))
                    continue
                star = self._starmap.get((row, col)) if r2 > 2.5 else None
                if star:
                    tw = 0.5 + 0.5 * sin(t * 1.5 + star[0])
                    if tw > 0.25:
                        chars.append(star[1])
                        styles.append(self._cstyle(70 * tw, 95 * tw, 130 * tw))
                        continue
                chars.append(" ")
                styles.append(None)

            i = 0
            while i < w:
                j, s = i + 1, styles[i]
                while j < w and styles[j] is s:
                    j += 1
                out.append("".join(chars[i:j]), s)
                i = j
            if row < h - 1:
                out.append("\n")
        return out


class LiveSphereApp(App):
    """Full-screen interactive conversational assistant with 3D audio-reactive sphere."""

    CSS = """
    Screen { background: #080b10; color: #b9d7e8; }
    #frame { width: 100%; height: 100%; border: round #253746; padding: 0 1; }
    #title { height: 1; content-align: center middle; color: #7dd3fc; text-style: bold; }
    #sphere { width: 100%; height: 55%; overflow: hidden; }
    #hud { height: 2; content-align: center middle; color: #5eead4; border-top: solid #1e293b; }
    #log { width: 100%; height: 1fr; background: #0c1017; border: round #1e293b; padding: 0 1; }
    #footer { height: 1; content-align: center middle; color: #64748b; }
    """

    BINDINGS = [
        Binding("q,escape,ctrl+c", "quit", "Quit"),
        Binding("i,enter", "interrupt", "Interrupt"),
        Binding("space", "pause", "Pause Spin"),
        Binding("t", "next_theme", "Theme"),
        Binding("m", "mute", "Mute"),
    ]

    def __init__(self, session: LiveConversationSession, **kwargs):
        super().__init__(**kwargs)
        self.session = session
        self.current_assistant_buffer = ""
        self.filter_notice = ""
        self.is_muted = False

    def compose(self) -> ComposeResult:
        with Vertical(id="frame"):
            yield Static("◈  AUDIO AGENT LIVE VOICE  ◈", id="title")
            yield SphereCanvas(id="sphere")
            yield Static("", id="hud")
            yield RichLog(id="log", highlight=True, markup=True)
            yield Static("ENTER / I interrupt · SPACE pause spin · T theme · M mute · Q quit", id="footer")

    def on_mount(self) -> None:
        # Wire session callbacks to Textual UI thread
        self.session.on_state_change = self._handle_state_change
        self.session.on_user_transcript = self._handle_user_transcript
        self.session.on_assistant_token = self._handle_assistant_token
        self.session.on_assistant_turn_complete = self._handle_assistant_complete
        self.session.on_audio_level = self._handle_audio_level
        self.session.on_barge_in = self._handle_barge_in
        self.session.on_filter_ignored = self._handle_filter_ignored

        self.query_one("#log", RichLog).write("[bold cyan]System:[/bold cyan] Connected to live voice agent. Speak naturally.")
        self.update_hud()
        self.set_interval(0.2, self.update_hud)

    @property
    def canvas(self) -> SphereCanvas:
        return self.query_one(SphereCanvas)

    def _handle_state_change(self, new_state: SessionState) -> None:
        def update():
            c = self.canvas
            if new_state == SessionState.LISTENING:
                c.theme = "aurora"
            elif new_state == SessionState.USER_SPEAKING:
                c.theme = "spectrum"
            elif new_state == SessionState.PROCESSING:
                c.theme = "matrix"
            elif new_state == SessionState.AGENT_SPEAKING:
                c.theme = "ember"
            elif new_state == SessionState.INTERRUPTED:
                c.trigger_shockwave()
            self.update_hud()

        self.call_from_thread(update)

    def _handle_user_transcript(self, text: str) -> None:
        def write():
            log = self.query_one("#log", RichLog)
            log.write(f"\n[bold yellow]You:[/bold yellow] {text}")
            self.current_assistant_buffer = ""

        self.call_from_thread(write)

    def _handle_assistant_token(self, token: str) -> None:
        def write():
            self.current_assistant_buffer += token

        self.call_from_thread(write)

    def _handle_assistant_complete(self, full_text: str) -> None:
        def write():
            log = self.query_one("#log", RichLog)
            log.write(f"[bold green]Agent:[/bold green] {full_text}")
            self.current_assistant_buffer = ""

        self.call_from_thread(write)

    def _handle_audio_level(self, level: float, bands: list[float], is_output: bool) -> None:
        if not self.is_muted:
            self.call_from_thread(self.canvas.push_audio, level, bands)

    def _handle_barge_in(self) -> None:
        def write():
            log = self.query_one("#log", RichLog)
            log.write("[bold red]⚡ [Barge-In: Playback stopped, listening to you...][/bold red]")
            self.canvas.trigger_shockwave()

        self.call_from_thread(write)

    def _handle_filter_ignored(self, label: str) -> None:
        def write():
            self.filter_notice = f"Filter: {label} ignored"
            log = self.query_one("#log", RichLog)
            log.write(f"[dim italic][Acoustic Filter: {label} ignored][/dim italic]")

        self.call_from_thread(write)

    def update_hud(self) -> None:
        c = self.canvas
        st = self.session.state.value
        state_colors = {
            "IDLE": "white",
            "LISTENING": "cyan",
            "USER_SPEAKING": "bright_yellow",
            "PROCESSING": "bright_blue",
            "AGENT_SPEAKING": "bright_green",
            "INTERRUPTED": "bright_red",
        }
        color = state_colors.get(st, "white")
        meter = "".join(SPARK[min(7, int(b * 8))] for b in c.bands)

        notice = f" · [bold magenta]{self.filter_notice}[/bold magenta]" if self.filter_notice else ""
        muted_str = " · [red][MUTED][/red]" if self.is_muted else ""
        hud_text = (
            f"State: [bold {color}]● {st}[/bold {color}] · Theme: {c.theme} "
            f"· Level: {meter} · FPS: {c.fps:.0f}{notice}{muted_str}"
        )
        self.query_one("#hud", Static).update(hud_text)

    def action_pause(self) -> None:
        self.canvas.running = not self.canvas.running

    def action_next_theme(self) -> None:
        c = self.canvas
        c.theme = THEME_NAMES[(THEME_NAMES.index(c.theme) + 1) % len(THEME_NAMES)]

    def action_interrupt(self) -> None:
        """Manually interrupts assistant speech and returns to listening."""
        self.session.interrupt()

    def action_mute(self) -> None:
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.canvas.push_audio(0.0)

