# """
# Animated ASCII sphere for the terminal (v2).

# Run:
#     uv add textual rich
#     uv run python sphere.py

# Optional, for real microphone reactivity:
#     uv add numpy sounddevice

# Keys:
#     q / Esc / Ctrl+C : quit            Space   : pause / resume spin
#     t                : next theme      1-4     : pick theme
#     g                : next glyph set  a       : audio source (idle / demo / mic)
#     + / -            : spin speed      mouse   : drag the sphere to rotate it

# Hooking it to your TTS: call SphereCanvas.push_audio(level, bands) with a 0..1
# loudness and an optional list of 16 band energies; it switches to live mode.
# """

# from __future__ import annotations

# import math
# import random
# import time

# from rich.style import Style
# from rich.text import Text
# from textual import events
# from textual.app import App, ComposeResult
# from textual.containers import Vertical
# from textual.widgets import Static

# try:  # optional: real microphone input
#     import numpy as np
#     import sounddevice as sd
# except Exception:  # pragma: no cover
#     np = sd = None

# TAU = math.tau
# BANDS = 16
# # Band edges for a 512-sample FFT (257 bins), log-spaced, strictly increasing.
# EDGES = [2]
# for _i in range(1, BANDS + 1):
#     EDGES.append(max(EDGES[-1] + 1, int(2 * 128 ** (_i / BANDS))))

# # Glyph sets, ordered sparse -> dense (except "classic", the original mix).
# RAMPS = {
#     "ramp": " .'`,:;-~=+*^!?oOxX#%@",
#     "classic": " .,:-~=+*^!|/\\()[]{}<>?abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$%&@",
#     "blocks": " ░▒▓█",
#     "dots": " ·∙•●",
# }
# # Color stops; last stop repeats the first so the gradient wraps smoothly.
# THEMES = {
#     "aurora": [(70, 244, 255), (40, 153, 255), (74, 75, 255), (70, 244, 255)],
#     "spectrum": [(44, 235, 224), (93, 230, 255), (255, 207, 93), (255, 115, 125),
#                  (222, 112, 255), (105, 123, 255), (44, 235, 224)],
#     "ember": [(255, 80, 40), (255, 150, 40), (255, 214, 102), (255, 90, 90), (255, 80, 40)],
#     "matrix": [(20, 200, 90), (120, 255, 160), (10, 140, 70), (20, 200, 90)],
# }
# THEME_NAMES = list(THEMES)
# RING_GLYPHS = "·∙:+*"
# SPARK = "▁▂▃▄▅▆▇█"


# def mix(a, b, t):
#     t = max(0.0, min(1.0, t))
#     return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


# def grad(stops, pos):
#     f = (pos % 1.0) * (len(stops) - 1)
#     i = int(f)
#     return mix(stops[i], stops[i + 1], f - i)


# def demo_audio(t: float) -> tuple[float, list[float]]:
#     """Speech-like envelope: phrases made of syllable bursts, with pauses."""
#     gate = max(0.0, math.sin(t * 0.45)) ** 0.6
#     syll = (0.5 + 0.5 * math.sin(t * (7 + math.sin(t * 0.7) * 2) + math.sin(t * 2.3) * 2)) ** 1.3
#     lvl = gate * syll
#     bands = [lvl * (0.35 + 0.65 * abs(math.sin(t * (1.7 + i * 0.37) + i))) * (1 - i / BANDS * 0.55)
#              for i in range(BANDS)]
#     return lvl, bands


# class Mic:
#     """Reads the default microphone and exposes level + 16 log-spaced bands."""

#     def __init__(self) -> None:
#         self.level = 0.0
#         self.bands = [0.0] * BANDS
#         self.stream = None
#         self.error = ""

#     def start(self) -> bool:
#         if sd is None:
#             self.error = "pip install numpy sounddevice"
#             return False
#         if self.stream:
#             return True
#         try:
#             self.stream = sd.InputStream(channels=1, samplerate=16000, blocksize=512, callback=self._cb)
#             self.stream.start()
#             self.error = ""
#             return True
#         except Exception as exc:
#             self.stream, self.error = None, str(exc)[:40]
#             return False

#     def stop(self) -> None:
#         if self.stream:
#             try:
#                 self.stream.stop()
#                 self.stream.close()
#             except Exception:
#                 pass
#             self.stream = None

#     def _cb(self, data, frames, time_info, status) -> None:
#         x = data[:, 0]
#         spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
#         self.bands = [min(1.0, float(spec[EDGES[i]:EDGES[i + 1]].mean() / 6.0) ** 0.6) for i in range(BANDS)]
#         self.level = min(1.0, float(np.sqrt(np.mean(x * x))) * 12) ** 0.7


# class SphereCanvas(Static):
#     """Rotating, shaded sphere with halo, shockwave rings, stars and audio reactivity."""

#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.ay, self.ax_off = 0.0, 0.0
#         self.spin, self.running = 0.7, True
#         self.theme, self.ramp = "aurora", "ramp"
#         self.mode = "idle"
#         self.mic = Mic()
#         self.level, self.bands = 0.0, [0.0] * BANDS
#         self.rings: list[list[float]] = []
#         self.fps = 30.0
#         self._tstart = self._tprev = time.monotonic()
#         self._extaudio = (0.0, [0.0] * BANDS)
#         self._dragpos = None
#         self._grid = (0, 0)
#         self._starmap: dict = {}
#         self._scache: dict = {}

#     # ----- control -----
#     def on_mount(self) -> None:
#         self.set_interval(1 / 30, self.tick)

#     def on_unmount(self) -> None:
#         self.mic.stop()

#     def set_mode(self, mode: str) -> None:
#         if mode == "mic" and not self.mic.start():
#             mode = "idle"
#         if mode != "mic":
#             self.mic.stop()
#         self.mode = mode

#     def cycle_mode(self) -> None:
#         modes = ["idle", "demo"] + (["mic"] if sd is not None else [])
#         self.set_mode(modes[(modes.index(self.mode) + 1) % len(modes)] if self.mode in modes else "idle")

#     def push_audio(self, level: float, bands: list[float] | None = None) -> None:
#         self.mode, self._extaudio = "live", (level, bands or [level] * BANDS)

#     # ----- mouse drag to rotate -----
#     def on_mouse_down(self, e: events.MouseDown) -> None:
#         self._dragpos = (e.x, e.y)
#         self.capture_mouse()

#     def on_mouse_move(self, e: events.MouseMove) -> None:
#         if self._dragpos:
#             self.ay += (e.x - self._dragpos[0]) * 0.09
#             self.ax_off = max(-1.0, min(1.0, self.ax_off + (e.y - self._dragpos[1]) * 0.07))
#             self._dragpos = (e.x, e.y)

#     def on_mouse_up(self, e: events.MouseUp) -> None:
#         self._dragpos = None
#         self.release_mouse()

#     # ----- animation -----
#     def tick(self) -> None:
#         now = time.monotonic()
#         dt, self._tprev = min(0.1, now - self._tprev), now
#         self.fps += (1 / max(dt, 1e-3) - self.fps) * 0.1
#         t = now - self._tstart
#         if self.mode == "demo":
#             lv, bs = demo_audio(t)
#         elif self.mode == "mic":
#             lv, bs = self.mic.level, self.mic.bands
#         elif self.mode == "live":
#             lv, bs = self._extaudio
#         else:
#             lv, bs = 0.0, [0.0] * BANDS
#         ease = lambda cur, tgt: cur + (tgt - cur) * (0.55 if tgt > cur else 0.12)  # fast attack, slow release
#         if lv - self.level > 0.18 and len(self.rings) < 4:
#             self.rings.append([1.0, 0.8])
#         self.level = ease(self.level, lv)
#         self.bands = [ease(c, b) for c, b in zip(self.bands, bs)]
#         self.rings = [[r + dt * 0.9, a * 0.93] for r, a in self.rings if a > 0.04 and r < 1.7]
#         if self.running and not self._dragpos:
#             self.ay += self.spin * dt * (1 + self.level * 2.5)
#         self.refresh()

#     def _cstyle(self, r: float, g: float, b: float) -> Style:
#         k = (min(31, int(r) >> 3), min(31, int(g) >> 3), min(31, int(b) >> 3))
#         s = self._scache.get(k)
#         if s is None:
#             s = self._scache[k] = Style(color=f"#{(k[0] << 3) | 4:02x}{(k[1] << 3) | 4:02x}{(k[2] << 3) | 4:02x}")
#         return s

#     def render(self) -> Text:
#         w, h = max(30, self.size.width), max(12, self.size.height)
#         if (w, h) != self._grid:
#             self._grid = (w, h)
#             self._starmap = {(random.randrange(h), random.randrange(w)): (random.random() * TAU, random.choice(".·+"))
#                            for _ in range(max(10, w * h // 45))}
#         t, level, bands, rings = time.monotonic() - self._tstart, self.level, self.bands, self.rings
#         stops, ramp = THEMES[self.theme], RAMPS[self.ramp]
#         nr = len(ramp) - 1
#         cx, cy = (w - 1) / 2, (h - 1) / 2
#         radius = min(w * 0.36, h * 0.8) * (1 + 0.06 * level + 0.012 * math.sin(t * 1.3))
#         inv, aspect = 1 / max(radius, 1), 0.48          # cells are taller than wide
#         ay, ax = self.ay, 0.22 + self.ax_off + math.sin(t * 0.27) * 0.16
#         cyaw, syaw, cp, sp = math.cos(ay), math.sin(ay), math.cos(ax), math.sin(ax)
#         LX, LY, LZ = -0.42, -0.5, 0.755                  # light: upper-left, toward viewer
#         drift, sin, cos, sqrt, atan2, acos, exp = t * 0.03, math.sin, math.cos, math.sqrt, math.atan2, math.acos, math.exp
#         out = Text(no_wrap=True)

#         for row in range(h):
#             ys = (row - cy) * inv / aspect
#             ys2 = ys * ys
#             chars: list[str] = []
#             styles: list[Style | None] = []
#             for col in range(w):
#                 xs = (col - cx) * inv
#                 r2 = xs * xs + ys2
#                 if r2 <= 1.0:
#                     z = sqrt(1.0 - r2)
#                     x2, z2 = xs * cyaw - z * syaw, xs * syaw + z * cyaw
#                     y2, z3 = ys * cp - z2 * sp, ys * sp + z2 * cp
#                     lon = (atan2(x2, z3) / TAU + 0.5) % 1.0
#                     lat = acos(max(-1.0, min(1.0, y2))) / math.pi
#                     e = bands[min(BANDS - 1, int(lat * BANDS))]       # this latitude's audio band
#                     diff = max(0.0, xs * LX + ys * LY + z * LZ)
#                     spec = diff ** 20
#                     shade = (0.16 + 0.62 * diff + 0.30 * (1 - z) ** 2.2 + 0.55 * spec) * (0.8 + 0.45 * e)
#                     base = grad(stops, lon + drift)
#                     lum = 0.25 + 0.75 * min(1.0, shade)
#                     white = min(1.0, spec * 0.9 + level * 0.18 * diff + e * 0.25)
#                     r, g, b = (c * lum + (255 - c * lum) * white for c in base)
#                     idx = int(max(0.0, min(1.0, shade * 0.95 + e * 0.2)) * nr)
#                     wave = sin((lon * 9 + lat * 5) * TAU + ay * 2)      # texture that rotates with the surface
#                     idx = min(nr, idx + 2) if wave > 0.68 else max(1, idx - 1) if wave < -0.68 else max(1, idx)
#                     chars.append(ramp[idx])
#                     styles.append(self._cstyle(r, g, b))
#                     continue

#                 glow, r = 0.0, 0.0
#                 if r2 < 2.72:
#                     r = sqrt(r2)
#                     if level > 0.03 and r < 1.3:
#                         glow = level * 0.5 * (1 - (r - 1) / 0.3) ** 2
#                     for ring_r, ring_a in rings:
#                         d = (r - ring_r) / 0.05
#                         if d * d < 9:
#                             glow += ring_a * exp(-d * d)
#                 if glow > 0.05:
#                     c = grad(stops, atan2(ys, xs) / TAU + 0.5 + drift)
#                     k = min(1.0, glow)
#                     chars.append(RING_GLYPHS[min(4, int(k * 5))])
#                     styles.append(self._cstyle(c[0] * k + 30, c[1] * k + 30, c[2] * k + 40))
#                     continue
#                 star = self._starmap.get((row, col)) if r2 > 2.7 else None
#                 if star:
#                     tw = 0.5 + 0.5 * sin(t * 1.5 + star[0])
#                     if tw > 0.25:
#                         chars.append(star[1])
#                         styles.append(self._cstyle(70 * tw, 95 * tw, 130 * tw))
#                         continue
#                 chars.append(" ")
#                 styles.append(None)

#             i = 0
#             while i < w:                                  # merge equal-style runs: far fewer segments
#                 j, s = i + 1, styles[i]
#                 while j < w and styles[j] is s:
#                     j += 1
#                 out.append("".join(chars[i:j]), s)
#                 i = j
#             if row < h - 1:
#                 out.append("\n")
#         return out


# class SphereApp(App):
#     CSS = """
#     Screen { background: #080b10; color: #b9d7e8; }
#     #frame { width: 100%; height: 100%; border: round #253746; padding: 0 1; }
#     #title { height: 1; content-align: center middle; color: #7dd3fc; text-style: bold; }
#     #sphere { width: 100%; height: 1fr; overflow: hidden; }
#     #status { height: 1; content-align: center middle; color: #5eead4; }
#     #footer { height: 1; content-align: center middle; color: #8297aa; }
#     """
#     BINDINGS = [
#         ("q,escape,ctrl+c", "quit", "Quit"),
#         ("space", "pause", "Pause"),
#         ("t", "next_theme", "Theme"),
#         ("1", "theme(0)", "aurora"), ("2", "theme(1)", "spectrum"),
#         ("3", "theme(2)", "ember"), ("4", "theme(3)", "matrix"),
#         ("g", "next_glyphs", "Glyphs"),
#         ("a", "next_audio", "Audio"),
#         ("plus,equals_sign", "speed(0.2)", "Faster"),
#         ("minus", "speed(-0.2)", "Slower"),
#     ]

#     def compose(self) -> ComposeResult:
#         with Vertical(id="frame"):
#             yield Static("◈  SPHERE  ◈", id="title")
#             yield SphereCanvas(id="sphere")
#             yield Static("", id="status")
#             yield Static("SPACE pause · T theme · G glyphs · A audio · +/- speed · drag to rotate · Q quit", id="footer")

#     def on_mount(self) -> None:
#         self.set_interval(0.2, self.update_status)

#     @property
#     def canvas(self) -> SphereCanvas:
#         return self.query_one(SphereCanvas)

#     def update_status(self) -> None:
#         c = self.canvas
#         meter = "".join(SPARK[min(7, int(b * 8))] for b in c.bands)
#         err = f" ({c.mic.error})" if c.mode == "idle" and c.mic.error else ""
#         state = "paused" if not c.running else f"spin x{c.spin / 0.7:.1f}"
#         self.query_one("#status", Static).update(
#             f"{c.theme} · {c.ramp} · audio: {c.mode}{err} · {state} · {c.fps:.0f} fps   {meter}")

#     def action_pause(self) -> None:
#         self.canvas.running = not self.canvas.running

#     def action_theme(self, i: int) -> None:
#         self.canvas.theme = THEME_NAMES[i]

#     def action_next_theme(self) -> None:
#         c = self.canvas
#         c.theme = THEME_NAMES[(THEME_NAMES.index(c.theme) + 1) % len(THEME_NAMES)]

#     def action_next_glyphs(self) -> None:
#         c, names = self.canvas, list(RAMPS)
#         c.ramp = names[(names.index(c.ramp) + 1) % len(names)]

#     def action_next_audio(self) -> None:
#         self.canvas.cycle_mode()

#     def action_speed(self, delta: float) -> None:
#         self.canvas.spin = max(0.0, min(4.0, self.canvas.spin + delta))


# if __name__ == "__main__":
#     SphereApp().run()





"""
Animated ASCII sphere for the terminal (v2).

Run:
    uv add textual rich
    uv run python sphere.py

Optional, for real microphone reactivity:
    uv add numpy sounddevice

Keys:
    q / Esc / Ctrl+C : quit            Space   : pause / resume spin
    t                : next theme      1-4     : pick theme
    g                : next glyph set  a       : audio source (idle / demo / mic)
    d                : density         + / -   : spin speed
    m                : mouse drag      b       : background (stars, halo, rings)
    h                : show / hide the status bar
    Mouse drag-rotate, background and status bar are all OFF by default.

Hooking it to your TTS: call SphereCanvas.push_audio(level, bands) with a 0..1
loudness and an optional list of 16 band energies; it switches to live mode.
"""

from __future__ import annotations

import math
import random
import time

from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import Static

try:  # optional: real microphone input
    import numpy as np
    import sounddevice as sd
except Exception:  # pragma: no cover
    np = sd = None

TAU = math.tau
BANDS = 16
# Band edges for a 512-sample FFT (257 bins), log-spaced, strictly increasing.
EDGES = [2]
for _i in range(1, BANDS + 1):
    EDGES.append(max(EDGES[-1] + 1, int(2 * 128 ** (_i / BANDS))))

# Glyph sets, ordered sparse -> dense (except "classic", the original mix).
RAMPS = {
    "ramp": " .'`,:;-~=+*^!?oOxX#%@",
    "classic": " .,:-~=+*^!|/\\()[]{}<>?abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$%&@",
    "blocks": " ░▒▓█",
    "dots": " ·∙•●",
}
# Color stops; last stop repeats the first so the gradient wraps smoothly.
THEMES = {
    "aurora": [(70, 244, 255), (40, 153, 255), (74, 75, 255), (70, 244, 255)],
    "spectrum": [(44, 235, 224), (93, 230, 255), (255, 207, 93), (255, 115, 125),
                 (222, 112, 255), (105, 123, 255), (44, 235, 224)],
    "ember": [(255, 80, 40), (255, 150, 40), (255, 214, 102), (255, 90, 90), (255, 80, 40)],
    "matrix": [(20, 200, 90), (120, 255, 160), (10, 140, 70), (20, 200, 90)],
}
THEME_NAMES = list(THEMES)
RING_GLYPHS = "·∙:+*"
SPARK = "▁▂▃▄▅▆▇█"


def mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def grad(stops, pos):
    f = (pos % 1.0) * (len(stops) - 1)
    i = int(f)
    return mix(stops[i], stops[i + 1], f - i)


def demo_audio(t: float) -> tuple[float, list[float]]:
    """Speech-like envelope: phrases made of syllable bursts, with pauses."""
    gate = max(0.0, math.sin(t * 0.45)) ** 0.6
    syll = (0.5 + 0.5 * math.sin(t * (7 + math.sin(t * 0.7) * 2) + math.sin(t * 2.3) * 2)) ** 1.3
    lvl = gate * syll
    bands = [lvl * (0.35 + 0.65 * abs(math.sin(t * (1.7 + i * 0.37) + i))) * (1 - i / BANDS * 0.55)
             for i in range(BANDS)]
    return lvl, bands


class Mic:
    """Reads the default microphone and exposes level + 16 log-spaced bands."""

    def __init__(self) -> None:
        self.level = 0.0
        self.bands = [0.0] * BANDS
        self.stream = None
        self.error = ""

    def start(self) -> bool:
        if sd is None:
            self.error = "pip install numpy sounddevice"
            return False
        if self.stream:
            return True
        try:
            self.stream = sd.InputStream(channels=1, samplerate=16000, blocksize=512, callback=self._cb)
            self.stream.start()
            self.error = ""
            return True
        except Exception as exc:
            self.stream, self.error = None, str(exc)[:40]
            return False

    def stop(self) -> None:
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _cb(self, data, frames, time_info, status) -> None:
        x = data[:, 0]
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        self.bands = [min(1.0, float(spec[EDGES[i]:EDGES[i + 1]].mean() / 6.0) ** 0.6) for i in range(BANDS)]
        self.level = min(1.0, float(np.sqrt(np.mean(x * x))) * 12) ** 0.7


class SphereCanvas(Static):
    """Rotating, shaded sphere with halo, shockwave rings, stars and audio reactivity."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ay, self.ax_off = 0.0, 0.0
        self.spin, self.running = 0.7, True
        self.theme, self.ramp = "aurora", "ramp"
        self.density = 1          # 0 sparse, 1 normal, 2 dense
        self.drag_enabled = False
        self.bg = False           # stars, halo and shockwave rings
        self.mode = "idle"
        self.mic = Mic()
        self.level, self.bands = 0.0, [0.0] * BANDS
        self.rings: list[list[float]] = []
        self.fps = 30.0
        self._tstart = self._tprev = time.monotonic()
        self._extaudio = (0.0, [0.0] * BANDS)
        self._dragpos = None
        self._grid = (0, 0)
        self._starmap: dict = {}
        self._scache: dict = {}

    # ----- control -----
    def on_mount(self) -> None:
        self.set_interval(1 / 30, self.tick)

    def on_unmount(self) -> None:
        self.mic.stop()

    def set_mode(self, mode: str) -> None:
        if mode == "mic" and not self.mic.start():
            mode = "idle"
        if mode != "mic":
            self.mic.stop()
        self.mode = mode

    def cycle_mode(self) -> None:
        modes = ["idle", "demo"] + (["mic"] if sd is not None else [])
        self.set_mode(modes[(modes.index(self.mode) + 1) % len(modes)] if self.mode in modes else "idle")

    def push_audio(self, level: float, bands: list[float] | None = None) -> None:
        self.mode, self._extaudio = "live", (level, bands or [level] * BANDS)

    # ----- mouse drag to rotate -----
    def on_mouse_down(self, e: events.MouseDown) -> None:
        if not self.drag_enabled:
            return
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

    # ----- animation -----
    def tick(self) -> None:
        now = time.monotonic()
        dt, self._tprev = min(0.1, now - self._tprev), now
        self.fps += (1 / max(dt, 1e-3) - self.fps) * 0.1
        t = now - self._tstart
        if self.mode == "demo":
            lv, bs = demo_audio(t)
        elif self.mode == "mic":
            lv, bs = self.mic.level, self.mic.bands
        elif self.mode == "live":
            lv, bs = self._extaudio
        else:
            lv, bs = 0.0, [0.0] * BANDS
        ease = lambda cur, tgt: cur + (tgt - cur) * (0.55 if tgt > cur else 0.12)  # fast attack, slow release
        if lv - self.level > 0.18 and len(self.rings) < 4:
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
            s = self._scache[k] = Style(color=f"#{(k[0] << 3) | 4:02x}{(k[1] << 3) | 4:02x}{(k[2] << 3) | 4:02x}")
        return s

    def render(self) -> Text:
        w, h = max(30, self.size.width), max(12, self.size.height)
        if (w, h) != self._grid:
            self._grid = (w, h)
            self._starmap = {(random.randrange(h), random.randrange(w)): (random.random() * TAU, random.choice(".·+"))
                           for _ in range(max(10, w * h // 45))}
        t, level, bands, rings = time.monotonic() - self._tstart, self.level, self.bands, self.rings
        stops, ramp = THEMES[self.theme], RAMPS[self.ramp]
        nr = len(ramp) - 1
        dens_off, min_idx = ((-0.12, 0), (0.0, 1), (0.25, max(2, nr // 4)))[self.density]
        lum_floor = 0.25 + 0.1 * self.density
        cx, cy = (w - 1) / 2, (h - 1) / 2
        radius = min(w * 0.42, h * 0.92) * (1 + 0.06 * level + 0.012 * math.sin(t * 1.3))
        inv, aspect = 1 / max(radius, 1), 0.48          # cells are taller than wide
        ay, ax = self.ay, 0.22 + self.ax_off + math.sin(t * 0.27) * 0.16
        cyaw, syaw, cp, sp = math.cos(ay), math.sin(ay), math.cos(ax), math.sin(ax)
        LX, LY, LZ = -0.42, -0.5, 0.755                  # light: upper-left, toward viewer
        drift, sin, cos, sqrt, atan2, acos, exp = t * 0.03, math.sin, math.cos, math.sqrt, math.atan2, math.acos, math.exp
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
                    e = bands[min(BANDS - 1, int(lat * BANDS))]       # this latitude's audio band
                    diff = max(0.0, xs * LX + ys * LY + z * LZ)
                    shade = (0.30 + 0.62 * diff) * (0.85 + 0.35 * e)     # soft diffuse only: no shine
                    base = grad(stops, lon + drift)
                    lum = min(1.0, lum_floor + (1 - lum_floor) * shade)
                    r, g, b = (c * lum for c in base)
                    idx = int(max(0.0, min(1.0, shade * 0.95 + e * 0.2 + dens_off)) * nr)
                    wave = sin((lon * 9 + lat * 5) * TAU + ay * 2)      # texture that rotates with the surface
                    idx = min(nr, idx + 2) if wave > 0.68 else max(min_idx, idx - 1) if wave < -0.68 else max(min_idx, idx)
                    chars.append(ramp[idx])
                    styles.append(self._cstyle(r, g, b))
                    continue

                glow, r = 0.0, 0.0
                if self.bg and r2 < 2.72:
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
                star = self._starmap.get((row, col)) if self.bg and r2 > 2.7 else None
                if star:
                    tw = 0.5 + 0.5 * sin(t * 1.5 + star[0])
                    if tw > 0.25:
                        chars.append(star[1])
                        styles.append(self._cstyle(70 * tw, 95 * tw, 130 * tw))
                        continue
                chars.append(" ")
                styles.append(None)

            i = 0
            while i < w:                                  # merge equal-style runs: far fewer segments
                j, s = i + 1, styles[i]
                while j < w and styles[j] is s:
                    j += 1
                out.append("".join(chars[i:j]), s)
                i = j
            if row < h - 1:
                out.append("\n")
        return out


class SphereApp(App):
    CSS = """
    Screen { background: #080b10; color: #b9d7e8; }
    #sphere { width: 100%; height: 1fr; overflow: hidden; }
    #hud { height: 1; display: none; content-align: center middle; color: #5eead4; background: #0c1219; }
    """
    BINDINGS = [
        ("q,escape,ctrl+c", "quit", "Quit"),
        ("space", "pause", "Pause"),
        ("t", "next_theme", "Theme"),
        ("1", "theme(0)", "aurora"), ("2", "theme(1)", "spectrum"),
        ("3", "theme(2)", "ember"), ("4", "theme(3)", "matrix"),
        ("g", "next_glyphs", "Glyphs"),
        ("a", "next_audio", "Audio"),
        ("d", "next_density", "Density"),
        ("m", "toggle_drag", "Mouse"),
        ("b", "toggle_bg", "Background"),
        ("h", "toggle_hud", "HUD"),
        ("plus,equals_sign", "speed(0.2)", "Faster"),
        ("minus", "speed(-0.2)", "Slower"),
    ]

    def compose(self) -> ComposeResult:
        yield SphereCanvas(id="sphere")
        yield Static("", id="hud")

    def on_mount(self) -> None:
        self.set_interval(0.2, self.update_status)

    @property
    def canvas(self) -> SphereCanvas:
        return self.query_one(SphereCanvas)

    def update_status(self) -> None:
        c = self.canvas
        meter = "".join(SPARK[min(7, int(b * 8))] for b in c.bands)
        onoff = lambda v: "on" if v else "off"
        err = f" ({c.mic.error})" if c.mode == "idle" and c.mic.error else ""
        self.query_one("#hud", Static).update(
            f"{c.theme} · {c.ramp} · density {c.density + 1}/3 · audio {c.mode}{err} · drag {onoff(c.drag_enabled)}"
            f" · bg {onoff(c.bg)} · {c.fps:.0f}fps {meter}   T G D A M B H Q")

    def action_next_density(self) -> None:
        self.canvas.density = (self.canvas.density + 1) % 3

    def action_toggle_drag(self) -> None:
        self.canvas.drag_enabled = not self.canvas.drag_enabled

    def action_toggle_bg(self) -> None:
        self.canvas.bg = not self.canvas.bg

    def action_toggle_hud(self) -> None:
        hud = self.query_one("#hud", Static)
        hud.display = not hud.display

    def action_pause(self) -> None:
        self.canvas.running = not self.canvas.running

    def action_theme(self, i: int) -> None:
        self.canvas.theme = THEME_NAMES[i]

    def action_next_theme(self) -> None:
        c = self.canvas
        c.theme = THEME_NAMES[(THEME_NAMES.index(c.theme) + 1) % len(THEME_NAMES)]

    def action_next_glyphs(self) -> None:
        c, names = self.canvas, list(RAMPS)
        c.ramp = names[(names.index(c.ramp) + 1) % len(names)]

    def action_next_audio(self) -> None:
        self.canvas.cycle_mode()

    def action_speed(self, delta: float) -> None:
        self.canvas.spin = max(0.0, min(4.0, self.canvas.spin + delta))


if __name__ == "__main__":
    SphereApp().run()