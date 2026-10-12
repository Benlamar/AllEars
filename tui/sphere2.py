"""
Music-reactive "planet": a tiny fixed core + translucent layers that ripple outward.

Core   : stays at the exact centre. Quiet = small dense glowing nucleus.
         Louder = brighter, whiter, tighter... down to a single intense dot.
         It never grows - it only concentrates.
Layers : compact and close together in silence. Each beat sends a wave
         outward through the layers (inner layers react first, outer ones echo);
         louder music spreads them farther (more volume), softer music
         lets them contract slowly. Always fits inside the window.

Particles are drawn with Braille sub-dots (2x4 per cell) so they look like grains.

Install:
    uv add textual rich numpy soundcard
    uv add sounddevice          # only needed for the macOS fallback

Run:
    uv run python sphere.py

System-audio capture (not the mic):
    Windows : works out of the box (WASAPI loopback of default speakers).
    Linux   : PulseAudio / PipeWire monitor of the default output.
    macOS   : install BlackHole, use a Multi-Output device as your output.

Keys:
    q / Esc : quit            d : toggle demo / system audio
    g : toggle GPU / CPU      f : show real-time FPS
    t : next theme (or 1-4)   Space : pause / resume spin
    Up / Down : reactivity    Left / Right : spin speed
"""

from __future__ import annotations

import asyncio
import collections
import ctypes
import math
import threading
import time
import warnings

# Suppress recording buffer warnings so they don't stutter threads
warnings.filterwarnings("ignore")

# Request 1ms timer resolution on Windows for locked, jitter-free 60 FPS
try:
    if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "winmm"):
        ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

import numpy as np
from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Static

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if HAS_CUDA else "CUDA Device"
except Exception:
    torch = None
    HAS_TORCH = False
    HAS_CUDA = False
    GPU_NAME = ""

try:
    import soundcard as sc
except Exception:  # pragma: no cover
    sc = None
try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None

TAU = math.tau
BANDS = 16
SR = 44100
BLOCK = 512

# log-spaced FFT bin edges (257 bins for a 512 block), strictly increasing
EDGES = [1]
for _i in range(1, BANDS + 1):
    EDGES.append(max(EDGES[-1] + 1, int(round(240 ** (_i / BANDS)))))

THEMES = {
    "aurora": [(70, 244, 255), (40, 153, 255), (74, 75, 255), (70, 244, 255)],
    "spectrum": [(44, 235, 224), (93, 230, 255), (255, 207, 93), (255, 115, 125),
                 (222, 112, 255), (105, 123, 255), (44, 235, 224)],
    "ember": [(255, 80, 40), (255, 150, 40), (255, 214, 102), (255, 90, 90), (255, 80, 40)],
    "matrix": [(20, 200, 90), (120, 255, 160), (10, 140, 70), (20, 200, 90)],
}
THEME_NAMES = list(THEMES)


def build_lut(stops) -> np.ndarray:
    st = np.array(stops, dtype=float)
    xs, sx = np.linspace(0, 1, 256), np.linspace(0, 1, len(st))
    return np.stack([np.interp(xs, sx, st[:, c]) for c in range(3)], axis=1)


LUTS = {name: build_lut(s) for name, s in THEMES.items()}

# Braille dot bit for sub-pixel (row 0..3, col 0..1)
BITS = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]], dtype=np.int64)


# --------------------------------------------------------------------------- #
# System audio (loopback) capture + loudness / beat analysis
# --------------------------------------------------------------------------- #
class SystemAudio:
    def __init__(self) -> None:
        self.level = 0.0                  # adaptive 0..1 loudness
        self.bands = np.zeros(BANDS)
        self.beat_id = 0                  # increments on every detected beat
        self.beat_str = 0.0               # strength of the latest beat 0..1
        self.sens = 0.0                   # gain in dB (Up/Down)
        self.density = 0.0                # 0..1 spectral density (sparse solo -> full orchestra)
        self.intimacy = 0.0               # 0..1 delicate solo vocal / instrument presence
        self.tension = 0.0                # 0..1 suspense & build-up anticipation
        self.drop_burst = 0.0             # 0..1 sudden drop impact discharge
        self.error = ""
        self.source = ""
        self.ok = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak, self._floor = -60.0, -60.0
        self._bmax = np.full(BANDS, 1e-6)
        self._onhist: collections.deque = collections.deque(maxlen=40)
        self._spec_prev: np.ndarray | None = None
        self._bass_prev = 0.0
        self._lastbeat = 0.0
        self._lvl_long = 0.0
        self._prev_intimacy = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.error, self.ok = "", False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _open_soundcard(self):
        if sc is None:
            raise RuntimeError("soundcard not installed")
        spk = sc.default_speaker()
        mic = sc.get_microphone(spk.id, include_loopback=True)
        rec = mic.recorder(samplerate=SR, channels=2, blocksize=BLOCK)
        rec.__enter__()
        self.source = f"loopback: {spk.name}"
        return (lambda: rec.record(numframes=BLOCK)), (lambda: rec.__exit__(None, None, None))

    def _open_sounddevice(self):
        if sd is None:
            raise RuntimeError("sounddevice not installed")
        keys = ("blackhole", "loopback", "soundflower", "monitor", "stereo mix", "what u hear")
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and any(k in d["name"].lower() for k in keys):
                stream = sd.InputStream(device=i, channels=min(2, d["max_input_channels"]),
                                        samplerate=SR, blocksize=BLOCK)
                stream.start()
                self.source = f"device: {d['name']}"
                return (lambda: stream.read(BLOCK)[0]), (lambda: (stream.stop(), stream.close()))
        raise RuntimeError("no loopback/monitor input device found")

    def _run(self) -> None:
        reader = closer = None
        errs = []
        for opener in (self._open_soundcard, self._open_sounddevice):
            try:
                reader, closer = opener()
                break
            except Exception as exc:
                errs.append(str(exc)[:60])
        if reader is None:
            self.error = " / ".join(errs) or "no audio source"
            return
        self.ok = True
        try:
            while not self._stop.is_set():
                data = np.asarray(reader(), dtype=np.float32)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if len(data) >= 256:
                    self._analyse(data)
        except Exception as exc:
            self.error, self.ok = str(exc)[:60], False
        finally:
            try:
                closer()
            except Exception:
                pass

    def _analyse(self, x: np.ndarray) -> None:
        n = len(x)
        dt = n / SR
        rms = float(np.sqrt(np.mean(x * x)))
        db = 20.0 * math.log10(rms + 1e-9)

        # --- adaptive loudness: tracks floor & peak dynamically across any OS volume level ---
        if db > self._peak:
            self._peak = db
        else:
            self._peak = max(self._peak - 5.0 * dt, db)            # peak decays 5 dB/s
        if db < self._floor:
            self._floor = db
        else:
            self._floor = min(self._floor + 3.0 * dt, self._peak - 6.0)  # floor rises 3 dB/s
        self._floor = max(self._floor, -85.0)
        span = max(self._peak - self._floor, 8.0)
        norm = 0.0 if db < self._floor + 1.2 else max(0.0, (db - self._floor) / span)
        gain = 10 ** (self.sens / 20.0)
        self.level = float(np.clip((norm * gain) ** 1.05, 0.0, 1.0))

        # --- spectrum bands, normalized against recent max with rich dynamic range ---
        spec = np.abs(np.fft.rfft(x[:BLOCK] * np.hanning(min(n, BLOCK)))) / (n / 4)
        spec = np.pad(spec, (0, max(0, EDGES[-1] + 1 - len(spec))))
        b = np.array([spec[EDGES[i]:EDGES[i + 1]].mean() for i in range(BANDS)])
        self._bmax = np.maximum(self._bmax * 0.985, b + 1e-6)
        band_norm = np.clip(b / self._bmax, 0.0, 1.0)
        self.bands = np.clip(band_norm ** 1.05 * (0.30 + 0.70 * self.level), 0.0, 1.0)

        # --- EDM beat & transient detection using low-frequency spectral flux + bass onset ---
        spec_low = spec[1:10]
        if self._spec_prev is None or len(self._spec_prev) != len(spec_low):
            self._spec_prev = spec_low.copy()
        flux = float(np.sum(np.maximum(0.0, spec_low - self._spec_prev)))
        self._spec_prev = spec_low.copy()

        bass_e = float(np.mean(spec[1:4]))
        bass_diff = max(0.0, bass_e - self._bass_prev)
        self._bass_prev = bass_e

        onset = flux * 1.8 + bass_diff * 3.0
        avg_onset = float(np.mean(self._onhist)) if len(self._onhist) > 0 else 0.0
        std_onset = float(np.std(self._onhist)) if len(self._onhist) > 5 else 0.0
        thresh = avg_onset + max(1.1 * std_onset, avg_onset * 0.28) + 1e-5

        now = time.monotonic()
        if (len(self._onhist) >= 6 and (now - self._lastbeat > 0.15)
                and (onset > thresh) and (db > self._floor + 2.0)):
            self.beat_str = float(np.clip((onset - avg_onset) / (std_onset * 2.0 + 1e-6), 0.5, 1.0))
            self.beat_id += 1
            self._lastbeat = now
        self._onhist.append(onset)

        # --- Musical intelligence: Density, Intimacy, Tension & Drop ---
        active_ratio = float(np.sum(band_norm > 0.20) / BANDS)
        self.density += (active_ratio - self.density) * (1.0 - math.exp(-dt / 0.30))

        mid_presence = float(np.mean(band_norm[3:9]))
        sub_presence = float(np.mean(band_norm[0:3]))
        # Vocal / solo instrument intimacy: active mid frequencies, quiet sub-bass, sparse density
        raw_intimacy = float(np.clip(mid_presence * 2.0 * (1.0 - sub_presence * 1.3) * (1.0 - self.density * 0.7), 0.0, 1.0))
        self.intimacy += (raw_intimacy - self.intimacy) * (1.0 - math.exp(-dt / 0.20))

        # Tension / Build-up detection:
        # Energy climbing steadily over time while sub-bass is filtered out and mids/highs escalate
        self._lvl_long += (self.level - self._lvl_long) * (1.0 - math.exp(-dt / 2.0))
        e_slope = max(0.0, (self.level - self._lvl_long) / (self._lvl_long + 0.12))
        buildup_signature = (mid_presence > 0.26 and sub_presence < 0.42 and e_slope > 0.08)

        if buildup_signature:
            self.tension = min(1.0, self.tension + dt * 0.45 * (1.0 + e_slope))
        else:
            self.tension = max(0.0, self.tension - dt * 0.40)

        # Drop detection: sudden sub-bass surge while tension or intimacy was high
        if (self.tension > 0.30 or self._prev_intimacy > 0.4) and (bass_diff > 0.30 or (sub_presence > 0.60 and self.density > 0.50)):
            self.drop_burst = 1.0
            self.tension = 0.0

        self.drop_burst *= math.exp(-dt / 1.4)
        self._prev_intimacy = self.intimacy


def demo_audio(t: float):
    """32s emotional arc: solo vocal intro -> groove -> tense buildup -> massive drop."""
    loop_t = t % 32.0
    bpm = 128.0
    beat_t = loop_t * (bpm / 60.0)
    beat_idx = int(beat_t)
    ph = beat_t % 1.0

    bands = np.zeros(BANDS)
    bstr = 0.0
    density = 0.0
    intimacy = 0.0
    tension = 0.0
    drop_burst = 0.0

    if loop_t < 8.0:
        # Phase 1: Delicate Solo Vocal / Melody Intro
        # Quiet, sparse spectrum, intimate vocal phrase breathing in center
        vocal_pitch = 0.5 + 0.5 * math.sin(loop_t * 1.4)
        v_idx = int(4 + vocal_pitch * 3)
        lvl = 0.24 + 0.10 * math.sin(loop_t * 2.2)
        bands[v_idx] = 0.85
        bands[v_idx - 1] = 0.45
        bands[v_idx + 1] = 0.40
        density = 0.20
        intimacy = 0.85
        tension = 0.0
    elif loop_t < 16.0:
        # Phase 2: Steady Groove Entrance
        kick = math.exp(-ph * 8.0)
        sub = 0.5 + 0.5 * math.sin(beat_t * 4.0 * math.tau)
        hat = math.exp(-abs(ph - 0.5) * 12.0)
        lvl = 0.45 + 0.35 * kick
        bands[0:3] = np.clip(kick * 0.85 + sub * 0.30, 0.0, 1.0)
        bands[3:8] = 0.35 * np.abs(np.sin(loop_t * 2.0 + np.arange(5)))
        bands[8:16] = hat * 0.60
        bstr = 0.85 if kick > 0.5 else 0.0
        density = 0.50
        intimacy = 0.0
        tension = 0.0
    elif loop_t < 24.0:
        # Phase 3: Suspenseful Build-up (snare rolls, rising riser, sub-bass cut, rising tension)
        build_p = (loop_t - 16.0) / 8.0
        roll_rate = 1.0 + build_p * 7.0
        roll_ph = (beat_t * roll_rate) % 1.0
        snare = math.exp(-roll_ph * 6.0)
        riser_pos = int(np.clip(build_p * 14.0, 2, 15))
        lvl = 0.35 + 0.50 * build_p + 0.15 * snare
        bands[riser_pos] = 0.90
        bands[max(0, riser_pos - 1)] = 0.60
        bands[4:9] += snare * 0.70
        bstr = 0.75 if snare > 0.6 else 0.0
        density = 0.45 + 0.20 * build_p
        intimacy = 0.0
        tension = build_p
    else:
        # Phase 4: Full Instrumental Drop! (massive sub-bass, all instruments together)
        drop_elapsed = loop_t - 24.0
        kick = math.exp(-ph * 9.0)
        clap = math.exp(-max(0.0, ph - 0.02) * 12.0) if (beat_idx % 2 == 1) else 0.0
        sub = 0.6 + 0.4 * math.sin(beat_t * 4.0 * math.tau)
        hat = math.exp(-abs(ph - 0.5) * 14.0)
        lvl = np.clip(0.65 + 0.35 * kick, 0.0, 1.0)
        bands[0:3] = np.clip(kick * 1.0 + sub * 0.5, 0.0, 1.0)
        bands[3:9] = np.clip(clap * 0.90 + 0.60, 0.0, 1.0)
        bands[9:16] = np.clip(hat * 0.85 + 0.50, 0.0, 1.0)
        bstr = 1.0 if kick > 0.5 else 0.0
        density = 0.95
        intimacy = 0.0
        tension = 0.0
        drop_burst = math.exp(-drop_elapsed * 0.6)

    return lvl, bands, beat_idx, bstr, density, intimacy, tension, drop_burst


# --------------------------------------------------------------------------- #
# Planet model: fixed dense core + translucent layers that ripple outward
# --------------------------------------------------------------------------- #
K = 7                                   # number of outer layers
DF = 1                                  # frames of delay between neighbouring layers (crisp travelling shockwave)
BASE = np.array([0.15 + 0.028 * k for k in range(K)])                    # compact when silent
MAXF = np.array([0.32 + 0.65 * (k / (K - 1)) ** 0.85 for k in range(K)]) # reach when loud
KK = np.array([260.0 - 15.0 * k for k in range(K)])                       # snappy responsive springs
CC = 2.0 * 0.58 * np.sqrt(KK)                                           # snappy elastic damping
LAYER_SPEEDS = np.array([1.5, -1.1, 0.85, -0.65, 0.50, -0.40, 0.32])   # differential orbital velocities

_rng = np.random.default_rng(7)


def _shell(n: int) -> np.ndarray:
    """n evenly spread points on a unit sphere, randomly oriented."""
    i = np.arange(n) + 0.5
    y = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    a = math.pi * (3.0 - math.sqrt(5.0)) * np.arange(n) + _rng.random() * TAU
    pts = np.stack([np.cos(a) * r, y, np.sin(a) * r], axis=1)
    q, _ = np.linalg.qr(_rng.normal(size=(3, 3)))
    return pts @ q


_counts = [900 + 250 * k for k in range(K)]            # outer layers get more grains
S_DIRS = np.concatenate([_shell(n) for n in _counts])
S_ID = np.concatenate([np.full(n, k) for k, n in enumerate(_counts)])
M = len(S_ID)
S_PHASE = _rng.random(M) * TAU
S_FREQ = 0.8 + _rng.random(M) * 1.6
S_THICK = _rng.normal(0.0, 1.0, M)
S_BAND = np.minimum(BANDS - 1, (np.arccos(np.clip(S_DIRS[:, 1], -1, 1)) / math.pi * BANDS).astype(int))
S_ZF = 0.5 + 0.5 * S_DIRS[:, 2]                        # front/back for depth shading

NC = 360                                                # core grains
C_G = _rng.normal(size=(NC, 2))
C_D = np.linalg.norm(C_G, axis=1)                       # distance in sigma units

# Pre-allocated GPU-resident tensors for zero-copy CUDA acceleration
DEV_CUDA = None
if HAS_CUDA and torch is not None:
    try:
        DEV_CUDA = torch.device("cuda")
        S_DIRS_GPU = torch.from_numpy(S_DIRS.astype(np.float32)).to(DEV_CUDA)
        S_ID_GPU = torch.from_numpy(S_ID.astype(np.int64)).to(DEV_CUDA)
        S_ID_GPU_FLOAT = S_ID_GPU.float()
        S_ID_EXP = torch.exp(-S_ID_GPU_FLOAT * 0.25)
        S_ID_FACTOR = S_ID_GPU_FLOAT / (K - 1) * 0.70
        S_PHASE_GPU = torch.from_numpy(S_PHASE.astype(np.float32)).to(DEV_CUDA)
        S_FREQ_GPU = torch.from_numpy(S_FREQ.astype(np.float32)).to(DEV_CUDA)
        S_THICK_GPU = torch.from_numpy(S_THICK.astype(np.float32)).to(DEV_CUDA)
        S_BAND_GPU = torch.from_numpy(S_BAND.astype(np.int64)).to(DEV_CUDA)
        BASE_GPU = torch.from_numpy(BASE.astype(np.float32)).to(DEV_CUDA)
        _gpu_yaws = torch.zeros(K, device=DEV_CUDA, dtype=torch.float32)
        _gpu_bands = torch.zeros(BANDS, device=DEV_CUDA, dtype=torch.float32)
        _gpu_R = torch.zeros(K, device=DEV_CUDA, dtype=torch.float32)
        _gpu_Lk = torch.zeros(K, device=DEV_CUDA, dtype=torch.float32)
        _gpu_Pk = torch.zeros(K, device=DEV_CUDA, dtype=torch.float32)
    except Exception:
        HAS_CUDA = False
        DEV_CUDA = None


class SphereCanvas(Static):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.audio = SystemAudio()
        self.mode = "system"
        self.theme = "aurora"
        self.use_gpu = bool(HAS_CUDA and DEV_CUDA is not None)
        self.device_name = f"GPU: {GPU_NAME}" if self.use_gpu else "CPU (Vectorized NumPy)"
        self.lvl_fast = 0.0      # drives core intensity
        self.lvl_slow = 0.0      # drives volume (layer spread)
        self.pulse = 0.0         # beat impulse
        self.bands = np.zeros(BANDS)
        self.hist: collections.deque = collections.deque(maxlen=K * DF + 1)
        self.R = BASE.copy()
        self.RV = np.zeros(K)
        self.Pk = np.zeros(K)    # per-layer (delayed) pulse -> travelling bright ring
        self.Lk = np.zeros(K)
        self.rr = BASE[S_ID].copy()
        self.density = 0.0
        self.intimacy = 0.0
        self.tension = 0.0
        self.drop_burst = 0.0
        self.intensity = 0.0
        self.layer_yaws = np.zeros(K)
        self.spin_scale = 1.0    # spin speed multiplier
        self.color_phase = 0.0
        self.proj_x = S_DIRS[:, 0].copy()
        self.proj_y = S_DIRS[:, 1].copy()
        self.proj_z = S_DIRS[:, 2].copy()
        self.fps = 60.0
        self._fps_count = 0
        self._fps_t0 = time.perf_counter()
        self._stopped = False
        self._frame_task: asyncio.Task | None = None
        self._t0 = self._tprev = time.monotonic()
        self._scache: dict[int, Style] = {}
        self._warned = self._announced = False
        self._last_bid = 0

    def on_mount(self) -> None:
        self.audio.start()
        self._stopped = False
        self._frame_task = asyncio.create_task(self._frame_loop())
        if self.use_gpu:
            self._say(f"60 FPS | {self.device_name} ('g': GPU/CPU, 'f': FPS)")
        else:
            self._say("60 FPS | CPU mode ('g': GPU/CPU, 'f': FPS)")

    async def _frame_loop(self) -> None:
        target_dt = 1.0 / 60.0
        t_next = time.perf_counter()
        while not self._stopped:
            self.tick()
            t_next += target_dt
            now = time.perf_counter()
            wait = t_next - now
            if wait > 0:
                await asyncio.sleep(wait)
            else:
                t_next = now
                await asyncio.sleep(0.001)

    def on_unmount(self) -> None:
        self._stopped = True
        if self._frame_task and not self._frame_task.done():
            self._frame_task.cancel()
        self.audio.stop()

    def toggle_device(self) -> str:
        if not HAS_CUDA or DEV_CUDA is None:
            self.use_gpu = False
            self.device_name = "CPU (GPU / CUDA not available)"
            return self.device_name
        self.use_gpu = not self.use_gpu
        self.device_name = f"GPU: {GPU_NAME}" if self.use_gpu else "CPU (Vectorized NumPy)"
        return self.device_name

    def toggle_mode(self) -> None:
        if self.mode == "system":
            self.mode = "demo"
        else:
            self.mode, self._warned, self._announced = "system", False, False
            self.audio.start()

    def _say(self, msg: str) -> None:
        try:
            self.app.notify(msg)
        except Exception:
            pass

    # ----- physics ----- #
    def tick(self) -> None:
        now_perf = time.perf_counter()
        self._fps_count += 1
        if now_perf - self._fps_t0 >= 0.5:
            self.fps = self._fps_count / (now_perf - self._fps_t0)
            self._fps_count = 0
            self._fps_t0 = now_perf

        now = time.monotonic()
        dt, self._tprev = min(0.035, now - self._tprev), now
        t = now - self._t0

        if self.mode == "demo":
            lv, bs, bid, bstr, den, intim, tens, drop = demo_audio(t)
        else:
            a = self.audio
            lv, bs, bid, bstr = a.level, a.bands, a.beat_id, a.beat_str
            den, intim, tens, drop = a.density, a.intimacy, a.tension, a.drop_burst
            if a.ok and not self._announced:
                self._announced = True
                self._say(f"capturing {a.source}")
            if a.error and not a.ok and not self._warned:
                self._warned = True
                self.mode = "demo"
                self._say(f"no system audio ({a.error}) - demo mode; 'd' retries")

        # Responsive musical intelligence state tracking
        self.density += (den - self.density) * (1.0 - math.exp(-dt / 0.06))
        self.intimacy += (intim - self.intimacy) * (1.0 - math.exp(-dt / 0.06))
        self.tension += (tens - self.tension) * (1.0 - math.exp(-dt / 0.06))
        self.drop_burst = max(drop, self.drop_burst * math.exp(-dt / 0.8))

        if bid != self._last_bid:
            self._last_bid = bid
            self.pulse = 1.0
            # Instant real-time punch: direct radius pop + velocity blast
            kick_mag = 3.5 * bstr * (1.0 + 0.5 * self.drop_burst)
            self.RV += kick_mag * np.exp(-np.arange(K) * 0.35)
            self.R += (0.07 * bstr * (1.0 + 0.4 * self.drop_burst)) * np.exp(-np.arange(K) * 0.30)
        self.pulse *= math.exp(-dt / 0.13)

        # Core intensity: ZERO attack lag (instant attack, fast clean decay)
        if lv > self.lvl_fast:
            self.lvl_fast = lv
        else:
            self.lvl_fast += (lv - self.lvl_fast) * (1.0 - math.exp(-dt / 0.07))

        # Volume swell: immediate responsive swell, smooth release
        if lv > self.lvl_slow:
            self.lvl_slow += (lv - self.lvl_slow) * 0.80
        else:
            self.lvl_slow += (lv - self.lvl_slow) * (1.0 - math.exp(-dt / 0.20))

        # Spectrum bands: ZERO attack lag across all 16 bands
        bs_arr = np.asarray(bs)
        up_mask = bs_arr > self.bands
        self.bands[up_mask] = bs_arr[up_mask]
        down_k = 1.0 - math.exp(-dt / 0.09)
        self.bands[~up_mask] += (bs_arr[~up_mask] - self.bands[~up_mask]) * down_k

        self.intensity = min(1.0, self.lvl_fast * 0.85 + 0.45 * self.pulse + 0.30 * self.drop_burst)

        # 3D orbital swirl: layers rotate in differential orbits, surging with the beat and drop
        spin_mult = self.spin_scale * (
            0.65 + 1.2 * self.lvl_fast + 1.5 * self.density + 2.0 * self.pulse + 3.0 * self.drop_burst
        )
        self.layer_yaws += dt * LAYER_SPEEDS * spin_mult
        self.color_phase = (self.color_phase + dt * (0.04 + 0.12 * self.lvl_fast + 0.22 * self.pulse + 0.30 * self.drop_burst)) % 1.0

        yaws = self.layer_yaws[S_ID]
        cy, sy = np.cos(yaws), np.sin(yaws)
        pitch = 0.30 + 0.06 * math.sin(t * 0.45)
        cp, sp = math.cos(pitch), math.sin(pitch)

        x, y, z = S_DIRS[:, 0], S_DIRS[:, 1], S_DIRS[:, 2]
        x1 = x * cy - z * sy
        z1 = x * sy + z * cy
        x2 = x1
        y2 = y * cp - z1 * sp
        z2 = y * sp + z1 * cp
        self.proj_x, self.proj_y, self.proj_z = x2, y2, z2

        # delay line: layer k hears the music k*DF frames late -> waves travel outward
        self.hist.appendleft((self.lvl_slow, self.pulse))
        n = len(self.hist)
        picks = [self.hist[min(k_ * DF, n - 1)] for k_ in range(K)]
        self.Lk = np.array([min(1.0, p[0]) ** 0.85 for p in picks])
        self.Pk = np.array([p[1] for p in picks])
        ripple = (0.18 + 0.14 * self.drop_burst) * self.Pk * (0.6 + 0.4 * np.arange(K) / (K - 1))

        # Layer expansion: Driven by instrumental density + volume + drop burst
        # Quiet solo vocal stays compact and intimate; full instrumentation blooms wide
        fullness = 0.22 * self.Lk + 0.48 * self.density + 0.40 * self.drop_burst
        max_reach = MAXF * (0.85 + 0.35 * self.drop_burst)
        target = BASE + (max_reach - BASE) * fullness + ripple

        acc = KK * (target - self.R) - CC * self.RV                # snappy elastic springs
        self.RV += acc * dt
        self.R += self.RV * dt

        # Dynamic physical deformations:
        # 1. Suspenseful build-up: tidal stretching along vertical axis into a tense prolate shape
        y_coord = S_DIRS[:, 1]
        stretch_tidal = 1.0 + 0.38 * self.tension * (2.5 * (y_coord ** 2) - 0.8)

        # 2. Suspenseful build-up: trembling high-frequency vibration
        tremble = self.tension * 0.040 * np.sin(t * 38.0 + S_PHASE)

        # 3. Solo vocal intimacy: delicate harmonic standing ripples that resonate with pitch
        theta_lat = np.arccos(np.clip(y_coord, -1.0, 1.0))
        vocal_ripple = 0.09 * self.intimacy * np.cos(3.0 * theta_lat + t * 4.5) * self.bands[S_BAND]

        # 4. Volumetric layer spread & thickness based on musical density
        spread = np.maximum(self.R - BASE, 0.0)
        rr = (self.R[S_ID] * stretch_tidal
              * (1.0 + 0.025 * np.sin(t * S_FREQ * 5.0 + S_PHASE)
                 + tremble
                 + vocal_ripple
                 + 0.15 * self.bands[S_BAND]
                 + (0.10 + 0.15 * self.drop_burst) * self.pulse * np.exp(-S_ID * 0.25))
              + S_THICK * (0.003 + (0.020 + 0.045 * self.density) * spread[S_ID]))
        rr = np.maximum(rr, 0.0)
        hi = rr > 0.92                                              # soft ceiling, never > 1.0
        rr[hi] = 0.92 + 0.08 * np.tanh((rr[hi] - 0.92) / 0.08)
        self.rr = rr
        self.refresh()

    # ----- drawing (Braille sub-cell dots) ----- #
    def _style(self, key: int) -> Style:
        s = self._scache.get(key)
        if s is None:
            r, g, b = ((key >> 8) & 15) << 4 | 8, ((key >> 4) & 15) << 4 | 8, (key & 15) << 4 | 8
            s = self._scache[key] = Style(color=f"#{r:02x}{g:02x}{b:02x}")
        return s

    def _render_outer_cpu(
        self, cxs: float, cys: float, rmax: float, W: int, H: int, sigma: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        D_cam = 2.7
        depth = D_cam - self.proj_z * self.rr
        persp = D_cam / np.maximum(depth, 0.4)

        sx = np.rint(cxs + self.proj_x * self.rr * rmax * persp).astype(np.int64)
        sy = np.rint(cys + self.proj_y * self.rr * rmax * persp).astype(np.int64)

        # 3D Directional Lighting: Key light from upper-left toward viewer
        LX, LY, LZ = -0.38, -0.55, 0.74
        diff = np.maximum(0.0, self.proj_x * LX + self.proj_y * LY + self.proj_z * LZ)
        spec = diff ** 7
        zf = np.clip((self.proj_z + 1.0) * 0.5, 0.0, 1.0)
        fog = zf ** 1.5
        rim = np.clip((1.0 - np.abs(self.proj_z)) ** 2.5 * 0.40, 0.0, 0.40)
        shade_3d = (0.22 + 0.55 * diff + 0.45 * spec + rim) * (0.28 + 0.72 * fog)

        # Volumetric depth filtering: front hemisphere is crisp; far-back hemisphere has soft depth
        clear = 1.8 + 2.0 * sigma
        dist_core_sq = (sx - cxs) ** 2 + (sy - cys) ** 2
        core_clear = (self.proj_z > 0.10) | (dist_core_sq > (clear * clear))
        depth_filter = (self.proj_z > -0.15) | (np.abs(S_PHASE) < (math.tau * (0.30 + 0.70 * zf)))
        ok = ((sx >= 0) & (sx < W) & (sy >= 0) & (sy < H) & core_clear & depth_filter)

        sx, sy, kid, z_val = sx[ok], sy[ok], S_ID[ok], self.proj_z[ok]
        alpha = 0.45 + 0.025 * kid + 0.25 * self.Lk[kid] + 0.85 * self.Pk[kid]
        bright = np.clip(alpha * shade_3d[ok], 0.0, 0.95)
        cpos = (kid / (K - 1) * 0.70 + self.color_phase) % 1.0
        return sx, sy, kid, z_val, bright, cpos

    def _render_outer_gpu(
        self, cxs: float, cys: float, rmax: float, W: int, H: int, sigma: float, t: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pitch = 0.30 + 0.06 * math.sin(t * 0.45)
        cp, sp = math.cos(pitch), math.sin(pitch)

        _gpu_yaws.copy_(torch.from_numpy(self.layer_yaws.astype(np.float32, copy=False)))
        yaws = _gpu_yaws[S_ID_GPU]
        cy, sy = torch.cos(yaws), torch.sin(yaws)

        x, y, z = S_DIRS_GPU[:, 0], S_DIRS_GPU[:, 1], S_DIRS_GPU[:, 2]
        x1 = x * cy - z * sy
        z1 = x * sy + z * cy
        x2 = x1
        y2 = y * cp - z1 * sp
        z2 = y * sp + z1 * cp

        y_coord = S_DIRS_GPU[:, 1]
        stretch_tidal = 1.0 + 0.38 * self.tension * (2.5 * (y_coord ** 2) - 0.8)
        tremble = self.tension * 0.040 * torch.sin(t * 38.0 + S_PHASE_GPU)
        theta_lat = torch.acos(torch.clamp(y_coord, -1.0, 1.0))
        _gpu_bands.copy_(torch.from_numpy(self.bands.astype(np.float32, copy=False)))
        vocal_ripple = 0.09 * self.intimacy * torch.cos(3.0 * theta_lat + t * 4.5) * _gpu_bands[S_BAND_GPU]

        _gpu_R.copy_(torch.from_numpy(self.R.astype(np.float32, copy=False)))
        spread = torch.clamp(_gpu_R - BASE_GPU, min=0.0)
        rr = (_gpu_R[S_ID_GPU] * stretch_tidal
              * (1.0 + 0.025 * torch.sin(t * S_FREQ_GPU * 5.0 + S_PHASE_GPU)
                 + tremble
                 + vocal_ripple
                 + 0.15 * _gpu_bands[S_BAND_GPU]
                 + (0.10 + 0.15 * self.drop_burst) * self.pulse * S_ID_EXP)
              + S_THICK_GPU * (0.003 + (0.020 + 0.045 * self.density) * spread[S_ID_GPU]))
        rr = torch.clamp(rr, min=0.0)
        hi = rr > 0.92
        rr = torch.where(hi, 0.92 + 0.08 * torch.tanh((rr - 0.92) / 0.08), rr)

        D_cam = 2.7
        depth = D_cam - z2 * rr
        persp = D_cam / torch.clamp(depth, min=0.4)

        sx = torch.round(cxs + x2 * rr * rmax * persp).long()
        sy = torch.round(cys + y2 * rr * rmax * persp).long()

        LX, LY, LZ = -0.38, -0.55, 0.74
        diff = torch.clamp(x2 * LX + y2 * LY + z2 * LZ, min=0.0)
        spec = diff ** 7
        zf = torch.clamp((z2 + 1.0) * 0.5, 0.0, 1.0)
        fog = zf ** 1.5
        rim = torch.clamp((1.0 - torch.abs(z2)) ** 2.5 * 0.40, 0.0, 0.40)
        shade_3d = (0.22 + 0.55 * diff + 0.45 * spec + rim) * (0.28 + 0.72 * fog)

        clear = 1.8 + 2.0 * sigma
        dist_core_sq = (sx.float() - cxs) ** 2 + (sy.float() - cys) ** 2
        core_clear = (z2 > 0.10) | (dist_core_sq > (clear * clear))
        depth_filter = (z2 > -0.15) | (torch.abs(S_PHASE_GPU) < (math.tau * (0.30 + 0.70 * zf)))
        ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H) & core_clear & depth_filter

        _gpu_Lk.copy_(torch.from_numpy(self.Lk.astype(np.float32, copy=False)))
        _gpu_Pk.copy_(torch.from_numpy(self.Pk.astype(np.float32, copy=False)))
        kid = S_ID_GPU[ok]
        alpha = 0.45 + 0.025 * S_ID_GPU_FLOAT[ok] + 0.25 * _gpu_Lk[kid] + 0.85 * _gpu_Pk[kid]
        bright = torch.clamp(alpha * shade_3d[ok], 0.0, 0.95)
        cpos = (S_ID_FACTOR[ok] + self.color_phase) % 1.0

        return (
            sx[ok].cpu().numpy(),
            sy[ok].cpu().numpy(),
            kid.cpu().numpy(),
            z2[ok].cpu().numpy(),
            bright.cpu().numpy(),
            cpos.cpu().numpy(),
        )

    def render(self) -> Text:
        w, h = max(20, self.size.width), max(10, self.size.height)
        W, H = w * 2, h * 4
        cxs, cys = (W - 1) / 2, (H - 1) / 2
        rmax = max(4.0, min(cxs, cys) - 1.0)
        t = time.monotonic() - self._t0
        lut = LUTS[self.theme]
        I = self.intensity

        # Core spread:
        # - Intimate vocal/solo: soft, warm breathing nucleus
        # - Full drop / instrumental climax: collapses into a blinding white diamond singularity
        singularity = float(np.clip(0.55 * self.density + 0.65 * self.drop_burst + 0.40 * I, 0.0, 1.0))
        sigma = 0.22 + (1.0 - singularity) ** 1.8 * (0.65 + 0.025 * rmax)

        # ---- outer layers (GPU or CPU 3D perspective projection + directional lighting) ----
        if self.use_gpu and DEV_CUDA is not None:
            try:
                sx, sy, kid, z_val, bright, cpos = self._render_outer_gpu(
                    cxs, cys, rmax, W, H, sigma, t
                )
            except Exception:
                self.use_gpu = False
                sx, sy, kid, z_val, bright, cpos = self._render_outer_cpu(
                    cxs, cys, rmax, W, H, sigma
                )
        else:
            sx, sy, kid, z_val, bright, cpos = self._render_outer_cpu(
                cxs, cys, rmax, W, H, sigma
            )

        # ---- core: quantum spin + pulse brightness at center (z = 0.02) ----
        core_spin = t * (2.5 + 5.0 * self.pulse + 8.0 * self.drop_burst)
        cs, sn = math.cos(core_spin), math.sin(core_spin)
        cg_x = C_G[:, 0] * cs - C_G[:, 1] * sn
        cg_y = C_G[:, 0] * sn + C_G[:, 1] * cs
        cx_ = np.floor(cxs + 0.5 + cg_x * sigma).astype(np.int64)
        cy_ = np.floor(cys + 0.5 + cg_y * sigma).astype(np.int64)
        white = min(1.0, 0.15 + 0.85 * singularity)
        core_power = 0.70 + 0.30 * singularity
        cb = core_power * (1.0 - 0.50 * np.minimum(1.0, C_D / 2.5))
        # Warm celestial tint during solo intimacy, blinding pure white during drops
        warm_core = lut[35] * (1.0 - white) + np.array([255.0, 255.0, 255.0]) * white
        ccol = warm_core[None, :] * cb[:, None]

        # ---- combine into Braille cells with true 3D Z-buffering ----
        cell_s = (sy >> 2) * w + (sx >> 1)
        cell_c = (cy_ >> 2) * w + (cx_ >> 1)
        subs = np.concatenate([sy * W + sx, cy_ * W + cx_])
        cells = np.concatenate([cell_s, cell_c])
        bits = np.concatenate([BITS[sy & 3, sx & 1], BITS[cy_ & 3, cx_ & 1]])
        _, first = np.unique(subs, return_index=True)
        mask = np.bincount(cells[first], weights=bits[first], minlength=w * h).astype(np.int64)

        rgb_s = lut[(cpos * 255).astype(int)] * bright[:, None]
        # Core has depth z = 0.02: background particles (z < 0.02) stay behind core,
        # foreground particles (z >= 0.02) orbit in front of core!
        z_c = np.full(NC, 0.02)

        all_cells = np.concatenate([cell_s, cell_c])
        all_z = np.concatenate([z_val, z_c])
        all_rgb = np.concatenate([rgb_s, ccol])

        # Depth order: furthest drawn first, nearest drawn last (front particles occlude back)
        z_order = np.argsort(all_z)
        rgb = np.zeros((w * h, 3))
        rgb[all_cells[z_order]] = all_rgb[z_order]

        rgb = np.clip(rgb, 0, 255).astype(int) >> 4
        keys = ((rgb[:, 0] << 8) | (rgb[:, 1] << 4) | rgb[:, 2]).tolist()
        ml = mask.tolist()
        chars = [chr(0x2800 + m) if m else " " for m in ml]

        out = Text(no_wrap=True)
        for row in range(h):
            base, col = row * w, 0
            while col < w:
                if not ml[base + col]:
                    j = col + 1
                    while j < w and not ml[base + j]:
                        j += 1
                    out.append(" " * (j - col))
                    col = j
                    continue
                kk, j = keys[base + col], col + 1
                while j < w and ml[base + j] and keys[base + j] == kk:
                    j += 1
                out.append("".join(chars[base + col:base + j]), self._style(kk))
                col = j
            if row < h - 1:
                out.append("\n")
        return out


class SphereApp(App):
    CSS = """
    Screen { background: #05070c; }
    #sphere { width: 100%; height: 100%; overflow: hidden; }
    """
    BINDINGS = [
        ("q,escape,ctrl+c", "quit", "Quit"),
        ("d", "toggle_mode", "Demo/System"),
        ("up,plus,equals_sign", "sens(3)", "More reactive"),
        ("down,minus", "sens(-3)", "Less reactive"),
        ("left,bracket_left", "spin(-0.25)", "Slower spin"),
        ("right,bracket_right", "spin(0.25)", "Faster spin"),
        ("space", "toggle_spin", "Pause/Resume spin"),
        ("g", "toggle_device", "Toggle GPU/CPU"),
        ("f", "show_fps", "Show FPS"),
        ("t", "next_theme", "Theme"),
        ("1", "theme(0)", "aurora"), ("2", "theme(1)", "spectrum"),
        ("3", "theme(2)", "ember"), ("4", "theme(3)", "matrix"),
    ]

    def compose(self) -> ComposeResult:
        yield SphereCanvas(id="sphere")

    @property
    def canvas(self) -> SphereCanvas:
        return self.query_one(SphereCanvas)

    def action_show_fps(self) -> None:
        c = self.canvas
        self.notify(f"Framerate: {c.fps:.1f} FPS (Target: 60.0 FPS) | {c.device_name}")

    def action_toggle_device(self) -> None:
        status = self.canvas.toggle_device()
        self.notify(f"Compute Engine: {status} ({self.canvas.fps:.1f} FPS)")

    def action_toggle_mode(self) -> None:
        self.canvas.toggle_mode()
        self.notify(f"audio: {self.canvas.mode}")

    def action_sens(self, db: float) -> None:
        a = self.canvas.audio
        a.sens = max(-12.0, min(18.0, a.sens + db))
        self.notify(f"reactivity gain {a.sens:+.0f} dB")

    def action_spin(self, delta: float) -> None:
        c = self.canvas
        c.spin_scale = max(0.0, min(4.0, c.spin_scale + delta))
        self.notify(f"spin speed {c.spin_scale:.2f}x")

    def action_toggle_spin(self) -> None:
        c = self.canvas
        c.spin_scale = 0.0 if c.spin_scale > 0.0 else 1.0
        self.notify(f"spin {'active' if c.spin_scale > 0 else 'paused'}")

    def action_theme(self, i: int) -> None:
        self.canvas.theme = THEME_NAMES[i]

    def action_next_theme(self) -> None:
        c = self.canvas
        c.theme = THEME_NAMES[(THEME_NAMES.index(c.theme) + 1) % len(THEME_NAMES)]


if __name__ == "__main__":
    SphereApp().run()