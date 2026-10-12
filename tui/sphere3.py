"""
Music-reactive "planet" - fast CPU-only terminal renderer (no Textual, no GPU).

Why this is smoother than the Textual version
---------------------------------------------
* Talks to the terminal directly with ANSI escapes. Mouse reporting is never
  enabled (and is explicitly switched off), so moving the mouse over the
  window generates no events and can't cause hitches.
* On Windows, console QuickEdit / mouse-input modes are disabled. (QuickEdit
  can *freeze* output while the mouse is over / clicking the console.)
* One write() per frame, wrapped in synchronized-output markers (DEC 2026),
  and only rows that changed since the last frame are re-sent.
* Braille cell masks are built with a vectorised boolean grid (no np.unique).
* Adaptive quality: if a frame takes too long (slow PC / huge window), the
  number of grains drawn is reduced automatically and restored when there's
  headroom, so the frame rate and motion stay fluid on any machine.
* The garbage collector is paused during the run (no GC hiccups).

Install:
    uv add numpy soundcard
    uv add sounddevice          # only needed for the macOS fallback

Run:
    uv run python sphere3.py

Keys:
    q / Esc : quit            d : toggle demo / system audio
    f : show/hide FPS         t : next theme (or 1-4)
    Space : pause / resume spin
    Up / Down : reactivity    Left / Right : spin speed
"""

from __future__ import annotations

import collections
import ctypes
import gc
import math
import os
import shutil
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore")

IS_WIN = os.name == "nt"

# 1 ms timer resolution on Windows -> accurate sleeps / steady 60 FPS
if IS_WIN:
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

import numpy as np

try:
    import soundcard as sc
except Exception:  # pragma: no cover
    sc = None
try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None

TAU = math.tau
F32 = np.float32
BANDS = 16
SR = 44100
BLOCK = 512

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
    return np.stack([np.interp(xs, sx, st[:, c]) for c in range(3)], axis=1).astype(F32)


LUTS = {name: build_lut(s) for name, s in THEMES.items()}

# Braille dot weights for a (4 rows x 2 cols) cell, shaped for reshape(h,4,w,2)
BITW = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]],
                dtype=np.int32).reshape(1, 4, 1, 2)
BRAILLE = [chr(0x2800 + m) for m in range(256)]
# 4096 pre-built colour escape codes (4 bits per channel)
SGR = [f"\x1b[38;2;{((k >> 8) & 15) << 4 | 8};{((k >> 4) & 15) << 4 | 8};{(k & 15) << 4 | 8}m"
       for k in range(4096)]


# --------------------------------------------------------------------------- #
# System audio (loopback) capture + loudness / beat analysis  (unchanged)
# --------------------------------------------------------------------------- #
class SystemAudio:
    def __init__(self) -> None:
        self.level = 0.0
        self.bands = np.zeros(BANDS)
        self.beat_id = 0
        self.beat_str = 0.0
        self.sens = 0.0
        self.density = 0.0
        self.intimacy = 0.0
        self.tension = 0.0
        self.drop_burst = 0.0
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

        if db > self._peak:
            self._peak = db
        else:
            self._peak = max(self._peak - 5.0 * dt, db)
        if db < self._floor:
            self._floor = db
        else:
            self._floor = min(self._floor + 3.0 * dt, self._peak - 6.0)
        self._floor = max(self._floor, -85.0)
        span = max(self._peak - self._floor, 8.0)
        norm = 0.0 if db < self._floor + 1.2 else max(0.0, (db - self._floor) / span)
        gain = 10 ** (self.sens / 20.0)
        self.level = float(np.clip((norm * gain) ** 1.05, 0.0, 1.0))

        spec = np.abs(np.fft.rfft(x[:BLOCK] * np.hanning(min(n, BLOCK)))) / (n / 4)
        spec = np.pad(spec, (0, max(0, EDGES[-1] + 1 - len(spec))))
        b = np.array([spec[EDGES[i]:EDGES[i + 1]].mean() for i in range(BANDS)])
        self._bmax = np.maximum(self._bmax * 0.985, b + 1e-6)
        band_norm = np.clip(b / self._bmax, 0.0, 1.0)
        self.bands = np.clip(band_norm ** 1.05 * (0.30 + 0.70 * self.level), 0.0, 1.0)

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

        active_ratio = float(np.sum(band_norm > 0.20) / BANDS)
        self.density += (active_ratio - self.density) * (1.0 - math.exp(-dt / 0.30))

        mid_presence = float(np.mean(band_norm[3:9]))
        sub_presence = float(np.mean(band_norm[0:3]))
        raw_intimacy = float(np.clip(mid_presence * 2.0 * (1.0 - sub_presence * 1.3)
                                     * (1.0 - self.density * 0.7), 0.0, 1.0))
        self.intimacy += (raw_intimacy - self.intimacy) * (1.0 - math.exp(-dt / 0.20))

        self._lvl_long += (self.level - self._lvl_long) * (1.0 - math.exp(-dt / 2.0))
        e_slope = max(0.0, (self.level - self._lvl_long) / (self._lvl_long + 0.12))
        buildup_signature = (mid_presence > 0.26 and sub_presence < 0.42 and e_slope > 0.08)

        if buildup_signature:
            self.tension = min(1.0, self.tension + dt * 0.45 * (1.0 + e_slope))
        else:
            self.tension = max(0.0, self.tension - dt * 0.40)

        if (self.tension > 0.30 or self._prev_intimacy > 0.4) and (
                bass_diff > 0.30 or (sub_presence > 0.60 and self.density > 0.50)):
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
        vocal_pitch = 0.5 + 0.5 * math.sin(loop_t * 1.4)
        v_idx = int(4 + vocal_pitch * 3)
        lvl = 0.24 + 0.10 * math.sin(loop_t * 2.2)
        bands[v_idx] = 0.85
        bands[v_idx - 1] = 0.45
        bands[v_idx + 1] = 0.40
        density = 0.20
        intimacy = 0.85
    elif loop_t < 16.0:
        kick = math.exp(-ph * 8.0)
        sub = 0.5 + 0.5 * math.sin(beat_t * 4.0 * math.tau)
        hat = math.exp(-abs(ph - 0.5) * 12.0)
        lvl = 0.45 + 0.35 * kick
        bands[0:3] = np.clip(kick * 0.85 + sub * 0.30, 0.0, 1.0)
        bands[3:8] = 0.35 * np.abs(np.sin(loop_t * 2.0 + np.arange(5)))
        bands[8:16] = hat * 0.60
        bstr = 0.85 if kick > 0.5 else 0.0
        density = 0.50
    elif loop_t < 24.0:
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
        tension = build_p
    else:
        drop_elapsed = loop_t - 24.0
        kick = math.exp(-ph * 9.0)
        clap = math.exp(-max(0.0, ph - 0.02) * 12.0) if (beat_idx % 2 == 1) else 0.0
        sub = 0.6 + 0.4 * math.sin(beat_t * 4.0 * math.tau)
        hat = math.exp(-abs(ph - 0.5) * 14.0)
        lvl = float(np.clip(0.65 + 0.35 * kick, 0.0, 1.0))
        bands[0:3] = np.clip(kick * 1.0 + sub * 0.5, 0.0, 1.0)
        bands[3:9] = np.clip(clap * 0.90 + 0.60, 0.0, 1.0)
        bands[9:16] = np.clip(hat * 0.85 + 0.50, 0.0, 1.0)
        bstr = 1.0 if kick > 0.5 else 0.0
        density = 0.95
        drop_burst = math.exp(-drop_elapsed * 0.6)

    return lvl, bands, beat_idx, bstr, density, intimacy, tension, drop_burst


# --------------------------------------------------------------------------- #
# Planet model
# --------------------------------------------------------------------------- #
K = 7
DF = 1
BASE = np.array([0.15 + 0.028 * k for k in range(K)])
MAXF = np.array([0.32 + 0.65 * (k / (K - 1)) ** 0.85 for k in range(K)])
KK = np.array([260.0 - 15.0 * k for k in range(K)])
CC = 2.0 * 0.58 * np.sqrt(KK)
LAYER_SPEEDS = np.array([1.5, -1.1, 0.85, -0.65, 0.50, -0.40, 0.32])

_rng = np.random.default_rng(7)


def _shell(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    y = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    a = math.pi * (3.0 - math.sqrt(5.0)) * np.arange(n) + _rng.random() * TAU
    pts = np.stack([np.cos(a) * r, y, np.sin(a) * r], axis=1)
    q, _ = np.linalg.qr(_rng.normal(size=(3, 3)))
    return pts @ q


_counts = [900 + 250 * k for k in range(K)]
_dirs = np.concatenate([_shell(n) for n in _counts])
_id = np.concatenate([np.full(n, k) for k, n in enumerate(_counts)])
M = len(_id)
_phase = _rng.random(M) * TAU
_freq = 0.8 + _rng.random(M) * 1.6
_thick = _rng.normal(0.0, 1.0, M)
_band = np.minimum(BANDS - 1, (np.arccos(np.clip(_dirs[:, 1], -1, 1)) / math.pi * BANDS).astype(int))

# Shuffle once so any prefix of the arrays is a uniform sample of every layer.
# This lets adaptive quality simply draw the first N grains.
_perm = _rng.permutation(M)
DX = np.ascontiguousarray(_dirs[_perm, 0], dtype=F32)
DY = np.ascontiguousarray(_dirs[_perm, 1], dtype=F32)
DZ = np.ascontiguousarray(_dirs[_perm, 2], dtype=F32)
S_ID = _id[_perm].astype(np.intp)
S_PHASE = _phase[_perm].astype(F32)
S_FREQ = _freq[_perm].astype(F32)
S_THICK = _thick[_perm].astype(F32)
S_BAND = _band[_perm].astype(np.intp)
# Constant per-grain terms
THETA_LAT = np.arccos(np.clip(DY, -1.0, 1.0)).astype(F32)
TIDAL_BASE = (2.5 * DY * DY - 0.8).astype(F32)
S_ID_EXP = np.exp(-S_ID * 0.25).astype(F32)
ALPHA_BASE = (0.45 + 0.025 * S_ID).astype(F32)
CPOS_BASE = (S_ID / (K - 1) * 0.70).astype(F32)
ZPHASE_OK = S_PHASE / F32(TAU)          # 0..1 compare against (0.30 + 0.70*zf)

NC = 360
C_G = _rng.normal(size=(NC, 2)).astype(F32)
C_D = np.linalg.norm(C_G, axis=1)
C_FALL = (1.0 - 0.50 * np.minimum(1.0, C_D / 2.5)).astype(F32)
C_Z = np.full(NC, 0.02, dtype=F32)


class Planet:
    def __init__(self) -> None:
        self.audio = SystemAudio()
        self.mode = "demo"
        self.theme = "aurora"
        self.lvl_fast = 0.0
        self.lvl_slow = 0.0
        self.pulse = 0.0
        self.bands = np.zeros(BANDS)
        self.hist: collections.deque = collections.deque(maxlen=K * DF + 1)
        self.R = BASE.copy()
        self.RV = np.zeros(K)
        self.Pk = np.zeros(K)
        self.Lk = np.zeros(K)
        self.density = 0.0
        self.intimacy = 0.0
        self.tension = 0.0
        self.drop_burst = 0.0
        self.intensity = 0.0
        self.layer_yaws = np.zeros(K)
        self.spin_scale = 1.0
        self.color_phase = 0.0
        self.quality = 1.0            # fraction of grains drawn (adaptive)
        self.n = M
        self.px = DX.copy()
        self.py = DY.copy()
        self.pz = DZ.copy()
        self.rr = np.full(M, 0.2, dtype=F32)
        self._t0 = self._tprev = time.monotonic()
        self._warned = self._announced = False
        self._last_bid = 0
        self.toast = ""
        self.toast_until = 0.0

    def say(self, msg: str, secs: float = 2.5) -> None:
        self.toast, self.toast_until = msg, time.monotonic() + secs

    def set_quality(self, q: float) -> None:
        self.quality = max(0.30, min(1.0, q))
        self.n = int(M * self.quality)

    def toggle_mode(self) -> None:
        if self.mode == "system":
            self.mode = "demo"
        else:
            self.mode, self._warned, self._announced = "system", False, False
            self.audio.start()
        self.say(f"audio: {self.mode}")

    # ----- physics ----- #
    def tick(self) -> None:
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
                self.say(f"capturing {a.source}")
            if a.error and not a.ok and not self._warned:
                self._warned = True
                self.mode = "demo"
                self.say(f"no system audio ({a.error}) - demo mode; 'd' retries", 5)

        self.density += (den - self.density) * (1.0 - math.exp(-dt / 0.06))
        self.intimacy += (intim - self.intimacy) * (1.0 - math.exp(-dt / 0.06))
        self.tension += (tens - self.tension) * (1.0 - math.exp(-dt / 0.06))
        self.drop_burst = max(drop, self.drop_burst * math.exp(-dt / 0.8))

        if bid != self._last_bid:
            self._last_bid = bid
            self.pulse = 1.0
            kick_mag = 3.5 * bstr * (1.0 + 0.5 * self.drop_burst)
            self.RV += kick_mag * np.exp(-np.arange(K) * 0.35)
            self.R += (0.07 * bstr * (1.0 + 0.4 * self.drop_burst)) * np.exp(-np.arange(K) * 0.30)
        self.pulse *= math.exp(-dt / 0.13)

        if lv > self.lvl_fast:
            self.lvl_fast = lv
        else:
            self.lvl_fast += (lv - self.lvl_fast) * (1.0 - math.exp(-dt / 0.07))

        if lv > self.lvl_slow:
            self.lvl_slow += (lv - self.lvl_slow) * 0.80
        else:
            self.lvl_slow += (lv - self.lvl_slow) * (1.0 - math.exp(-dt / 0.20))

        bs_arr = np.asarray(bs)
        up = bs_arr > self.bands
        self.bands[up] = bs_arr[up]
        down_k = 1.0 - math.exp(-dt / 0.09)
        self.bands[~up] += (bs_arr[~up] - self.bands[~up]) * down_k

        self.intensity = min(1.0, self.lvl_fast * 0.85 + 0.45 * self.pulse + 0.30 * self.drop_burst)

        spin_mult = self.spin_scale * (
            0.65 + 1.2 * self.lvl_fast + 1.5 * self.density + 2.0 * self.pulse + 3.0 * self.drop_burst)
        self.layer_yaws += dt * LAYER_SPEEDS * spin_mult
        self.color_phase = (self.color_phase + dt * (
            0.04 + 0.12 * self.lvl_fast + 0.22 * self.pulse + 0.30 * self.drop_burst)) % 1.0

        # delay line: layer k hears the music k*DF frames late
        self.hist.appendleft((self.lvl_slow, self.pulse))
        nh = len(self.hist)
        picks = [self.hist[min(k_ * DF, nh - 1)] for k_ in range(K)]
        self.Lk = np.array([min(1.0, p[0]) ** 0.85 for p in picks])
        self.Pk = np.array([p[1] for p in picks])
        ripple = (0.18 + 0.14 * self.drop_burst) * self.Pk * (0.6 + 0.4 * np.arange(K) / (K - 1))

        fullness = 0.22 * self.Lk + 0.48 * self.density + 0.40 * self.drop_burst
        max_reach = MAXF * (0.85 + 0.35 * self.drop_burst)
        target = BASE + (max_reach - BASE) * fullness + ripple

        # sub-stepped spring integration: stable & smooth even if a frame is slow
        steps = 2
        h_ = dt / steps
        for _ in range(steps):
            acc = KK * (target - self.R) - CC * self.RV
            self.RV += acc * h_
            self.R += self.RV * h_

        # ---- per-grain geometry (only the first n grains) ---- #
        n = self.n
        sid = S_ID[:n]
        yaws = self.layer_yaws.astype(F32)[sid]
        cy, sy = np.cos(yaws), np.sin(yaws)
        pitch = 0.30 + 0.06 * math.sin(t * 0.45)
        cp, sp = F32(math.cos(pitch)), F32(math.sin(pitch))
        x, y, z = DX[:n], DY[:n], DZ[:n]
        x1 = x * cy - z * sy
        z1 = x * sy + z * cy
        self.px = x1
        self.py = y * cp - z1 * sp
        self.pz = y * sp + z1 * cp

        phase = S_PHASE[:n]
        bnd = self.bands.astype(F32)[S_BAND[:n]]
        stretch = 1.0 + F32(0.38 * self.tension) * TIDAL_BASE[:n]
        tremble = F32(self.tension * 0.040) * np.sin(F32(t * 38.0) + phase)
        vocal = F32(0.09 * self.intimacy) * np.cos(3.0 * THETA_LAT[:n] + F32(t * 4.5)) * bnd
        R32 = self.R.astype(F32)
        spread = np.maximum(R32 - BASE.astype(F32), 0.0)
        wob = 0.025 * np.sin(S_FREQ[:n] * F32(t * 5.0) + phase)
        pulse_term = F32((0.10 + 0.15 * self.drop_burst) * self.pulse) * S_ID_EXP[:n]
        rr = (R32[sid] * stretch * (1.0 + wob + tremble + vocal + 0.15 * bnd + pulse_term)
              + S_THICK[:n] * (F32(0.003) + (F32(0.020 + 0.045 * self.density)) * spread[sid]))
        np.maximum(rr, 0.0, out=rr)
        hi = rr > 0.92
        if hi.any():
            rr[hi] = 0.92 + 0.08 * np.tanh((rr[hi] - 0.92) / 0.08)
        self.rr = rr

    # ----- rendering: returns one escape-string per terminal row ----- #
    def render_rows(self, w: int, h: int) -> list[str]:
        W, H = w * 2, h * 4
        cxs, cys = (W - 1) / 2, (H - 1) / 2
        rmax = max(4.0, min(cxs, cys) - 1.0)
        t = time.monotonic() - self._t0
        lut = LUTS[self.theme]
        I = self.intensity
        n = self.n

        singularity = float(np.clip(0.55 * self.density + 0.65 * self.drop_burst + 0.40 * I, 0.0, 1.0))
        sigma = 0.22 + (1.0 - singularity) ** 1.8 * (0.65 + 0.025 * rmax)

        px, py, pz, rr = self.px, self.py, self.pz, self.rr
        D_cam = 2.7
        persp = D_cam / np.maximum(D_cam - pz * rr, 0.4)
        scale = rr * F32(rmax) * persp
        sx = np.rint(px * scale + F32(cxs)).astype(np.int32)
        sy = np.rint(py * scale + F32(cys)).astype(np.int32)

        LX, LY, LZ = -0.38, -0.55, 0.74
        diff = np.maximum(0.0, px * F32(LX) + py * F32(LY) + pz * F32(LZ))
        spec = diff ** 7
        zf = np.clip((pz + 1.0) * 0.5, 0.0, 1.0)
        fog = zf * np.sqrt(zf)
        rim = np.clip((1.0 - np.abs(pz)) ** 2.5 * 0.40, 0.0, 0.40)
        shade = (0.22 + 0.55 * diff + 0.45 * spec + rim) * (0.28 + 0.72 * fog)

        clear = 1.8 + 2.0 * sigma
        dsq = (sx - cxs) ** 2 + (sy - cys) ** 2
        core_clear = (pz > 0.10) | (dsq > clear * clear)
        depth_ok = (pz > -0.15) | (ZPHASE_OK[:n] < (0.30 + 0.70 * zf))
        ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H) & core_clear & depth_ok

        sx, sy = sx[ok], sy[ok]
        kid = S_ID[:n][ok]
        zv = pz[ok]
        alpha = ALPHA_BASE[:n][ok] + F32(0.25) * self.Lk.astype(F32)[kid] + F32(0.85) * self.Pk.astype(F32)[kid]
        bright = np.clip(alpha * shade[ok], 0.0, 0.95)
        cpos = (CPOS_BASE[:n][ok] + F32(self.color_phase)) % 1.0

        # ---- core ----
        core_spin = t * (2.5 + 5.0 * self.pulse + 8.0 * self.drop_burst)
        cs, sn = math.cos(core_spin), math.sin(core_spin)
        cgx = C_G[:, 0] * cs - C_G[:, 1] * sn
        cgy = C_G[:, 0] * sn + C_G[:, 1] * cs
        cx_ = np.clip(np.floor(cxs + 0.5 + cgx * sigma), 0, W - 1).astype(np.int32)
        cy_ = np.clip(np.floor(cys + 0.5 + cgy * sigma), 0, H - 1).astype(np.int32)
        white = min(1.0, 0.15 + 0.85 * singularity)
        core_power = 0.70 + 0.30 * singularity
        warm = lut[35] * (1.0 - white) + F32(255.0) * white
        ccol = warm[None, :] * (core_power * C_FALL)[:, None]

        # ---- braille mask via boolean dot grid ----
        dots = np.zeros(H * W, dtype=bool)
        dots[sy * W + sx] = True
        dots[cy_ * W + cx_] = True
        mask = (dots.reshape(h, 4, w, 2) * BITW).sum(axis=(1, 3), dtype=np.int32).ravel()

        cell_s = (sy >> 2) * w + (sx >> 1)
        cell_c = (cy_ >> 2) * w + (cx_ >> 1)
        rgb_s = lut[(cpos * 255).astype(np.intp)] * bright[:, None]

        all_cells = np.concatenate([cell_s, cell_c])
        all_z = np.concatenate([zv, C_Z])
        all_rgb = np.concatenate([rgb_s, ccol])
        order = np.argsort(all_z, kind="stable")           # far first, near last
        rgb = np.zeros((w * h, 3), dtype=F32)
        rgb[all_cells[order]] = all_rgb[order]
        q = np.clip(rgb, 0, 255).astype(np.int32) >> 4
        keys = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]

        nz = np.flatnonzero(mask)
        rows_l = (nz // w).tolist()
        cols_l = (nz % w).tolist()
        keys_l = keys[nz].tolist()
        mask_l = mask[nz].tolist()

        out = [""] * h
        parts: list[str] = []
        cur, last_col, last_key = -1, 0, -1
        for r_, c_, k_, m_ in zip(rows_l, cols_l, keys_l, mask_l):
            if r_ != cur:
                if cur >= 0:
                    out[cur] = "".join(parts)
                cur, parts, last_col, last_key = r_, [], 0, -1
            gap = c_ - last_col
            if gap > 0:
                parts.append(" " * gap if gap < 5 else f"\x1b[{c_ + 1}G")
            if k_ != last_key:
                parts.append(SGR[k_])
                last_key = k_
            parts.append(BRAILLE[m_])
            last_col = c_ + 1
        if cur >= 0:
            out[cur] = "".join(parts)
        return out


# --------------------------------------------------------------------------- #
# Terminal plumbing
# --------------------------------------------------------------------------- #
class Keys:
    """Background keyboard reader -> queue of key names. No mouse input at all."""

    def __init__(self) -> None:
        self.q: collections.deque = collections.deque()
        self._old = None
        self._fd = None

    def start(self) -> None:
        if not IS_WIN:
            import termios
            import tty
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        threading.Thread(target=self._win if IS_WIN else self._posix, daemon=True).start()

    def stop(self) -> None:
        if not IS_WIN and self._old is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def _win(self) -> None:
        import msvcrt
        names = {"H": "up", "P": "down", "K": "left", "M": "right"}
        while True:
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                self.q.append(names.get(msvcrt.getwch(), ""))
            elif ch == "\x1b":
                self.q.append("esc")
            elif ch == "\x03":
                self.q.append("q")
            else:
                self.q.append("space" if ch == " " else ch.lower())

    def _posix(self) -> None:
        import select
        names = {"A": "up", "B": "down", "D": "left", "C": "right"}
        fd = self._fd
        while True:
            ch = os.read(fd, 1).decode("latin1")
            if ch == "\x1b":
                if select.select([fd], [], [], 0.03)[0]:
                    nxt = os.read(fd, 1).decode("latin1")
                    if nxt in "[O" and select.select([fd], [], [], 0.03)[0]:
                        self.q.append(names.get(os.read(fd, 1).decode("latin1"), ""))
                else:
                    self.q.append("esc")
            elif ch == "\x03":
                self.q.append("q")
            else:
                self.q.append("space" if ch == " " else ch.lower())


def _win_console_setup():
    """Enable ANSI, disable QuickEdit/mouse input. Returns a restore function."""
    if not IS_WIN:
        return lambda: None
    try:
        k = ctypes.windll.kernel32
        hin, hout = k.GetStdHandle(-10), k.GetStdHandle(-11)
        min_, mout = ctypes.c_uint32(), ctypes.c_uint32()
        k.GetConsoleMode(hin, ctypes.byref(min_))
        k.GetConsoleMode(hout, ctypes.byref(mout))
        k.SetConsoleMode(hout, mout.value | 0x0004)                    # VT processing
        k.SetConsoleMode(hin, (min_.value & ~0x0040 & ~0x0010) | 0x0080)  # no QuickEdit / mouse

        def restore():
            k.SetConsoleMode(hin, min_.value)
            k.SetConsoleMode(hout, mout.value)
        return restore
    except Exception:
        return lambda: None


def main() -> None:
    restore_console = _win_console_setup()
    out = sys.stdout.buffer

    def write(s: str) -> None:
        out.write(s.encode("utf-8", "replace"))
        out.flush()

    planet = Planet()
    keys = Keys()
    planet.mode = "system"
    planet.audio.start()
    keys.start()
    gc.collect()
    gc.disable()                                         # no GC pauses mid-animation

    # alt screen, hide cursor, mouse reporting OFF, wrap OFF
    write("\x1b[?1049h\x1b[?25l\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?7l\x1b[2J")

    show_fps = False
    fps, frames, fps_t0 = 60.0, 0, time.perf_counter()
    work_ema = 0.008
    target = 1.0 / 60.0
    prev: list[str | None] = []
    prev_size = (0, 0)
    status_prev = ""
    t_next = time.perf_counter()
    running = True

    try:
        while running:
            f0 = time.perf_counter()

            # ---- input ----
            while keys.q:
                k = keys.q.popleft()
                a = planet.audio
                if k in ("q", "esc"):
                    running = False
                elif k == "d":
                    planet.toggle_mode()
                elif k == "f":
                    show_fps = not show_fps
                elif k == "t":
                    planet.theme = THEME_NAMES[(THEME_NAMES.index(planet.theme) + 1) % len(THEME_NAMES)]
                elif k in ("1", "2", "3", "4"):
                    planet.theme = THEME_NAMES[int(k) - 1]
                elif k in ("up", "+", "="):
                    a.sens = min(18.0, a.sens + 3)
                    planet.say(f"reactivity gain {a.sens:+.0f} dB")
                elif k in ("down", "-"):
                    a.sens = max(-12.0, a.sens - 3)
                    planet.say(f"reactivity gain {a.sens:+.0f} dB")
                elif k in ("left", "["):
                    planet.spin_scale = max(0.0, planet.spin_scale - 0.25)
                    planet.say(f"spin speed {planet.spin_scale:.2f}x")
                elif k in ("right", "]"):
                    planet.spin_scale = min(4.0, planet.spin_scale + 0.25)
                    planet.say(f"spin speed {planet.spin_scale:.2f}x")
                elif k == "space":
                    planet.spin_scale = 0.0 if planet.spin_scale > 0 else 1.0
                    planet.say("spin " + ("active" if planet.spin_scale else "paused"))
            if not running:
                break

            cols, rows = shutil.get_terminal_size((100, 40))
            w, h = max(20, cols), max(8, rows - 1)       # last row = status line
            if (cols, rows) != prev_size:
                prev_size = (cols, rows)
                prev = [None] * h
                status_prev = ""
                write("\x1b[2J")

            planet.tick()
            lines = planet.render_rows(w, h)

            # ---- diff: only changed rows ----
            buf = ["\x1b[?2026h"]                          # begin synchronized update
            for r, s in enumerate(lines):
                if r >= len(prev) or s != prev[r]:
                    buf.append(f"\x1b[{r + 1};1H\x1b[2K{s}\x1b[0m")
            prev = list(lines)

            now_m = time.monotonic()
            if now_m < planet.toast_until:
                status = planet.toast
            elif show_fps:
                status = f"{fps:5.1f} FPS | quality {planet.quality * 100:3.0f}% | {planet.mode} | {planet.theme}"
            else:
                status = ""
            if status != status_prev:
                status_prev = status
                buf.append(f"\x1b[{rows};1H\x1b[2K\x1b[38;2;120;130;150m {status[:cols - 2]}\x1b[0m")
            buf.append("\x1b[?2026l")
            write("".join(buf))

            # ---- adaptive quality: keep the frame inside its time budget ----
            work = time.perf_counter() - f0
            work_ema += (work - work_ema) * 0.1
            if work_ema > target * 0.85:
                planet.set_quality(planet.quality - 0.03)
            elif work_ema < target * 0.55 and planet.quality < 1.0:
                planet.set_quality(planet.quality + 0.01)

            frames += 1
            now_p = time.perf_counter()
            if now_p - fps_t0 >= 0.5:
                fps, frames, fps_t0 = frames / (now_p - fps_t0), 0, now_p

            # ---- precise frame pacing ----
            t_next += target
            wait = t_next - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                t_next = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        planet.audio.stop()
        keys.stop()
        write("\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l")
        restore_console()
        gc.enable()


if __name__ == "__main__":
    main()