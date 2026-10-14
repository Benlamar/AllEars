"""Low-latency gapless audio streaming player for real-time speech output.

A single persistent ``sounddevice.OutputStream`` is fed from a lock-protected sample
queue by the audio callback. Chunks are played back-to-back with no stream
re-open between them. Edge fades remove boundary clicks, smooth ducking enables
interruption candidate evaluation, and generation ID tagging prevents stale audio
from playing after cancellations.
"""

from dataclasses import dataclass
import threading
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from audio_agent.providers.base import AudioOutput

try:  # sounddevice needs PortAudio; keep the module importable without it
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None  # type: ignore


@dataclass
class _QueuedPiece:
    data: np.ndarray
    generation_id: Optional[int] = None


class AudioStreamPlayer(AudioOutput):
    """Plays enqueued audio chunks gaplessly on one persistent output stream."""

    def __init__(
        self,
        sample_rate: int = 24000,
        fade_ms: float = 8.0,
        open_stream: bool = True,
        blocksize: int = 0,
    ):
        self.sample_rate = sample_rate
        self.fade_samples = int(sample_rate * fade_ms / 1000.0)
        self._open_stream = open_stream
        self._blocksize = blocksize

        self._lock = threading.Lock()
        self._chunks: Deque[_QueuedPiece] = deque()
        self._cur: Optional[_QueuedPiece] = None
        self._pos = 0
        self._buffered = 0  # samples queued but not yet played

        # Ducking & volume smoothing
        self._volume: float = 1.0
        self._target_volume: float = 1.0
        self._volume_ramp_step: float = 0.0

        self._stream = None
        self._is_running = False
        self._finishing = False
        self._has_played = False
        self._drained = threading.Event()
        self._drained.set()
        self._null_sink_thread: Optional[threading.Thread] = None
        self.underruns = 0

    # ------------------------------------------------------------------ state
    @property
    def is_playing(self) -> bool:
        """True while queued audio is still being sent to the speakers."""
        return self._is_running and self._buffered > 0

    @property
    def buffered_seconds(self) -> float:
        return self._buffered / float(self.sample_rate)

    @property
    def current_volume(self) -> float:
        return self._volume

    # -------------------------------------------------------------- ducking
    def duck(self, target_volume: float = 0.25, ramp_ms: float = 40.0) -> None:
        """Smoothly ducks playback volume to target_volume over ramp_ms."""
        with self._lock:
            self._target_volume = max(0.0, min(1.0, float(target_volume)))
            ramp_samples = max(1, int(self.sample_rate * (ramp_ms / 1000.0)))
            self._volume_ramp_step = (self._target_volume - self._volume) / float(ramp_samples)

    def restore(self, ramp_ms: float = 60.0) -> None:
        """Smoothly restores playback volume to 1.0 over ramp_ms."""
        with self._lock:
            self._target_volume = 1.0
            ramp_samples = max(1, int(self.sample_rate * (ramp_ms / 1000.0)))
            self._volume_ramp_step = (self._target_volume - self._volume) / float(ramp_samples)

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Opens the persistent output stream (idempotent)."""
        if self._is_running:
            return
        self._is_running = True
        self._finishing = False
        self._has_played = False
        self._volume = 1.0
        self._target_volume = 1.0
        self._volume_ramp_step = 0.0
        if not self._open_stream or sd is None:
            return
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._blocksize,
                callback=self._callback,
            )
            self._stream.start()
        except Exception:
            # No usable output device: discard audio in (fast) real time instead of blocking
            self._stream = None
            self._null_sink_thread = threading.Thread(target=self._null_sink, daemon=True)
            self._null_sink_thread.start()

    def enqueue(
        self,
        audio_chunk: np.ndarray,
        pause_after_s: float = 0.0,
        generation_id: Optional[int] = None,
    ) -> None:
        """Queues an audio array; optionally followed by ``pause_after_s`` of silence."""
        if audio_chunk is None or len(audio_chunk) == 0:
            return
        if not self._is_running:
            self.start()
        data = np.asarray(audio_chunk, dtype=np.float32).reshape(-1).copy()
        n = min(self.fade_samples, len(data) // 2)
        if n > 0:
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
            data[:n] *= ramp
            data[-n:] *= ramp[::-1]
        pieces = [_QueuedPiece(data=data, generation_id=generation_id)]
        if pause_after_s > 0:
            pause_samples = int(pause_after_s * self.sample_rate)
            pieces.append(
                _QueuedPiece(
                    data=np.zeros(pause_samples, dtype=np.float32),
                    generation_id=generation_id,
                )
            )
        with self._lock:
            for p in pieces:
                self._chunks.append(p)
                self._buffered += len(p.data)
            self._drained.clear()

    def cancel_generation(self, generation_id: int) -> int:
        """Purges any queued or currently playing audio matching generation_id.
        
        Returns the number of discarded audio samples.
        """
        with self._lock:
            discarded = 0
            surviving: Deque[_QueuedPiece] = deque()
            for piece in self._chunks:
                if piece.generation_id == generation_id:
                    discarded += len(piece.data)
                else:
                    surviving.append(piece)
            self._chunks = surviving

            if self._cur is not None and self._cur.generation_id == generation_id:
                rem = len(self._cur.data) - self._pos
                discarded += max(0, rem)
                self._cur = None
                self._pos = 0

            self._buffered = max(0, self._buffered - discarded)
            if self._buffered <= 0:
                self._drained.set()
            return discarded

    def finish(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Signals no more audio for this turn; optionally waits for playback to drain."""
        if not self._is_running:
            return
        self._finishing = True
        if wait:
            self._drained.wait(timeout)
            self._close_stream()
            self._is_running = False

    def stop(self) -> None:
        """Immediately interrupts playback and discards queued audio."""
        with self._lock:
            self._chunks.clear()
            self._cur = None
            self._pos = 0
            self._buffered = 0
            self._drained.set()
        self._close_stream(abort=True)
        self._is_running = False

    def _close_stream(self, abort: bool = False) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.abort() if abort else stream.stop()
                stream.close()
            except Exception:
                pass

    # ------------------------------------------------------------- audio path
    def _fill(self, frames: int) -> np.ndarray:
        """Returns exactly ``frames`` samples, zero-padded on underrun. Thread-safe."""
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < frames:
                if self._cur is None:
                    if not self._chunks:
                        break
                    self._cur = self._chunks.popleft()
                    self._pos = 0
                take = min(frames - filled, len(self._cur.data) - self._pos)
                out[filled : filled + take] = self._cur.data[self._pos : self._pos + take]
                filled += take
                self._pos += take
                self._buffered -= take
                if self._pos >= len(self._cur.data):
                    self._cur = None
            if filled > 0:
                self._has_played = True
            if filled < frames:
                if self._buffered <= 0:
                    self._drained.set()
                    if self._has_played and not self._finishing and filled == 0:
                        self.underruns += 1

            # Apply gain and smooth volume ramps
            if filled > 0:
                if self._volume_ramp_step != 0.0 or self._volume != 1.0 or self._target_volume != 1.0:
                    for i in range(filled):
                        if self._volume_ramp_step != 0.0:
                            self._volume += self._volume_ramp_step
                            if (self._volume_ramp_step > 0 and self._volume >= self._target_volume) or (
                                self._volume_ramp_step < 0 and self._volume <= self._target_volume
                            ):
                                self._volume = self._target_volume
                                self._volume_ramp_step = 0.0
                        out[i] *= self._volume

        return out

    def _callback(self, outdata, frames, time_info, status) -> None:  # pragma: no cover - device
        outdata[:, 0] = self._fill(frames)

    def _null_sink(self) -> None:
        while self._is_running:
            self._fill(self.sample_rate // 20)
            threading.Event().wait(0.01)
