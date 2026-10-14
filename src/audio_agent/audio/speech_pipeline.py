"""Streaming speech pipeline: LLM text deltas -> sentence chunks -> TTS -> gapless player.

The LLM loop only calls :meth:`push_text` (non-blocking). A worker thread synthesizes
chunks while earlier chunks are still playing, so the player is not starved by
synthesis latency and LLM token consumption is never blocked by TTS.
Generation IDs and cancellation events ensure stale audio is dropped immediately.
"""

import queue
import threading
import time
from typing import Callable, List, Optional

import numpy as np

from audio_agent.audio.player import AudioStreamPlayer
from audio_agent.audio.speech_chunker import ChunkerConfig, SpeechChunker

_SENTENCE_END = (".", "!", "?", "\u2026", "\n")


class StreamingSpeechPipeline:
    """Coordinates chunking, lookahead synthesis and gapless playback for one turn."""

    def __init__(
        self,
        tts,
        player: AudioStreamPlayer,
        chunker_config: Optional[ChunkerConfig] = None,
        lookahead: int = 2,
        sentence_pause_s: float = 0.12,
        on_first_audio: Optional[Callable[[np.ndarray, int], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        generation_id: int = 0,
    ):
        self.tts = tts
        self.player = player
        self.config = chunker_config or ChunkerConfig()
        self.chunker = SpeechChunker(self.config)
        self.sentence_pause_s = sentence_pause_s
        self.on_first_audio = on_first_audio
        self.cancel_event = cancel_event or threading.Event()
        self.on_error = on_error
        self.generation_id = generation_id

        # bounded text chunk handoff to worker
        self._text_q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._lookahead = max(1, lookahead)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._last_push = time.monotonic()
        self._started = False
        self._first_audio_done = False
        self.chunks_synthesized = 0
        self.stale_chunks_rejected = 0
        self.synthesized_texts: List[str] = []
        self.synth_seconds: list[float] = []
        self.first_audio_latency_s: Optional[float] = None
        self._t0 = time.monotonic()

    # ----------------------------------------------------------------- input
    def _ensure_started(self) -> None:
        if not self._started:
            self._started = True
            self._t0 = time.monotonic()
            self._worker.start()

    def push_text(self, delta: str) -> None:
        """Adds streamed LLM text; never blocks on TTS."""
        if self.cancel_event.is_set() or not delta:
            return
        self._ensure_started()
        self._last_push = time.monotonic()
        for chunk in self.chunker.push(delta):
            self._text_q.put(chunk)

    def poll_stall(self) -> None:
        """Call periodically; releases buffered text if the LLM stalled (max_wait_s)."""
        if (
            self._started
            and self.chunker.has_pending
            and time.monotonic() - self._last_push >= self.config.max_wait_s
            and self._text_q.qsize() == 0
        ):
            chunk = self.chunker.flush_stalled()
            if chunk:
                self._text_q.put(chunk)

    def finish(self, wait: bool = True) -> None:
        """Flushes the tail, drains synthesis and (optionally) waits for playback."""
        if self.cancel_event.is_set():
            return
        tail = self.chunker.flush()
        if tail:
            self._ensure_started()
            self._text_q.put(tail)
        if not self._started:
            return
        self._text_q.put(None)
        if wait:
            self._worker.join()
            if not self.cancel_event.is_set():
                self.player.finish(wait=True)

    def cancel(self) -> None:
        """Barge-in: drop pending text and audio immediately."""
        self.cancel_event.set()
        while True:
            try:
                self._text_q.get_nowait()
            except queue.Empty:
                break
        self._text_q.put(None)
        self.player.cancel_generation(self.generation_id)
        self.player.stop()

    # ---------------------------------------------------------------- worker
    def _synthesize(self, text: str) -> Optional[np.ndarray]:
        res = self.tts.synthesize(text)
        if isinstance(res, tuple):
            res = res[0]
        if res is None or len(res) == 0:
            return None
        return np.asarray(res, dtype=np.float32)

    def _run(self) -> None:
        while not self.cancel_event.is_set():
            try:
                text = self._text_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if text is None:
                break

            # Lookahead throttle: do not synthesize far ahead of playback
            while (
                not self.cancel_event.is_set()
                and self.player.buffered_seconds > self._lookahead * 8.0
            ):
                time.sleep(0.05)

            if self.cancel_event.is_set():
                self.stale_chunks_rejected += 1
                break

            t0 = time.monotonic()
            try:
                audio = self._synthesize(text)
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
                continue

            if self.cancel_event.is_set():
                self.stale_chunks_rejected += 1
                break
            if audio is None:
                continue

            self.synth_seconds.append(time.monotonic() - t0)
            self.chunks_synthesized += 1
            self.synthesized_texts.append(text)

            if not self._first_audio_done:
                self._first_audio_done = True
                self.first_audio_latency_s = time.monotonic() - self._t0
                if self.on_first_audio:
                    try:
                        self.on_first_audio(audio, getattr(self.tts, "sample_rate", 24000))
                    except Exception:
                        pass

            pause = self.sentence_pause_s if text.rstrip().endswith(_SENTENCE_END) else 0.0
            self.player.enqueue(audio, pause_after_s=pause, generation_id=self.generation_id)
