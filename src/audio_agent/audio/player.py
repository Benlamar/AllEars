"""Low-latency audio streaming player for real-time speech output."""

import queue
import threading
from typing import Optional
import numpy as np
import sounddevice as sd


class AudioStreamPlayer:
    """Consumes and plays synthesized audio chunks sequentially in a background thread."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._is_running = False
        self._stop_event = threading.Event()

    @property
    def is_playing(self) -> bool:
        """Returns True if audio is actively playing through speakers."""
        if not self._is_running:
            return False
        try:
            stream = sd.get_stream()
            return bool(stream and stream.active)
        except Exception:
            return False

    def start(self) -> None:
        """Starts the background audio playback thread."""
        if self._is_running:
            return
        self._is_running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    def enqueue(self, audio_chunk: np.ndarray) -> None:
        """Enqueues an audio array for playback."""
        if not self._is_running:
            self.start()
        if audio_chunk is not None and len(audio_chunk) > 0:
            self._queue.put(audio_chunk)

    def finish(self, wait: bool = True) -> None:
        """Signals that no more audio will be enqueued."""
        if self._is_running:
            self._queue.put(None) # Sentinel to stop thread
            if wait and self._thread is not None:
                self._thread.join()
            self._is_running = False

    def stop(self) -> None:
        """Immediately interrupts and halts playback."""
        self._stop_event.set()
        try:
            sd.stop()
        except Exception:
            pass
        # Clear any remaining items
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._is_running = False

    def _playback_loop(self) -> None:
        """Worker loop that plays queued audio chunks."""
        while not self._stop_event.is_set():
            try:
                chunk = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk is None: # Finished sentinel
                break

            if self._stop_event.is_set():
                break

            try:
                sd.play(chunk, self.sample_rate)
                # Wait for this chunk to finish playing before starting the next
                while sd.get_stream() and sd.get_stream().active:
                    if self._stop_event.is_set():
                        sd.stop()
                        break
                    sd.sleep(10)
            except Exception:
                pass
            finally:
                self._queue.task_done()

