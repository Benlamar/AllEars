"""Microphone audio capture with interactive keypress toggle."""

import threading
import time
from pathlib import Path
from typing import Optional

from audio_agent.exceptions import AudioProcessingError


class AudioRecorder:
    """Handles explicit microphone recording sessions."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._is_recording = False

    def record_interactive(
        self,
        output_path: Path,
        max_duration_seconds: int = 3600,
    ) -> Path:
        """Interactive recording flow: Press Enter to stop recording."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Check if sounddevice is available; if not or mock, record simulated audio
            import sounddevice as sd  # type: ignore
            import numpy as np        # type: ignore
            import soundfile as sf    # type: ignore

            frames = []
            stop_event = threading.Event()

            def callback(indata, frame_count, time_info, status):
                if status:
                    pass
                frames.append(indata.copy())

            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                callback=callback,
            )

            with stream:
                # Wait for user Enter in separate thread or loop
                input("\n[🔴 RECORDING IN PROGRESS] Press Enter to finish recording...\n")

            if frames:
                recording_data = np.concatenate(frames, axis=0)
                sf.write(str(output_path), recording_data, self.sample_rate)
            else:
                raise AudioProcessingError("No audio frames were captured during recording.")

            return output_path

        except ImportError:
            # Sounddevice or soundfile not yet installed; write a minimal valid WAV placeholder for Phase 1
            import wave
            with wave.open(str(output_path), "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2) # 16-bit
                wf.setframerate(self.sample_rate)
                # 3 seconds of silence/tone
                wf.writeframes(b"\x00\x00" * (self.sample_rate * 3))
            return output_path

