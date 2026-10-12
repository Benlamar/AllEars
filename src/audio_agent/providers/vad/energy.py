"""Energy-based RMS VAD provider for lightweight, zero-dependency speech activity detection."""

from pathlib import Path
from typing import List, Optional
import numpy as np
import soundfile as sf

from audio_agent.exceptions import AudioProcessingError
from audio_agent.providers.base import VADProvider
from audio_agent.schemas import SpeechRegion


class EnergyVADProvider(VADProvider):
    """Zero-overhead Voice Activity Detection based on RMS energy thresholding."""

    provider_name: str = "energy"
    execution_mode: str = "local"

    def __init__(self, threshold: float = 0.015, sampling_rate: int = 16000, **kwargs):
        # Allow custom threshold up to 0.15; if someone passes e.g. 0.5 (probability), scale to safe default 0.018
        if threshold <= 0.15:
            self.threshold = threshold
        else:
            self.threshold = 0.018
        self.sampling_rate = sampling_rate

    def is_available(self) -> bool:
        return True

    def stream_prob(self, chunk: np.ndarray) -> float:
        """Evaluates a frame and returns speech probability scaled to RMS."""
        if chunk is None or len(chunk) == 0:
            return 0.0
        x = chunk.astype(np.float32)
        rms = float(np.sqrt(np.mean(x * x)))
        if rms < self.threshold:
            return 0.0
        # Smoothly scale from 0.55 at threshold to 1.0 at 2x threshold
        return min(1.0, 0.55 + 0.45 * (rms - self.threshold) / max(0.01, self.threshold))

    def reset_states(self) -> None:
        """No recurrent hidden states needed for pure energy VAD."""
        pass

    def detect_speech_regions(self, audio_path: Path) -> List[SpeechRegion]:
        """Detects speech regions by chunking audio into 32ms frames and thresholding RMS."""
        path = Path(audio_path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        try:
            data, sr = sf.read(str(path))
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            data = data.astype(np.float32)

            frame_size = int(sr * 0.032) # 32ms
            if frame_size <= 0:
                return []

            regions: List[SpeechRegion] = []
            in_speech = False
            start_s = 0.0

            for i in range(0, len(data), frame_size):
                chunk = data[i : i + frame_size]
                if len(chunk) == 0:
                    continue
                rms = float(np.sqrt(np.mean(chunk * chunk)))
                t_now = i / float(sr)

                if rms >= self.threshold:
                    if not in_speech:
                        in_speech = True
                        start_s = t_now
                else:
                    if in_speech:
                        end_s = t_now
                        if end_s - start_s >= 0.06: # ignore tiny spikes < 60ms
                            regions.append(
                                SpeechRegion(
                                    start=round(start_s, 2),
                                    end=round(end_s, 2),
                                    probability=0.85,
                                )
                            )
                        in_speech = False

            if in_speech:
                end_s = len(data) / float(sr)
                if end_s - start_s >= 0.06:
                    regions.append(
                        SpeechRegion(
                            start=round(start_s, 2),
                            end=round(end_s, 2),
                            probability=0.85,
                        )
                    )

            return regions

        except Exception as e:
            raise AudioProcessingError(f"Failed to detect speech regions with Energy VAD: {e}") from e

