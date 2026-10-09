"""Silero VAD provider using ONNX runtime and SoundFile."""

from pathlib import Path
from typing import List, Optional
import numpy as np
import scipy.signal as signal
import soundfile as sf
import torch

from audio_agent.exceptions import AudioProcessingError, ProviderError
from audio_agent.providers.base import VADProvider
from audio_agent.schemas import SpeechRegion


class SileroVADProvider(VADProvider):
    """Voice Activity Detection using the Silero VAD ONNX model."""

    provider_name: str = "silero"
    execution_mode: str = "local"

    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16000, device: str = "cpu"):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from silero_vad import load_silero_vad
                use_cuda = (self.device == "cuda") and torch.cuda.is_available()
                if use_cuda:
                    self._model = load_silero_vad(onnx=False).to("cuda")
                else:
                    self._model = load_silero_vad(onnx=True)
            except Exception as e:
                raise ProviderError(f"Failed to load Silero VAD model: {e}") from e
        return self._model

    def is_available(self) -> bool:
        try:
            import silero_vad
            return True
        except ImportError:
            return False

    def detect_speech_regions(self, audio_path: Path) -> List[SpeechRegion]:
        path = Path(audio_path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        try:
            data, sr = sf.read(str(path))
            # Convert multi-channel to mono
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # Resample to 16000 Hz if necessary
            target_sr = 16000
            if sr != target_sr:
                num_samples = int(len(data) * target_sr / sr)
                data = signal.resample(data, num_samples).astype(np.float32)
            else:
                data = data.astype(np.float32)

            tensor_wav = torch.from_numpy(data)
            if (self.device == "cuda") and torch.cuda.is_available():
                tensor_wav = tensor_wav.to("cuda")
            model = self._get_model()

            from silero_vad import get_speech_timestamps
            timestamps = get_speech_timestamps(
                tensor_wav,
                model,
                threshold=self.threshold,
                sampling_rate=target_sr,
                return_seconds=True,
            )

            regions = [
                SpeechRegion(
                    start=round(ts["start"], 2),
                    end=round(ts["end"], 2),
                    probability=self.threshold,
                )
                for ts in timestamps
            ]
            return regions

        except Exception as e:
            raise ProviderError(f"Silero VAD processing failed: {e}") from e

