"""FireRedVAD provider using FireRedStreamVad and FireRedVad (DFSMN architecture)."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import scipy.signal as signal
import soundfile as sf
import torch

from audio_agent.exceptions import AudioProcessingError, ProviderError
from audio_agent.providers.base import VADProvider
from audio_agent.schemas import SpeechRegion


class FireRedVADProvider(VADProvider):
    """Voice Activity Detection using FireRedVAD (DFSMN industrial-grade VAD)."""

    provider_name: str = "firered"
    execution_mode: str = "local"

    def __init__(
        self,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        device: str = "cpu",
        model_dir: Optional[str] = None,
    ):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.device = device
        self.model_dir = model_dir
        self._stream_model = None
        self._offline_model = None
        self._model_paths: Optional[Tuple[str, str]] = None

    def _resolve_model_paths(self) -> Tuple[str, str]:
        """Resolves local paths to Stream-VAD and VAD model weight directories."""
        if self._model_paths is not None:
            return self._model_paths

        candidates = [self.model_dir, "models/firered_vad", "models/FireRedVAD"]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                stream_dir = os.path.join(candidate, "Stream-VAD")
                vad_dir = os.path.join(candidate, "VAD")
                if os.path.exists(stream_dir) and os.path.exists(vad_dir):
                    self._model_paths = (stream_dir, vad_dir)
                    return self._model_paths

        try:
            from huggingface_hub import snapshot_download

            root = snapshot_download(
                "FireRedTeam/FireRedVAD",
                allow_patterns=["Stream-VAD/*", "VAD/*"],
            )
            stream_dir = os.path.join(root, "Stream-VAD")
            vad_dir = os.path.join(root, "VAD")
            self._model_paths = (stream_dir, vad_dir)
            return self._model_paths
        except Exception as e:
            raise ProviderError(f"Failed to download FireRedVAD weights from HuggingFace: {e}") from e

    def _get_stream_model(self):
        if self._stream_model is None:
            try:
                from fireredvad import FireRedStreamVad, FireRedStreamVadConfig

                stream_dir, _ = self._resolve_model_paths()
                use_gpu = (self.device == "cuda") and torch.cuda.is_available()
                cfg = FireRedStreamVadConfig(
                    use_gpu=use_gpu,
                    speech_threshold=self.threshold,
                )
                self._stream_model = FireRedStreamVad.from_pretrained(stream_dir, config=cfg)
            except Exception as e:
                raise ProviderError(f"Failed to initialize FireRedStreamVad: {e}") from e
        return self._stream_model

    def _get_offline_model(self):
        if self._offline_model is None:
            try:
                from fireredvad import FireRedVad, FireRedVadConfig

                _, vad_dir = self._resolve_model_paths()
                use_gpu = (self.device == "cuda") and torch.cuda.is_available()
                cfg = FireRedVadConfig(
                    use_gpu=use_gpu,
                    speech_threshold=self.threshold,
                )
                self._offline_model = FireRedVad.from_pretrained(vad_dir, config=cfg)
            except Exception as e:
                raise ProviderError(f"Failed to initialize FireRedVad: {e}") from e
        return self._offline_model

    def is_available(self) -> bool:
        try:
            import fireredvad
            return True
        except ImportError:
            return False

    def stream_prob(self, chunk: np.ndarray) -> float:
        """Evaluates an audio frame or chunk and returns the smoothed speech probability [0.0 - 1.0]."""
        model = self._get_stream_model()

        # Handle channels
        if len(chunk.shape) > 1:
            chunk = np.mean(chunk, axis=1)

        # FireRedVAD expects 16-bit PCM integer range [-32768, 32767]
        if np.issubdtype(chunk.dtype, np.floating):
            chunk = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
        elif chunk.dtype != np.int16:
            chunk = chunk.astype(np.int16)

        try:
            res = model.detect_chunk(chunk)
            if not res:
                return 0.0
            return float(res[-1].smoothed_prob)
        except Exception:
            return 0.0

    def reset_states(self) -> None:
        """Resets streaming recurrent cache state between turns."""
        if self._stream_model is not None:
            self._stream_model.reset()

    def detect_speech_regions(self, audio_path: Path) -> List[SpeechRegion]:
        path = Path(audio_path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        try:
            data, sr = sf.read(str(path))
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            target_sr = 16000
            if sr != target_sr:
                num_samples = int(len(data) * target_sr / sr)
                data = signal.resample(data, num_samples)

            if np.issubdtype(data.dtype, np.floating):
                data = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
            elif data.dtype != np.int16:
                data = data.astype(np.int16)

            model = self._get_offline_model()
            res, _ = model.detect(data)

            regions: List[SpeechRegion] = []
            if res and "timestamps" in res:
                for start_s, end_s in res["timestamps"]:
                    regions.append(
                        SpeechRegion(
                            start=float(start_s),
                            end=float(end_s),
                            probability=self.threshold,
                        )
                    )
            return regions
        except Exception as e:
            raise AudioProcessingError(f"FireRedVAD speech detection failed: {e}") from e
