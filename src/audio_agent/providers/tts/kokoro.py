"""Kokoro-82M ONNX Text-to-Speech provider with streaming synthesis."""

import os
from pathlib import Path
from typing import Iterator, Optional, Tuple
import urllib.request
import numpy as np

from audio_agent.audio.chunker import StreamingClauseChunker
from audio_agent.audio.sanitizer import SpeechTextSanitizer
from audio_agent.exceptions import AudioProcessingError, ProviderError
from audio_agent.providers.base import TTSProvider


class KokoroTTSProvider(TTSProvider):
    """High-quality 24kHz neural Text-to-Speech using Kokoro-82M ONNX."""

    provider_name: str = "kokoro"
    capability: str = "tts"
    sample_rate: int = 24000
    execution_mode: str = "local"

    MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
    VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        model_name: str = "kokoro-v1.0.onnx",
        voice: str = "af_sarah",
        speed: float = 1.0,
        device: str = "cpu",
    ):
        if model_dir:
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = Path.home() / ".cache" / "audio_agent" / "models" / "kokoro"

        self.model_name = model_name
        self.voice = voice
        self.speed = speed
        self.device = device
        self._kokoro = None

    def _ensure_weights(self) -> Tuple[Path, Path]:
        """Ensures that ONNX model and voice embeddings are present locally."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / "kokoro-v1.0.onnx"
        voices_path = self.model_dir / "voices-v1.0.bin"

        def _download_file(url: str, dest: Path) -> None:
            if dest.exists() and dest.stat().st_size > 0:
                return
            temp_path = dest.with_suffix(".tmp")
            try:
                urllib.request.urlretrieve(url, str(temp_path))
                temp_path.replace(dest)
            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                raise ProviderError(f"Failed to download Kokoro asset from {url}: {e}") from e

        if not model_path.exists() or model_path.stat().st_size == 0:
            _download_file(self.MODEL_URL, model_path)

        if not voices_path.exists() or voices_path.stat().st_size == 0:
            _download_file(self.VOICES_URL, voices_path)

        return model_path, voices_path

    def _get_engine(self):
        """Lazy loads Kokoro-ONNX instance with optimized session options."""
        if self._kokoro is None:
            try:
                import kokoro_onnx
                import onnxruntime as ort

                model_path, voices_path = self._ensure_weights()

                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_opts.intra_op_num_threads = 2
                sess_opts.inter_op_num_threads = 1

                providers = ["CPUExecutionProvider"]
                if (self.device == "cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

                session = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=providers)
                instance = kokoro_onnx.Kokoro.__new__(kokoro_onnx.Kokoro)
                instance._setup(
                    session=session,
                    model_path=str(model_path),
                    voices_path=str(voices_path),
                    espeak_config=None,
                    vocab_config=None,
                )
                self._kokoro = instance
            except Exception as e:
                raise ProviderError(f"Failed to initialize Kokoro TTS engine: {e}") from e
        return self._kokoro

    def is_available(self) -> bool:
        """Returns True if kokoro_onnx and phonemizer dependencies are available."""
        try:
            import kokoro_onnx
            return True
        except ImportError:
            return False

    def synthesize(self, text: str, output_path: Optional[Path] = None) -> np.ndarray:
        """Synthesizes text to a 24kHz float32 audio waveform."""
        clean_text = SpeechTextSanitizer.sanitize(text)
        if not clean_text:
            return np.zeros(0, dtype=np.float32)

        engine = self._get_engine()
        try:
            samples, sr = engine.create(clean_text, voice=self.voice, speed=self.speed)
            samples = samples.astype(np.float32)

            if output_path:
                import soundfile as sf
                out_p = Path(output_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(out_p), samples, sr)

            return samples
        except Exception as e:
            raise ProviderError(f"Kokoro TTS synthesis failed: {e}") from e

    def stream_synthesize(self, text_stream: Iterator[str]) -> Iterator[Tuple[np.ndarray, int]]:
        """Consumes a stream of tokens, chunks into natural phrases, and yields (samples, sample_rate)."""
        engine = self._get_engine()
        chunker = StreamingClauseChunker()

        for phrase in chunker.chunk_stream(text_stream):
            if not phrase:
                continue
            try:
                samples, sr = engine.create(phrase, voice=self.voice, speed=self.speed)
                samples = samples.astype(np.float32)
                yield samples, sr
            except Exception as e:
                raise ProviderError(f"Kokoro streaming TTS synthesis failed on phrase '{phrase}': {e}") from e

