"""Faster-Whisper ASR Provider with word-level timestamps and CUDA acceleration."""

import time
from pathlib import Path
from typing import List, Optional, Tuple

from audio_agent.exceptions import AudioProcessingError, ProviderError
from audio_agent.providers.base import ASRProvider
from audio_agent.schemas import Word


class FasterWhisperASRProvider(ASRProvider):
    """Local ASR provider powered by faster-whisper (CTranslate2)."""

    provider_name: str = "faster_whisper"
    execution_mode: str = "local"

    def __init__(
        self,
        model_name: str = "base.en",
        device: str = "cuda",
        compute_type: str = "float16",
        download_root: Optional[str] = None,
        beam_size: int = 1,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self.beam_size = beam_size
        self._model = None

    def _get_model(self):
        """Lazy loads the WhisperModel instance into memory."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                
                # Check device fallback if CUDA isn't available
                dev = self.device
                comp = self.compute_type
                if dev == "cuda":
                    import torch
                    if not torch.cuda.is_available():
                        dev = "cpu"
                        comp = "int8"

                self._model = WhisperModel(
                    self.model_name,
                    device=dev,
                    compute_type=comp,
                    download_root=self.download_root,
                )
            except Exception as e:
                raise ProviderError(
                    f"Failed to load faster-whisper model '{self.model_name}' on {self.device}: {e}"
                ) from e
        return self._model

    def is_available(self) -> bool:
        """Returns True if faster-whisper package is installed."""
        try:
            import faster_whisper
            return True
        except ImportError:
            return False

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = "en",
    ) -> Tuple[str, List[Word]]:
        """Transcribes audio file and returns text and timestamped words."""
        path = Path(audio_path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        model = self._get_model()

        try:
            segments, info = model.transcribe(
                str(path),
                language=language,
                beam_size=self.beam_size,
                condition_on_previous_text=False,
                word_timestamps=True,
                vad_filter=True, # Built-in Silero VAD filtering
            )

            words: List[Word] = []
            text_chunks: List[str] = []

            for segment in segments:
                text_chunks.append(segment.text.strip())
                if segment.words:
                    for w in segment.words:
                        cleaned = w.word.strip()
                        if cleaned:
                            words.append(
                                Word(
                                    text=cleaned,
                                    start=round(w.start, 2),
                                    end=round(w.end, 2),
                                    confidence=round(w.probability, 2) if w.probability is not None else None,
                                )
                            )

            full_transcript = " ".join(text_chunks).strip()
            return full_transcript, words

        except Exception as e:
            raise ProviderError(f"Faster-Whisper transcription failed: {e}") from e

