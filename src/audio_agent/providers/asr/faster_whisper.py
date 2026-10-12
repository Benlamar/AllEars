"""Faster-Whisper ASR Provider with word-level timestamps and CUDA acceleration."""

import time
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

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
        initial_prompt: Optional[str] = None,
        vad_filter: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.vad_filter = vad_filter
        self._model = None

    def warmup(self) -> None:
        """Loads the model and runs a tiny decode so the first real turn is fast."""
        try:
            model = self._get_model()
            segs, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language="en", beam_size=1)
            list(segs)
        except Exception:
            pass

    def _get_model(self):
        """Lazy loads the WhisperModel instance into memory."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                
                dev = self.device
                comp = self.compute_type
                if dev == "cuda":
                    import sys, os
                    import torch
                    if not torch.cuda.is_available():
                        dev = "cpu"
                        comp = "int8"
                    elif sys.platform == "win32":
                        torch_lib = Path(torch.__file__).parent / "lib"
                        if torch_lib.exists():
                            try:
                                os.add_dll_directory(str(torch_lib))
                            except Exception:
                                pass
                            os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")

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
        audio_input: Union[Path, str, np.ndarray],
        language: Optional[str] = "en",
    ) -> Tuple[str, List[Word]]:
        """Transcribes audio file or in-memory numpy array and returns text and timestamped words."""
        if isinstance(audio_input, np.ndarray):
            audio_source = audio_input.astype(np.float32)
        else:
            path = Path(audio_input)
            if not path.exists():
                raise AudioProcessingError(f"Audio file not found: {path}")
            audio_source = str(path)

        model = self._get_model()

        try:
            segments, info = model.transcribe(
                audio_source,
                language=language,
                beam_size=self.beam_size,
                condition_on_previous_text=False,
                word_timestamps=True,
                vad_filter=self.vad_filter, # Built-in Silero VAD filtering
                initial_prompt=self.initial_prompt,
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

