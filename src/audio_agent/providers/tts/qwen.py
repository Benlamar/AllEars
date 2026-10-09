"""Qwen3-TTS provider using OpenAI-compatible /v1/audio/speech endpoint (vLLM)."""

import io
import json
from pathlib import Path
from typing import Iterator, Optional, Tuple
import urllib.error
import urllib.request
import numpy as np
import soundfile as sf

from audio_agent.audio.chunker import StreamingClauseChunker
from audio_agent.audio.sanitizer import SpeechTextSanitizer
from audio_agent.exceptions import AudioProcessingError, ProviderError, ProviderUnavailableError
from audio_agent.providers.base import TTSProvider


class QwenTTSProvider(TTSProvider):
    """High-fidelity neural Text-to-Speech using Qwen3-TTS VoiceDesign on vLLM."""

    provider_name: str = "qwen"
    capability: str = "tts"
    sample_rate: int = 24000
    execution_mode: str = "local" # local network vLLM server

    PRESET_VOICES = [
        "vivian", "serena", "ryan", "eric", "aiden", "dylan",
        "uncle_fu", "ono_anna", "sohee", "default"
    ]

    def __init__(
        self,
        endpoint: str = "http://192.168.1.6:8091/v1",
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        task_type: str = "CustomVoice",
        voice: str = "vivian",
        instructions: Optional[str] = "A calm, friendly voice with clear articulation.",
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        ref_audio: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/v1"):
            self.speech_url = f"{self.endpoint}/v1/audio/speech"
            self.models_url = f"{self.endpoint}/v1/models"
            self.voices_url = f"{self.endpoint}/v1/audio/voices"
        else:
            self.speech_url = f"{self.endpoint}/audio/speech"
            self.models_url = f"{self.endpoint}/models"
            self.voices_url = f"{self.endpoint}/audio/voices"

        self.model_name = model_name
        self.task_type = task_type
        self.voice = voice.lower() if voice else "vivian"
        self.instructions = instructions
        self.seed = seed
        self.temperature = temperature
        self.ref_audio = ref_audio
        self.timeout = timeout

    def is_available(self) -> bool:
        """Checks if the vLLM TTS server is reachable."""
        try:
            req = urllib.request.Request(self.models_url)
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("id") for m in data.get("data", [])]
                if models and self.model_name not in models:
                    # Auto-update active model name from server
                    self.model_name = models[0]
                return len(models) > 0
        except Exception:
            return False

    def synthesize(self, text: str, output_path: Optional[Path] = None) -> np.ndarray:
        """Synthesizes text into audio waveform via vLLM /v1/audio/speech."""
        clean_text = SpeechTextSanitizer.sanitize(text)
        if not clean_text:
            return np.zeros(0, dtype=np.float32)

        payload = {
            "model": self.model_name,
            "input": clean_text,
        }

        is_custom_voice = (
            "customvoice" in self.model_name.lower()
            or (self.task_type and self.task_type.lower() == "customvoice")
        )

        if is_custom_voice:
            payload["voice"] = (self.voice or "vivian").lower()
            if self.instructions:
                payload["instructions"] = self.instructions
        else:
            if self.task_type:
                payload["task_type"] = self.task_type
            if self.instructions:
                payload["instructions"] = self.instructions
            if self.seed is not None:
                payload["seed"] = self.seed
            if self.temperature is not None:
                payload["temperature"] = self.temperature
            if self.ref_audio is not None:
                payload["ref_audio"] = self.ref_audio

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.speech_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                wav_bytes = resp.read()

            audio_data, sr = sf.read(io.BytesIO(wav_bytes))
            samples = audio_data.astype(np.float32)
            self.sample_rate = sr

            if output_path:
                out_p = Path(output_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                with open(out_p, "wb") as f:
                    f.write(wav_bytes)

            return samples

        except urllib.error.URLError as e:
            raise ProviderUnavailableError(f"Failed to connect to Qwen TTS server at {self.speech_url}: {e}") from e
        except Exception as e:
            raise ProviderError(f"Qwen TTS synthesis failed: {e}") from e

    def stream_synthesize(self, text_stream: Iterator[str]) -> Iterator[Tuple[np.ndarray, int]]:
        """Consumes a stream of tokens, chunks into natural phrases, and yields (samples, sample_rate)."""
        chunker = StreamingClauseChunker()

        for phrase in chunker.chunk_stream(text_stream):
            if not phrase:
                continue
            samples = self.synthesize(phrase)
            if samples is not None and len(samples) > 0:
                yield samples, self.sample_rate

