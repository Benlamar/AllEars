"""Registry for looking up and instantiating capability providers."""

from typing import Any, Dict, Type
from audio_agent.exceptions import ProviderUnavailableError
from audio_agent.providers.acoustic import ParselmouthAcousticProvider
from audio_agent.providers.asr import FasterWhisperASRProvider
from audio_agent.providers.base import BaseProvider
from audio_agent.providers.llm import OpenAICompatibleLLMProvider
from audio_agent.providers.diarization import NemotronDiarizationProvider
from audio_agent.providers.mocks import (
    MockASRProvider,
    MockAcousticProvider,
    MockDiarizationProvider,
    MockLLMProvider,
    MockSoundEventProvider,
    MockSpeakerVerificationProvider,
    MockTTSProvider,
    MockVADProvider,
)
from audio_agent.providers.sound_events import YAMNetSoundEventProvider
from audio_agent.providers.speaker_verification import AcousticSpeakerVerificationProvider
from audio_agent.providers.tts import KokoroTTSProvider, QwenTTSProvider
from audio_agent.providers.vad import EnergyVADProvider, FireRedVADProvider, SileroVADProvider


class ProviderRegistry:
    """Central registry mapping (capability, provider_name) to provider classes."""

    _registry: Dict[str, Dict[str, Type[BaseProvider]]] = {
        "asr": {
            "mock": MockASRProvider,
            "faster_whisper": FasterWhisperASRProvider,
            "faster-whisper": FasterWhisperASRProvider,
        },
        "vad": {
            "mock": MockVADProvider,
            "silero": SileroVADProvider,
            "silero_vad": SileroVADProvider,
            "silero-vad": SileroVADProvider,
            "firered": FireRedVADProvider,
            "firered_vad": FireRedVADProvider,
            "firered-vad": FireRedVADProvider,
            "fireredvad": FireRedVADProvider,
            "energy": EnergyVADProvider,
            "none": EnergyVADProvider,
        },
        "diarization": {
            "mock": MockDiarizationProvider,
            "nemotron": NemotronDiarizationProvider,
        },
        "sound_events": {
            "mock": MockSoundEventProvider,
            "yamnet": YAMNetSoundEventProvider,
        },
        "acoustic": {
            "mock": MockAcousticProvider,
            "parselmouth": ParselmouthAcousticProvider,
            "parselmouth_librosa": ParselmouthAcousticProvider,
        },
        "llm": {
            "mock": MockLLMProvider,
            "openai_compatible": OpenAICompatibleLLMProvider,
            "tensorrt_llm": OpenAICompatibleLLMProvider,
            "vllm": OpenAICompatibleLLMProvider,
        },
        "tts": {
            "mock": MockTTSProvider,
            "kokoro": KokoroTTSProvider,
            "qwen": QwenTTSProvider,
            "vllm": QwenTTSProvider,
            "qwen_tts": QwenTTSProvider,
        },
        "speaker_verification": {
            "mock": MockSpeakerVerificationProvider,
            "acoustic": AcousticSpeakerVerificationProvider,
            "acoustic_baseline": AcousticSpeakerVerificationProvider,
        },
    }

    @classmethod
    def register(cls, capability: str, provider_name: str, provider_class: Type[BaseProvider]) -> None:
        if capability not in cls._registry:
            cls._registry[capability] = {}
        cls._registry[capability][provider_name] = provider_class

    @classmethod
    def get(cls, capability: str, provider_name: str, **kwargs: Any) -> BaseProvider:
        cap_providers = cls._registry.get(capability, {})
        provider_cls = cap_providers.get(provider_name)
        if not provider_cls:
            raise ProviderUnavailableError(
                f"No provider '{provider_name}' registered for capability '{capability}'. "
                f"Available: {list(cap_providers.keys())}"
            )
        return provider_cls(**kwargs)

    @classmethod
    def list_available(cls) -> Dict[str, list[str]]:
        return {cap: list(providers.keys()) for cap, providers in cls._registry.items()}

