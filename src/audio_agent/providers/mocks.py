"""Mock capability providers for testing and scaffolding without external model downloads."""

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union
import numpy as np

from audio_agent.providers.base import (
    ASRProvider,
    AcousticProvider,
    DiarizationProvider,
    LLMProvider,
    SoundEventProvider,
    TTSProvider,
    VADProvider,
)
from audio_agent.schemas import (
    AcousticFeatures,
    SoundEvent,
    SpeakerSegment,
    SpeechRegion,
    Word,
)


class MockASRProvider(ASRProvider):
    provider_name: str = "mock"
    model_name: str = "mock-whisper"

    def is_available(self) -> bool:
        return True

    def transcribe(self, audio_path: Path, language: Optional[str] = "en") -> Tuple[str, List[Word]]:
        words = [
            Word(text="Hello", start=0.0, end=0.4, confidence=0.98),
            Word(text="and", start=0.42, end=0.6, confidence=0.95),
            Word(text="welcome", start=0.62, end=1.1, confidence=0.97),
            Word(text="to", start=1.12, end=1.25, confidence=0.99),
            Word(text="the", start=1.26, end=1.4, confidence=0.96),
            Word(text="meeting.", start=1.42, end=1.95, confidence=0.98),
            Word(text="Let's", start=2.5, end=2.8, confidence=0.94),
            Word(text="review", start=2.82, end=3.2, confidence=0.97),
            Word(text="the", start=3.22, end=3.35, confidence=0.99),
            Word(text="action", start=3.36, end=3.7, confidence=0.98),
            Word(text="items.", start=3.72, end=4.1, confidence=0.96),
        ]
        text = "Hello and welcome to the meeting. Let's review the action items."
        return text, words


class MockVADProvider(VADProvider):
    provider_name: str = "mock"
    model_name: str = "mock-silero"

    def __init__(self, **kwargs):
        self._mock_prob = 0.8

    def is_available(self) -> bool:
        return True

    def stream_prob(self, chunk: np.ndarray) -> float:
        return self._mock_prob

    def reset_states(self) -> None:
        pass

    def detect_speech_regions(self, audio_path: Path) -> List[SpeechRegion]:
        return [
            SpeechRegion(start=0.0, end=2.0, probability=0.99),
            SpeechRegion(start=2.4, end=4.2, probability=0.98),
        ]


class MockDiarizationProvider(DiarizationProvider):
    provider_name: str = "mock"
    model_name: str = "mock-nemotron"

    def is_available(self) -> bool:
        return True

    def diarize(self, audio_path: Path, num_speakers: Optional[int] = None) -> List[SpeakerSegment]:
        return [
            SpeakerSegment(speaker_id="speaker_1", start=0.0, end=2.1, confidence=0.95),
            SpeakerSegment(speaker_id="speaker_2", start=2.4, end=4.3, confidence=0.92),
        ]


class MockSoundEventProvider(SoundEventProvider):
    provider_name: str = "mock"
    model_name: str = "mock-yamnet"

    def __init__(self, **kwargs):
        self._mock_events = [
            SoundEvent(label="Speech", start=0.0, end=2.0, score=0.96),
            SoundEvent(label="Cough", start=2.1, end=2.4, score=0.88, description="Isolated dry cough"),
            SoundEvent(label="Speech", start=2.4, end=4.2, score=0.94),
        ]

    def is_available(self) -> bool:
        return True

    def detect_events_waveform(self, wav: np.ndarray, sr: int = 16000) -> List[SoundEvent]:
        return self._mock_events

    def detect_events(self, audio_path: Path) -> List[SoundEvent]:
        return self._mock_events


class MockAcousticProvider(AcousticProvider):
    provider_name: str = "mock"
    model_name: str = "mock-parselmouth"

    def is_available(self) -> bool:
        return True

    def extract_features(self, audio_path: Path) -> AcousticFeatures:
        return AcousticFeatures(
            pitch_mean_hz=145.2,
            pitch_min_hz=98.0,
            pitch_max_hz=215.0,
            pitch_std_hz=24.5,
            intensity_mean_db=68.4,
            speaking_rate_syllables_per_sec=4.2,
            pause_count=1,
            pause_duration_ratio=0.07,
            jitter_local=0.012,
            shimmer_local=0.038,
            hnr_db=18.5,
            notes="Valid acoustic measurements extracted across speech intervals.",
        )


class MockLLMProvider(LLMProvider):
    provider_name: str = "mock"
    model_name: str = "mock-qwen2.5"

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        prompt_str = prompt if isinstance(prompt, str) else " ".join(m.get("content", "") for m in prompt)
        if "action items" in prompt_str.lower() or "summary" in prompt_str.lower():
            return (
                "Executive Summary:\n"
                "- Speaker 1 opened the meeting from 0.0s to 2.0s.\n"
                "- A brief cough sound event was detected at 2.1s - 2.4s.\n"
                "- Speaker 2 initiated review of the action items starting at 2.4s.\n"
                "- Overall acoustic metrics show steady pitch (mean 145.2 Hz) and standard speaking rate."
            )
        return (
            "Grounded Response based on analyzed evidence:\n"
            f"Question answered using structured timeline observations: '{prompt_str[:60]}...'."
        )

    def generate_stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        full_text = self.generate(prompt, system_prompt, temperature)
        # Yield words as streaming tokens
        tokens = full_text.split(" ")
        for i, token in enumerate(tokens):
            suffix = " " if i < len(tokens) - 1 else ""
            yield token + suffix



class MockTTSProvider(TTSProvider):
    provider_name: str = "mock"
    model_name: str = "mock-kokoro"
    sample_rate: int = 24000

    def is_available(self) -> bool:
        return True

    def synthesize(self, text: str, output_path: Optional[Path] = None) -> np.ndarray:
        duration = max(0.2, len(text.split()) * 0.15)
        num_samples = int(self.sample_rate * duration)
        samples = np.zeros(num_samples, dtype=np.float32)
        if output_path:
            import soundfile as sf
            sf.write(str(output_path), samples, self.sample_rate)
        return samples

    def stream_synthesize(self, text_stream: Iterator[str]) -> Iterator[Tuple[np.ndarray, int]]:
        for phrase in text_stream:
            clean = phrase.strip()
            if clean:
                yield self.synthesize(clean), self.sample_rate


