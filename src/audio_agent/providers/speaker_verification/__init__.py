"""Speaker verification providers package."""

from audio_agent.providers.base import SpeakerVerificationProvider
from audio_agent.providers.speaker_verification.acoustic_baseline import AcousticSpeakerVerificationProvider

__all__ = ["SpeakerVerificationProvider", "AcousticSpeakerVerificationProvider"]
