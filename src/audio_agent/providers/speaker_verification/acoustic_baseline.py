"""Acoustic pitch and harmonicity speaker verification provider (in-memory baseline)."""

from typing import Any, List, Optional, Tuple
import numpy as np

from audio_agent.audio.voice_anchor import UserVoiceAnchor
from audio_agent.providers.base import SpeakerVerificationProvider


class AcousticSpeakerVerificationProvider(SpeakerVerificationProvider):
    """Verifies speaker using fundamental pitch F0 and HNR (harmonicity) in memory."""

    provider_name: str = "acoustic_baseline"
    execution_mode: str = "local"

    def __init__(
        self,
        pitch_tolerance_ratio: float = 0.45,
        min_hnr_db: float = 5.0,
        pitch_floor_hz: float = 65.0,
        pitch_ceiling_hz: float = 500.0,
    ):
        self._anchor = UserVoiceAnchor(
            pitch_tolerance_ratio=pitch_tolerance_ratio,
            min_hnr_db=min_hnr_db,
            pitch_floor_hz=pitch_floor_hz,
            pitch_ceiling_hz=pitch_ceiling_hz,
        )

    def is_available(self) -> bool:
        return True

    @property
    def is_enrolled(self) -> bool:
        return self._anchor.is_enrolled

    def enroll(self, utterances: List[Any], sample_rate: int = 16000) -> bool:
        """Enrolls the user using clean speech waveforms."""
        if not utterances:
            return False
        # Aggregate or take the first clean voiced waveform
        for audio in utterances:
            if audio is not None and len(audio) > 0:
                success = self._anchor.enroll_user(audio, sr=sample_rate)
                if success:
                    return True
        return False

    def verify(self, candidate_audio: Any, sample_rate: int = 16000) -> Tuple[bool, float]:
        """Verifies candidate audio against the enrolled voice profile.
        
        Returns (is_match, similarity_score).
        """
        if not self.is_enrolled:
            return True, 1.0

        if candidate_audio is None or len(candidate_audio) == 0:
            return False, 0.0

        matched = self._anchor.matches(candidate_audio, sr=sample_rate, strict=False)
        # Score proportional to pitch closeness
        cand_pitch, _ = self._anchor._extract_pitch_and_hnr(candidate_audio, sample_rate)
        if cand_pitch is None or self._anchor.user_pitch_mean is None:
            score = 0.7 if matched else 0.0
        else:
            diff = abs(cand_pitch - self._anchor.user_pitch_mean) / self._anchor.user_pitch_mean
            score = max(0.0, 1.0 - diff)
        return matched, float(score)

    def reset_session(self) -> None:
        """Clears in-memory session reference (zero persistence)."""
        self._anchor = UserVoiceAnchor()
