"""User voice anchoring and acoustic profile matching to filter background noise and speaker echo."""

import math
from typing import Optional, Tuple
import numpy as np


class UserVoiceAnchor:
    """Anchors the primary speaker's voice to filter out background noise and TTS speaker echo."""

    def __init__(
        self,
        pitch_tolerance_ratio: float = 0.45,
        min_hnr_db: float = 5.0,
        pitch_floor_hz: float = 65.0,
        pitch_ceiling_hz: float = 500.0,
    ):
        self.pitch_tolerance_ratio = pitch_tolerance_ratio
        self.min_hnr_db = min_hnr_db
        self.pitch_floor_hz = pitch_floor_hz
        self.pitch_ceiling_hz = pitch_ceiling_hz

        self.is_enrolled = False
        self.user_pitch_mean: Optional[float] = None
        self.user_pitch_min: Optional[float] = None
        self.user_pitch_max: Optional[float] = None
        self.user_hnr_mean: Optional[float] = None
        self.agent_pitch_mean: Optional[float] = None

    def _extract_pitch_and_hnr(
        self, samples: np.ndarray, sr: int = 16000
    ) -> Tuple[Optional[float], Optional[float]]:
        """Extracts mean F0 pitch and HNR (harmonicity) using Parselmouth or fast autocorrelation fallback."""
        if samples is None or len(samples) < int(sr * 0.15):
            return None, None

        x = samples.astype(np.float32)

        try:
            import parselmouth
            from parselmouth.praat import call

            sound = parselmouth.Sound(x, sampling_frequency=sr)
            pitch = sound.to_pitch(pitch_floor=self.pitch_floor_hz, pitch_ceiling=self.pitch_ceiling_hz)
            f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")

            harm = sound.to_harmonicity()
            hnr_mean = call(harm, "Get mean", 0, 0)

            mean_pitch = float(f0_mean) if not math.isnan(f0_mean) and f0_mean > 0 else None
            mean_hnr = float(hnr_mean) if not math.isnan(hnr_mean) else None
            return mean_pitch, mean_hnr

        except Exception:
            # Fallback: Normalized Autocorrelation for pitch estimation
            r = np.correlate(x, x, mode="full")
            r = r[len(r) // 2 :]
            # Search lag corresponding to pitch range
            min_lag = int(sr / self.pitch_ceiling_hz)
            max_lag = int(sr / self.pitch_floor_hz)
            if max_lag < len(r) and min_lag < max_lag:
                peak_lag = min_lag + np.argmax(r[min_lag:max_lag])
                est_f0 = float(sr / peak_lag) if peak_lag > 0 else None
                # Estimate HNR
                r0 = r[0] + 1e-6
                r_peak = r[peak_lag]
                hnr_est = 10.0 * math.log10(max(1e-3, r_peak / max(1e-3, r0 - r_peak)))
                return est_f0, hnr_est

        return None, None

    def enroll_user(self, audio_samples: np.ndarray, sr: int = 16000) -> bool:
        """Extracts and anchors the user's voice profile from an accepted utterance."""
        pitch, hnr = self._extract_pitch_and_hnr(audio_samples, sr)
        if pitch is not None and pitch > self.pitch_floor_hz:
            self.user_pitch_mean = pitch
            self.user_hnr_mean = hnr
            self.is_enrolled = True
            return True
        return False

    def enroll_agent(self, audio_samples: np.ndarray, sr: int = 16000) -> None:
        """Registers the agent's synthetic voice pitch to detect and reject speaker bleed."""
        pitch, _ = self._extract_pitch_and_hnr(audio_samples, sr)
        if pitch is not None and pitch > self.pitch_floor_hz:
            self.agent_pitch_mean = pitch

    def matches(self, candidate_samples: np.ndarray, sr: int = 16000, strict: bool = False) -> bool:
        """Determines if candidate speech audio matches the enrolled user rather than echo or background noise."""
        if not self.is_enrolled or self.user_pitch_mean is None:
            # If not yet enrolled, accept voiced speech with reasonable harmonicity
            _, hnr = self._extract_pitch_and_hnr(candidate_samples, sr)
            return hnr is None or hnr >= self.min_hnr_db

        cand_pitch, cand_hnr = self._extract_pitch_and_hnr(candidate_samples, sr)

        # 1. Check if audio matches the agent's synthetic voice (speaker echo)
        if cand_pitch is not None and self.agent_pitch_mean is not None:
            agent_diff = abs(cand_pitch - self.agent_pitch_mean) / self.agent_pitch_mean
            if agent_diff < 0.12:  # Candidate audio is virtually identical to the agent's voice!
                return False

        # 2. Non-strict mode (used for real-time barge-in trigger on short ~250ms chunks)
        if not strict:
            if cand_pitch is None:
                return True
            user_diff = abs(cand_pitch - self.user_pitch_mean) / self.user_pitch_mean
            return user_diff <= (self.pitch_tolerance_ratio * 1.25)

        # 3. Strict mode (evaluated on full completed utterance)
        if cand_pitch is None or (cand_hnr is not None and cand_hnr < self.min_hnr_db):
            return False

        user_diff = abs(cand_pitch - self.user_pitch_mean) / self.user_pitch_mean
        return user_diff <= self.pitch_tolerance_ratio

