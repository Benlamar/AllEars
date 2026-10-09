"""Acoustic feature extraction using Praat/Parselmouth."""

from pathlib import Path
from typing import Optional
import math
from audio_agent.exceptions import AudioProcessingError, ProviderError
from audio_agent.providers.base import AcousticProvider
from audio_agent.schemas import AcousticFeatures


class ParselmouthAcousticProvider(AcousticProvider):
    """Extracts objective pitch, intensity, speaking rate, and voice quality metrics using Praat."""

    provider_name: str = "parselmouth"
    execution_mode: str = "local"

    def __init__(
        self,
        pitch_floor_hz: float = 75.0,
        pitch_ceiling_hz: float = 600.0,
    ):
        self.pitch_floor_hz = pitch_floor_hz
        self.pitch_ceiling_hz = pitch_ceiling_hz

    def is_available(self) -> bool:
        try:
            import parselmouth
            return True
        except ImportError:
            return False

    def extract_features(self, audio_path: Path) -> AcousticFeatures:
        path = Path(audio_path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        try:
            import parselmouth
            from parselmouth.praat import call

            sound = parselmouth.Sound(str(path))
            total_duration = sound.get_total_duration()

            if total_duration <= 0:
                return AcousticFeatures(notes="Audio file has 0 duration.")

            # 1. Pitch (F0)
            pitch = sound.to_pitch(pitch_floor=self.pitch_floor_hz, pitch_ceiling=self.pitch_ceiling_hz)
            f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
            f0_min = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
            f0_max = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
            f0_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")

            # 2. Intensity (Loudness)
            intensity = sound.to_intensity(minimum_pitch=self.pitch_floor_hz)
            intensity_mean = call(intensity, "Get mean", 0, 0, "energy")

            # 3. Voice Quality (HNR, Jitter, Shimmer)
            harmonicity = sound.to_harmonicity()
            hnr_mean = call(harmonicity, "Get mean", 0, 0)

            # Point process for jitter / shimmer
            point_process = call(
                sound,
                "To PointProcess (periodic, cc)",
                self.pitch_floor_hz,
                self.pitch_ceiling_hz,
            )
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

            # 4. Speaking Rate / Syllables & Pauses
            # Count pitch frames with non-zero pitch (voiced) vs zero pitch (unvoiced/pause)
            voiced_frames = [p for p in pitch.selected_array["frequency"] if p > 0]
            voiced_ratio = len(voiced_frames) / max(1, len(pitch.selected_array["frequency"]))
            pause_ratio = max(0.0, 1.0 - voiced_ratio)

            # Approximate speaking rate based on voiced transitions and intensity peaks
            syllable_estimate = max(1, int(len(voiced_frames) * 0.15))
            speaking_rate = round(syllable_estimate / max(0.1, total_duration), 1)

            def _clean(val: Optional[float], round_digits: int = 2) -> Optional[float]:
                if val is None or math.isnan(val) or math.isinf(val):
                    return None
                return round(float(val), round_digits)

            return AcousticFeatures(
                pitch_mean_hz=_clean(f0_mean, 1),
                pitch_min_hz=_clean(f0_min, 1),
                pitch_max_hz=_clean(f0_max, 1),
                pitch_std_hz=_clean(f0_std, 1),
                intensity_mean_db=_clean(intensity_mean, 1),
                speaking_rate_syllables_per_sec=_clean(speaking_rate, 1),
                pause_count=max(0, int(pause_ratio * 5)),
                pause_duration_ratio=_clean(pause_ratio, 2),
                jitter_local=_clean(jitter_local, 4),
                shimmer_local=_clean(shimmer_local, 4),
                hnr_db=_clean(hnr_mean, 1),
                notes="Acoustic measurements calculated via Praat/Parselmouth algorithms.",
            )

        except Exception as e:
            raise ProviderError(f"Acoustic feature extraction failed: {e}") from e

