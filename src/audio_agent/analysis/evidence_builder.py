"""Evidence building and structured synthesis for downstream LLM reasoning."""

from pathlib import Path
from typing import Dict, List, Optional
import json

from audio_agent.analysis.alignment import align_words_with_speakers
from audio_agent.schemas import (
    AcousticFeatures,
    AnalysisEvidence,
    Provenance,
    SoundEvent,
    SpeakerSegment,
    SpeechRegion,
    UncertaintyNotice,
    Word,
)


class EvidenceBuilder:
    """Consolidates observations across pipeline stages into a unified evidence object."""

    def __init__(self, recording_id: str, filepath: Path, duration_seconds: float):
        self.recording_id = recording_id
        self.filepath = filepath
        self.duration_seconds = duration_seconds
        self.words: List[Word] = []
        self.transcript: str = ""
        self.speaker_segments: List[SpeakerSegment] = []
        self.speech_regions: List[SpeechRegion] = []
        self.sound_events: List[SoundEvent] = []
        self.acoustic_features = AcousticFeatures()
        self.uncertainties: List[UncertaintyNotice] = []
        self.provenance: Dict[str, Provenance] = {}
        self.speaker_aliases: Dict[str, str] = {}

    def set_transcript(self, transcript: str, words: List[Word], prov: Provenance) -> None:
        self.transcript = transcript
        self.words = words
        self.provenance["asr"] = prov

    def set_diarization(self, segments: List[SpeakerSegment], prov: Provenance) -> None:
        self.speaker_segments = segments
        self.provenance["diarization"] = prov

    def set_vad(self, regions: List[SpeechRegion], prov: Provenance) -> None:
        self.speech_regions = regions
        self.provenance["vad"] = prov

    def set_sound_events(self, events: List[SoundEvent], prov: Provenance) -> None:
        self.sound_events = events
        self.provenance["sound_events"] = prov

    def set_acoustics(self, features: AcousticFeatures, prov: Provenance) -> None:
        self.acoustic_features = features
        self.provenance["acoustic"] = prov

    def add_speaker_alias(self, speaker_id: str, alias: str) -> None:
        self.speaker_aliases[speaker_id] = alias

    def build(self) -> AnalysisEvidence:
        # Align words with speakers
        aligned_words, align_uncertainties = align_words_with_speakers(
            self.words, self.speaker_segments
        )
        all_uncertainties = list(self.uncertainties) + align_uncertainties

        return AnalysisEvidence(
            recording_id=self.recording_id,
            filename=self.filepath.name,
            duration_seconds=self.duration_seconds,
            transcript=self.transcript,
            words=aligned_words,
            speaker_segments=self.speaker_segments,
            speech_regions=self.speech_regions,
            sound_events=self.sound_events,
            acoustic_features=self.acoustic_features,
            speaker_aliases=self.speaker_aliases,
            uncertainties=all_uncertainties,
            provenance=self.provenance,
        )

    @staticmethod
    def format_llm_context(evidence: AnalysisEvidence) -> str:
        """Formats the evidence as a concise, structured markdown prompt context for the LLM."""
        sections = [
            f"# Audio Recording Evidence: {evidence.filename} (ID: {evidence.recording_id})",
            f"- Duration: {evidence.duration_seconds:.2f} seconds",
            "",
            "## Attributed Transcript & Speaker Turns:",
        ]
        
        attributed = evidence.get_attributed_transcript()
        sections.append(attributed if attributed else "(No speech transcribed)")
        
        if evidence.sound_events:
            sections.append("\n## Detected Sound Events:")
            for ev in evidence.sound_events:
                desc = f" ({ev.description})" if ev.description else ""
                sections.append(f"- [{ev.start:.2f}s - {ev.end:.2f}s] {ev.label} (confidence: {ev.score:.2f}){desc}")

        ac = evidence.acoustic_features
        sections.append("\n## Objective Acoustic Measurements:")
        if ac.pitch_mean_hz:
            sections.append(f"- Pitch (F0): mean {ac.pitch_mean_hz:.1f} Hz (min: {ac.pitch_min_hz:.1f}, max: {ac.pitch_max_hz:.1f})")
        if ac.intensity_mean_db:
            sections.append(f"- Intensity/Loudness: mean {ac.intensity_mean_db:.1f} dB")
        if ac.speaking_rate_syllables_per_sec:
            sections.append(f"- Speaking Rate: {ac.speaking_rate_syllables_per_sec:.1f} syllables/sec")
        if ac.jitter_local is not None and ac.shimmer_local is not None:
            sections.append(f"- Voice Quality: Jitter {ac.jitter_local:.3f}, Shimmer {ac.shimmer_local:.3f}, HNR {ac.hnr_db or 0.0:.1f} dB")

        if evidence.uncertainties:
            sections.append("\n## Uncertainty Notices / Caveats:")
            for unc in evidence.uncertainties:
                timing = f" [{unc.start:.2f}s - {unc.end:.2f}s]" if unc.start is not None else ""
                sections.append(f"- ({unc.stage}){timing} {unc.message}")

        return "\n".join(sections)

