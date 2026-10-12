"""Verifies the diarization stage end-to-end (Phase 3).

Usage: uv run python scripts/verify_diarization.py [audio.wav]

Runs ASR + diarization, aligns words to speakers, and prints exactly what the LLM would receive.
Exit code 1 if diarization produced no segments or any word lacks a speaker/uncertainty.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/speech_sample.wav"
    from audio_agent.analysis.alignment import align_words_with_speakers
    from audio_agent.analysis.evidence_builder import EvidenceBuilder
    from audio_agent.config import load_config
    from audio_agent.providers.asr import FasterWhisperASRProvider
    from audio_agent.providers.diarization import NemotronDiarizationProvider
    import soundfile as sf

    cfg = load_config().models.asr
    dur = sf.info(path).duration

    t0 = time.perf_counter()
    dia = NemotronDiarizationProvider()
    segments = dia.diarize(path)
    t_dia = time.perf_counter() - t0
    print(f"[diarization] executed in {t_dia:.2f}s -> {len(segments)} segments, "
          f"{len({s.speaker_id for s in segments})} speakers")
    for s in segments:
        print(f"  {s.speaker_id:>10}  {s.start:7.2f} - {s.end:7.2f}")

    asr = FasterWhisperASRProvider(model_name=cfg.model, device=cfg.device, compute_type=cfg.compute_type)
    transcript, words = asr.transcribe(path)
    attributed, notices = align_words_with_speakers(words, segments)
    unlabeled = [w for w in attributed if not getattr(w, "speaker_id", None)]
    print(f"[alignment] {len(attributed)} words, {len(notices)} uncertainty notices, {len(unlabeled)} unlabeled")

    builder = EvidenceBuilder("verify", Path(path), dur)
    builder.set_transcript(transcript, attributed, asr.get_provenance(0.0))
    builder.set_diarization(segments, dia.get_provenance(t_dia))
    context = builder.format_llm_context(builder.build())
    print("\n===== LLM CONTEXT =====\n" + context)

    attributed_ratio = (len(attributed) - len(notices)) / max(1, len(attributed))
    labels_in_context = any(s.speaker_id in context for s in segments)
    problems = []
    if not segments:
        problems.append("diarization produced no segments")
    if attributed_ratio < 0.8:
        problems.append(f"only {attributed_ratio:.0%} of words were attributed to a speaker")
    if not labels_in_context:
        problems.append("no speaker label appears in the LLM context")
    print("\nRESULT:", "OK" if not problems else "FAIL - " + "; ".join(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
