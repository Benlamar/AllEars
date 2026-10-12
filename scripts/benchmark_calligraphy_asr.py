"""Benchmark script evaluating Faster-Whisper on the Calligraphy failure sequence.

Evaluates:
1. Baseline Faster-Whisper (base.en, beam=1) on the 3 calligraphy utterances.
2. The effect of beam size (beam=1 vs beam=3).
3. The effect of contextual initial_prompt ("calligraphy, art, penmanship").
4. Word-level confidence scores (w.confidence) to see if misrecognitions are detectable.
5. Behavior with added ambient noise.
"""

from pathlib import Path
import numpy as np
import soundfile as sf

from audio_agent.providers.asr.faster_whisper import FasterWhisperASRProvider
from audio_agent.providers.tts.kokoro import KokoroTTSProvider


UTTERANCES = [
    ("turn1_topic", "So, let's talk about calligraphy. How good should a person be at calligraphy?"),
    ("turn2_clarification", "I'm talking about calligraphy."),
    ("turn3_repair", "Calligraphy!"),
]


def add_gaussian_noise(audio: np.ndarray, snr_db: float = 18.0) -> np.ndarray:
    """Adds white noise to audio array at specified SNR in dB."""
    rms_signal = np.sqrt(np.mean(audio ** 2))
    if rms_signal == 0:
        return audio
    rms_noise = rms_signal / (10 ** (snr_db / 20.0))
    noise = np.random.normal(0, rms_noise, audio.shape).astype(np.float32)
    return audio + noise


def synthesize_benchmark_audio(out_dir: Path):
    """Generates audio for each utterance using Kokoro TTS."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tts = KokoroTTSProvider(voice="am_adam")
    audio_paths = {}

    for tag, text in UTTERANCES:
        path_clean = out_dir / f"{tag}_clean.wav"
        path_noisy = out_dir / f"{tag}_noisy.wav"

        # Synthesize clean
        samples = tts.synthesize(text, output_path=path_clean)
        
        # Create noisy version (SNR = 16 dB ambient noise)
        noisy = add_gaussian_noise(samples, snr_db=16.0)
        sf.write(str(path_noisy), noisy, tts.sample_rate)

        audio_paths[tag] = {"clean": path_clean, "noisy": path_noisy, "text": text}

    return audio_paths


def run_benchmark():
    audio_dir = Path("data/benchmark_audio")
    print("\n[1/3] Synthesizing benchmark audio utterances...")
    audio_map = synthesize_benchmark_audio(audio_dir)

    print("\n[2/3] Benchmarking Faster-Whisper (base.en) under different configurations...\n")
    print("=" * 95)
    print(f"{'Condition':<14} | {'Config':<28} | {'Recognized Transcript':<38} | {'Min Conf'}")
    print("=" * 95)

    configs = [
        ("Base (beam=1)", {"beam_size": 1, "initial_prompt": None}),
        ("Base (beam=3)", {"beam_size": 3, "initial_prompt": None}),
        ("Base + Context Prompt", {"beam_size": 1, "initial_prompt": "Topic: calligraphy, handwriting, art"}),
        ("Base + Context (beam=3)", {"beam_size": 3, "initial_prompt": "Topic: calligraphy, handwriting, art"}),
    ]

    for tag, data in audio_map.items():
        print(f"\n--- Utterance: {tag} (\" {data['text']} \") ---")
        for cond in ["clean", "noisy"]:
            wav_path = data[cond]
            for label, cfg in configs:
                provider = FasterWhisperASRProvider(
                    model_name="base.en",
                    device="cuda",
                    beam_size=cfg["beam_size"],
                    initial_prompt=cfg["initial_prompt"],
                )
                transcript, words = provider.transcribe(wav_path)
                confidences = [w.confidence for w in words if w.confidence is not None]
                min_c = min(confidences) if confidences else 0.0
                avg_c = np.mean(confidences) if confidences else 0.0

                print(f"{cond:<14} | {label:<28} | {transcript[:36]:<38} | {min_c:.2f} (avg {avg_c:.2f})")
                
                # Check how "calligraphy" specifically was recognized
                calligraphy_words = [w for w in words if "calligraph" in w.text.lower()]
                if calligraphy_words:
                    for cw in calligraphy_words:
                        pass
                else:
                    # Look for candidate words
                    pass

    print("\n" + "=" * 95)
    print("[3/3] Benchmark completed successfully.")


if __name__ == "__main__":
    run_benchmark()

