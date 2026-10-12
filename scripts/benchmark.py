"""Per-component latency benchmark (Phase 0/4).

Usage:
    uv run python scripts/benchmark.py [--audio data/speech_sample.wav] [--runs 5]
        [--only asr,vad,diarization,sound_events,acoustic,llm,tts_qwen,tts_kokoro] [--save]
"""

import argparse
import statistics
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TTS_TEXT = (
    "Preheat your oven to three fifty. Then mix the flour, sugar, and eggs until smooth. "
    "Pour the batter into a greased pan and bake for thirty minutes."
)


def timed(fn, runs):
    fn()  # warmup (model load / caches)
    xs = []
    for _ in range(runs):
        t0 = time.perf_counter()
        extra = fn()
        xs.append((time.perf_counter() - t0, extra))
    return xs


def stats(xs):
    ts = sorted(x[0] for x in xs)
    p50 = statistics.median(ts)
    p95 = ts[min(len(ts) - 1, int(round(0.95 * (len(ts) - 1))))]
    return p50, p95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="data/speech_sample.wav")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}

    import soundfile as sf
    from audio_agent.providers.registry import ProviderRegistry  # noqa: F401  (registers providers)
    from audio_agent.providers.acoustic import ParselmouthAcousticProvider
    from audio_agent.providers.asr import FasterWhisperASRProvider
    from audio_agent.providers.diarization import NemotronDiarizationProvider
    from audio_agent.providers.llm import OpenAICompatibleLLMProvider
    from audio_agent.providers.sound_events import YAMNetSoundEventProvider
    from audio_agent.providers.tts.kokoro import KokoroTTSProvider
    from audio_agent.providers.tts.qwen import QwenTTSProvider
    from audio_agent.providers.vad import SileroVADProvider
    from audio_agent.config import load_config

    audio = Path(args.audio)
    info = sf.info(str(audio)) if audio.exists() else None
    dur = info.duration if info else 0.0
    cfg = load_config()
    asr_cfg, llm_cfg, tts_cfg = cfg.models.asr, cfg.models.llm, cfg.models.tts

    rows = []

    def bench(name, fn, rtf_audio=None, note=""):
        if only and name not in only:
            return
        try:
            xs = timed(fn, args.runs)
            p50, p95 = stats(xs)
            rtf = f"{p50 / rtf_audio:.2f}" if rtf_audio else "-"
            rows.append((name, f"{p50 * 1000:.0f}", f"{p95 * 1000:.0f}", rtf, note))
        except Exception as e:  # keep going; report the failure
            rows.append((name, "ERR", "ERR", "-", f"{type(e).__name__}: {str(e)[:80]}"))

    if audio.exists():
        asr = FasterWhisperASRProvider(model_name=asr_cfg.model, device=asr_cfg.device,
                                       compute_type=asr_cfg.compute_type, beam_size=asr_cfg.beam_size)
        bench("asr", lambda: asr.transcribe(str(audio)), dur, f"{asr_cfg.model} beam={asr_cfg.beam_size}")
        vad = SileroVADProvider()
        bench("vad", lambda: vad.detect_speech_regions(str(audio)), dur)
        dia = NemotronDiarizationProvider()
        bench("diarization", lambda: dia.diarize(str(audio)), dur)
        sed = YAMNetSoundEventProvider()
        bench("sound_events", lambda: sed.detect_events(str(audio)), dur)
        ac = ParselmouthAcousticProvider()
        bench("acoustic", lambda: ac.extract_features(str(audio)), dur)
    else:
        print(f"[skip] audio file not found: {audio}")

    llm = OpenAICompatibleLLMProvider(endpoint=llm_cfg.endpoint, model_name=llm_cfg.model)

    def llm_ttft():
        t0 = time.perf_counter()
        n = 0
        first = None
        for _ in llm.generate_stream("Give me a two sentence cake tip."):
            n += 1
            if first is None:
                first = time.perf_counter() - t0
        return first

    if not only or "llm" in only:
        if llm.is_available():
            xs = timed(llm_ttft, max(1, args.runs // 2))
            ttft = sorted(x[1] for x in xs)
            rows.append(("llm_ttft", f"{statistics.median(ttft) * 1000:.0f}", f"{max(ttft) * 1000:.0f}", "-", llm_cfg.model))
        else:
            rows.append(("llm_ttft", "N/A", "N/A", "-", "endpoint unreachable"))

    def tts_bench(label, prov):
        if only and label not in only:
            return
        if not prov.is_available():
            rows.append((label, "N/A", "N/A", "-", "unavailable"))
            return
        secs = {}

        def run():
            a = prov.synthesize(TTS_TEXT)
            secs["d"] = len(a) / float(prov.sample_rate)

        xs = timed(run, args.runs)
        p50, p95 = stats(xs)
        rows.append((label, f"{p50 * 1000:.0f}", f"{p95 * 1000:.0f}", f"{p50 / secs['d']:.2f}", f"{secs['d']:.1f}s audio"))

    tts_bench("tts_qwen", QwenTTSProvider(endpoint=tts_cfg.endpoint or "http://192.168.1.6:8091/v1",
                                          voice=tts_cfg.voice or "ryan"))
    tts_bench("tts_kokoro", KokoroTTSProvider())

    header = "| component | p50 ms | p95 ms | RTF (<1 = faster than real time) | note |\n|---|---|---|---|---|\n"
    table = header + "\n".join("| " + " | ".join(r) + " |" for r in rows)
    print(table)
    if args.save:
        out = Path("docs/benchmarks")
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{date.today().isoformat()}.md").write_text(f"# Benchmark {date.today()}\n\naudio={audio} ({dur:.1f}s) runs={args.runs}\n\n{table}\n", encoding="utf-8")


if __name__ == "__main__":
    main()

