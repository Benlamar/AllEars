"""Compare ASR configs on casual speech (Phase 2).

Usage:
    uv run python scripts/asr_eval.py tests/data/asr_casual.jsonl \
        [--models base.en,small.en,distil-large-v3] [--beams 1,5]

Each JSONL line: {"audio": "path.wav", "reference": "I'm beat for today. I need to take a rest."}
Also works on files saved by `audio-agent live --debug-dump` (add the reference text yourself).
Prints WER, latency, and the lowest-confidence words per config.
"""

import argparse
import json
import re
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PROMPT = "Casual conversation in English. The speaker uses everyday idioms."


def norm(s):
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[-1] / max(1, len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--models", default="base.en,small.en")
    ap.add_argument("--beams", default="1,5")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from audio_agent.providers.asr import FasterWhisperASRProvider

    items = [json.loads(l) for l in Path(args.dataset).read_text(encoding="utf-8").splitlines() if l.strip()]
    print("| model | beam | vad | prompt | WER | avg s | lowest-confidence words |\n|---|---|---|---|---|---|---|")
    for model in args.models.split(","):
        for beam, vad, use_prompt in product([int(b) for b in args.beams.split(",")], [True, False], [False, True]):
            p = FasterWhisperASRProvider(model_name=model, device=args.device, beam_size=beam,
                                         vad_filter=vad, initial_prompt=PROMPT if use_prompt else None)
            errs, secs, low = [], [], []
            for it in items:
                t0 = time.perf_counter()
                text, words = p.transcribe(it["audio"])
                secs.append(time.perf_counter() - t0)
                errs.append(wer(it["reference"], text))
                low += [f"{w.text}({w.confidence})" for w in sorted(words, key=lambda w: w.confidence or 1)[:2]]
            print(f"| {model} | {beam} | {vad} | {use_prompt} | {sum(errs)/len(errs):.3f} | "
                  f"{sum(secs)/len(secs):.2f} | {', '.join(low[:4])} |")


if __name__ == "__main__":
    main()

