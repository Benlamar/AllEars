"""Unit tests for the gapless player and the streaming speech pipeline."""

import threading
import time

import numpy as np

from audio_agent.audio.player import AudioStreamPlayer
from audio_agent.audio.speech_pipeline import StreamingSpeechPipeline


def make_player(**kw):
    return AudioStreamPlayer(sample_rate=1000, fade_ms=0, open_stream=False, **kw)


def test_player_is_gapless_across_chunks():
    p = make_player()
    p._is_running = True
    p.enqueue(np.ones(100, dtype=np.float32))
    p.enqueue(np.ones(100, dtype=np.float32))
    out = np.concatenate([p._fill(30) for _ in range(7)])
    assert np.all(out[:200] == 1.0)  # no zeros between chunk boundary
    assert np.all(out[200:] == 0.0)
    assert not p.is_playing


def test_player_fade_removes_edge_clicks():
    p = AudioStreamPlayer(sample_rate=1000, fade_ms=10, open_stream=False)
    p._is_running = True
    p.enqueue(np.ones(100, dtype=np.float32))
    out = p._fill(100)
    assert out[0] == 0.0 and out[-1] == 0.0 and out[50] == 1.0


def test_player_pause_after_and_stop():
    p = make_player()
    p._is_running = True
    p.enqueue(np.ones(10, dtype=np.float32), pause_after_s=0.05)
    assert p._buffered == 10 + 50
    p.stop()
    assert p._buffered == 0 and not p.is_playing


class SlowTTS:
    sample_rate = 1000

    def __init__(self, delay=0.05):
        self.delay = delay
        self.texts = []
        self.times = []

    def synthesize(self, text):
        self.texts.append(text)
        self.times.append(time.monotonic())
        time.sleep(self.delay)
        return np.ones(100, dtype=np.float32)


def test_pipeline_streams_and_preserves_order():
    tts = SlowTTS()
    p = make_player()
    pipe = StreamingSpeechPipeline(tts, p, sentence_pause_s=0.0)
    for tok in "One sentence is here now. Another sentence follows right after it. Done".split(" "):
        pipe.push_text(tok + " ")
    pipe.finish(wait=False)
    pipe._worker.join(timeout=3)
    assert len(tts.texts) >= 2
    assert " ".join(tts.texts).split() == "One sentence is here now. Another sentence follows right after it. Done".split()
    assert pipe.first_audio_latency_s is not None


def test_push_text_does_not_block_on_tts():
    tts = SlowTTS(delay=0.3)
    pipe = StreamingSpeechPipeline(tts, make_player())
    t0 = time.monotonic()
    pipe.push_text("This is the first full sentence of the answer. ")
    pipe.push_text("This is the second full sentence of the answer. ")
    assert time.monotonic() - t0 < 0.1
    pipe.cancel()


def test_cancel_clears_queues_and_stops_worker():
    tts = SlowTTS(delay=0.05)
    p = make_player()
    pipe = StreamingSpeechPipeline(tts, p, cancel_event=threading.Event())
    for _ in range(6):
        pipe.push_text("A complete sentence that is long enough to emit. ")
    pipe.cancel()
    pipe._worker.join(timeout=2)
    assert not pipe._worker.is_alive()
    assert p._buffered == 0
    before = len(tts.texts)
    time.sleep(0.2)
    assert len(tts.texts) == before


def test_synthesis_overlaps_with_playback():
    """Chunk N+1 must start synthesizing without waiting for chunk N to finish playing."""
    tts = SlowTTS(delay=0.02)
    p = make_player()  # 100 samples at 1kHz = 0.1s per chunk, nobody consumes => 'playing'
    pipe = StreamingSpeechPipeline(tts, p, sentence_pause_s=0.0)
    for _ in range(3):
        pipe.push_text("Another complete sentence that is long enough. ")
    pipe.finish(wait=False)
    pipe._worker.join(timeout=3)
    assert len(tts.times) == 3
    assert tts.times[2] - tts.times[0] < 0.2  # all synthesized while audio still queued

