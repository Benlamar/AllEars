"""Unit tests for the sentence-aware SpeechChunker."""

import random

from audio_agent.audio.speech_chunker import ChunkerConfig, SpeechChunker

RECIPE = (
    "Here is a simple vanilla cake. First, preheat your oven to 350 degrees. "
    "Then mix 2.5 cups of flour, e.g. all-purpose, with 1 cup of sugar, and a pinch of salt. "
    "Add three eggs, one at a time, beating well after each. "
    "Pour the batter into a greased pan and bake for 30 minutes. Enjoy!"
)


def run(text, sizes=None, cfg=None):
    ch = SpeechChunker(cfg)
    out = []
    i = 0
    rng = random.Random(1)
    while i < len(text):
        n = rng.randint(1, 6) if sizes is None else sizes
        out.extend(ch.push(text[i : i + n]))
        i += n
    tail = ch.flush()
    if tail:
        out.append(tail)
    return out


def test_no_split_on_decimal_or_abbreviation():
    chunks = run(RECIPE)
    joined = " ".join(chunks)
    assert "2.5 cups" in joined
    assert not any(c.endswith("e.g.") for c in chunks)
    assert not any(c.endswith("2.") for c in chunks)


def test_chunks_are_sentence_aligned_after_first():
    chunks = run(RECIPE)
    for c in chunks[1:]:
        assert c[-1] in ".!?", c


def test_short_sentences_merge():
    chunks = run("Sure. Okay. Got it. Let me explain how this works in detail today.")
    # later chunks must reach min_chars; "Sure." alone must never be a non-first chunk
    assert all(len(c) >= 20 for c in chunks[:-1])


def test_first_chunk_released_early_on_clause():
    ch = SpeechChunker()
    out = ch.push("Well, when you think about it, ")
    assert out and out[0].endswith(",")


def test_output_independent_of_token_split():
    a = run(RECIPE, sizes=1)
    b = run(RECIPE, sizes=7)
    c = run(RECIPE, sizes=len(RECIPE))
    assert a == b == c


def test_text_is_preserved():
    chunks = run(RECIPE)
    assert " ".join(chunks).split() == RECIPE.split()


def test_max_chars_splits_on_word_boundary():
    cfg = ChunkerConfig(max_chars=60, min_chars=20)
    text = "word " * 60
    chunks = run(text, cfg=cfg)
    assert all(len(c) <= 61 for c in chunks)
    assert all(not c.startswith("ord") and c.split()[-1] == "word" for c in chunks)


def test_stall_flush_releases_whole_words_only():
    ch = SpeechChunker()
    assert ch.push("this is a long phrase without") == []
    out = ch.flush_stalled()
    assert out == "this is a long phrase"
    assert ch.flush() == "without"


def test_newline_ends_chunk():
    ch = SpeechChunker(ChunkerConfig(first_chunk_min_chars=5))
    out = ch.push("First paragraph here\nSecond")
    assert out == ["First paragraph here"]

