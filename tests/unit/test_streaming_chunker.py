"""Unit tests for StreamingClauseChunker."""

import pytest
from audio_agent.audio.chunker import StreamingClauseChunker


def test_chunk_stream_sentence_boundaries():
    tokens = ["Hello", " world.", " This", " is", " sentence", " two!"]
    chunker = StreamingClauseChunker()
    chunks = list(chunker.chunk_stream(iter(tokens)))

    assert len(chunks) == 2
    assert chunks[0] == "Hello world."
    assert chunks[1] == "This is sentence two!"


def test_chunk_stream_strips_control_tags_on_the_fly():
    tokens = ["Because", " they", " make", " up", " everything!<turn|>"]
    chunker = StreamingClauseChunker()
    chunks = list(chunker.chunk_stream(iter(tokens)))

    assert len(chunks) == 1
    assert chunks[0] == "Because they make up everything!"


def test_chunk_stream_clause_split():
    tokens = [
        "Although", " it", " was", " raining", " heavily", " outside,",
        " we", " still", " went", " for", " a", " walk."
    ]
    chunker = StreamingClauseChunker(min_words_for_clause=5)
    chunks = list(chunker.chunk_stream(iter(tokens)))

    assert len(chunks) == 2
    assert chunks[0].endswith(",")
    assert chunks[1].endswith(".")


def test_chunk_stream_tail_flush():
    tokens = ["No", " punctuation", " at", " the", " end"]
    chunker = StreamingClauseChunker()
    chunks = list(chunker.chunk_stream(iter(tokens)))

    assert len(chunks) == 1
    assert chunks[0] == "No punctuation at the end"

