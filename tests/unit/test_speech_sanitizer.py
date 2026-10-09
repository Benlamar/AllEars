"""Unit tests for SpeechTextSanitizer."""

import pytest
from audio_agent.audio.sanitizer import SpeechTextSanitizer


def test_sanitize_control_tags():
    raw = "Because they make up everything!<turn|>"
    clean = SpeechTextSanitizer.sanitize(raw)
    assert clean == "Because they make up everything!"


def test_sanitize_various_llm_tags():
    raw = "<start_of_turn>model\nHello there!<end_of_turn><|im_end|>"
    clean = SpeechTextSanitizer.sanitize(raw)
    assert "start_of_turn" not in clean
    assert "end_of_turn" not in clean
    assert "im_end" not in clean
    assert "Hello there!" in clean


def test_sanitize_think_tags():
    raw = "<think>Let me evaluate the speakers.</think>Speaker 1 spoke first."
    clean = SpeechTextSanitizer.sanitize(raw)
    assert "think" not in clean
    assert clean == "Speaker 1 spoke first."


def test_sanitize_markdown():
    raw = "## Executive Summary:\n**Speaker 1** said: *Hello* and `status_ok`."
    clean = SpeechTextSanitizer.sanitize(raw)
    assert "**" not in clean
    assert "##" not in clean
    assert "Executive Summary:" in clean
    assert "Speaker 1 said: Hello and status_ok." in clean


def test_sanitize_bullets_and_brackets():
    raw = "[*] Key findings:\n* First point.\n- Second point."
    clean = SpeechTextSanitizer.sanitize(raw)
    assert "[*]" not in clean
    assert "*" not in clean
    assert "First point." in clean
    assert "Second point." in clean


def test_sanitize_emojis_and_arrows():
    raw = "Audio summary 🔊: Process completed -> all good! ✨"
    clean = SpeechTextSanitizer.sanitize(raw)
    assert "🔊" not in clean
    assert "✨" not in clean
    assert "->" not in clean
    assert "Audio summary: Process completed all good!" in clean

