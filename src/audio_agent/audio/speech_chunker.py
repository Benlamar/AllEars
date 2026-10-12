"""Sentence-aware speech chunker for natural-sounding streaming TTS.

Unlike :class:`audio_agent.audio.chunker.StreamingClauseChunker` (word-count driven,
breaks mid-sentence), this chunker only cuts at linguistic boundaries so that each
chunk handed to the TTS engine is one or more complete sentences. TTS prosody
(intonation, phrase-final lowering) is then correct inside every chunk.
"""

import re
from dataclasses import dataclass
from typing import List

from audio_agent.audio.sanitizer import SpeechTextSanitizer


@dataclass
class ChunkerConfig:
    """Tunable buffering behaviour.

    Attributes:
        first_chunk_min_words: the very first chunk may end at a clause boundary
            (, ; : —) once it holds this many words, to keep time-to-first-audio low.
        first_chunk_min_chars: minimum length for the first chunk when it ends at a sentence boundary.
        min_chars: later chunks merge short sentences ("Sure.", "Okay.") until this length.
        max_chars: hard cap; longer text is split at the last clause boundary, then at a space.
        max_wait_s: consumers should call :meth:`SpeechChunker.flush_stalled` if no new text
            arrived for this long (LLM stall) so audio does not starve.
    """

    first_chunk_min_words: int = 5
    first_chunk_min_chars: int = 20
    min_chars: int = 40
    max_chars: int = 400
    max_wait_s: float = 1.2


_ABBREVIATIONS = {
    "e.g", "i.e", "dr", "mr", "mrs", "ms", "vs", "st", "jr", "sr", "prof",
    "no", "approx", "inc", "ltd", "fig", "min", "max", "tbsp", "tsp",
}

# terminator run, optional closing quotes/brackets, then whitespace  |  or a bare newline
_BOUNDARY_RE = re.compile(r"([.!?\u2026]+)([\"')\]]*)(?=\s)|\n")
_CLAUSE_RE = re.compile(r"[,;:\u2014](?=\s)")


class SpeechChunker:
    """Incrementally turns streamed LLM text deltas into speech-ready chunks."""

    def __init__(self, config: ChunkerConfig | None = None):
        self.config = config or ChunkerConfig()
        self._buf = ""
        self._emitted_any = False

    # ------------------------------------------------------------------ helpers
    @property
    def has_pending(self) -> bool:
        return bool(self._buf.strip())

    def _is_real_boundary(self, match: re.Match) -> bool:
        if match.group(0) == "\n":
            return True
        terminators = match.group(1)
        if len(terminators) > 1 and set(terminators) <= {".", "\u2026"}:
            return False  # ellipsis: usually mid-sentence
        if terminators != ".":
            return True  # ! ? are never abbreviations
        before = self._buf[: match.start()]
        word_match = re.search(r"(\S+)$", before)
        if not word_match:
            return True
        word = word_match.group(1).lower().lstrip("([\"'")
        if word in _ABBREVIATIONS:
            return False
        if len(word) == 1 and word.isalpha():
            return False  # initial, e.g. "J."
        if word.isdigit():
            line_start = before[: word_match.start()].rsplit("\n", 1)[-1]
            if not line_start.strip():
                return False  # list marker "1."
        return True

    def _emit(self, end: int) -> str | None:
        text = self._buf[:end]
        self._buf = self._buf[end:].lstrip(" \t")
        self._emitted_any = True
        clean = SpeechTextSanitizer.sanitize(text)
        return clean or None

    def _extract_one(self) -> str | None:
        cfg = self.config
        min_len = cfg.min_chars if self._emitted_any else cfg.first_chunk_min_chars

        # 1. sentence boundary with enough text (merges short sentences)
        for m in _BOUNDARY_RE.finditer(self._buf):
            if not self._is_real_boundary(m):
                continue
            end = m.end(2) if m.group(0) != "\n" else m.end()
            if len(self._buf[:end].strip()) >= min_len:
                return self._emit(end) or ""

        # 2. fast start: first chunk may end at a clause boundary
        if not self._emitted_any:
            for m in _CLAUSE_RE.finditer(self._buf):
                if len(self._buf[: m.start()].split()) >= cfg.first_chunk_min_words:
                    return self._emit(m.end()) or ""

        # 3. hard cap
        if len(self._buf) > cfg.max_chars:
            window = self._buf[: cfg.max_chars]
            clauses = list(_CLAUSE_RE.finditer(window))
            if clauses and clauses[-1].end() >= cfg.min_chars:
                return self._emit(clauses[-1].end()) or ""
            space = window.rfind(" ")
            return self._emit(space if space > 0 else cfg.max_chars) or ""
        return None

    # --------------------------------------------------------------------- API
    def push(self, delta: str) -> List[str]:
        """Adds streamed text; returns zero or more complete speech chunks."""
        self._buf += delta
        out: List[str] = []
        while True:
            chunk = self._extract_one()
            if chunk is None:
                break
            if chunk:
                out.append(chunk)
        return out

    def flush_stalled(self) -> str | None:
        """Releases buffered text up to the last complete word (used on LLM stall)."""
        if not self._buf.strip():
            return None
        if self._buf[-1].isspace():
            end = len(self._buf)
        else:
            end = self._buf.rfind(" ")
            if end <= 0:
                return None
        if len(self._buf[:end].split()) < 3:
            return None
        return self._emit(end)

    def flush(self) -> str | None:
        """Releases whatever remains when the stream ends."""
        text = SpeechTextSanitizer.sanitize(self._buf)
        self._buf = ""
        return text or None

