"""Streaming sentence and clause boundary chunker for low-latency speech synthesis."""

import re
from typing import Iterator

from audio_agent.audio.sanitizer import SpeechTextSanitizer


class StreamingClauseChunker:
    """Aggregates incoming streaming tokens into natural speech clauses.

    Splits incoming text on sentence boundaries (. ? ! \\n) or long clause boundaries (, ; :)
    to deliver phrases to the TTS engine as fast as possible (~150-250ms).
    """

    # Sentence boundary regex
    _SENTENCE_SPLIT_REGEX = re.compile(r"([\.!\?\n]+)")
    # Clause boundary regex (commas, semicolons, dashes)
    _CLAUSE_SPLIT_REGEX = re.compile(r"([,;:—]+)")

    def __init__(self, min_words_for_clause: int = 6, max_words_before_break: int = 14):
        self.min_words_for_clause = min_words_for_clause
        self.max_words_before_break = max_words_before_break

    def chunk_stream(self, token_stream: Iterator[str]) -> Iterator[str]:
        """Consumes a stream of tokens and yields speech-ready phrases."""
        buffer = ""

        for raw_token in token_stream:
            buffer += raw_token

            # Check if current buffer contains a sentence terminator
            match = self._SENTENCE_SPLIT_REGEX.search(buffer)
            if match:
                end_pos = match.end()
                phrase = buffer[:end_pos]
                buffer = buffer[end_pos:]

                clean_phrase = SpeechTextSanitizer.sanitize(phrase)
                if clean_phrase:
                    yield clean_phrase
                continue

            # Check if buffer has exceeded clause word count and contains a clause terminator
            words = buffer.strip().split()
            if len(words) >= self.min_words_for_clause:
                clause_match = self._CLAUSE_SPLIT_REGEX.search(buffer)
                if clause_match:
                    end_pos = clause_match.end()
                    phrase = buffer[:end_pos]
                    buffer = buffer[end_pos:]

                    clean_phrase = SpeechTextSanitizer.sanitize(phrase)
                    if clean_phrase:
                        yield clean_phrase
                    continue

            # Fallback break for very long clauses without punctuation
            if len(words) >= self.max_words_before_break:
                # Find last space
                last_space = buffer.rfind(" ")
                if last_space > 0:
                    phrase = buffer[:last_space]
                    buffer = buffer[last_space:].lstrip()
                    clean_phrase = SpeechTextSanitizer.sanitize(phrase)
                    if clean_phrase:
                        yield clean_phrase

        # Flush any remaining text in buffer when stream ends
        remaining = SpeechTextSanitizer.sanitize(buffer)
        if remaining:
            yield remaining

