"""Speech text sanitizer for cleaning LLM control tags, markdown syntax, and symbols for TTS."""

import re
from typing import Optional


class SpeechTextSanitizer:
    """Intelligently cleanses raw LLM generated text for natural speech synthesis.

    Removes:
    - Control/turn tags like <turn|>, <start_of_turn>, <|im_end|>, <think>...</think>
    - Markdown formatting like **bold**, *italic*, headers (##), bullet points (*, -)
    - Code blocks and backticks
    - Decorative brackets ([*], [+], [?]) and emojis
    """

    # Matches special LLM control tags and tokens
    _CONTROL_TAGS_PATTERN = re.compile(
        r"(<turn\|>|<start_of_turn>|<end_of_turn>|<\|im_end\|>|<\|im_start\|>|<\|eot_id\|>|\[INST\]|\[/INST\]|<think>.*?</think>|<[^>]+>)",
        re.DOTALL | re.IGNORECASE,
    )

    # Matches markdown links: [link text](url) -> link text
    _MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^\)]+\)")

    # Matches markdown images: ![alt text](url) -> empty
    _MD_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^\)]+\)")

    # Matches bold/italic: **text** or *text* or __text__ or _text_
    _MD_BOLD_ITALIC_PATTERN = re.compile(r"(\*\*|__|\*|_)(.*?)\1")

    # Matches code backticks: `code` or ```code```
    _MD_CODE_PATTERN = re.compile(r"(`{1,3})(.*?)\1", re.DOTALL)

    # Matches markdown headers: ## Header -> Header
    _MD_HEADER_PATTERN = re.compile(r"^\s*#{1,6}\s*", re.MULTILINE)

    # Matches bullet points at start of lines: * item, - item, + item
    _MD_BULLET_PATTERN = re.compile(r"^\s*[\*\-\+]\s+", re.MULTILINE)

    # Matches decorative prompt brackets: [*], [+], [-], [?]
    _BRACKET_ICONS_PATTERN = re.compile(r"\[[\*\+\-\?!]\]")

    # Matches arrows: ->, -->, ==>, =>
    _ARROW_PATTERN = re.compile(r"[-=]+>")

    # Matches common emojis / symbols that phonemizers stumble over
    _EMOJI_PATTERN = re.compile(
        r"[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\ud83c-\ud83e][\ud000-\udfff]",
        re.UNICODE,
    )

    @classmethod
    def sanitize(cls, text: str) -> str:
        """Sanitizes text by removing non-speech tokens, markdown, and unpronounceable characters."""
        if not text:
            return ""

        s = text

        # 1. Remove XML/control tags (including <turn|>, <think>...</think>)
        s = cls._CONTROL_TAGS_PATTERN.sub(" ", s)

        # 2. Remove markdown images and transform links
        s = cls._MD_IMAGE_PATTERN.sub(" ", s)
        s = cls._MD_LINK_PATTERN.sub(r"\1", s)

        # 3. Remove code formatting
        s = cls._MD_CODE_PATTERN.sub(r"\2", s)

        # 4. Remove bold / italic markers (repeatedly in case of nesting)
        for _ in range(2):
            s = cls._MD_BOLD_ITALIC_PATTERN.sub(r"\2", s)

        # 5. Remove headers and bullet points
        s = cls._MD_HEADER_PATTERN.sub("", s)
        s = cls._MD_BULLET_PATTERN.sub("", s)

        # 6. Remove decorative symbols and arrows
        s = cls._BRACKET_ICONS_PATTERN.sub(" ", s)
        s = cls._ARROW_PATTERN.sub(" ", s)
        s = cls._EMOJI_PATTERN.sub(" ", s)

        # 7. Normalize whitespace around punctuation and collapse spaces
        s = re.sub(r"\s+([,\.\?!;:])", r"\1", s) # fix "word ." -> "word."
        s = re.sub(r"\s+", " ", s) # collapse multiple spaces
        s = s.strip()

        return s

    # Convenience alias for clean_for_speech
    clean_for_speech = sanitize

