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

    # Set of common passive conversational backchannels, acknowledgements, and vocalizations
    _BACKCHANNEL_PHRASES = {
        "ok", "okay", "okay okay", "ok ok", "yeah", "yep", "yup", "yeah yeah",
        "uh-huh", "uh huh", "mhm", "mm-hmm", "mm hmm", "hmm", "hm", "hum",
        "right", "sure", "got it", "i see", "cool", "nod", "alright", "all right",
        "yes", "ah", "ha", "aha", "sure sure", "right right", "yep yep",
        "sigh", "*sigh*", "[sigh]", "(sigh)",
        "cough", "*cough*", "[cough]", "(cough)",
        "throat clearing", "[clearing throat]",
        "um", "uh", "er",
    }

    @classmethod
    def is_passive_backchannel(cls, text: str) -> bool:
        """Determines if a transcribed utterance is merely a passive backchannel, nod, or vocalization."""
        if not text:
            return True
        clean = re.sub(r"[^\w\s\*\-\[\]\(\)]", "", text).strip().lower()
        if not clean:
            return True
        if clean in cls._BACKCHANNEL_PHRASES:
            return True
        words = clean.split()
        if words and all(w in {"ok", "okay", "yeah", "yep", "yup", "mhm", "hmm", "hm", "hum", "uh", "um", "ah", "right", "sure"} for w in words):
            return True
        return False

    # Explicit commands to stop speaking or be quiet
    _EXPLICIT_STOP_COMMANDS = {
        "stop", "stop talking", "be quiet", "shut up", "hush", "quiet",
        "stop please", "stop it", "stop now", "please stop",
        "stop and listen", "stop and listen to me", "listen to me",
        "hold on", "wait", "wait a minute", "wait a second", "pause",
        "pause please", "cancel", "shh", "shhh",
    }

    @classmethod
    def is_explicit_stop_command(cls, text: str) -> bool:
        """Determines if the transcribed utterance is an explicit command to stop talking/listen."""
        if not text:
            return False
        clean = re.sub(r"[^\w\s]", "", text).strip().lower()
        if not clean:
            return False
        if clean in cls._EXPLICIT_STOP_COMMANDS:
            return True
        for cmd in ("stop talking", "be quiet", "shut up", "stop and listen", "hold on a sec", "wait a second"):
            if clean.startswith(cmd):
                return True
        return False

    # Words and connectors that indicate a speaker is pausing mid-thought
    _INCOMPLETE_TRAILING_WORDS = {
        # Prepositions
        "in", "on", "at", "to", "for", "with", "from", "of", "about", "into",
        "through", "by", "over", "under", "between", "against", "during", "without",
        "before", "after", "like",
        # Conjunctions & Connectives
        "and", "or", "but", "because", "so", "if", "when", "while", "that", "which",
        "although", "since", "as", "unless", "whether", "than", "then",
        # Determiners & Articles
        "the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her",
        "our", "their", "some", "any", "every",
        # Auxiliaries & Modals
        "is", "are", "was", "were", "have", "has", "had", "can", "could", "will",
        "would", "shall", "should", "might", "must", "do", "does", "did",
        # Hesitation / Fillers
        "um", "uh", "er", "ah",
    }

    @classmethod
    def is_syntactically_incomplete(cls, text: str) -> bool:
        """Detects if an utterance is cut off mid-thought or ending on a trailing connector."""
        if not text:
            return False
        raw = text.strip()
        # Trailing ellipsis or dash explicitly indicates continuation
        if raw.endswith("...") or raw.endswith("..") or raw.endswith("—") or raw.endswith("--"):
            return True
        clean = re.sub(r"[^\w\s]", "", raw).strip().lower()
        if not clean:
            return False
        words = clean.split()
        if not words:
            return False
        last_word = words[-1]
        return last_word in cls._INCOMPLETE_TRAILING_WORDS

    # Convenience alias for clean_for_speech
    clean_for_speech = sanitize

