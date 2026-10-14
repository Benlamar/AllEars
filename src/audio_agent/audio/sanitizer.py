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
        "cancel", "shh", "shhh",
    }

    _EXPLICIT_PAUSE_COMMANDS = {
        "hold on", "wait", "wait a minute", "wait a second", "pause",
        "pause please", "give me a second", "one second", "hold up",
    }

    _EXPLICIT_RESET_COMMANDS = {
        "forget this conversation", "forget everything", "clear history",
        "start over", "reset conversation", "clear conversation",
    }

    @classmethod
    def is_explicit_stop_command(cls, text: str) -> bool:
        """Determines if the transcribed utterance is an explicit command to stop talking."""
        if not text:
            return False
        clean = re.sub(r"[^\w\s]", "", text).strip().lower()
        if not clean:
            return False
        if clean in cls._EXPLICIT_STOP_COMMANDS:
            return True
        for cmd in ("stop talking", "be quiet", "shut up", "stop and listen"):
            if clean.startswith(cmd):
                return True
        return False

    @classmethod
    def is_pause_command(cls, text: str) -> bool:
        """Determines if the user wants the assistant to pause / wait."""
        if not text:
            return False
        clean = re.sub(r"[^\w\s]", "", text).strip().lower()
        if not clean:
            return False
        if clean in cls._EXPLICIT_PAUSE_COMMANDS:
            return True
        for cmd in ("hold on", "wait a sec", "wait a minute", "give me a sec"):
            if clean.startswith(cmd):
                return True
        return False

    @classmethod
    def is_reset_command(cls, text: str) -> bool:
        """Determines if the user is asking to clear / reset conversation history."""
        if not text:
            return False
        clean = re.sub(r"[^\w\s]", "", text).strip().lower()
        if not clean:
            return False
        if clean in cls._EXPLICIT_RESET_COMMANDS:
            return True
        for cmd in ("forget this conversation", "start over", "clear history"):
            if clean.startswith(cmd):
                return True
        return False

    # Words and connectors that genuinely indicate a speaker is pausing mid-thought
    _INCOMPLETE_TRAILING_WORDS = {
        # Conjunctions & Connectives that cannot end a complete thought
        "and", "or", "but", "because", "if", "when", "while", "although",
        "since", "unless", "whether", "whereas",
        # Subordinating prepositions strictly requiring an object
        "into", "onto", "during", "against", "between", "among", "about", "in",
        # Articles & Determiners requiring a following noun
        "the", "a", "an", "every",
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
        # Explicit completed terminal punctuation indicates question or statement is complete
        if raw.endswith("?") or raw.endswith("!"):
            return False
        clean = re.sub(r"[^\w\s]", "", raw).strip().lower()
        if not clean:
            return False
        words = clean.split()
        if not words:
            return False
        last_word = words[-1]
        return last_word in cls._INCOMPLETE_TRAILING_WORDS


    @classmethod
    def classify_intent(cls, text: str) -> str:
        """Classifies the conversational utterance into a high-level intent category."""
        if not text or not text.strip():
            return "EMPTY"
        if cls.is_reset_command(text):
            return "RESET"
        if cls.is_explicit_stop_command(text):
            return "STOP"
        if cls.is_pause_command(text):
            return "PAUSE"
        if cls.is_passive_backchannel(text):
            return "BACKCHANNEL"
        if cls.is_syntactically_incomplete(text):
            return "INCOMPLETE"
        return "NORMAL"

    _INTERRUPTION_STOP_WORDS = {
        "stop", "cancel", "halt", "freeze", "quit", "quiet", "hush", "shut up",
    }

    _INTERRUPTION_WAIT_WORDS = {
        "wait", "pause", "hold on", "hang on", "hold up", "give me a second",
        "one second", "just a second", "just a minute", "wait a second", "wait a minute",
    }

    _INTERRUPTION_CORRECTION_PHRASES = [
        "actually wait", "no wait", "thats not what i meant", "that is not what i meant",
        "what i meant was", "what i meant", "thats wrong", "that is wrong", "not that",
        "not what i asked", "you misunderstood", "listen", "listen to me", "excuse me",
        "sorry to interrupt", "wait that is not", "wait thats not", "no no", "wait wait",
        "stop stop", "hold up hold up", "hold on hold on",
    ]

    _DIRECTED_QUESTION_STARTERS = {
        "can you", "could you", "tell me", "what about", "what if", "why did",
        "why does", "how come", "how does", "how do", "switch to", "instead of",
    }

    _NOISE_HALLUCINATIONS = {
        "you", "thank you", "thanks for watching", "subtitles by", "bye", "bye bye",
        "thank you for watching", ".", "oh", "ah",
    }

    @classmethod
    def classify_interruption_intent(cls, text: str, mode: str = "auto") -> "InterruptionDecision":
        """Decision table & fuzzy matcher to evaluate whether candidate speech was intended to interrupt."""
        from dataclasses import dataclass

        raw = (text or "").strip()
        if not raw:
            return InterruptionDecision(
                is_interruption=False,
                trigger_type="NOISE",
                reason="Empty or silent candidate audio",
                transcript="",
            )

        clean = re.sub(r"[^\w\s]", "", raw).strip().lower()
        if not clean or clean in cls._NOISE_HALLUCINATIONS:
            return InterruptionDecision(
                is_interruption=False,
                trigger_type="NOISE",
                reason="Noise blip or ASR hallucination artifact",
                transcript=raw,
            )

        # 1. Reject passive backchannels
        if cls.is_passive_backchannel(raw):
            return InterruptionDecision(
                is_interruption=False,
                trigger_type="BACKCHANNEL",
                reason="Passive acknowledgement / backchannel",
                transcript=raw,
            )

        words = clean.split()

        # 2. Check for explicit correction / redirection phrases
        for phrase in cls._INTERRUPTION_CORRECTION_PHRASES:
            if phrase in clean:
                return InterruptionDecision(
                    is_interruption=True,
                    trigger_type="CORRECTION",
                    matched_phrase=phrase,
                    reason=f"Matched correction trigger '{phrase}'",
                    transcript=raw,
                )

        # 3. Check for stop keywords
        for w in words:
            if w in cls._INTERRUPTION_STOP_WORDS:
                return InterruptionDecision(
                    is_interruption=True,
                    trigger_type="STOP_COMMAND",
                    matched_phrase=w,
                    reason=f"Matched stop keyword '{w}'",
                    transcript=raw,
                )

        # 4. Check for wait / hold keywords
        for w in words:
            if w in cls._INTERRUPTION_WAIT_WORDS:
                return InterruptionDecision(
                    is_interruption=True,
                    trigger_type="WAIT_COMMAND",
                    matched_phrase=w,
                    reason=f"Matched wait keyword '{w}'",
                    transcript=raw,
                )
        for phrase in ("hold on", "hang on", "hold up", "give me a second", "wait a minute", "wait a second"):
            if phrase in clean:
                return InterruptionDecision(
                    is_interruption=True,
                    trigger_type="WAIT_COMMAND",
                    matched_phrase=phrase,
                    reason=f"Matched wait phrase '{phrase}'",
                    transcript=raw,
                )

        # 5. Check for repetitions ("no no no", "wait wait")
        if len(words) >= 2 and all(w == words[0] for w in words):
            if words[0] in {"no", "wait", "stop", "hold"}:
                return InterruptionDecision(
                    is_interruption=True,
                    trigger_type="REPETITION",
                    matched_phrase=" ".join(words),
                    reason=f"Repeated emphatic cue '{words[0]}'",
                    transcript=raw,
                )

        # 6. Mode handling: "keywords" strictly requires keyword triggers
        if mode in {"keywords", "strict_keywords", "strict"}:
            return InterruptionDecision(
                is_interruption=False,
                trigger_type="UNRELATED",
                reason="Does not contain an interruption keyword",
                transcript=raw,
            )

        # 7. In "auto" / conversational mode: Check for directed question starters or substantive turns
        for starter in cls._DIRECTED_QUESTION_STARTERS:
            if clean.startswith(starter):
                return InterruptionDecision(
                    is_interruption=True,
                    trigger_type="DIRECTED_QUERY",
                    matched_phrase=starter,
                    reason=f"Directed question starter '{starter}'",
                    transcript=raw,
                )

        # If turn is substantial (> 4 words), treat as natural interruption in auto mode
        if len(words) >= 4:
            return InterruptionDecision(
                is_interruption=True,
                trigger_type="SPEECH_INTENT",
                matched_phrase=" ".join(words[:3]),
                reason="Substantive multi-word interruption turn",
                transcript=raw,
            )

        return InterruptionDecision(
            is_interruption=False,
            trigger_type="UNRELATED",
            reason="Ambiguous brief utterance without interruption intent",
            transcript=raw,
        )

    # Convenience alias for clean_for_speech
    clean_for_speech = sanitize


from dataclasses import dataclass

@dataclass
class InterruptionDecision:
    """Detailed decision output from semantic interruption classification."""
    is_interruption: bool
    trigger_type: str
    matched_phrase: Optional[str] = None
    reason: str = ""
    transcript: str = ""


