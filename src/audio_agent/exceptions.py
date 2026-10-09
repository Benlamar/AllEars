"""Custom exceptions for Audio Agent."""


class AudioAgentError(Exception):
    """Base exception for all Audio Agent errors."""
    pass


class ConfigurationError(AudioAgentError):
    """Raised when configuration is invalid or missing."""
    pass


class PrivacyViolationError(AudioAgentError):
    """Raised when an operation violates configured privacy policies (e.g. unapproved cloud call)."""
    pass


class AudioProcessingError(AudioAgentError):
    """Raised when decoding, loading, or processing audio fails."""
    pass


class ProviderError(AudioAgentError):
    """Raised when a model or capability provider fails."""
    pass


class ProviderUnavailableError(ProviderError):
    """Raised when a requested provider or model weights are missing or unreachable."""
    pass


class StorageError(AudioAgentError):
    """Raised when database or file storage operations fail."""
    pass

