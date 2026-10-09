"""Privacy policy validation and enforcement."""

from typing import Optional
from audio_agent.config import PrivacyConfig
from audio_agent.exceptions import PrivacyViolationError


class PrivacyPolicyEnforcer:
    """Enforces zero unapproved cloud egress and safe retention rules."""

    def __init__(self, config: PrivacyConfig):
        self.config = config

    def validate_provider_execution(self, capability: str, provider_name: str, execution_mode: str) -> None:
        """Validates that a provider execution mode is permitted by policy."""
        mode = execution_mode.lower()
        if mode == "cloud":
            if not self.config.cloud.get("allowed", False):
                raise PrivacyViolationError(
                    f"Privacy policy violation: Cloud execution requested for '{capability}' "
                    f"via provider '{provider_name}', but cloud processing is disabled in privacy.yaml."
                )

    def validate_fallback_attempt(self, capability: str, from_provider: str, to_provider: str, to_mode: str) -> None:
        """Enforces prohibition of silent fallback to cloud when a local provider fails."""
        if to_mode.lower() == "cloud":
            if not self.config.cloud.get("automatic_fallback", False):
                raise PrivacyViolationError(
                    f"Privacy policy violation: Automatic fallback to cloud provider '{to_provider}' "
                    f"for capability '{capability}' is strictly forbidden by policy."
                )

