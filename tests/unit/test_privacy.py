"""Unit tests for PrivacyPolicyEnforcer."""

import pytest
from audio_agent.config import PrivacyConfig
from audio_agent.exceptions import PrivacyViolationError
from audio_agent.security.privacy_policy import PrivacyPolicyEnforcer


def test_cloud_rejected_when_disallowed():
    config = PrivacyConfig(cloud={"allowed": False, "automatic_fallback": False})
    enforcer = PrivacyPolicyEnforcer(config)

    # Local is allowed
    enforcer.validate_provider_execution("asr", "mock", "local")

    # Cloud execution must raise PrivacyViolationError
    with pytest.raises(PrivacyViolationError, match="Cloud execution requested"):
        enforcer.validate_provider_execution("asr", "cloud_api", "cloud")


def test_automatic_fallback_prohibited():
    config = PrivacyConfig(cloud={"allowed": False, "automatic_fallback": False})
    enforcer = PrivacyPolicyEnforcer(config)

    with pytest.raises(PrivacyViolationError, match="Automatic fallback"):
        enforcer.validate_fallback_attempt("asr", "local_whisper", "cloud_whisper", "cloud")

