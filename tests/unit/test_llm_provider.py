"""Unit tests for OpenAI-compatible local LLM provider."""

import json
from unittest.mock import MagicMock, patch
from audio_agent.providers.llm import OpenAICompatibleLLMProvider


def test_openai_compatible_provider_generate():
    provider = OpenAICompatibleLLMProvider(
        endpoint="http://192.168.1.6:8001/v1",
        model_name="gemma-4-E4B",
    )

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [
            {"message": {"role": "assistant", "content": "Grounded answer text"}}
        ]
    }).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        answer = provider.generate("Summarize the recording")
        assert answer == "Grounded answer text"

