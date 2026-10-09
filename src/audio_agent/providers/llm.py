"""OpenAI-compatible local LLM provider for TensorRT-LLM, vLLM, Ollama, and LocalAI."""

import json
import urllib.error
import urllib.request
from typing import Iterator, Optional

from audio_agent.exceptions import ProviderError, ProviderUnavailableError
from audio_agent.providers.base import LLMProvider


class OpenAICompatibleLLMProvider(LLMProvider):
    """Connects to OpenAI-compatible chat completion endpoints (e.g. TensorRT-LLM on local network)."""

    provider_name: str = "openai_compatible"
    execution_mode: str = "local"

    def __init__(
        self,
        endpoint: str = "http://192.168.1.6:8001/v1",
        model_name: str = "gemma-4-E4B",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        api_key: Optional[str] = None,
    ):
        # Normalize endpoint (strip trailing slash)
        self.endpoint = endpoint.rstrip("/")
        # Ensure /chat/completions suffix
        if not self.endpoint.endswith("/v1") and not self.endpoint.endswith("/chat/completions"):
            self.chat_url = f"{self.endpoint}/v1/chat/completions"
        elif self.endpoint.endswith("/v1"):
            self.chat_url = f"{self.endpoint}/chat/completions"
        else:
            self.chat_url = self.endpoint

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key = api_key

    def is_available(self) -> bool:
        """Checks whether the server is reachable and lists the model."""
        base = self.chat_url.rsplit("/chat/completions", 1)[0]
        models_url = f"{base}/models"
        try:
            req = urllib.request.Request(models_url)
            if self.api_key:
                req.add_header("Authorization", f"Bearer {self.api_key}")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("id") for m in data.get("data", [])]
                return self.model_name in models or len(models) > 0
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.chat_url, data=data, headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                choices = result.get("choices", [])
                if not choices:
                    raise ProviderError(f"OpenAI-compatible endpoint returned no choices: {result}")
                message = choices[0].get("message", {})
                content = message.get("content", "")
                return content.strip()
        except urllib.error.URLError as e:
            raise ProviderUnavailableError(
                f"Failed to connect to LLM at {self.chat_url}: {e}"
            ) from e
        except Exception as e:
            raise ProviderError(f"Error during LLM generation: {e}") from e

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        """Streams token deltas from OpenAI-compatible endpoint as they are generated."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.chat_url, data=data, headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue
                    if line_str.startswith("data: "):
                        data_str = line_str[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
        except urllib.error.URLError as e:
            raise ProviderUnavailableError(
                f"Failed to connect to LLM stream at {self.chat_url}: {e}"
            ) from e
        except Exception as e:
            raise ProviderError(f"Error during LLM streaming: {e}") from e


