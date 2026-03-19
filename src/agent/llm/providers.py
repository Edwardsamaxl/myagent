from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import requests

from ..config import AgentConfig


Message = dict[str, str]


class ModelProvider(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
    ) -> str:
        raise NotImplementedError


class OllamaProvider(ModelProvider):
    def __init__(self, base_url: str, model_name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def generate(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]


class OpenAICompatibleProvider(ModelProvider):
    def __init__(self, base_url: str, api_key: str, model_name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name

    def generate(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
    ) -> str:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 为空，无法使用 openai_compatible 提供商。")

        url = f"{self.base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class MockProvider(ModelProvider):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
    ) -> str:
        _ = (temperature, max_tokens)
        last_user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "")
                break
        return f"[mock:{self.model_name}] 你刚才说的是：{last_user_message}"


def build_model_provider(config: AgentConfig) -> ModelProvider:
    provider = config.model_provider.lower()

    if provider == "ollama":
        return OllamaProvider(config.ollama_base_url, config.model_name)
    if provider == "openai_compatible":
        return OpenAICompatibleProvider(
            config.openai_base_url,
            config.openai_api_key,
            config.model_name,
        )
    if provider == "mock":
        return MockProvider(config.model_name)

    raise ValueError(
        "MODEL_PROVIDER 不支持。可选值: ollama | openai_compatible | mock"
    )
