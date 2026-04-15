from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import requests

from ..config import AgentConfig


Message = dict[str, str]

# 与 build_model_provider 分支一致；Web UI 与文档可共用，避免下拉框与配置漂移。
MODEL_PROVIDER_CHOICES: list[tuple[str, str]] = [
    ("ollama", "Ollama（本地）"),
    ("openai_compatible", "OpenAI 兼容 API"),
    ("anthropic_compatible", "Anthropic 兼容 / 中转"),
    ("mock", "mock（调试）"),
]


def supported_model_providers() -> list[dict[str, str]]:
    return [{"id": pid, "label": label} for pid, label in MODEL_PROVIDER_CHOICES]


class ModelProvider(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        raise NotImplementedError

    def generate_raw(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """返回原始 API 响应（供 tool calling 使用）。默认实现调用 generate 并包装。"""
        text = self.generate(messages, temperature, max_tokens, tools)
        return {"content": [{"type": "text", "text": text}]}


class OllamaProvider(ModelProvider):
    def __init__(self, base_url: str, model_name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def generate(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
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
        if tools:
            payload["tools"] = tools
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]


class _HTTPChatProvider(ModelProvider):
    """OpenAI/Anthropic 兼容 API 的公共基类"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        *,
        auth_header: str = "Authorization",
        auth_scheme: str = "Bearer",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self._auth_header = auth_header
        self._auth_scheme = auth_scheme
        self._extra_headers = extra_headers or {}

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            self._auth_header: f"{self._auth_scheme} {self.api_key}",
        }
        headers.update(self._extra_headers)
        return headers

    def _parse_response(self, data: dict[str, Any]) -> str:
        """子类可覆盖此方法以自定义响应解析"""
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        return str(data)

    def generate(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        if not self.api_key:
            raise ValueError(f"{self.__class__.__name__}: API key 为空")

        payload = self._build_payload(messages, temperature, max_tokens, tools)
        url = f"{self.base_url}{self._endpoint()}"
        response = requests.post(
            url,
            headers=self._build_headers(),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    def generate_raw(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """返回原始 API 响应（包含 stop_reason、content 等完整信息）。"""
        if not self.api_key:
            raise ValueError(f"{self.__class__.__name__}: API key 为空")

        payload = self._build_payload(messages, temperature, max_tokens, tools)
        url = f"{self.base_url}{self._endpoint()}"
        response = requests.post(
            url,
            headers=self._build_headers(),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def _endpoint(self) -> str:
        """子类返回 API 端点路径"""
        raise NotImplementedError

    def _build_payload(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """子类返回请求 payload"""
        raise NotImplementedError


class OpenAICompatibleProvider(_HTTPChatProvider):
    """OpenAI 兼容 API"""

    def _endpoint(self) -> str:
        return "/v1/chat/completions"

    def _build_payload(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        return payload


class AnthropicCompatibleProvider(_HTTPChatProvider):
    """Anthropic 兼容 API (如 api.minimaxi.com)"""

    def _endpoint(self) -> str:
        return "/v1/messages"

    def _build_payload(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        return payload

    def _build_headers(self) -> dict[str, str]:
        headers = super()._build_headers()
        headers["anthropic-version"] = "2023-06-01"
        return headers

    def _parse_response(self, data: dict[str, Any]) -> str:
        # 检查 base_resp 错误状态
        if "base_resp" in data and isinstance(data["base_resp"], dict):
            status_code = data["base_resp"].get("status_code", 0)
            status_msg = data["base_resp"].get("status_msg", "")
            if status_code != 0:
                raise ValueError(
                    f"MiniMax API error: status_code={status_code}, status_msg={status_msg}"
                )

        # Anthropic API 返回格式: {"content": [{"type": "text", "text": "..."}]}
        # MiniMax 可能返回 thinking 块: [{"type": "thinking", ...}, {"type": "text", "text": "..."}]
        # 也可能只有 thinking 没有 text
        if "content" in data and isinstance(data["content"], list):
            text_content = None
            thinking_content = None
            for item in data["content"]:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_content = item.get("text", "")
                        break
                    elif item.get("type") == "thinking":
                        thinking_content = item.get("thinking", "")
            if text_content is not None:
                return text_content
            # MiniMax 有时只有 thinking 没有 text，用 thinking 作为响应
            if thinking_content is not None:
                return thinking_content

        # OpenAI 兼容格式 fallback
        if "choices" in data and data["choices"] is not None:
            return data["choices"][0]["message"]["content"]

        # 无有效响应
        raise ValueError(f"Unexpected API response format: {str(data)[:200]}")


class MockProvider(ModelProvider):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        _ = (temperature, max_tokens, tools)
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
    if provider == "anthropic_compatible":
        return AnthropicCompatibleProvider(
            config.anthropic_base_url,
            config.anthropic_api_key,
            config.model_name,
        )
    if provider == "mock":
        return MockProvider(config.model_name)

    raise ValueError(
        "MODEL_PROVIDER 不支持。可选值: ollama | openai_compatible | anthropic_compatible | mock"
    )
