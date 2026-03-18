from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AgentConfig:
    model_provider: str
    model_name: str
    ollama_base_url: str
    openai_base_url: str
    openai_api_key: str
    max_steps: int
    temperature: float
    max_tokens: int

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            model_provider=os.getenv("MODEL_PROVIDER", "ollama").strip(),
            model_name=os.getenv("MODEL_NAME", "qwen2.5:7b").strip(),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com").strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            max_steps=int(os.getenv("MAX_STEPS", "6")),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
        )

