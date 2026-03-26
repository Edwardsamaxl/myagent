from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Any

import requests

from ..config import AgentConfig


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OllamaEmbeddingProvider(EmbeddingProvider):
    # 并行 worker 数量
    _MAX_WORKERS = 8

    def __init__(self, base_url: str, model_name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not texts:
            return []
        vectors: list[list[float]] = [None] * len(texts)

        def _embed_one(idx_and_text: tuple[int, str]) -> tuple[int, list[float]]:
            idx, text = idx_and_text
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            vec = data.get("embedding")
            if not isinstance(vec, list):
                raise ValueError("Ollama embeddings 返回格式异常：缺少 embedding 列表。")
            return idx, [float(x) for x in vec]

        with ThreadPoolExecutor(max_workers=self._MAX_WORKERS) as executor:
            futures = {executor.submit(_embed_one, (i, t)): i for i, t in enumerate(texts)}
            for future in as_completed(futures):
                idx, vec = future.result()
                vectors[idx] = vec

        return vectors


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, base_url: str, api_key: str, model_name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 为空，无法使用 openai_compatible embedding。")
        payload: dict[str, Any] = {"model": self.model_name, "input": texts}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{self.base_url}/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        if not isinstance(data, list):
            raise ValueError("OpenAI embeddings 返回格式异常：缺少 data 列表。")
        vectors: list[list[float]] = []
        for item in data:
            vec = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vec, list):
                raise ValueError("OpenAI embeddings 返回格式异常：embedding 不是列表。")
            vectors.append([float(x) for x in vec])
        return vectors


class MockEmbeddingProvider(EmbeddingProvider):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            bucket = [0.0] * 32
            coarse_tokens = re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower())
            fine_tokens = re.findall(r"[\u4e00-\u9fff]", text.lower())
            for token in coarse_tokens + fine_tokens:
                bucket[hash(token) % 32] += 1.0
            out.append(bucket)
        return out


def build_embedding_provider(config: AgentConfig) -> EmbeddingProvider | None:
    if not config.embedding_enabled:
        return None

    provider = config.embedding_provider or config.model_provider.lower()
    if provider == "ollama":
        return OllamaEmbeddingProvider(config.ollama_base_url, config.embedding_model)
    if provider == "openai_compatible":
        return OpenAICompatibleEmbeddingProvider(
            config.openai_base_url,
            config.openai_api_key,
            config.embedding_model,
        )
    if provider == "mock":
        return MockEmbeddingProvider()
    return None
