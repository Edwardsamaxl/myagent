from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentConfig:
    model_provider: str
    model_name: str
    ollama_base_url: str
    openai_base_url: str
    openai_api_key: str
    anthropic_base_url: str
    anthropic_api_key: str
    max_steps: int
    temperature: float
    max_tokens: int
    web_host: str
    web_port: int
    data_dir: Path
    workspace_dir: Path
    skills_dir: Path
    memory_file: Path
    sessions_file: Path
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    rerank_top_k: int
    rerank_enabled: bool
    rerank_provider: str
    rerank_model: str
    rerank_base_url: str
    rerank_cascade: bool
    use_coordinator: bool
    retrieval_fusion_mode: str
    retrieval_lexical_weight: float
    retrieval_tfidf_weight: float
    retrieval_embedding_weight: float
    embedding_enabled: bool
    embedding_provider: str
    embedding_model: str
    embedding_top_k: int
    rag_enabled: bool
    trace_file: Path
    eval_records_file: Path

    @classmethod
    def from_env(cls) -> "AgentConfig":
        data_dir = Path(os.getenv("DATA_DIR", "./runtime")).resolve()
        workspace_dir = Path(os.getenv("WORKSPACE_DIR", "./workspace")).resolve()
        skills_dir = workspace_dir / "skills"
        memory_file = workspace_dir / "MEMORY.md"
        sessions_file = data_dir / "sessions.json"
        trace_file = data_dir / "traces.jsonl"
        eval_records_file = data_dir / "eval_records.jsonl"
        return cls(
            model_provider=os.getenv("MODEL_PROVIDER", "anthropic_compatible").strip(),
            model_name=os.getenv("MODEL_NAME", "MiniMax-M2.7").strip(),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com").strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            anthropic_api_key=os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip(),
            anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic").strip(),
            max_steps=int(os.getenv("MAX_STEPS", "6")),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("MAX_TOKENS", "768")),
            web_host=os.getenv("WEB_HOST", "127.0.0.1").strip(),
            web_port=int(os.getenv("WEB_PORT", "7860")),
            data_dir=data_dir,
            workspace_dir=workspace_dir,
            skills_dir=skills_dir,
            memory_file=memory_file,
            sessions_file=sessions_file,
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "6")),
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "3")),
            rerank_enabled=os.getenv("RERANK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
            rerank_provider=os.getenv("RERANKER_PROVIDER", "huggingface").strip().lower(),
            rerank_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3").strip(),
            rerank_base_url=os.getenv("RERANKER_BASE_URL", "http://localhost:11434").strip(),
            rerank_cascade=os.getenv("RERANK_CASCADE", "true").strip().lower() in {"1", "true", "yes", "on"},
            retrieval_fusion_mode=os.getenv("RETRIEVAL_FUSION_MODE", "weighted_sum").strip().lower(),
            retrieval_lexical_weight=float(os.getenv("RETRIEVAL_LEXICAL_WEIGHT", "0.35")),
            retrieval_tfidf_weight=float(os.getenv("RETRIEVAL_TFIDF_WEIGHT", "0.25")),
            retrieval_embedding_weight=float(os.getenv("RETRIEVAL_EMBEDDING_WEIGHT", "0.40")),
            embedding_enabled=os.getenv("EMBEDDING_ENABLED", "true").strip().lower()
            in {"1", "true", "yes", "on"},
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "").strip().lower(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "qwen3-embedding").strip(),
            embedding_top_k=int(os.getenv("EMBEDDING_TOP_K", "12")),
            rag_enabled=os.getenv("RAG_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
            use_coordinator=os.getenv("USE_COORDINATOR", "false").strip().lower() in {"1", "true", "yes", "on"},
            trace_file=trace_file,
            eval_records_file=eval_records_file,
        )

