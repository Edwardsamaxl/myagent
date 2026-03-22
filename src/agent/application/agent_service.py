from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .rag_agent_service import RagAgentService
from ..config import AgentConfig
from ..core.agent_loop import SimpleAgent
from ..core.memory_store import MemoryStore
from ..core.session_store import SessionStore
from ..core.skill_store import SkillStore
from ..llm.providers import build_model_provider
from ..tools.registry import default_tools


class AgentService:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._ensure_dirs()
        self.memory_store = MemoryStore(self.config.memory_file)
        self.skill_store = SkillStore(self.config.skills_dir)
        self.session_store = SessionStore(self.config.sessions_file)
        self.model = build_model_provider(self.config)
        self.tools = default_tools(self.memory_store, self.skill_store, self.config.workspace_dir)
        self.agent = SimpleAgent(config=self.config, model=self.model, tools=self.tools)
        self.rag = RagAgentService(config=self.config, model=self.model)

    def _ensure_dirs(self) -> None:
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.config.skills_dir.mkdir(parents=True, exist_ok=True)

    def build_long_context(self) -> str:
        memory = self.memory_store.read()
        skills = self.skill_store.render_for_prompt()
        return f"[MEMORY.md]\n{memory}\n\n[SKILLS]\n{skills}"

    def chat(self, session_id: str, user_message: str) -> dict[str, Any]:
        rag_result: dict[str, Any] | None = None
        if self.config.rag_enabled:
            rag_result = self.rag.answer(user_message)
        history = self.session_store.get_history(session_id)
        user_input = user_message
        if rag_result and rag_result.get("retrieval_hits"):
            context_block = self._build_retrieval_context(rag_result["retrieval_hits"])
            user_input = f"{user_message}\n\n[检索证据]\n{context_block}"
        result = self.agent.run(
            user_input=user_input,
            history_messages=history,
            long_context=self.build_long_context(),
        )
        self.session_store.set_history(session_id, result.messages)
        return {
            "answer": result.answer,
            "steps_used": result.steps_used,
            "tool_calls": result.tool_calls,
            "session_id": session_id,
            "rag": rag_result,
        }

    def update_model(self, provider: str, model_name: str) -> dict[str, str]:
        provider = provider.strip().lower()
        model_name = model_name.strip()
        if not provider or not model_name:
            raise ValueError("provider 和 model_name 不能为空。")

        self.config.model_provider = provider
        self.config.model_name = model_name
        self.model = build_model_provider(self.config)
        self.agent = SimpleAgent(config=self.config, model=self.model, tools=self.tools)
        self.rag.update_model(self.model)
        return {"provider": self.config.model_provider, "model_name": self.config.model_name}

    def get_state(self) -> dict[str, Any]:
        return {
            "model_provider": self.config.model_provider,
            "model_name": self.config.model_name,
            "sessions": self.session_store.list_session_ids(),
            "skills": self.skill_store.list_skills(),
            "memory_path": str(self.config.memory_file),
            "skills_path": str(self.config.skills_dir),
            "workspace_path": str(self.config.workspace_dir),
            "framework": {
                "rag_enabled": self.config.rag_enabled,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "retrieval_top_k": self.config.retrieval_top_k,
                "rerank_top_k": self.config.rerank_top_k,
            },
            "config": asdict(self.config) | {
                "data_dir": str(self.config.data_dir),
                "workspace_dir": str(self.config.workspace_dir),
                "skills_dir": str(self.config.skills_dir),
                "memory_file": str(self.config.memory_file),
                "sessions_file": str(self.config.sessions_file),
                "trace_file": str(self.config.trace_file),
                "eval_records_file": str(self.config.eval_records_file),
            },
        }

    def get_memory(self) -> str:
        return self.memory_store.read()

    def save_memory(self, content: str) -> None:
        content = content.strip() + "\n"
        self.config.memory_file.write_text(content, encoding="utf-8")

    def get_skill(self, name: str) -> str:
        return self.skill_store.read_skill(name)

    def save_skill(self, name: str, content: str) -> str:
        return self.skill_store.upsert_skill(name, content)

    def ingest_document(self, doc_id: str, source: str, content: str) -> dict[str, Any]:
        return self.rag.ingest_document(doc_id=doc_id, source=source, content=content)

    def rag_answer(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        return self.rag.answer(question=question, top_k=top_k)

    def get_metrics(self) -> dict[str, float | int]:
        return self.rag.get_metrics()

    @staticmethod
    def _build_retrieval_context(hits: list[dict[str, Any]]) -> str:
        lines = []
        for idx, hit in enumerate(hits, start=1):
            lines.append(
                f"[{idx}] source={hit.get('source', '')} score={hit.get('score', 0)}\n"
                f"{hit.get('text_preview', '')}"
            )
        return "\n\n".join(lines)
