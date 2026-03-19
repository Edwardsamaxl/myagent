from __future__ import annotations

from dataclasses import asdict
from typing import Any

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

    def _ensure_dirs(self) -> None:
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.config.skills_dir.mkdir(parents=True, exist_ok=True)

    def build_long_context(self) -> str:
        memory = self.memory_store.read()
        skills = self.skill_store.render_for_prompt()
        return f"[MEMORY.md]\n{memory}\n\n[SKILLS]\n{skills}"

    def chat(self, session_id: str, user_message: str) -> dict[str, Any]:
        history = self.session_store.get_history(session_id)
        result = self.agent.run(
            user_input=user_message,
            history_messages=history,
            long_context=self.build_long_context(),
        )
        self.session_store.set_history(session_id, result.messages)
        return {
            "answer": result.answer,
            "steps_used": result.steps_used,
            "tool_calls": result.tool_calls,
            "session_id": session_id,
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
            "config": asdict(self.config) | {
                "data_dir": str(self.config.data_dir),
                "workspace_dir": str(self.config.workspace_dir),
                "skills_dir": str(self.config.skills_dir),
                "memory_file": str(self.config.memory_file),
                "sessions_file": str(self.config.sessions_file),
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
