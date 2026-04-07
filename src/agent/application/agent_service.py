from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from .rag_agent_service import RagAgentService
from ..config import AgentConfig
from ..core.agent_loop import SimpleAgent
from ..core.dialogue import SessionMetaStore, classify_intent
from ..core.dialogue.intent_schema import IntentKind
from ..core.memory_store import MemoryStore
from ..core.planning import Coordinator, build_turn_plan
from ..core.planning.langgraph_agent import LangGraphAgent
from ..core.session_store import SessionStore
from ..core.skill_store import SkillStore
from ..llm.providers import build_model_provider, supported_model_providers
from ..tools.registry import default_tools


class AgentService:
    """组合 RAG 与 LangGraphAgent：编排见 `docs/agent-design/rag-bridge.md` 与 `dialogue-planning.md`。

    - 对话终答**始终**由 `SimpleAgent` 产出；语料型且开启 RAG 时调用 `RagAgentService.answer` 做检索（及同路径内的 grounded 生成），
      但仅将 `retrieval_hits` 格式化为「[检索证据]」注入用户消息，**不把** RAG 返回的 `answer` 当作 `chat` 的最终回复。
    - 无命中或拒答时仍可走 `SimpleAgent`（无证据块或仅有空检索结果），由工具循环与自然语言策略兜底。
    - 是否调用检索仅由 ``RAG_ENABLED`` / 单次请求的 ``use_rag`` 决定，**不再**按「工具/闲聊」意图二次跳过。
    - 检索 query 经 ``rewrite_for_rag``（归一化 + 轻量指代拼接）；直连 ``/api/rag`` 仍返回完整 RAG 结果。
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._ensure_dirs()
        self.memory_store = MemoryStore(self.config.memory_file)
        self.skill_store = SkillStore(self.config.skills_dir)
        self.session_store = SessionStore(self.config.sessions_file)
        self.session_meta_store = SessionMetaStore(self.config.data_dir / "sessions_meta.json")
        self.model = build_model_provider(self.config)
        self.query_rewrite_mode = os.getenv("QUERY_REWRITE_MODE", "hybrid").strip().lower()
        self.query_rewrite_temperature = float(os.getenv("QUERY_REWRITE_TEMPERATURE", "0.0"))
        self.query_rewrite_max_tokens = int(os.getenv("QUERY_REWRITE_MAX_TOKENS", "128"))
        # 先创建 rag service，再创建 tools（以便传入 rag 工具）
        self.rag = RagAgentService(config=self.config, model=self.model)
        self.tools = default_tools(
            self.memory_store, self.skill_store, self.config.workspace_dir, rag_service=self.rag
        )
        self.agent = LangGraphAgent(config=self.config, model=self.model, tools=self.tools)
        # Coordinator（多 Agent 模式）
        self.coordinator = Coordinator(
            config=self.config,
            model=self.model,
            tools=self.tools,
        )

    def _ensure_dirs(self) -> None:
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.config.skills_dir.mkdir(parents=True, exist_ok=True)

    def build_long_context(self) -> str:
        memory = self.memory_store.read()
        skills = self.skill_store.render_for_prompt()
        return f"[MEMORY.md]\n{memory}\n\n[SKILLS]\n{skills}"

    def _append_turn_to_session(
        self, session_id: str, history: list[dict[str, str]], user_text: str, assistant_text: str
    ) -> None:
        messages = list(history)
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
        self.session_store.set_history(session_id, messages)

    def _should_use_coordinator(self, user_input: str) -> bool:
        """判断是否使用 Coordinator 模式

        启发式规则：
        - 问题较长（> 50 字符）
        - 包含多个关键词（数据、年份、计算等）
        """
        if not self.config.use_coordinator:
            return False
        return len(user_input) > 50 or any(
            k in user_input.lower()
            for k in ["多少", "增长", "同比", "年度", "报告", "计算", "对比", "分析"]
        )

    def chat(
        self,
        session_id: str,
        user_message: str,
        *,
        use_rag: bool | None = None,
    ) -> dict[str, Any]:
        """`use_rag` 为 ``None`` 时遵循 ``config.rag_enabled``；否则仅本请求覆盖是否走语料检索。"""
        rag_on = self.config.rag_enabled if use_rag is None else bool(use_rag)
        history = self.session_store.get_history(session_id)
        meta = self.session_meta_store.get(session_id)
        if meta.phase == "awaiting_clarification" and meta.pending_context.strip():
            turn_text = f"{meta.pending_context.strip()}\n{user_message.strip()}".strip()
        else:
            turn_text = user_message.strip()

        intent = classify_intent(turn_text, history)

        if intent.intent == IntentKind.AMBIGUOUS:
            plan = build_turn_plan(intent=intent, rag_will_run=False)
            meta.phase = "awaiting_clarification"
            meta.pending_context = turn_text
            meta.last_intent = intent.intent.value
            meta.last_plan_summary = plan.summary()
            self.session_meta_store.put(session_id, meta)
            clarify = (intent.clarify_prompt or "请补充更多信息。").strip()
            self._append_turn_to_session(session_id, history, user_message, clarify)
            return {
                "answer": clarify,
                "steps_used": 0,
                "tool_calls": [],
                "session_id": session_id,
                "rag": None,
            }

        meta.pending_context = ""
        meta.phase = "idle"
        plan = build_turn_plan(intent=intent, rag_will_run=rag_on)
        meta.last_intent = intent.intent.value
        meta.last_plan_summary = plan.summary()
        self.session_meta_store.put(session_id, meta)

        # 检查是否使用 Coordinator 模式
        if self._should_use_coordinator(turn_text):
            coord_result = self.coordinator.run(
                user_input=turn_text,
                history_messages=history,
                long_context=self.build_long_context(),
            )
            self._append_turn_to_session(session_id, history, user_message, coord_result.answer)
            return {
                "answer": coord_result.answer,
                "steps_used": coord_result.total_steps,
                "tool_calls": coord_result.tool_calls,
                "session_id": session_id,
                "plan_id": coord_result.plan_id,
                "rag": None,
            }

        result = self.agent.run(
            user_input=turn_text,
            history_messages=history,
            long_context=self.build_long_context(),
        )
        self.session_store.set_history(session_id, result.messages)
        return {
            "answer": result.answer,
            "steps_used": result.steps_used,
            "tool_calls": result.tool_calls,
            "session_id": session_id,
            "rag": None,
        }

    def update_model(self, provider: str, model_name: str) -> dict[str, str]:
        provider = provider.strip().lower()
        model_name = model_name.strip()
        if not provider or not model_name:
            raise ValueError("provider 和 model_name 不能为空。")

        self.config.model_provider = provider
        self.config.model_name = model_name
        self.model = build_model_provider(self.config)
        self.agent = LangGraphAgent(config=self.config, model=self.model, tools=self.tools)
        self.rag.update_model(self.model)
        return {"provider": self.config.model_provider, "model_name": self.config.model_name}

    def get_state(self) -> dict[str, Any]:
        return {
            "model_provider": self.config.model_provider,
            "model_name": self.config.model_name,
            "model_providers": supported_model_providers(),
            "sessions": self.session_store.list_session_ids(),
            "skills": self.skill_store.list_skills(),
            "memory_path": str(self.config.memory_file),
            "skills_path": str(self.config.skills_dir),
            "workspace_path": str(self.config.workspace_dir),
            "framework": {
                "rag_enabled": self.config.rag_enabled,
                "query_rewrite_mode": self.query_rewrite_mode,
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

    def list_chat_sessions(self) -> list[dict[str, str | int]]:
        return self.session_store.list_sessions_meta()

    def get_chat_history(self, session_id: str) -> list[dict[str, str]]:
        return list(self.session_store.get_history(session_id))

    def ensure_chat_session(self, session_id: str) -> None:
        self.session_store.ensure_session(session_id)

    def delete_chat_session(self, session_id: str) -> bool:
        ok = self.session_store.delete_session(session_id)
        if ok:
            self.session_meta_store.delete(session_id)
        return ok

    def get_memory(self) -> str:
        return self.memory_store.read()

    def save_memory(self, content: str) -> None:
        content = content.strip() + "\n"
        self.config.memory_file.write_text(content, encoding="utf-8")

    def get_skill(self, name: str) -> str:
        return self.skill_store.read_skill(name)

    def save_skill(self, name: str, content: str) -> str:
        return self.skill_store.upsert_skill(name, content)

    def ingest_document(
        self,
        doc_id: str,
        source: str,
        content: str,
        doc_metadata: dict[str, str] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        dedup_across_docs: bool = False,
    ) -> dict[str, Any]:
        return self.rag.ingest_document(
            doc_id=doc_id,
            source=source,
            content=content,
            doc_metadata=doc_metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            dedup_across_docs=dedup_across_docs,
        )

    def rag_answer(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        return self.rag.answer(question=question, top_k=top_k)

    def get_metrics(self) -> dict[str, float | int | None]:
        return self.rag.get_metrics()

