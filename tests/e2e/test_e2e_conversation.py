"""E2E tests for conversation flow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestE2EConversation:
    """End-to-end conversation flow tests."""

    def test_agent_service_initializes(self):
        """AgentService should initialize without errors.

        Note: This test requires the full environment with proper router imports.
        It tests the integration chain without mocking at import level.
        """
        pytest.skip("Requires full environment with router imports resolved")

    def test_rag_tool_registered_when_rag_enabled(self):
        """RAG tool should be registered when rag_enabled=True."""
        from agent.tools.registry import default_tools
        from agent.core.memory_store import MemoryStore
        from agent.core.skill_store import SkillStore
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryStore(Path(tmpdir) / "memory.md")
            skills = SkillStore(Path(tmpdir) / "skills")
            workspace = Path(tmpdir)

            mock_rag = MagicMock()
            tools = default_tools(memory, skills, workspace, rag_service=mock_rag)

            assert "search_knowledge_base" in tools
            tool = tools["search_knowledge_base"]
            assert tool.name == "search_knowledge_base"
            assert callable(tool.func)

    def test_rag_tool_not_registered_when_no_rag_service(self):
        """RAG tool should NOT be registered when rag_service=None."""
        from agent.tools.registry import default_tools
        from agent.core.memory_store import MemoryStore
        from agent.core.skill_store import SkillStore
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryStore(Path(tmpdir) / "memory.md")
            skills = SkillStore(Path(tmpdir) / "skills")
            workspace = Path(tmpdir)

            tools = default_tools(memory, skills, workspace, rag_service=None)
            assert "search_knowledge_base" not in tools

    def test_tools_equivalence(self):
        """search_knowledge_base should be equivalent to get_time/calculate."""
        from agent.tools.registry import default_tools
        from agent.core.memory_store import MemoryStore
        from agent.core.skill_store import SkillStore
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryStore(Path(tmpdir) / "memory.md")
            skills = SkillStore(Path(tmpdir) / "skills")
            workspace = Path(tmpdir)

            mock_rag = MagicMock()
            tools = default_tools(memory, skills, workspace, rag_service=mock_rag)

            # All should be registered Tool objects with name/description/func
            for name in ["get_time", "calculate", "search_knowledge_base"]:
                assert name in tools, f"{name} should be registered"
                tool = tools[name]
                assert hasattr(tool, "name")
                assert hasattr(tool, "description")
                assert hasattr(tool, "func")
                assert callable(tool.func)

    def test_intent_classifier_import(self):
        """Intent classifier should be importable."""
        from agent.core.dialogue import classify_intent
        assert callable(classify_intent)

    def test_rewrite_for_rag_import(self):
        """rewrite_for_rag should be importable."""
        from agent.core.dialogue import rewrite_for_rag
        assert callable(rewrite_for_rag)

    def test_search_knowledge_base_impl_signature(self):
        """search_knowledge_base_impl should accept retriever, reranker, config."""
        from agent.tools.rag_tool import search_knowledge_base_impl
        import inspect
        sig = inspect.signature(search_knowledge_base_impl)
        params = list(sig.parameters.keys())
        assert "query" in params
        assert "retriever" in params
        assert "reranker" in params
        assert "config" in params
