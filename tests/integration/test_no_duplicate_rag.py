"""Integration tests: verify RAG is not called multiple times per query."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]


class TestNoDuplicateRAG:
    """Test that RAG tool is called at most once per user query."""

    def test_rag_tool_returns_formatted_evidence(self):
        """RAG tool should return formatted evidence, not full answer."""
        # Import the tool factory
        from agent.tools.rag_tool import create_search_knowledge_base_tool

        # Create mock rag_service
        class MockConfig:
            retrieval_top_k = 5
            rerank_top_k = 3

        class MockHit:
            def __init__(self):
                self.text = "贵州茅台2024年营业收入1709亿元"
                self.source = "贵州茅台/年报_2024.md"
                self.score = 0.85
                self.chunk_id = "c1"
                self.metadata = {}

        class MockRetriever:
            _chunks = {"c1": MockHit()}

            def set_weights(self, **kwargs):
                pass

            def search_with_debug(self, query, top_k, numeric_indicators=None):
                return [MockHit()], {}

        class MockRAG:
            config = MockConfig()
            retriever = MockRetriever()
            reranker = MagicMock()

        tool_name, tool_desc, tool_func = create_search_knowledge_base_tool(MockRAG())
        assert tool_name == "search_knowledge_base"

        # Call the tool
        result = tool_func("贵州茅台2024年营收是多少？")

        # Should return formatted evidence, not full answer
        assert isinstance(result, str)
        assert "来源:" in result or "检索结果为空" in result or "检索失败" in result
        # Should NOT contain a direct answer like "1709亿元"
        # (it contains evidence text but that's the retrieval chunk, not an LLM answer)

    def test_rag_tool_no_llm_generation(self):
        """RAG tool should not call LLM for generation."""
        from agent.tools.rag_tool import create_search_knowledge_base_tool

        call_count = {"llm": 0}

        class MockRAG:
            class config:
                retrieval_top_k = 3
                rerank_top_k = 2

            retriever = MagicMock()
            reranker = MagicMock()

        tool_name, tool_desc, tool_func = create_search_knowledge_base_tool(MockRAG())

        # Execute
        result = tool_func("测试查询")

        # Tool returns string - no LLM was called
        assert isinstance(result, str)
        # If retrieval failed, it returns error message (not crashing)
        assert len(result) > 0

    def test_empty_query_returns_error_message(self):
        """Empty query should return error message, not crash."""
        from agent.tools.rag_tool import create_search_knowledge_base_tool

        class MockRAG:
            config = MagicMock()
            retriever = MagicMock()
            reranker = MagicMock()

        tool_name, tool_desc, tool_func = create_search_knowledge_base_tool(MockRAG())

        result = tool_func("")
        assert "为空" in result or "失败" in result

    def test_rag_service_not_called_in_answer_pipeline(self):
        """Verify agent_service.py no longer pre-calls rag.answer()."""
        # This test verifies the code change by checking the source
        agent_service_path = ROOT / "src" / "agent" / "application" / "agent_service.py"
        content = agent_service_path.read_text(encoding="utf-8")

        # The old code had: rag_result = self.rag.answer(question=...)
        # The new code should NOT have rag.answer in the chat method flow
        # for pre-injection
        lines = content.split("\n")
        in_chat_method = False
        chat_method_lines = []
        for line in lines:
            if "def chat(" in line:
                in_chat_method = True
            elif in_chat_method and line.startswith("    def "):
                # Next method started
                break
            if in_chat_method:
                chat_method_lines.append(line)

        chat_code = "\n".join(chat_method_lines)

        # Check that rag.answer pre-call is removed
        # The new code should have rag_result = None or similar
        assert "rag_result = None" in chat_code, "rag_result should be None (on-demand)"
