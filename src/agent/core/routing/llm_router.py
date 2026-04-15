"""LLM Router Agent - determines task type and tool set."""

import time
import json
from typing import Any

from .route_decision import RouteDecision, RouteType


class LLMRouterAgent:
    """Uses LLM to classify task type (single_step/multi_step/clarify) and select tools."""

    def __init__(self, model_provider: Any, config: dict[str, Any]) -> None:
        self.model = model_provider
        self.config = config

    def route(self, query: str, context: dict[str, Any]) -> RouteDecision:
        """Determine route type, tools, and optional RAG config using LLM."""
        start = time.time()

        # Build messages for LLM to classify the query
        messages = self._build_routing_messages(query, context)

        # Call LLM with proper parameters
        response = self.model.generate(
            messages=messages,
            temperature=0.1,
            max_tokens=256,
        )
        parsed = self._parse_llm_response(response)

        latency_ms = int((time.time() - start) * 1000)

        return RouteDecision(
            route_type=parsed["route_type"],
            selected_tools=parsed["selected_tools"],
            rag_chain=parsed.get("rag_chain"),
            reasoning=parsed.get("reasoning", ""),
            confidence=parsed.get("confidence", 0.0),
            router_latency_ms=latency_ms,
        )

    def _build_routing_messages(self, query: str, context: dict[str, Any]) -> list[dict[str, str]]:
        """Build messages for LLM to classify the task."""
        available_tools = context.get("available_tools", [])
        tools_str = ", ".join(available_tools) if available_tools else "none"
        history = context.get("history")

        system_prompt = """你是一个路由专家。请将用户查询分类到正确的路由类型。

## 路由类型

**SINGLE_STEP（单步）**: 只需要一次工具调用的简单查询。
- "现在几点" → 需要 get_time 工具
- "茅台是哪家公司" → 需要 search_knowledge_base（一次检索）
- "计算 1+1" → 需要 calculate 工具
- "五粮液主营什么" → 需要 search_knowledge_base（一次回答）

**MULTI_STEP（多步）**: 需要两次或多次顺序工具调用的复杂查询。
- "查茅台营收并计算增长率" → search_knowledge_base 然后 calculate（2步）
- "对比茅台和五粮液的营收" → search_knowledge_base 查两个然后对比（2+步）
- "检索财务数据并分析原因" → search 然后分析（2步）
- "查营收并生成报告" → search_knowledge_base 然后生成报告（2步）

**CLARIFY（澄清）**: 模糊的、不完整的查询，需要用户进一步明确。
- "那个" → 指代不明，需要澄清
- "继续" → 不清楚要继续什么
- "怎么了" → 缺少上下文，需要澄清
- "嗯" → 只是语气词，不是问题
- "说一下" → 没有指定主题
- "你好" → 寒暄，不需要工具
- "谢谢" → 只是感谢，不需要工具

## 示例

Query: "茅台是哪家公司" → {"route_type": "SINGLE_STEP", "selected_tools": ["search_knowledge_base"], "reasoning": "简单知识问答", "confidence": 0.9}
Query: "查茅台营收并计算增长率" → {"route_type": "MULTI_STEP", "selected_tools": ["search_knowledge_base", "calculate"], "reasoning": "需要先检索再计算", "confidence": 0.85}
Query: "那个" → {"route_type": "CLARIFY", "selected_tools": [], "reasoning": "指代不明需要澄清", "confidence": 0.95}
Query: "现在几点" → {"route_type": "SINGLE_STEP", "selected_tools": ["get_time"], "reasoning": "简单工具调用", "confidence": 0.95}
Query: "你好" → {"route_type": "CLARIFY", "selected_tools": [], "reasoning": "寒暄不需要工具", "confidence": 0.95}
Query: "检索财务数据并分析原因" → {"route_type": "MULTI_STEP", "selected_tools": ["search_knowledge_base"], "reasoning": "需要检索和分析两步", "confidence": 0.8}

## 你的任务

请分类这个查询并以 JSON 格式回复：
{"route_type": "SINGLE_STEP|MULTI_STEP|CLARIFY", "selected_tools": [], "rag_chain": null, "reasoning": "", "confidence": 0.0}"""

        if history:
            user_prompt = f"""Query: {query}
Available tools: {tools_str}
Conversation history: {history}"""
        else:
            user_prompt = f"""Query: {query}
Available tools: {tools_str}"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_llm_response(self, response: str) -> dict[str, Any]:
        """Parse LLM JSON response into structured routing decision."""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end]
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end]
            else:
                # Find the first { and try to parse the entire response as JSON
                json_start = response.find('{')
                if json_start != -1:
                    response = response[json_start:]

            data = json.loads(response.strip())

            # Map uppercase LLM response to lowercase enum values
            route_type_str = data.get("route_type", "single_step").upper()
            route_type_map = {
                "SINGLE_STEP": RouteType.SINGLE_STEP,
                "MULTI_STEP": RouteType.MULTI_STEP,
                "CLARIFY": RouteType.CLARIFY,
            }
            route_type = route_type_map.get(route_type_str, RouteType.SINGLE_STEP)

            return {
                "route_type": route_type,
                "selected_tools": data.get("selected_tools", []),
                "rag_chain": data.get("rag_chain"),
                "reasoning": data.get("reasoning", ""),
                "confidence": float(data.get("confidence", 0.5)),
            }
        except (json.JSONDecodeError, ValueError, KeyError):
            # Fallback: treat as CLARIFY since unparseable responses often mean unclear query
            return {
                "route_type": RouteType.CLARIFY,
                "selected_tools": [],
                "rag_chain": None,
                "reasoning": "无法解析LLM响应，默认澄清",
                "confidence": 0.0,
            }
