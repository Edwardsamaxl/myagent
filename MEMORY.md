# MEMORY

## 用户偏好
- 用户信任我的判断，可在说明方案后直接执行
- 用户配置了丰富的 agent 库在 `C:/Users/Ed/.claude/agents/`，工作时按需加载

## 可用 Agent 索引

工作时遇到以下类型任务，读取对应 agent 文件并以其角色执行：

| 任务类型 | Agent 文件 | 用途 |
|---------|-----------|------|
| AI/ML/RAG/LLM | `engineering-ai-engineer.md` | 模型、向量检索、ML系统 |
| 代码审查 | `engineering-code-reviewer.md` | 安全、正确性、可维护性 |
| 系统架构/DDD | `engineering-software-architect.md` | 架构决策、ADRs、边界设计 |

Agent 文件路径：`C:/Users/Ed/.claude/agents/<name>.md`

---

## 2026-03-31 工作记录

### 已完成

#### P0: LangGraph ReAct Agent（feature/langgraph-agent → merged to main）
- 新文件：`planning/state.py`、`planning/nodes.py`、`planning/agent_graph.py`、`planning/langgraph_agent.py`
- `ModelProviderWrapper`：将项目 `ModelProvider`（支持 anthropic_compatible）适配为 LangChain `BaseChatModel`
- `LangGraphAgent`：封装 `create_react_agent`，支持 MiniMax-M2.7 native tool-calling
- `AgentService`：SimpleAgent → LangGraphAgent
- **状态**：代码已 merge，但有兼容性问题（见下方"未解决问题"）

#### P1: BGE Reranker（feature/reranker → merged to main）
- 新增 `HuggingFaceReranker`：基于 `sentence-transformers` 的 CrossEncoder，模型 `BAAI/bge-reranker-v2-m3`
- 新增 `BGEReranker`：Ollama `/api/rerank` 版本（实验性）
- `build_reranker()` 工厂：支持 `huggingface` / `ollama` provider
- config 新增：`rerank_enabled`、`rerank_provider`、`rerank_model`
- **状态**：代码已 merge，但 sentence-transformers 环境问题未解决

#### P2: 评估指标体系（feature/eval-framework → merged to main）
- `recall_at_k`、`hit_rate_at_k`、`mean_reciprocal_rank`
- `GroundednessEvaluator`、`RelevanceEvaluator`（LLM-based）
- `scripts/eval_retrieval.py`、`scripts/run_ablation.py`
- `data/eval/retrieval_test_set.json`（25 条贵州茅台标注问答）
- **状态**：框架代码完整，评估数据格式有问题（expected_answer 与 chunk 原文格式不匹配）

#### 模型信息更新
- CLAUDE.md：`MODEL_PROVIDER=anthropic_compatible`（MiniMax-M2.7），非 ollama/qwen
- MiniMax-M2.7 支持 Anthropic-compatible tool-calling（已测试验证）

#### 环境修复
- requirements.txt 新增：`langgraph>=0.2.0`、`sentence-transformers>=3.0.0`
- 解决了 3 个版本兼容问题：
  1. `@langchain_tool(name=...)` → `StructuredTool.from_function()`
  2. `create_react_agent(max_iterations=)` → 移除该参数
  3. `AgentState` 新增 `remaining_steps` 字段

---

### 未解决问题

#### 1. LangGraphAgent 运行时错误（BLOCKING）
**问题**：`create_react_agent` 调用时 `AttributeError: 'list' object has no attribute 'message'`
**原因**：LangChain `BaseChatModel._generate_with_cache` 内部缓存检查时，`generation` 变成了 `list` 而非 `ChatGeneration` 对象——`ModelProviderWrapper._generate` 返回的 `LLMResult` 格式与 LangChain 内部缓存机制不兼容
**影响**：Web UI 提问返回"服务内部错误"
**RAG 链路验证**：单独测 `rag.answer()` 正常工作，说明 RAG pipeline 没问题
**修复方案**：放弃 `create_react_agent`，改用**手写 ReAct 循环**：

```python
while step < max_steps:
    response = model.bind_tools(tools).invoke(messages)
    if not response.tool_calls:
        return response.content  # 最终答案
    for tc in response.tool_calls:
        result = execute_tool(tc.name, tc.arguments)
        messages.append(ToolMessage(content=str(result), tool_call_id=tc.id))
    step += 1
```

#### 2. sentence-transformers / PyTorch 版本
**问题**：scoop 全局 Python 的 PyTorch 2.2 太旧，导致 transformers 5.4 报错
**注意**：用户用 `ai_env` 环境跑，`torch` 是 2.5.1，不应该有问题
**建议**：先解决 LangGraph 问题后再验证 reranker 是否正常

#### 3. 评估数据格式
**问题**：`expected_answer = "170899152276.34元或约1709亿元"` 但 chunk 里是"1709亿元"（不同格式），导致 recall=0
**修复**：调整 `retrieval_test_set.json` 里的 `expected_answers`，使用 chunk 里会实际出现的格式

---

### 项目当前架构

```
src/agent/
├── core/
│   ├── planning/
│   │   ├── state.py          # AgentState (remaining_steps, messages, ...)
│   │   ├── nodes.py          # _make_langchain_tools → StructuredTool
│   │   ├── agent_graph.py    # build_agent_graph + run_agent (create_react_agent)
│   │   ├── langgraph_agent.py # LangGraphAgent + ModelProviderWrapper
│   │   ├── plan_schema.py
│   │   └── planner.py
│   ├── rerank.py             # SimpleReranker + BGEReranker + HuggingFaceReranker
│   ├── retrieval.py          # InMemoryHybridRetriever (已支持持久化)
│   ├── evaluation.py          # recall_at_k / hit_rate_at_k / mrr / GroundednessEvaluator
│   └── schemas.py
└── application/
    └── agent_service.py       # AgentService → LangGraphAgent (待修复)
```

### 下一步顺序

1. **修复 LangGraphAgent**：用**手写 ReAct 循环**替代 `create_react_agent`（优先级 P0）
2. **验证 RAG + Agent 链路**：能正常检索 + 生成 + 回答
3. **解决 sentence-transformers**：确认 `pip install sentence-transformers` 在 `ai_env` 中成功
4. **跑 eval_retrieval.py**：验证 reranker + 评估框架
5. **修复评估数据格式**：统一 expected_answer 与 chunk 原文的格式
6. **commit 所有修复**

---

## 重要决策记录

- MiniMax-M2.7 使用 **Anthropic-compatible tool-calling**（非 ReAct 字符串解析）
- ReAct vs Tool-calling：选 Tool-calling（MiniMax 支持，效率更高）
- Reranker：选 **HuggingFace CrossEncoder**（`BAAI/bge-reranker-v2-m3`），不用 Ollama（Ollama 的 `/api/rerank` 只支持官方模型）
- embedding 模型：`qwen3-embedding`（4096 维，Ollama pull）

## 长期目标

面向算法工程师实习面试的 RAG+Agent 项目，展示：
- RAG pipeline 优化能力（混合检索 + Rerank）
- Agent 架构设计能力（状态机 + 工具调用）
- 工程化落地能力（模块化 + 可观测 + 评估体系）
