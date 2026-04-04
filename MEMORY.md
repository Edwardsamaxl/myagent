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

## 2026-04-04 工作记录

### Tool-calling 修复（已验证通过）✅

**问题**：`ModelProviderWrapper._generate` 中 `self._bound_tools` 未传递给 API，且 `generate()` 返回纯文本丢失 tool_use 信息。

**修复**（涉及 3 个文件）：

1. **`src/agent/llm/providers.py`**
   - `ModelProvider.generate()` 添加 `tools` 参数
   - `_HTTPChatProvider.generate()` 添加 `tools` 参数
   - `OpenAICompatibleProvider._build_payload()` 支持 tools
   - `AnthropicCompatibleProvider._build_payload()` 支持 tools
   - `OllamaProvider.generate()` 支持 tools
   - `MockProvider.generate()` 支持 tools
   - 新增 `generate_raw()` 方法返回完整 API 响应

2. **`src/agent/core/planning/langgraph_agent.py`**
   - `_generate()` 使用 `generate_raw()` 获取原始响应
   - 正确解析 `tool_use` 并构建 LangChain `ToolCall`
   - 使用 `AIMessage(tool_calls=[...])` 格式

3. **`src/agent/tools/registry.py`**
   - `get_time()`: `(_: str)` → `()`
   - `read_memory()`: `(_: str)` → `()`
   - `list_skills()`: `(_: str)` → `()`

**验证结果**：
- `get_time`: Answer = "当前时间是 **2026年4月4日 14:24:57**", Steps = 1
- `calculate`: Answer = "20", Steps = 3
- `read_memory`: 正确读取记忆文件内容, Steps = 1

### 待验证问题

1. **Reranker**：`HuggingFaceReranker` 在 `ai_env` 中是否正常
2. **评估数据格式**：`expected_answer` 与 chunk 原文格式不匹配

---

## 2026-04-03 工作记录

### 今日修复（已验证通过）

#### 1. LangGraph 兼容性错误 ✅
**问题**：Web UI 返回"服务内部错误"，错误为 `AttributeError: 'list' object has no attribute 'message'`
**根因**：`ModelProviderWrapper._generate` 返回 `LLMResult(generations=[[chat_gen]])`（2层嵌套），但 LangChain 内部期望 `ChatResult(generations=[chat_gen])`（1层）
**修复**：
- `langgraph_agent.py` 第 54 行：导入从 `LLMResult` 改为 `ChatResult`
- `langgraph_agent.py` 第 80 行：返回类型从 `LLMResult` 改为 `ChatResult`
- `langgraph_agent.py` 第 95 行：`generations` 格式从 `[[chat_gen]]` 改为 `[chat_gen]`

#### 2. 消息对象类型错误 ✅
**问题**：`session_store.set_history()` 期望 `dict`，但收到 `SystemMessage` 等 LangChain 对象
**修复**：`langgraph_agent.py` 的 `run()` 方法中添加 `msg_to_dict()` 转换函数

#### 3. 流式接口 `split("")` bug ✅
**问题**：`web_app.py` 第 244 行 `answer.split("")` 导致 `ValueError: empty separator`
**修复**：改为直接遍历字符 `for i, ch in enumerate(answer):`

### 当前状态
- Web UI：`/api/chat` 和 `/api/chat/stream` 均返回 200 OK
- API 认证：正常工作 (MiniMax-M2.7)
- 工具调用：**已验证通过**（2026-04-04）

### 待验证问题
1. **Tool-calling 功能**：✅ 已修复并验证
2. **Reranker**：`HuggingFaceReranker` 在 `ai_env` 中是否正常

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

#### 1. Tool-calling 功能未验证（需测试）
**问题**：`ModelProviderWrapper._generate` 中 `self._inner.generate()` 没有传递 `tools` 参数给 API
**当前状态**：`self._bound_tools` 存储了工具定义，但 API 调用时未注入
**影响**：Agent 无法调用工具，只能生成文本

#### 2. sentence-transformers / PyTorch 版本
**问题**：scoop 全局 Python 的 PyTorch 2.2 太旧，导致 transformers 5.4 报错
**注意**：用户用 `ai_env` 环境跑，`torch` 是 2.5.1，不应该有问题
**验证**：在 `ai_env` 中直接测试 `from sentence_transformers import CrossEncoder`

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
    └── agent_service.py       # AgentService → LangGraphAgent (已修复)
```

### 下一步顺序

1. **测试 Reranker**：`HuggingFaceReranker` 在 `ai_env` 中是否正常工作
2. **修复评估数据格式**：统一 expected_answer 与 chunk 原文的格式
3. **跑 eval_retrieval.py**：验证 reranker + 评估框架
4. **commit 所有修复**

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
