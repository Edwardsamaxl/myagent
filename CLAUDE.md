# CLAUDE.md

## 项目定位

**mini-evolving-agent** 是一个面向算法工程师实习面试的 RAG+Agent 项目。
目标：展示 RAG pipeline 优化能力、Agent 架构设计能力、工程实现能力。

## 核心问题（截至 2026-03-30）

### 1. Embedding 通道已修复

**问题**：`EMBEDDING_ENABLED=true`，配置 `embedding_weight=0.40`，但 `OllamaEmbeddingProvider` 调用的 `qwen3-embedding` 模型若未在 Ollama 中运行，则 `/api/embeddings` 请求静默失败，embedding 通道退化为空，整个混合检索变成纯词频（lexical + tfidf），语义权重 40% 形同虚设。

**修复**（`rag_agent_service.py`）：
- 在 `RagAgentService.__init__` 中增加 **embedding 健康检查**：用单句测试向量是否返回，失败则将 `embedding_weight` 设为 0.0，provider 置 None
- 降级行为有 `logging.warning` 明确提示原因和修复建议
- 不再静默失败

**当前 fallback 行为**：
- `ollama` 无 `qwen3-embedding` → 降级到纯词频（lexical + tfidf，总权重 60%）
- `embedding_provider=mock` → 使用 MockEmbeddingProvider（32 维哈希向量，测试用）

### 2. Agent 规划层缺失

**问题**：`SimpleAgent` 本质是"见到工具名就调用"，没有先规划再执行的机制。设计文档 `loop.md` 定义了 `plan → tool_call → observe` 状态机，但代码中没有实现 planner 角色。

**影响**：模型决策质量差，该检索时直答、工具选择随机、循环次数不可控。

### 3. RAG 与 Agent 桥接混乱

**问题**：`agent_service.py` 的 `chat()` 完全没有调用 RAG，`retrieval_hits` 注入了但没有在 `agent.run()` 之前做路由决策。意图分类和规划层存在（`classify_intent`、`build_turn_plan`），但与 RAG 调用完全脱节。

### 4. 架构文档 vs 实现

设计文档（`docs/agent-design/`）描述的系统远比代码实现完整。面试时若追问实现细节，需坦诚哪些是骨架实现。

## 后续整改优先级

1. **高优先级**：Agent 规划层（状态机、planner 角色）→ 可考虑迁移到 LangGraph
2. **高优先级**：RAG 与 Agent 协同（路由决策、证据注入时机、`/api/rag` vs `/api/chat` 分工）
3. **中优先级**：完善 embedding 通道（`qwen3-embedding` 已就位，4096 维；健康检查已实装）
4. **中优先级**：评估指标体系（当前 trace 有字段但不完整，缺少召回率等核心指标）
5. **低优先级**：持久化、并发、可观测 Dashboard

## 技术栈

- **模型**：anthropic_compatible（默认 MiniMax-M2.7，通过 api.minimaxi.com）、openai_compatible、ollama
- **Embedding**：`qwen3-embedding`（4096 维，阿里 Qwen3 官方 embedding 模型，ollama pull）；健康检查失败时自动降级为纯词频（lexical + tfidf）
- **检索**：混合检索（lexical + tfidf + embedding，RRF/weighted_sum 可配）
- **重排**：SimpleReranker（骨架实现）
- **生成**：GroundedGenerator（基于证据生成 + 拒答）
- **Agent**：SimpleAgent（工具循环，无规划层）

## 环境变量关键配置

```env
# Embedding 通道
EMBEDDING_ENABLED=true
EMBEDDING_PROVIDER=ollama          # 或 mock / openai_compatible
EMBEDDING_MODEL=qwen3-embedding

# 检索权重（embedding 不可用时自动降级）
RETRIEVAL_LEXICAL_WEIGHT=0.35
RETRIEVAL_TFIDF_WEIGHT=0.25
RETRIEVAL_EMBEDDING_WEIGHT=0.40    # 健康检查失败后自动置 0

# RAG
RAG_ENABLED=true
RETRIEVAL_TOP_K=6
RERANK_TOP_K=3
```

## 面试展示策略

| 维度 | 展示内容 | 注意事项 |
|------|---------|---------|
| RAG 原理 | 完整链路：ingestion → retrieval → rerank → generation | 已实现，但 rerank 是骨架 |
| 检索优化 | hybrid search + RRF fusion + 权重可配置 | embedding 通道已实装 qwen3-embedding（4096维） |
| Agent 架构 | 状态机设计、tool use 协议、规划层分离 | 当前缺规划层，慎谈 |
| 工程能力 | 模块化分层、trace 可观测、eval 机制 | 架构清晰，但 trace 字段不完整 |
