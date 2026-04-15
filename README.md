**Fin-agent**: 面向金融场景的 RAG + Agent 项目。

> 一个面向金融领域的 RAG + Agent 问答系统。展示 RAG pipeline 优化能力、Agent 架构设计能力与工程实现能力。

核心定位：**混合检索 + 多阶段重排 + 可观测 Agent**，默认使用 MiniMax-M2.7（Anthropic 兼容 API），支持本地模型热切换。

---

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                      Web UI (Flask)                     │
│   聊天 · Session 切换 · 模型热切换 · 记忆/技能编辑     │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│              Application Layer                           │
│   AgentService  ·  RagAgentService  ·  Coordinator       │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                  Agent Core                              │
│  ┌────────────────┐  ┌──────────┐  ┌──────────┐  ┌────┐│
│  │ AgentRouter   │  │ Planner  │  │  Worker  │  │Sync ││
│  │(路由决策)      │→ │(任务分解)│→ │Executor  │→ │     ││
│  └────────────────┘  └──────────┘  └──────────┘  └────┘│
│        │                                                   │
│   Route.ReAct ────→ LangGraph ReAct Agent（简单问题）      │
│   Route.Coordinator ──→ Coordinator 多步规划（复杂问题）    │
│   Route.Clarify ────→ 澄清回复（模糊问题）                │
└─────────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                 RAG Pipeline                            │
│  Ingestion → HybridRetriever → Reranker → Generation   │
│  (文档摄入)   (lexical+tfidf/       (Simple/BGE/Cascade│
│               BM25+embedding)        Reranker)         │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                   LLM Providers                          │
│  anthropic_compatible · openai_compatible · ollama      │
└─────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
myagent/
├─ main.py                          # 启动入口（Web 服务）
├─ requirements.txt
├─ .env.example
├─ runtime/                         # 运行时数据
│   ├─ sessions.json               # 会话历史
│   ├─ traces.jsonl                # 执行链路追踪
│   └─ eval_records.jsonl           # 评估记录
├─ workspace/                       # 可进化资产
│   ├─ MEMORY.md                   # 长期记忆
│   └─ skills/                     # 技能库（.md 文件）
└─ src/agent/
   ├─ config.py                    # 配置层（所有环境变量集中管理）
   ├─ core/
   │   ├─ agent_loop.py            # Agent 状态机循环（兼容）
   │   ├─ planning/
   │   │   ├─ coordinator.py       # 主调度器（Plan→Worker→Synthesize）
   │   │   ├─ planner.py          # 任务分解器（LLM 生成步骤计划）
   │   │   ├─ synthesizer.py       # 结果汇总器（基于证据生成回答）
   │   │   ├─ plan_schema.py       # Plan 数据结构
   │   │   ├─ task_notification.py # Worker 通知协议
   │   │   ├─ worker_result.py     # Worker 执行结果封装
   │   │   ├─ langgraph_agent.py  # LangGraph ReAct Agent 封装
   │   │   ├─ agent_graph.py      # LangGraph 图构建
   │   │   └─ state.py            # Agent 状态定义
   │   ├─ retrieval.py             # 混合检索（lexical + tfidf/BM25 + embedding）
   │   ├─ rerank.py                # 多级重排（Simple · BGE · HuggingFace · Cascade）
   │   ├─ router/
   │   │   └─ agent_router.py     # AgentRouter 路由决策（复杂度+困惑度判断）
   │   ├─ dialogue/
   │   │   ├─ intent_classifier.py # 四阶段意图分类
   │   │   ├─ intent_schema.py    # 意图数据结构
   │   │   └─ query_rewrite.py   # Query 重写（HyDE/expand）
   │   ├─ ingestion.py             # 文档清洗·切块·去重
   │   ├─ generation.py            # 基于证据生成 + 拒答
   │   ├─ evaluation.py            # RAG 评估（Recall@K · HitRate · MRR · LLM Groundedness）
   │   ├─ observability.py          # trace 事件记录
   │   └─ schemas.py              # 数据模型（DocumentChunk · RetrievalHit · EvalRecord）
   ├─ application/
   │   ├─ agent_service.py        # Agent 编排层（Router + Coordinator/LangGraph）
   │   └─ rag_agent_service.py     # RAG 链路编排（含 embedding 健康检查与降级）
   ├─ llm/
   │   ├─ providers.py             # 多 Provider 统一抽象
   │   └─ embeddings.py           # Embedding 提供者（Ollama qwen3-embedding · Mock）
   ├─ tools/
   │   ├─ registry.py             # 工具注册与调用协议
   │   ├─ rag_tool.py            # search_knowledge_base 工具函数
   │   ├─ schemas.py             # 工具数据结构
   │   ├─ builders.py             # 工具构建器
   │   ├─ context.py             # 工具上下文
   │   └─ mcp.py                 # MCP 工具管理
   ├─ interfaces/
   │   └─ web_app.py              # Flask Web UI + REST API
   └─ (compat) agent.py · providers.py · service.py · web.py
```

---

## 快速启动

```powershell
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
copy .env.example .env
# 编辑 .env，填入你的 API Key（若使用 anthropic_compatible 或 openai_compatible）

# 3. 启动
python main.py

# 4. 打开页面
open http://127.0.0.1:7860
```

**默认配置**：使用 `anthropic_compatible` provider + `MiniMax-M2.7` 模型，embedding 通道使用 Ollama `qwen3-embedding`（4096维）。

---

## 核心能力详解

### 1. 混合检索（Hybrid Retrieval）

三层检索通道，支持权重求和或 RRF fusion：

| 通道 | 权重（默认） | 说明 |
|------|-------------|------|
| Lexical（词元重叠） | 0.35 | query token 与 chunk token 重叠度 |
| TF-IDF / BM25 | 0.25 | 词频-逆文档频率（可切换 `tfidf` · `bm25` · `tfidf_bm25`） |
| Embedding（qwen3-embedding） | 0.40 | Ollama 本地向量（4096维，健康检查失败自动降级为纯词频） |

```env
RETRIEVAL_FUSION_MODE=weighted_sum   # 或 rrf
RETRIEVAL_LEXICAL_WEIGHT=0.35
RETRIEVAL_TFIDF_WEIGHT=0.25
RETRIEVAL_EMBEDDING_WEIGHT=0.40
EMBEDDING_ENABLED=true
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=qwen3-embedding
SPARSE_MODE=tfidf
```

### 2. 多级重排（Rerank）

| Reranker | 说明 | 部署要求 |
|----------|------|---------|
| `SimpleReranker` | 规则分（keyword/length/metadata/numeric bonus） | 默认，无额外依赖 |
| `BGEReranker` | Ollama BGE cross-encoder 精排 | `ollama pull dengcao/bge-reranker-v2-m3` |
| `HuggingFaceReranker` | sentence-transformers CrossEncoder | `pip install sentence-transformers` |
| `CascadeReranker` | 规则初筛（2×top_k）→ BGE/HF 精排，两阶段串联 | 上述任一 |

```env
RERANK_ENABLED=true
RERANK_CASCADE=true                  # 启用两阶段串联
RERANKER_PROVIDER=huggingface        # 或 ollama
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

### 3. AgentRouter 路由决策

`AgentRouter` 是统一入口，根据复杂度+困惑度将请求路由到不同执行路径：

```
用户问题 → AgentRouter.decide(query, history, intent_result)
    │
    ├── Route.ReAct  → LangGraph ReAct Agent（单步工具调用，简单问题）
    │                 例：时间查询、计算、闲聊
    │
    ├── Route.Coordinator → Coordinator 多步规划（复杂问题）
    │                       例：查营收+计算增长率、多工具协作
    │
    └── Route.Clarify → 澄清回复（模糊问题）
                          例：单独代词、无宾语疑问词
```

**路由判断维度**：
- **IntentTier**：MIXED/OOS → Coordinator；AMBIGUOUS/CHITCHAT → Clarify
- **复杂度信号**：多语义pattern（`并[且和]`、`对比`、`增长率`）≥2个 → Coordinator
- **置信度**：TOOL_ONLY + conf≥0.85 → ReAct；conf<0.60 + 非KNOWLEDGE → Clarify
- **单步信号**：`现在几点`、`计算\d+`等正则匹配 → ReAct

### 4. Coordinator 规划层

`Coordinator` 实现 **Plan → Execute → Synthesize** 三阶段：

```
用户问题
    │
    ▼
┌─────────┐  LLM 分解任务
│ Planner │ → { plan_id, goal, steps: [{action, detail, depends_on}] }
└────┬────┘
     │
     ▼
┌──────────────────┐  并行/串行执行（ThreadPoolExecutor）
│  WorkerExecutor  │  action: rag · calc · web · memory · synthesize
│  (工具调用)        │
└────┬─────────────┘
     │
     ▼
┌────────────┐
│ Synthesizer│  汇总 worker 结果 + 证据 → 生成最终回答
└────────────┘
```

- **action 类型**：`rag`（知识库检索）、`calc`（计算）、`web`（联网搜索）、`memory`（读写记忆）、`synthesize`（汇总生成）
- **依赖调度**：根据 `depends_on` 与 `parallel_with` 自动拓扑排序，独立任务并行执行
- **降级策略**：Planner JSON 解析失败时，自动降级为直接 synthesize

### 5. RAG 评估指标体系

| 指标 | 说明 |
|------|------|
| `Recall@K` | Top-K 检索结果中包含正确答案的比例 |
| `HitRate@K` | Top-K 是否存在任意命中文档（0/1） |
| `MRR` | 首个命中位置权重（1/rank） |
| `GroundednessEvaluator` | LLM 评估生成内容被证据支持的比例（0~1） |
| `RelevanceEvaluator` | LLM 评估回答与问题的相关程度（0~1） |
| `substring_match_rate` | 离线评估：参考答案子串命中比例 |

### 6. 可进化机制

- **长期记忆**：`workspace/MEMORY.md`，Agent 可读写
- **技能学习**：`workspace/skills/*.md`，按需加载
- **会话沉淀**：`runtime/sessions.json`
- **可观测**：`runtime/traces.jsonl` 记录完整执行链路

---

## API 概览

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/chat` | POST | Agent 对话（支持工具调用链路） |
| `/api/rag` | POST | 纯 RAG 链路（检索→重排→生成） |
| `/api/ingest` | POST | 文档摄入（PDF/TXT） |
| `/api/metrics` | GET | 在线评估聚合指标 |
| `/api/sessions` | GET | 列出所有会话 |
| `/:session_id` | GET | 切换 Session |

---

## 技术栈

| 层级 | 技术 |
|------|------|
| Web 框架 | Flask |
| LLM | anthropic_compatible / openai_compatible / ollama |
| Embedding | qwen3-embedding（4096维，阿里 Qwen3，Ollama） |
| Reranker | BAAI/bge-reranker-v2-m3（CrossEncoder） |
| Agent | LangGraph（ReAct）+ 自研 Coordinator + AgentRouter |
| 评估 | Recall@K · HitRate · MRR · LLM-based Groundedness |
| 向量存储 | 内存索引（惰性计算 + 磁盘持久化） |

---

## 关键环境变量

```env
# 模型（默认 anthropic_compatible + MiniMax-M2.7）
MODEL_PROVIDER=anthropic_compatible
MODEL_NAME=MiniMax-M2.7
ANTHROPIC_AUTH_TOKEN=your_token_here
ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic

# Embedding（Ollama 本地）
EMBEDDING_ENABLED=true
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=qwen3-embedding

# 检索权重（embedding 不可用时自动降级为纯词频）
RETRIEVAL_FUSION_MODE=weighted_sum
RETRIEVAL_LEXICAL_WEIGHT=0.35
RETRIEVAL_TFIDF_WEIGHT=0.25
RETRIEVAL_EMBEDDING_WEIGHT=0.40

# 重排
RERANK_ENABLED=true
RERANK_CASCADE=true
RERANKER_PROVIDER=huggingface
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Agent（USE_COORDINATOR=true 启用规划层）
USE_COORDINATOR=false
MAX_STEPS=6

# Web
WEB_HOST=127.0.0.1
WEB_PORT=7860
```
