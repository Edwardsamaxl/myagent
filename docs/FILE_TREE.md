# 项目文件结构

**Fin-agent**: 面向金融场景的 RAG + Agent 项目。
**更新日期**: 2026-04-12

---

## 目录树

```
E:\CursorProject\myagent
│
├── main.py                          # 项目入口，启动 Flask Web 服务
│
├── scripts/                         # 评估与工具脚本
│
├── src/agent/                      # Agent 核心模块
│   │
│   ├── __init__.py
│   ├── config.py                   # 配置管理（环境变量读取）
│   ├── service.py                  # Flask HTTP 服务
│   ├── web.py                     # Web UI 路由
│   ├── providers.py                 # 模型提供者工厂
│   ├── memory.py                   # 记忆管理（MemoryStore）
│   ├── sessions.py                 # 会话管理（SessionStore）
│   ├── skills.py                   # 技能管理（SkillStore）
│   │
│   ├── application/                 # 应用层
│   │   ├── agent_service.py        # **主入口**：chat() 对话入口，RAG+Agent 调度
│   │   └── rag_agent_service.py    # **RAG 编排**：检索→重排→生成，answer() 方法
│   │
│   ├── rag/                        # **RAG 全链路模块（2026-04-12 新建）**
│   │   ├── __init__.py
│   │   ├── schemas.py              # DocumentChunk, RetrievalHit, GenerationResult
│   │   ├── retrieval.py            # 混合检索：lexical + tfidf + embedding + numeric 融合
│   │   ├── rerank.py              # 重排：Simple / BGE / HuggingFace / CascadeReranker
│   │   ├── ingestion.py            # 文档摄取：分块 (chunk_size, chunk_overlap)
│   │   ├── generation.py           # 生成：GroundedGenerator（基于证据生成 + 拒答）
│   │   └── evidence_format.py      # 证据块格式化、引用提取、锚点覆盖率评估
│   │
│   ├── core/                       # 核心模块
│   │   ├── schemas.py              # 数据结构：仅 EvalRecord（RAG schemas 移至 rag/）
│   │   ├── evaluation.py            # 评估结果存储（JSONL）
│   │   ├── memory_store.py          # 长期记忆存储
│   │   ├── session_store.py         # 会话历史存储
│   │   ├── skill_store.py           # 技能库存储
│   │   │
│   │   ├── dialogue/               # 对话处理
│   │   │   ├── __init__.py
│   │   │   ├── query_rewrite.py     # 查询改写：rule/llm/hybrid/hyde/expand 模式
│   │   │   ├── intent_classifier.py # 意图分类：rule→embedding→LLM 三级分类
│   │   │   ├── intent_schema.py     # IntentKind 枚举，IntentResult 数据类
│   │   │   ├── clarify_policy.py    # 澄清策略
│   │   │   └── session_meta_store.py
│   │   │
│   │   ├── router/                 # 路由决策
│   │   │   ├── __init__.py
│   │   │   └── agent_router.py     # AgentRouter：ReAct / Coordinator / Clarify 路由
│   │   │
│   │   ├── planning/               # Agent 规划层
│   │   │   ├── __init__.py
│   │   │   ├── plan_schema.py      # PlanArtifact, PlanStep, PlanStepAction 数据类
│   │   │   ├── planner.py          # build_turn_plan()
│   │   │   ├── agent_graph.py      # LangGraph Agent 图结构
│   │   │   ├── langgraph_agent.py  # LangGraphAgent（基于 create_react_agent）
│   │   │   ├── nodes.py             # LangGraph 节点定义
│   │   │   ├── state.py             # Agent 状态管理
│   │   │   ├── coordinator.py       # Coordinator（多 Agent 协调）
│   │   │   ├── synthesizer.py       # 多 Agent 结果汇总
│   │   │   ├── worker_result.py    # WorkerResult 数据结构
│   │   │   └── worker.py           # Worker 执行器
│   │   │
│   │   ├── multi_agent/             # 多 Agent 协作（Coordinator 体系）
│   │   │   ├── __init__.py
│   │   │   ├── coordinator.py
│   │   │   ├── worker.py
│   │   │   ├── synthesizer.py
│   │   │   ├── task_notification.py
│   │   │   └── worker_result.py
│   │   │
│   │   └── observability/           # 可观测性体系
│   │       ├── __init__.py
│   │       ├── trace_record.py
│   │       ├── trace_store.py
│   │       ├── trace_logger.py
│   │       ├── dashboard.py
│   │       ├── analyzer.py
│   │       └── judge.py
│   │
│   ├── llm/                        # LLM 接口层
│   │   ├── providers.py             # 模型提供者工厂：build_model_provider()
│   │   └── embeddings.py            # Embedding 提供者：build_embedding_provider()
│   │
│   └── tools/                       # 工具注册
│       ├── __init__.py
│       ├── registry.py              # 工具注册表：default_tools()
│       ├── builders.py              # 工具构建器
│       ├── context.py               # 工具上下文 ToolUseContext
│       ├── mcp.py                   # MCP 工具集成
│       ├── rag_tool.py              # RAG 检索工具（search_knowledge_base）
│       └── schemas.py               # 工具参数 schema
│
└── data/                            # 数据目录
    ├── eval/eval_records/           # 评估结果记录（JSONL）
    └── (retrieval_index/, sessions/, skills/, workspace/, etc.)
```

---

## 模块依赖关系（调用链）

```
用户请求
    │
    ▼
AgentService.chat()
    │
    ├─► AgentRouter.decide()         # 路由决策：ReAct / Coordinator / Clarify
    │
    ├─► Route.ReAct:
    │       LangGraphAgent.run()
    │           └─► build_agent_graph()  # create_react_agent
    │               └─► tools (search_knowledge_base / memory / skills / ...)
    │
    └─► Route.Coordinator:
            Coordinator.run()
                └─► Planner.build_plan()
                    └─► Worker.execute() + Synthesizer.synthesize()

RAG 管线（search_knowledge_base 工具内部）:
    rewrite_for_rag() → retrieval.search() → rerank.rerank() → GroundedGenerator.generate()
```

---

## 关键文件说明

| 文件 | 功能 |
|------|------|
| `application/agent_service.py` | 主入口，编排 RAG 与 Agent 流程 |
| `application/rag_agent_service.py` | RAG 编排：检索→重排→生成 |
| `rag/retrieval.py` | 混合检索：lexical + tfidf + embedding + numeric |
| `rag/rerank.py` | 重排：Simple / BGE / HuggingFace / CascadeReranker |
| `rag/ingestion.py` | 文档分块 |
| `rag/generation.py` | GroundedGenerator：基于证据生成 + 拒答 |
| `rag/evidence_format.py` | 证据块格式化、引用提取、锚点覆盖率 |
| `core/dialogue/query_rewrite.py` | 查询改写：rule/llm/hybrid/hyde/expand 模式 |
| `core/dialogue/intent_classifier.py` | 意图分类 |
| `core/planning/langgraph_agent.py` | LangGraphAgent（基于 create_react_agent）|
| `core/planning/coordinator.py` | 多 Agent 协调 |
| `core/router/agent_router.py` | ReAct / Coordinator / Clarify 路由 |
| `providers.py` | 模型提供者工厂（MiniMax / OpenAI / Ollama）|
| `embeddings.py` | Embedding 提供者（qwen3-embedding）|
| `tools/registry.py` | 工具注册表 |

---

## 技术栈

- **Embedding**: `qwen3-embedding` (4096维, ollama)
- **检索**: 混合检索 lexical + tfidf + embedding + numeric（RRF / weighted_sum 可配）
- **重排**: HuggingFaceReranker / BGE Reranker / SimpleReranker / CascadeReranker
- **生成**: GroundedGenerator（基于证据生成 + 拒答）
- **Agent**: LangGraphAgent（基于 LangGraph `create_react_agent` + MiniMax M2.7 native tool-calling）
- **LLM**: anthropic_compatible (MiniMax-M2.7 via api.minimaxi.com)
- **Embedding Provider**: openai_compatible / ollama（支持 fallback）
