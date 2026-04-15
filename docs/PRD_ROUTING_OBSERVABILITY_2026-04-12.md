# PRD：LLM-Driven Routing + Multi-Agent Coordination System

**日期:** 2026-04-12
**状态:** Draft
**项目:** Fin-agent Routing Optimization

---

## 1. 背景与目标

### 现状问题

#### 问题一：脆弱的规则路由
当前路由逻辑（`router.py`、`agent_router.py`、`intent_classifier.py`）都是简单的 if-else 规则：
- `router.py`: 硬编码 regex patterns 判断 NUMERIC/ENTITY/SEMANTIC/HYBRID
- `agent_router.py`: 硬编码规则判断 ReAct/Coordinator/Clarify
- `intent_classifier.py`: 规则 → Embedding → LLM 三级fallback，但各阶段独立、无联动

**后果:**
- 无法理解模糊/边界 query
- 无法自适应调整
- 无法解释决策逻辑

#### 问题二：Coordinator/Planner/Synthesizer 职责不清
当前设计：
- `Planner`: 任务分解
- `Synthesizer`: 结果汇总
- `Coordinator`: 调度执行

**用户期望：**
- 规划 + 总结应由同一个 Agent 完成
- Coordinator 只负责协调（决定分几个 worker、分配任务、收集结果）
- Worker 通过 TaskNotification 与 Coordinator 通信
- Worker 完成后由 Synthesizer 做最终总结

### 目标
1. 用 LLM/Agent 替代 if-else 路由，实现意图理解 + 工具选择 + RAG 链路的智能化
2. 重构多 Agent 协调模式，实现严格的分层：Coordinator（协调） + Worker（执行） + Synthesizer（总结）
3. 完整的可观测性体系：每个环节的触发原因、延迟、作用全程可追踪

---

## 2. 核心架构

### 2.1 路由决策（LLM-Driven）

#### 新路由流程
```
Query → [LLM Router Agent]
              ↓
         分析任务类型（单步/多步/澄清）
              ↓
         选择工具集（tool_selection）
              ↓
         如果需要 RAG:
              ↓
         [Query Rewrite] → [RAG Retrieval] → [Rerank]
              ↓
         执行工具 or 返回澄清
```

#### LLM Router Agent 的职责
```python
class LLMRouterAgent:
    """用 LLM/Agent 做路由决策，替代 if-else 规则"""

    def route(self, query: str, context: dict) -> RouteDecision:
        """
        1. 分析任务类型：
           - 单步任务（直接执行工具）
           - 多步任务（需要规划）
           - 澄清任务（信息不足）

        2. 选择工具集：
           - 需要哪些工具（rag/calc/web/memory）
           - 工具调用顺序
           - 是否需要 RAG

        3. 如果需要 RAG：
           - 先做 query_rewrite
           - 再执行 RAG 链路
           - 返回检索结果

        4. 返回结构化决策：
           - route_type: single_step / multi_step / clarify
           - tool_set: list[Tool]
           - rag_chain: {rewrite_mode, top_k, ...} | None
           - reasoning: 决策理由（用于可观测性）
        """
```

#### RAG 链路集成
当 Router 判断需要 RAG 时，自动触发完整 RAG 链路：
```python
class RAGChain:
    """完整的 RAG 链路（可观测）"""

    def execute(self, query: str, rewrite_mode: str = "hybrid") -> RAGResult:
        # Step 1: Query Rewrite（每个环节可观测延迟）
        rewrite_start = time.time()
        rewritten = query_rewrite(query, mode=rewrite_mode)
        rewrite_latency = time.time() - rewrite_start

        # Step 2: Retrieval
        retrieval_start = time.time()
        hits = retrieval(rewritten)
        retrieval_latency = time.time() - retrieval_start

        # Step 3: Rerank
        rerank_start = time.time()
        reranked = rerank(hits)
        rerank_latency = time.time() - rerank_start

        return RAGResult(
            hits=reranked,
            latency_breakdown={
                "rewrite_ms": rewrite_latency * 1000,
                "retrieval_ms": retrieval_latency * 1000,
                "rerank_ms": rerank_latency * 1000,
            }
        )
```

### 2.2 多 Agent 协调模式（严格分层）

#### 架构图
```
┌─────────────────────────────────────────────────────────────┐
│                     Coordinator                             │
│  - 负责任务分解（决定分几个 worker）                          │
│  - 分配任务给各 worker                                       │
│  - 收集 worker 的 TaskNotification                           │
│  - 决定何时结束（所有任务完成/失败）                          │
│  - 调用 Synthesizer 做最终总结                                │
└─────────────────────────────────────────────────────────────┘
         ↑ TaskNotification                    ↓ TaskNotification
         │                                     │
    ┌────┴────┐                          ┌────┴────┐
    │ WorkerA │                          │ WorkerB │
    │ (task1) │                          │ (task2) │
    └─────────┘                          └─────────┘

    所有 Worker 完成后
         ↓
┌─────────────────────────────────────────────────────────────┐
│                     Synthesizer                             │
│  - 汇总各 Worker 的执行结果                                   │
│  - 生成最终回复                                              │
│  - 返回 answer + latency_breakdown                           │
└─────────────────────────────────────────────────────────────┘
```

#### Coordinator 职责（精简版）
```python
class Coordinator:
    """Coordinator：只负责协调，不负责规划和总结"""

    def run(self, query: str, route_decision: RouteDecision) -> CoordinatorResult:
        # 1. 基于 route_decision 决定分几个 worker
        workers = self._allocate_workers(route_decision)

        # 2. 并行/串行调度 worker 执行
        worker_results = self._execute_workers(workers)

        # 3. 汇总结果
        final_answer = self.synthesizer.synthesize(worker_results)

        return CoordinatorResult(
            answer=final_answer,
            worker_results=worker_results,
            total_latency_ms=sum(w.latency_ms for w in worker_results),
        )
```

#### Worker 职责
```python
class Worker:
    """Worker：只负责执行单一任务，通过 TaskNotification 通信"""

    def execute(self, task: Task) -> WorkerResult:
        result = self._do_work(task)

        # 发送完成通知给 Coordinator
        self._notify(
            NotificationType.COMPLETED,
            payload=result,
            dependencies_met=[task.id],
        )

        return result
```

#### TaskNotification 通信协议
```python
@dataclass
class TaskNotification:
    sender: str           # Worker ID
    type: NotificationType  # COMPLETED / ERROR / STATUS_UPDATE
    payload: Any          # 执行结果
    dependencies_met: list[str]  # 哪些依赖已满足
```

### 2.3 完整可观测性体系

#### 链路追踪结构
```python
@dataclass
class TraceRecord:
    trace_id: str              # 全链路唯一ID

    # Query 信息
    query: str
    query_rewrite_result: str | None
    query_rewrite_latency_ms: float

    # 路由决策
    route_decision: RouteDecision
    router_reasoning: str      # LLM 的决策理由
    router_latency_ms: float

    # RAG 链路（如果执行了）
    rag_retrieval_latency_ms: float
    rag_rerank_latency_ms: float
    rag_hits_count: int

    # Worker 执行
    workers: list[WorkerTrace]
    total_execution_latency_ms: float

    # 最终结果
    final_answer: str
    answer_quality_score: float | None  # LLM-as-judge

@dataclass
class WorkerTrace:
    worker_id: str
    task_type: str             # rag / calc / web / memory
    task_detail: str
    started_at: str            # ISO timestamp
    completed_at: str
    latency_ms: float
    success: bool
    output_preview: str        # 前100字符
```

#### Dashboard API
```
GET /api/observability/trace/{trace_id}     # 单次追踪详情
GET /api/observability/routing/distribution # 路由分布
GET /api/observability/routing/accuracy_trend  # 准确率趋势
GET /api/observability/routing/error_analysis # 错误分析
GET /api/observability/latency/breakdown    # 延迟拆解
```

#### 延迟拆解可视化
```json
{
  "query": "贵州茅台2024年营业收入是多少？",
  "latency_breakdown": {
    "router_llm_ms": 120,
    "query_rewrite_ms": 85,
    "retrieval_ms": 230,
    "rerank_ms": 45,
    "worker_execution_ms": 320,
    "synthesizer_ms": 180,
    "total_ms": 980
  },
  "stages": [
    {"stage": "router", "ms": 120, "作用": "判断任务类型和工具集"},
    {"stage": "query_rewrite", "ms": 85, "作用": "HyDE展开提升召回"},
    {"stage": "retrieval", "ms": 230, "作用": "混合检索（lexical+embedding）"},
    {"stage": "rerank", "ms": 45, "作用": "CrossEncoder重排"},
    {"stage": "worker_calc", "ms": 320, "作用": "执行计算任务"},
    {"stage": "synthesizer", "ms": 180, "作用": "汇总生成最终答案"}
  ]
}
```

---

## 3. 功能需求

### 3.1 LLM Router Agent

#### 3.1.1 任务类型判断
- **single_step**: 简单工具调用（查时间、计算等）
- **multi_step**: 需要多个工具协同（检索+计算+总结）
- **clarify**: 信息不足，需要澄清

#### 3.1.2 工具集选择
- 根据任务类型决定使用哪些工具
- 决定工具调用顺序和依赖关系
- 如果包含 rag，自动触发 query_rewrite

#### 3.1.3 RAG 链路集成
- query_rewrite: rule / hyde / expand / hyde_expand（可配置）
- retrieval: 混合检索（lexical + tfidf + embedding + numeric）
- rerank: CrossEncoder 重排

### 3.2 多 Agent 协调

#### 3.2.1 Coordinator 重构
- 去除 Planner 职责（由 LLM Router Agent 替代）
- 只保留任务分配和进度协调
- 通过 TaskNotification 收集 Worker 状态

#### 3.2.2 Worker 管理
- 支持并行/串行执行
- 每个 Worker 有独立 trace
- Worker 之间不直接通信，都通过 Coordinator

#### 3.2.3 Synthesizer
- 接收所有 Worker 的执行结果
- 生成最终回复
- 支持答案质量评估（LLM-as-judge）

### 3.3 可观测性体系

#### 3.3.1 链路录制
- 完整记录每个 query 的路由决策和执行过程
- 延迟拆解到每个 stage
- 支持离线回填历史数据

#### 3.3.2 质量分析
- 按 query_type 统计路由错误率
- 定期分析输出错误案例
- LLM-as-judge 答案质量评估

#### 3.3.3 Dashboard
- 路由分布饼图
- 准确率趋势折线图
- 延迟拆解柱状图
- A/B 策略对比表

---

## 4. 技术架构

### 4.1 模块结构
```
src/agent/core/
├── routing/                      # LLM 路由（新增）
│   ├── __init__.py
│   ├── llm_router.py            # LLM Router Agent
│   ├── route_decision.py        # RouteDecision 数据结构
│   └── rag_chain.py             # RAG 链路（可观测）
│
├── multi_agent/                  # 多 Agent 协调（重构）
│   ├── __init__.py
│   ├── coordinator.py          # Coordinator（精简版）
│   ├── worker.py                # Worker 执行器
│   ├── synthesizer.py           # Synthesizer（总结）
│   ├── task_notification.py     # TaskNotification 通信协议
│   └── worker_result.py         # WorkerResult 数据结构
│
├── observability/               # 可观测性（新增）
│   ├── __init__.py
│   ├── trace_record.py         # TraceRecord 数据结构
│   ├── trace_store.py          # In-memory + JSONL 存储
│   ├── analyzer.py             # 质量分析
│   ├── judge.py                # LLM-as-judge 评估
│   └── dashboard.py            # Dashboard API
│
├── router.py                   # 保留（兼容）
├── agent_router.py             # 保留（兼容）
├── planner.py                  # 废弃 → 职责移至 LLM Router
└── synthesizer.py             # 移动到 multi_agent/
```

### 4.2 数据流

```
User Query
    ↓
[LLM Router Agent]
    ↓ route_decision
┌─────────────────────────────────────────────────┐
│ 如果需要 RAG:                                    │
│   [Query Rewrite] → [Retrieval] → [Rerank]    │
│   ↓ rag_hits                                    │
└─────────────────────────────────────────────────┘
    ↓
[Coordinator]
    ↓ 分配 worker
[Worker A] ← TaskNotification → [Worker B]
    ↓                              ↓
[Worker Result] ←←←←←←←←←←←←←←←←←┘
    ↓
[Synthesizer]
    ↓ final_answer
[Trace Record] → JSONL + Dashboard API
```

### 4.3 技术约束

1. **Tools Registry 兼容**: 保持现有 `tools/registry.py` 接口不变
2. **In-memory 优先**: 存储层先 in-memory，支持后续迁 Redis/RabbitMQ
3. **不破坏现有 HTTP API**: 只新增 dashboard 端点
4. **OpenTelemetry 集成**: Trace / Metrics / Span 导出

---

## 5. 验收标准

### 5.1 LLM Router Agent
- [ ] LLM 判断任务类型（single_step/multi_step/clarify）
- [ ] LLM 选择工具集
- [ ] RAG 链路自动触发 query_rewrite
- [ ] 路由决策可解释（reasoning 字段）

### 5.2 多 Agent 协调
- [ ] Coordinator 只负责协调（不含规划）
- [ ] Worker 通过 TaskNotification 通信
- [ ] Synthesizer 汇总所有 Worker 结果
- [ ] 支持并行/串行 Worker 执行

### 5.3 可观测性
- [ ] 完整链路追踪（每个 stage 的延迟和作用）
- [ ] JSONL 持久化到 `data/routing_observability/traces/`
- [ ] Dashboard API 可查询

### 5.4 Dashboard
- [ ] `/api/observability/trace/{trace_id}` 单次追踪详情
- [ ] `/api/observability/routing/distribution` 路由分布
- [ ] `/api/observability/latency/breakdown` 延迟拆解

---

## 6. 实施计划

### Phase 1: LLM Router Agent（2天）
- [ ] 定义 RouteDecision 数据结构
- [ ] 实现 LLM Router Agent（用 LLM 判断任务类型 + 工具选择）
- [ ] 实现 RAG Chain（query_rewrite → retrieval → rerank，可观测延迟）
- [ ] 埋点接入 observability

### Phase 2: 多 Agent 协调重构（2天）
- [ ] 重构 Coordinator（去除 Planner 职责）
- [ ] 实现 Worker + TaskNotification 通信
- [ ] 重构 Synthesizer（汇总 Worker 结果）
- [ ] 端到端联调

### Phase 3: 可观测性体系（1天）
- [ ] 实现 TraceRecord + TraceStore
- [ ] 实现 Dashboard API
- [ ] 延迟拆解可视化

### Phase 4: 质量分析与 A/B（1天）
- [ ] 实现 RouterQualityAnalyzer
- [ ] LLM-as-judge 答案评估
- [ ] A/B routing 支持

---

## 7. 关键文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/agent/core/routing/llm_router.py` | 新增 | LLM Router Agent |
| `src/agent/core/routing/route_decision.py` | 新增 | 路由决策数据结构 |
| `src/agent/core/routing/rag_chain.py` | 新增 | RAG 链路（可观测） |
| `src/agent/core/multi_agent/coordinator.py` | 重构 | Coordinator（精简版） |
| `src/agent/core/multi_agent/worker.py` | 新增 | Worker 执行器 |
| `src/agent/core/multi_agent/synthesizer.py` | 移动 | Synthesizer（来自 planning/） |
| `src/agent/core/multi_agent/task_notification.py` | 移动 | TaskNotification |
| `src/agent/core/multi_agent/worker_result.py` | 新增 | WorkerResult 数据结构 |
| `src/agent/core/observability/trace_record.py` | 新增 | TraceRecord 数据结构 |
| `src/agent/core/observability/trace_store.py` | 新增 | 存储层 |
| `src/agent/core/observability/dashboard.py` | 新增 | Dashboard API |
| `src/agent/core/planner.py` | 废弃 | 职责移至 LLM Router |
| `src/agent/core/router.py` | 保留 | 兼容层 |
| `src/agent/core/agent_router.py` | 保留 | 兼容层 |
| `data/routing_observability/traces/` | 新增 | 追踪数据目录 |

---

## 8. 附录：关键设计决策

### 8.1 为什么用 LLM 替代 if-else 路由？
- if-else 无法处理模糊/边界 query
- LLM 可以理解语义，做出可解释的决策
- 方便后续微调和优化（可以分析哪些 query 路由错误）

### 8.2 为什么 Coordinator/Planner/Synthesizer 要合体？
- 规划（决定怎么做）和总结（生成最终答案）是紧密相关的
- 由同一个 Agent 完成可以保持上下文一致性
- Coordinator 只负责"分配任务 + 收集结果 + 协调进度"

### 8.3 TaskNotification 的作用？
- Worker 不直接通信，通过 Coordinator 中转
- 便于记录每个 Worker 的执行状态和延迟
- 支持故障恢复（Coordinator 知道哪个 Worker 失败了）
