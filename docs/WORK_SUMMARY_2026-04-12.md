# 工作总结 2026-04-12

**项目:** Fin-agent Routing Optimization
**日期:** 2026-04-12

---

## 一、PRD 制定

### 1.1 背景问题

#### 问题一：脆弱的规则路由
当前路由逻辑（`router.py`、`agent_router.py`、`intent_classifier.py`）都是简单的 if-else 规则：
- 无法理解模糊/边界 query
- 无法自适应调整
- 无法解释决策逻辑

#### 问题二：Coordinator/Planner/Synthesizer 职责不清
- Planner: 任务分解
- Synthesizer: 结果汇总
- Coordinator: 调度执行

用户期望：规划+总结应由同一 Agent 完成，Coordinator 只负责协调

### 1.2 目标
1. 用 LLM/Agent 替代 if-else 路由
2. 重构多 Agent 协调模式：Coordinator（协调） + Worker（执行） + Synthesizer（总结）
3. 完整的可观测性体系

### 1.3 新架构

```
Query → [LLM Router Agent]
              ↓
         分析任务类型（single_step/multi_step/clarify）
              ↓
         选择工具集 + 决定是否需要 RAG
              ↓
         如果需要 RAG: Query Rewrite → Retrieval → Rerank
              ↓
[Coordinator] → [Worker A / Worker B] → [Synthesizer]
     (协调)      (执行)                 (总结)
```

---

## 二、PRD 实施

### 2.1 团队分工

| 任务 | 负责人 | 产出 |
|------|--------|------|
| LLM Router Agent + RAG Chain | routing-expert | `src/agent/core/routing/` |
| 多Agent协调模块重构 | multi-agent-expert | `src/agent/core/multi_agent/` |
| 可观测性体系 | observability-expert / routing-expert | `src/agent/core/observability/` |

### 2.2 完成内容

#### Task #1: LLM Router Agent + RAG Chain
**产出:** `src/agent/core/routing/`

| 文件 | 功能 |
|------|------|
| `route_decision.py` | RouteDecision + RouteType + RAGChainConfig |
| `llm_router.py` | LLM判断任务类型 + 工具集选择 |
| `rag_chain.py` | 可观测RAG链路（rewrite→retrieval→rerank，各环节延迟追踪）|

#### Task #2: 多Agent协调模块重构
**产出:** `src/agent/core/multi_agent/`

| 文件 | 功能 |
|------|------|
| `coordinator.py` | 精简版Coordinator（只协调，无Planner职责）|
| `worker.py` | Worker执行器 + TaskNotification通信 |
| `task_notification.py` | 通信协议（COMPLETED/ERROR/STATUS_UPDATE）|
| `synthesizer.py` | 汇总Worker结果生成最终回复 |
| `worker_result.py` | WorkerResult数据结构 |

#### Task #3: 可观测性体系
**产出:** `src/agent/core/observability/`

| 文件 | 功能 |
|------|------|
| `trace_record.py` | TraceRecord + WorkerTrace 数据结构 |
| `trace_store.py` | In-memory + JSONL持久化 |
| `dashboard.py` | 4个API端点 |
| `analyzer.py` | 路由质量分析 |
| `judge.py` | LLM-as-judge答案质量评估 |

**数据路径:** `data/routing_observability/traces/`

---

## 三、文档更新

### 3.1 新增文档
- `docs/PRD_ROUTING_OBSERVABILITY_2026-04-12.md` - 完整PRD

### 3.2 更新文档
- `docs/FILE_TREE.md` - 更新项目文件结构

---

## 四、新模块结构

```
src/agent/core/
├── routing/                    # LLM Router + RAG Chain (新增)
│   ├── route_decision.py
│   ├── llm_router.py
│   └── rag_chain.py
├── multi_agent/               # 多Agent协调 (新增)
│   ├── coordinator.py
│   ├── worker.py
│   ├── synthesizer.py
│   ├── task_notification.py
│   └── worker_result.py
└── observability/            # 可观测性 (新增)
    ├── trace_record.py
    ├── trace_store.py
    ├── dashboard.py
    ├── analyzer.py
    └── judge.py
```

---

## 五、下一步建议

1. **集成测试** - 串联三个新模块端到端测试
2. **Dashboard前端** - 可视化界面（当前只有API）
3. **OpenTelemetry集成** - 导出到Jaeger/Zipkin
4. **现有代码适配** - 将现有 agent_service.py 等接入新架构

---

## 六、团队问题

团队清理遇到系统级问题（agents多次确认关闭但系统未识别）。不影响工作成果，可手动清理：
```bash
rm -rf ~/.claude/teams/prd-routing-impl/
rm -rf ~/.claude/tasks/prd-routing-impl/
```

---

## 七、下午工作：Bug修复与准确率提升

### 7.1 问题发现

LLM Router 集成到 `agent_service.py` 后，测试发现准确率仅 **26.1%**：

| 指标 | 数值 |
|------|------|
| 整体准确率 | 26.1% (5/23) |
| single_step | 62.5% (5/8) |
| multi_step | 0% (0/5) |
| clarify | 0% (0/10) |

### 7.2 根本原因

`RouteType` 枚举值大小写不匹配：

```python
# 枚举定义 (route_decision.py)
class RouteType(str, Enum):
    SINGLE_STEP = "single_step"   # 小写
    MULTI_STEP = "multi_step"     # 小写
    CLARIFY = "clarify"           # 小写

# LLM 返回的是大写
{"route_type": "CLARIFY", ...}   # ValueError!
```

`ValueError: 'CLARIFY' is not a valid RouteType` 被 `except` 捕获后 fallback 到 `SINGLE_STEP`。

### 7.3 修复方案

```python
# 修复后的 _parse_llm_response()
route_type_map = {
    "SINGLE_STEP": RouteType.SINGLE_STEP,
    "MULTI_STEP": RouteType.MULTI_STEP,
    "CLARIFY": RouteType.CLARIFY,
}
route_type = route_type_map.get(route_type_str, RouteType.SINGLE_STEP)
```

### 7.4 修复后测试结果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| **整体准确率** | 26.1% | **60.9%** |
| single_step | 62.5% | 62.5% |
| multi_step | 0% | **40.0%** |
| clarify | 0% | **70.0%** |

**平均延迟**: 6170ms (约6秒)

### 7.5 关键文件状态

| 文件 | 状态 |
|------|------|
| `src/agent/core/routing/llm_router.py` | ✅ 已修复 |
| `src/agent/core/routing/route_decision.py` | ✅ |
| `src/agent/core/routing/rag_chain.py` | ✅ 已集成真实组件 |
| `src/agent/core/observability/dashboard.py` | ✅ 3个新端点 |
| `src/agent/application/agent_service.py` | ✅ 已集成 LLM Router |
| `scripts/test_llm_router.py` | ✅ 23问题测试 |

### 7.6 待优化项

1. **准确率** (当前 60.9%)
   - multi_step 分类较弱 (40%)
   - 可通过更多 few-shot examples 改善

2. **API 错误**
   - 部分请求返回 "Unexpected API response format"
   - 可能是 MiniMax API 限流

3. **延迟**
   - 当前平均 6 秒
   - 可考虑缓存常见查询的路由结果

### 7.7 测试命令

```bash
# 运行 LLM Router 测试
python scripts/test_llm_router.py

# 运行单元测试
pytest tests/unit/test_routing_intent_classifier.py -v

# 查看测试报告
cat data/routing_observability/llm_router_report.json
```
