# 路由测试指南

> 本文档供其他 agent 使用，用于理解、执行和扩展路由测试。

## 快速开始

```bash
# 运行全部路由测试
cd E:\CursorProject\myagent
python -m pytest tests/unit/test_routing_intent_classifier.py -v

# 运行特定测试类
python -m pytest tests/unit/test_routing_intent_classifier.py::TestAgentRouterReAct -v
python -m pytest tests/unit/test_routing_intent_classifier.py::TestRoutingQuestionDataset -v

# 查看覆盖率
python -m pytest tests/unit/test_routing_intent_classifier.py --cov=src/agent/core/dialogue --cov=src/agent/core/router --cov=src/agent/core/multi_agent --cov=src/agent/core/observability
```

## 测试文件结构

```
tests/unit/test_routing_intent_classifier.py   # 101 tests
tests/unit/test_agent_router.py                 # 17 tests (existing)
```

## 测试覆盖的模块

| 模块 | 测试类 | 数量 |
|------|--------|------|
| `intent_classifier.py` (Stage 1-4) | `TestIntentClassifier*` | ~40 |
| `agent_router.py` | `TestAgentRouter*` | ~20 |
| `multi_agent/` (coordinator/worker/synthesizer) | `Test*Execution` / `TestSynthesizer` | ~25 |
| 端到端问题数据集 | `TestRoutingQuestionDataset` | 24 |
| 链路有效性 | `TestRoutingChainEffectiveness` | 4 |

## 路由链路说明

### 路由决策流程

```
Query → classify_intent() → AgentRouter.decide() → Route(ReAct/Coordinator/Clarify)
                                         ↓
                            Coordinator.run() → Worker.execute() → Synthesizer
```

### 三种路由目标

| Route | 触发条件 | 预期 steps |
|-------|----------|-----------|
| **ReAct** | TOOL_ONLY 高置信度，或简单 KNOWLEDGE 单实体查询 | 1 |
| **Coordinator** | MIXED tier，或多实体/低置信度/复杂语义 | ≥2 |
| **Clarify** | AMBIGUOUS tier，或短 query (<4字符) 无历史 | 0 |

### IntentTier → Route 映射

```
TOOL_ONLY  → ReAct (高置信度) 或 Clarify
KNOWLEDGE  → ReAct (简单) 或 Coordinator (复杂)
MIXED      → Coordinator
CHITCHAT   → Clarify
AMBIGUOUS  → Clarify
OOS        → Coordinator (graceful handling)
```

## 问题数据集 (24 queries)

位于 `TestRoutingQuestionDataset.QUESTION_DATASET`，每个 entry 为：

```python
(query, expected_tier, expected_route, description)
```

### 数据集覆盖矩阵

| Tier | Route | 示例 queries |
|------|-------|-------------|
| TOOL_ONLY | ReAct | "现在几点？", "计算 123*456" |
| KNOWLEDGE | ReAct | "茅台是哪家公司？" |
| KNOWLEDGE | Coordinator | "茅台和五粮液的营收对比" |
| MIXED | Coordinator | "查营收并计算同比增长率" |
| CHITCHAT | Clarify | "你好!" |
| AMBIGUOUS | Clarify | "那个", "继续" |
| OOS | ReAct | "如何制作炸弹" |

### 运行数据集测试

```bash
python -m pytest tests/unit/test_routing_intent_classifier.py::TestRoutingQuestionDataset -v
```

### 添加新测试 query

在 `QUESTION_DATASET` 列表中添加 tuple：

```python
("我的新查询", IntentTier.MIXED, Route.Coordinator, "描述"),
```

然后运行验证：

```bash
python -m pytest tests/unit/test_routing_intent_classifier.py::TestRoutingQuestionDataset -v
# 如果失败，查看推理是否合理
```

## 已知系统限制

> 以下行为是**设计决定**，不是 bug。测试中已标注。

### 1. RAG 互斥锁仅 per-worker

**当前行为**: `Coordinator._execute_rag_with_lock` 使用 `threading.Lock()`，每个 Worker 实例独立。
多个并行 RAG 任务会全部执行，不会合并。

**测试**: `test_coordinator_rag_mutex` 断言 `len(rag_calls) == 2`（反映当前行为）

**改进方向**: Coordinator 级别共享 `rag_lock`，或在任务分发层合并同质 RAG 任务

### 2. Stage 4 短 query 强制 AMBIGUOUS

**当前行为**: Stage 4 精细化调整中，`len(text) < 4` + 无历史 → 强制 AMBIGUOUS

这意味着：
- "你好" (2字符) → AMBIGUOUS（即使命中 CHITCHAT 规则）
- "你是谁" (3字符) → AMBIGUOUS（即使命中规则）
- "你好！" (3字符) → AMBIGUOUS

**不受影响的**: 4+ 字符的 greeting 如 "你好呀！" (4字符) → CHITCHAT ✓

**改进方向**: CHITCHAT 规则命中时，跳过 Stage 4 短 query 覆盖

### 3. `classify_intent()` 是同步 wrapper

`classify_intent(query, history)` 只执行 Stage 1 (规则) + Stage 4 (精细化)，不执行 Stage 2-3 (embedding/LLM)。

完整流水线使用 `classify_intent_async()`。

## 扩展测试指南

### 添加新的 IntentClassifier 规则测试

```python
class TestIntentClassifierNewFeature:
    def test_new_rule_hits_tool_only(self):
        result = classify_intent("新查询关键词", history=[])
        assert result.tier == IntentTier.TOOL_ONLY
        assert result.source == IntentSource.RULE
        assert result.confidence >= 0.80
```

### 添加新的 Router 决策测试

```python
class TestAgentRouterNewScenarios:
    def test_new_pattern_coordinator(self):
        router = AgentRouter()
        intent = _make_intent(IntentTier.MIXED, 0.80, sub=SubIntent.NEW_SUBINTENT)
        result = router.decide("新复杂查询", [], intent)
        assert result.route == Route.Coordinator
```

### 添加新的 MultiAgent 测试

```python
class TestCoordinatorNewBehavior:
    def test_new_task_type(self):
        mock_tools = {"my_tool": MagicMock(return_value="ok")}
        coord = Coordinator(MockConfig(), MockModelProvider(), mock_tools)
        result = coord.run([{"task_id": "t1", "task_type": "my_tool", "input": "x"}], "query")
        assert result.worker_results["t1"].success
```

## 调试失败的测试

```bash
# 查看详细输出
python -m pytest tests/unit/test_routing_intent_classifier.py -v --tb=short

# 运行单个测试
python -m pytest "tests/unit/test_routing_intent_classifier.py::TestAgentRouterReAct::test_simple_query_to_react[现在几点？]" -v

# 查看 print 输出
python -m pytest tests/unit/test_routing_intent_classifier.py -v -s
```

## 回归测试

```bash
# 完整回归（包含现有 router 测试）
python -m pytest tests/unit/test_routing_intent_classifier.py tests/unit/test_agent_router.py -v

# 快速检查（只跑 router 决策）
python -m pytest tests/unit/test_agent_router.py -v
```

## 依赖

测试使用 `unittest.mock.MagicMock`，不需要外部服务。

Coordinator/Synthesizer 测试使用 mock tool 函数，不依赖真实的 LLM provider 或 RAG service。
