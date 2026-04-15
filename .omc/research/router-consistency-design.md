# build_turn_plan 与 AgentRouter 路由一致性修复方案

## 1. 当前问题根因分析

### 1.1 问题描述

`build_turn_plan` 生成的 plan 用于可观测性摘要（`meta.last_plan_summary`），但与 `AgentRouter.decide()` 实际路由完全脱节：

| 组件 | 路由逻辑 | 输出 |
|------|---------|------|
| `build_turn_plan` | 按旧逻辑生成（RAG → AGENT_LOOP） | `PlanArtifact` 含 `PlanStepAction.RAG` / `AGENT_LOOP` |
| `AgentRouter.decide()` | 决定 `Route.ReAct` / `Route.Coordinator` / `Route.Clarify` | `RouterDecision` 含 `route`, `confidence`, `reasoning` |

两者没有关联，导致 `meta` 记录的计划与实际执行路径不一致。

### 1.2 代码层面的具体问题

**agent_service.py:167-169:**
```python
plan = build_turn_plan(intent=intent, rag_will_run=rag_on)
meta.last_intent = intent.intent
meta.last_plan_summary = plan.summary()  # 来自 build_turn_plan，与实际路由无关
```

**实际执行路径 (agent_service.py:181-211):**
```python
if decision.route == Route.Coordinator:
    coord_result = self.coordinator.run(...)  # Coordinator 路由
elif decision.route == Route.ReAct:
    result = self.agent.run(...)  # ReAct 路由
# plan.summary() 描述的是 RAG→AGENT_LOOP，与实际不符
```

### 1.3 PlanStepAction 与 Route 的语义鸿沟

| `PlanStepAction` (build_turn_plan) | `Route` (AgentRouter) | 实际执行组件 |
|-----------------------------------|----------------------|-------------|
| `RAG` | - | `RagAgentService.answer()` (可选预处理) |
| `AGENT_LOOP` | `ReAct` | `LangGraphAgent.run()` |
| - | `Coordinator` | `Coordinator.run()` |
| `CLARIFY` | `Clarify` | 直接返回澄清话术 |

**问题**：`build_turn_plan` 只输出 `RAG` + `AGENT_LOOP` 两种 action，无法表达 `Coordinator` 路由，也无法描述 `Clarify` 情况。

### 1.4 根本原因

1. **`build_turn_plan` 设计为"静态计划生成器"**，不考虑实际路由决策
2. **`AgentRouter.decide()` 独立运行**，没有向 `build_turn_plan` 传递路由结果
3. **两者使用不同的决策依据**：intent + rag_will_run vs query + history + intent_result

---

## 2. 修复方案

### 2.1 核心思路

**让 `build_turn_plan` 接受 `RouterDecision` 作为输入，基于实际路由生成对应的 plan steps**。

这样 `meta.last_plan_summary` 就能准确反映实际执行路径。

### 2.2 详细设计

#### 方案 A：扩展 `build_turn_plan` 签名（推荐）

**修改 `build_turn_plan` 函数签名**：

```python
# planner.py
from ..router import Route  # 新增 import

def build_turn_plan(
    *,
    intent: IntentResult,
    rag_will_run: bool,
    router_decision: RouterDecision | None = None,  # 新增参数
) -> PlanArtifact:
```

**根据 `router_decision.route` 生成不同的 steps**：

```python
def build_turn_plan(
    *,
    intent: IntentResult,
    rag_will_run: bool,
    router_decision: RouterDecision | None = None,
) -> PlanArtifact:
    pid = f"p_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
    goal = intent.normalized_query or ""
    steps: list[PlanStep] = []

    # Clarify 路由
    if router_decision and router_decision.route == Route.Clarify:
        steps.append(
            PlanStep("s1", PlanStepAction.CLARIFY, router_decision.reasoning, PlanStepStatus.DONE)
        )
        return PlanArtifact(pid, goal, steps)

    # Coordinator 路由（多步规划）
    if router_decision and router_decision.route == Route.Coordinator:
        steps.append(
            PlanStep(
                "s1",
                PlanStepAction.COORDINATOR,
                f"多步规划（{router_decision.estimated_steps}步）",
                PlanStepStatus.PENDING,
            )
        )
        return PlanArtifact(pid, goal, steps)

    # ReAct 路由（单步工具调用）
    # 保留原有 RAG 前置逻辑（如果启用）
    if rag_will_run:
        steps.append(
            PlanStep(
                "s1",
                PlanStepAction.RAG,
                "文档检索 + 重排",
                PlanStepStatus.PENDING,
            )
        )
    sid = f"s{len(steps) + 1}" if steps else "s1"
    steps.append(
        PlanStep(
            sid,
            PlanStepAction.REACT,  # 新增 REACT action 替代 AGENT_LOOP
            "单步工具调用",
            PlanStepStatus.PENDING,
        )
    )
    return PlanArtifact(pid, goal, steps)
```

#### 方案 B：在 `agent_service.py` 中统一路由决策后的 plan 生成

**修改 `agent_service.py` 的 `chat` 方法**：

```python
# agent_service.py
from ..router import AgentRouter, Route
from ..planning import build_turn_plan

# 在 chat 方法中：
intent = classify_intent(turn_text, history)
router = AgentRouter()
decision = router.decide(turn_text, history, intent)

if decision.route == Route.Clarify:
    # ... Clarify 处理 ...
    meta.last_plan_summary = f"clarify:{decision.reasoning}"
elif decision.route == Route.Coordinator:
    # ... Coordinator 处理 ...
    meta.last_plan_summary = f"coordinator:{decision.estimated_steps}steps"
elif decision.route == Route.ReAct:
    plan = build_turn_plan(intent=intent, rag_will_run=rag_on, router_decision=decision)
    meta.last_plan_summary = plan.summary()
```

---

### 2.3 `PlanStepAction` 枚举扩展

**plan_schema.py** 需要新增 `REACT` 和 `COORDINATOR` action：

```python
class PlanStepAction(str, Enum):
    """任务步骤动作类型"""
    RAG = "rag"                  # 文档检索
    CALC = "calc"                # 算术计算
    WEB = "web"                  # 网络搜索
    MEMORY = "memory"            # 记忆操作
    SYNTHESIZE = "synthesize"    # 汇总生成
    AGENT_LOOP = "agent_loop"    # 保留：通用工具循环（兼容旧代码）
    CLARIFY = "clarify"          # 澄清问题
    REACT = "react"               # 单步 ReAct 工具调用（替代 agent_loop）
    COORDINATOR = "coordinator"   # 多步规划协调
```

---

## 3. 具体代码改动方案

### 3.1 文件：`src/agent/core/planning/plan_schema.py`

**改动**：为 `PlanStepAction` 枚举添加 `REACT` 和 `COORDINATOR` 两个值。

```python
class PlanStepAction(str, Enum):
    RAG = "rag"
    CALC = "calc"
    WEB = "web"
    MEMORY = "memory"
    SYNTHESIZE = "synthesize"
    AGENT_LOOP = "agent_loop"    # 保留，向后兼容
    CLARIFY = "clarify"
    REACT = "react"               # 新增
    COORDINATOR = "coordinator"   # 新增
```

### 3.2 文件：`src/agent/core/planning/planner.py`

**改动 1**：添加必要的 import。

```python
from ..router import Route  # 新增
from ..router.agent_router import RouterDecision  # 新增
```

**改动 2**：修改 `build_turn_plan` 函数签名和实现。

```python
def build_turn_plan(
    *,
    intent: IntentResult,
    rag_will_run: bool,
    router_decision: RouterDecision | None = None,
) -> PlanArtifact:
    """基于实际路由决策生成计划。

    Args:
        intent: 意图分类结果
        rag_will_run: 是否会运行 RAG
        router_decision: AgentRouter 的决策结果（可选，用于生成准确的 plan）
    """
    pid = f"p_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
    goal = intent.normalized_query or ""
    steps: list[PlanStep] = []

    # 无 router_decision 时退化为原有逻辑（向后兼容）
    if router_decision is None:
        if intent.intent == IntentKind.AMBIGUOUS:
            steps.append(
                PlanStep("s1", PlanStepAction.CLARIFY, "返回澄清问句", PlanStepStatus.DONE)
            )
            return PlanArtifact(pid, goal, steps)
        if rag_will_run:
            steps.append(
                PlanStep("s1", PlanStepAction.RAG, "检索+重排", PlanStepStatus.PENDING)
            )
        sid = f"s{len(steps) + 1}" if steps else "s1"
        steps.append(
            PlanStep(sid, PlanStepAction.AGENT_LOOP, "SimpleAgent 工具循环", PlanStepStatus.PENDING)
        )
        return PlanArtifact(pid, goal, steps)

    # 基于 router_decision.route 生成对应 plan
    route = router_decision.route

    if route == Route.Clarify:
        steps.append(
            PlanStep(
                "s1",
                PlanStepAction.CLARIFY,
                router_decision.reasoning,
                PlanStepStatus.DONE,
            )
        )
    elif route == Route.Coordinator:
        steps.append(
            PlanStep(
                "s1",
                PlanStepAction.COORDINATOR,
                f"多步规划（预估{router_decision.estimated_steps}步）",
                PlanStepStatus.PENDING,
            )
        )
    elif route == Route.ReAct:
        if rag_will_run:
            steps.append(
                PlanStep(
                    "s1",
                    PlanStepAction.RAG,
                    "文档检索 + 重排",
                    PlanStepStatus.PENDING,
                )
            )
        sid = f"s{len(steps) + 1}" if steps else "s1"
        steps.append(
            PlanStep(
                sid,
                PlanStepAction.REACT,
                router_decision.reasoning,
                PlanStepStatus.PENDING,
            )
        )
    else:
        # 兜底：不应该到达这里
        steps.append(
            PlanStep("s1", PlanStepAction.AGENT_LOOP, "工具循环", PlanStepStatus.PENDING)
        )

    return PlanArtifact(pid, goal, steps)
```

### 3.3 文件：`src/agent/application/agent_service.py`

**改动**：修改 `chat` 方法中 plan 的生成逻辑，传入 `router_decision`。

```python
# agent_service.py 第 167-169 行（修改前）：
plan = build_turn_plan(intent=intent, rag_will_run=rag_on)
meta.last_intent = intent.intent
meta.last_plan_summary = plan.summary()

# 修改后：
# 注意：Clarify 和 Coordinator 路由已经在各自分支中设置了 meta.last_plan_summary
# 这里只在 ReAct 路由时生成 plan
if decision.route == Route.ReAct:
    plan = build_turn_plan(intent=intent, rag_will_run=rag_on, router_decision=decision)
    meta.last_plan_summary = plan.summary()
else:
    # 对于 Clarify 和 Coordinator，plan 由各自分支设置
    # 这里确保 meta.last_plan_summary 被设置
    if decision.route == Route.Clarify:
        meta.last_plan_summary = f"clarify:{decision.reasoning}"
    elif decision.route == Route.Coordinator:
        meta.last_plan_summary = f"coordinator:{decision.estimated_steps}steps"
```

**更简洁的方案**：在所有分支统一使用 `build_turn_plan`：

```python
# 在 chat 方法中，所有路由分支之后：
plan = build_turn_plan(intent=intent, rag_will_run=rag_on, router_decision=decision)
meta.last_intent = intent.intent
meta.last_plan_summary = plan.summary()
self.session_meta_store.put(session_id, meta)
```

但需要注意：`meta.phase` 的设置时机可能需要调整。

---

## 4. 风险评估

### 4.1 向后兼容性风险

| 风险 | 级别 | 缓解措施 |
|------|------|---------|
| `PlanStepAction.AGENT_LOOP` 被新 action 替代 | 低 | 保留 `AGENT_LOOP`，标记为"保留" |
| `build_turn_plan` 签名变化 | 中 | 添加默认参数 `router_decision=None`，退化为原有逻辑 |
| 依赖 `meta.last_plan_summary` 的下游代码 | 低 | 先搜索确认无其他消费者 |

### 4.2 一致性风险

| 风险 | 级别 | 缓解措施 |
|------|------|---------|
| `build_turn_plan` 内部逻辑与 `AgentRouter.decide` 不一致 | 中 | `build_turn_plan` 直接接受 `RouterDecision`，不重新计算 |
| 某些边界情况 `router_decision` 为 None | 低 | 提供 `None` 时的向后兼容逻辑 |

### 4.3 测试风险

| 风险 | 级别 | 缓解措施 |
|------|------|---------|
| 现有测试依赖旧的 plan summary 格式 | 中 | 更新测试断言，使用新的 action 名称 |
| Coordinator 路由的 plan 生成未覆盖 | 低 | 添加针对 Coordinator 路由的单元测试 |

---

## 5. 实施步骤

1. **修改 `plan_schema.py`**：添加 `REACT` 和 `COORDINATOR` 到 `PlanStepAction` 枚举
2. **修改 `planner.py`**：扩展 `build_turn_plan` 函数，支持 `router_decision` 参数
3. **修改 `agent_service.py`**：将 `router.decision` 传入 `build_turn_plan`
4. **验证**：确保 `meta.last_plan_summary` 与实际路由一致

---

## 6. 验证方法

### 6.1 单元测试

```python
def test_build_turn_plan_with_react_route():
    decision = RouterDecision(route=Route.ReAct, confidence=0.95, reasoning="单步工具调用", estimated_steps=1)
    plan = build_turn_plan(intent=mock_intent, rag_will_run=True, router_decision=decision)
    assert "react" in plan.summary().lower() or "rag" in plan.summary().lower()

def test_build_turn_plan_with_coordinator_route():
    decision = RouterDecision(route=Route.Coordinator, confidence=0.90, reasoning="多步任务", estimated_steps=3)
    plan = build_turn_plan(intent=mock_intent, rag_will_run=False, router_decision=decision)
    assert "coordinator" in plan.summary().lower()

def test_build_turn_plan_with_clarify_route():
    decision = RouterDecision(route=Route.Clarify, confidence=0.95, reasoning="需要澄清", estimated_steps=0)
    plan = build_turn_plan(intent=mock_intent, rag_will_run=False, router_decision=decision)
    assert "clarify" in plan.summary().lower()
```

### 6.2 集成验证

在 `agent_service.chat()` 调用后检查 `meta.last_plan_summary` 与实际返回的 `route` 是否匹配。
