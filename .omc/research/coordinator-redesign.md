# Coordinator RAG 去重与 TaskNotification 调度驱动架构

> 规划文档 v1.0 | 日期: 2026-04-10 | 状态: 设计中

---

## 一、根因分析

### 1.1 RAG 重复调用的根因

**问题位置**: `coordinator.py` 第 421-488 行 `_execute_plan()` 方法

**现象**: 当 Planner 生成包含多个独立 RAG 步骤的计划时（如 `[RAG(s1), RAG(s2), CALC(s3)]` 且三者无依赖），`_execute_plan` 的第一轮迭代会将 `s1`、`s2` 同时放入 `ready` 列表，并通过 `ThreadPoolExecutor` **真正并行**执行。

```python
# coordinator.py 第 457-469 行
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {
        executor.submit(
            self.worker_executor.execute,
            step,
            results,
            rag_hits=shared_rag if step.action == PlanStepAction.RAG else None,  # shared_rag 初始为空!
        ): step
        for step in ready  # s1(RAG) 和 s2(RAG) 同时在 ready 中
    }
```

**根因链条**:
1. `shared_rag` 初始化为空 `list(self.shared_rag_hits)`（第 427 行）
2. 两个 RAG Worker 并行启动，**都**拿到空 `shared_rag`（因为另一个还没执行完）
3. 各自独立调用 `search_knowledge_base` 工具
4. 结果写入 `shared_rag` 时已有两个独立查询结果（第一个 RAG Worker 完成后写入，但第二个已经在跑了）

**共享机制失效**: `worker_executor.execute` 中的这段逻辑：
```python
# 第 216-226 行
if step.action == PlanStepAction.RAG and rag_hits is not None:
    output = self._format_rag_hits(rag_hits)
    # ... 使用共享 hits，跳过实际调用
    return result, None
```
由于 `rag_hits` 初始为空列表 `[]`（而非 `None`），条件不成立，**永远走到实际调用分支**。

---

### 1.2 TaskNotification 降级为日志的根因

**问题位置**: `coordinator.py` 第 362-378 行 `_handle_notification()` + 第 357-360 行 `WorkerExecutor` 初始化

```python
# 第 362-363 行
def _handle_notification(self, notification: TaskNotification) -> None:
    """处理 Worker 发来的 TaskNotification（当前未注册为回调，仅供调试）"""
```

```python
# 第 357-360 行
self.worker_executor = WorkerExecutor(
    tools,
    shared_rag_hits={"hits": self.shared_rag_hits},  # 注意：这是 dict，但 execute() 期望 list
)
```

**根因链条**:
1. `WorkerExecutor.__init__` 接收 `notify_callback` 参数（第 165 行），存储为 `self.notify_callback`
2. 但 `Coordinator.__init__` 调用时 **没有传递 `notify_callback`**：
   ```python
   self.worker_executor = WorkerExecutor(
       tools,
       shared_rag_hits={"hits": self.shared_rag_hits},  # 没有 notify_callback!
   )
   ```
3. 因此 `WorkerExecutor._send_notification()`（第 255-270 行）永远 `self.notify_callback is None`，通知从未发出
4. `_handle_notification` 方法体保留但从未被调用，只打印日志

**调度 vs 可观测性错位**: PRD 说"通知协议保留用于可观测性"，这意味着 TaskNotification 协议本身设计正确，但未被用于驱动调度。当前调度完全依赖静态 DAG 扫描（第 433-441 行），每次迭代重新计算 `ready` 列表，不依赖任何动态事件。

---

### 1.3 数据类型不匹配

**问题**: `shared_rag_hits` 在 `Coordinator.__init__` 中初始化为 `list[dict]`：
```python
self.shared_rag_hits: list[dict] = []
```

但传给 `WorkerExecutor` 时包装成了 `dict`：
```python
shared_rag_hits={"hits": self.shared_rag_hits},  # dict，不是 list
```

而 `WorkerExecutor.execute()` 第 176 行签名和第 216 行检查都是针对 `list[dict] | None`：
```python
def execute(self, step: PlanStep, context: dict[str, WorkerResult], rag_hits: list[dict] | None = None):
    if step.action == PlanStepAction.RAG and rag_hits is not None:  # rag_hits 是 dict 时永远非 None
```

---

## 二、架构设计

### 2.1 目标架构图

```
用户问题
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│                      AgentRouter                                  │
│  简单问题 ──→ LangGraph ReAct（单轮工具调用）                      │
│  复杂问题 ──→ Coordinator                                          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Planner                                      │
│  生成 Plan: [Step1, Step2, Step3]（DAG 结构）                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
         ┌───────────────────┼────────────────────┐
         │                   │                    │
         ▼                   ▼                    ▼
   ┌──────────┐       ┌──────────┐        ┌──────────┐
   │ RAG      │       │ CALC     │        │ WEB      │
   │ Worker   │       │ Worker   │        │ Worker   │
   │ (MUTEX)  │       │          │        │          │
   └────┬─────┘       └────┬─────┘        └────┬─────┘
        │                  │                   │
        └──────────────────┼───────────────────┘
                           │ TaskNotification
                           ▼
                    ┌──────────────┐
                    │ Coordinator  │ ← 调度驱动核心
                    │ (事件驱动)   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Synthesizer  │
                    └──────────────┘
```

### 2.2 RAG 互斥机制

**方案选择**: `threading.Lock`（优于状态标记，因为 Lock 语义更清晰且自动可重置）

```python
class Coordinator:
    def __init__(...):
        # ...existing code...
        self._rag_lock = threading.Lock()  # RAG 互斥锁
        self._rag_completed_hits: list[dict] = []  # 已完成的 RAG 结果

    def _execute_plan(self, plan: PlanArtifact) -> dict[str, WorkerResult]:
        # ...修改执行逻辑...
        with ThreadPoolExecutor(max_workers=4) as executor:
            for step in ready:
                if step.action == PlanStepAction.RAG:
                    # RAG 步骤必须获取锁
                    executor.submit(self._execute_rag_with_lock, step, results)
                else:
                    # 非 RAG 步骤正常并行
                    executor.submit(self.worker_executor.execute, step, results, None)
```

**替代简化方案**（无需 Lock）: 静态合并所有 RAG 步骤为单一 RAG 步骤

```python
# 在 _execute_plan 开始时，将多个 RAG 步骤合并为一个
rag_steps = [s for s in plan.steps if s.action == PlanStepAction.RAG]
if len(rag_steps) > 1:
    # 将多个 RAG detail 合并为一个查询
    merged_rag = PlanStep(
        id="rag_merged",
        action=PlanStepAction.RAG,
        detail="; ".join(s.detail for s in rag_steps),
        depends_on=[],
    )
    # 替换原多个 RAG 步骤
```

**推荐方案**: Lock + 首次查询共享结果（见 2.3）

---

### 2.3 TaskNotification 恢复为调度驱动

**目标**: Worker 完成时通知 Coordinator，Coordinator 动态决定下一步

**当前流程**（静态迭代）:
```
for iteration in range(max_iterations):
    ready = compute_ready_steps(pending, results)  # 静态计算
    parallel_execute(ready)                        # 不等待通知
```

**改造后流程**（事件驱动）:
```
1. 初始扫描 ready 步骤，启动所有无依赖的 Workers
2. 每个 Worker 完成 → 发送 TaskNotification
3. Coordinator 收到通知 → 重新扫描 pending → 启动新 ready Workers
4. 直到所有步骤完成
```

**实现方案**: 使用 `asyncio` + `ConcurrentFutures` 的回调机制

```python
def _execute_plan(self, plan: PlanArtifact) -> dict[str, WorkerResult]:
    results: dict[str, WorkerResult] = {}
    pending = {s.id: s for s in plan.steps}
    completed_events: dict[str, asyncio.Event] = {}

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def on_worker_done(step_id: str, future: concurrent.futures.Future):
        """Worker 完成时的回调"""
        result = future.result()
        results[step_id] = result

        # 发送 TaskNotification（驱动调度）
        self._handle_notification(TaskNotification(
            sender=step_id,
            type=NotificationType.COMPLETED,
            payload=result,
            dependencies_met=[step_id],
        ))

        # 设置事件，触发下一轮调度
        if step_id in completed_events:
            completed_events[step_id].set()

    # 启动初始 ready Workers
    initial_ready = self._get_ready_steps(pending, results)
    for step in initial_ready:
        self._submit_worker(step, results, loop, on_worker_done, completed_events)

    # 事件循环直到所有完成
    loop.run_until_complete(self._wait_all(completed_events.values()))
    return results
```

---

### 2.4 Coordinator Worker 优先级模型

**PRD 要求**: RAG Worker 优先执行，其他 Worker（CALC/WEB）从属，最后 Synthesizer 汇总

**执行顺序语义**:
1. **RAG 优先**: 所有无依赖的 RAG 步骤最先启动（使用 Lock 保证只有一个真正执行）
2. **CALC/WEB 从属**: 等 RAG 完成后（获得共享 rag_hits）才执行
3. **Synthesizer 最后**: 所有步骤完成后执行

**优先级实现**:

```python
ACTION_PRIORITY = {
    PlanStepAction.RAG: 0,       # 最高优先级
    PlanStepAction.CALC: 1,
    PlanStepAction.WEB: 1,
    PlanStepAction.MEMORY: 1,
    PlanStepAction.SYNTHESIZE: 2,  # 最后执行
}

def _get_ready_steps(self, pending: dict[str, PlanStep], results: dict[str, WorkerResult]) -> list[PlanStep]:
    """返回所有依赖已满足且未执行的步骤，按优先级排序"""
    ready = []
    for step_id, step in pending.items():
        if all(dep in results and results[dep].is_success() for dep in step.depends_on):
            ready.append(step)
    # RAG 优先排序
    return sorted(ready, key=lambda s: ACTION_PRIORITY.get(s.action, 99))
```

---

## 三、代码改动方案

### 3.1 coordinator.py 改动

#### 改动 1: 添加 RAG Lock 和共享状态

**位置**: `Coordinator.__init__`（第 344-360 行）

```python
import threading
from concurrent.futures import Future

class Coordinator:
    def __init__(...):
        # ...existing code...
        self._rag_lock = threading.Lock()  # 新增: RAG 互斥锁
        self._rag_completed_hits: list[dict] = []  # 新增: 已完成的 RAG 结果
        self._rag_executed: bool = False  # 新增: RAG 是否已执行标志
```

#### 改动 2: 添加 RAG Lock 执行方法

**位置**: 在 `Coordinator` 类中添加新方法

```python
def _execute_rag_with_lock(
    self,
    step: PlanStep,
    context: dict[str, WorkerResult],
) -> tuple[WorkerResult, list[dict] | None]:
    """使用 RAG Lock 执行 RAG 步骤，保证只有一个 RAG 调用"""
    with self._rag_lock:
        # 如果已经有 RAG 结果，直接使用（跳过重复调用）
        if self._rag_executed and self._rag_completed_hits:
            output = self.worker_executor._format_rag_hits(self._rag_completed_hits)
            result = WorkerResult(
                step_id=step.id,
                action=step.action,
                status="success",
                output=output,
                metadata={"rag_hits_shared": True, "chunk_count": len(self._rag_completed_hits)},
            )
            return result, None

        # 执行 RAG 调用
        result, new_rag_hits = self.worker_executor.execute(step, context, rag_hits=None)

        # 共享 RAG 结果
        if new_rag_hits:
            self._rag_completed_hits = new_rag_hits
            self._rag_executed = True

        return result, new_rag_hits
```

#### 改动 3: 修改 `_execute_plan` 方法

**位置**: `Coordinator._execute_plan`（第 421-488 行）

**主要改动**:
1. 将 `shared_rag` 从参数传递改为类属性共享
2. RAG 步骤使用 `_execute_rag_with_lock`
3. 非 RAG 步骤并行执行，使用 `rag_hits=self._rag_completed_hits if self._rag_executed else None`

```python
def _execute_plan(self, plan: PlanArtifact) -> dict[str, WorkerResult]:
    """执行计划：处理依赖关系，事件驱动调度"""
    results: dict[str, WorkerResult] = {}
    pending = {s.id: s for s in plan.steps}
    max_iterations = len(plan.steps) * 2

    for _ in range(max_iterations):
        if not pending:
            break

        # 找出所有依赖已满足的步骤（按优先级排序）
        ready = self._get_ready_steps(pending, results)

        if not ready:
            # 无法继续执行，标记剩余为失败
            for step_id, step in pending.items():
                results[step_id] = WorkerResult(
                    step_id=step_id,
                    action=step.action,
                    status="failed",
                    output="",
                    error="依赖步骤执行失败或循环依赖",
                )
            pending.clear()
            break

        # 按优先级分组执行
        from concurrent.futures import ThreadPoolExecutor, as_completed

        rag_steps = [s for s in ready if s.action == PlanStepAction.RAG]
        other_steps = [s for s in ready if s.action != PlanStepAction.RAG]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}

            # RAG 步骤：使用 Lock 保证互斥
            for step in rag_steps:
                futures[executor.submit(self._execute_rag_with_lock, step, results)] = step

            # 非 RAG 步骤：并行执行，传入已完成的 RAG 结果
            for step in other_steps:
                hits = self._rag_completed_hits if self._rag_executed else None
                futures[executor.submit(self.worker_executor.execute, step, results, hits)] = step

            for future in as_completed(futures):
                step = futures[future]
                try:
                    result, new_rag_hits = future.result()
                    if new_rag_hits:
                        self._rag_completed_hits.extend(new_rag_hits)
                        self._rag_executed = True
                except Exception as exc:
                    result = WorkerResult(
                        step_id=step.id,
                        action=step.action,
                        status="failed",
                        output="",
                        error=str(exc),
                    )
                results[step.id] = result
                del pending[step.id]

    return results
```

#### 改动 4: 添加 `_get_ready_steps` 辅助方法

```python
ACTION_PRIORITY = {
    PlanStepAction.RAG: 0,
    PlanStepAction.CALC: 1,
    PlanStepAction.WEB: 1,
    PlanStepAction.MEMORY: 1,
    PlanStepAction.SYNTHESIZE: 2,
}

def _get_ready_steps(self, pending: dict[str, PlanStep], results: dict[str, WorkerResult]) -> list[PlanStep]:
    """返回所有依赖已满足的步骤，按优先级排序"""
    ready = []
    for step_id, step in pending.items():
        deps_met = all(
            dep in results and results[dep].is_success()
            for dep in step.depends_on
        )
        if deps_met:
            ready.append(step)
    return sorted(ready, key=lambda s: ACTION_PRIORITY.get(s.action, 99))
```

#### 改动 5: 连接 TaskNotification 回调（可选，兼容可观测性）

**位置**: `Coordinator.__init__`

```python
def __init__(...):
    # ...existing code...
    # 连接通知回调，用于日志和可观测性
    self.worker_executor = WorkerExecutor(
        tools,
        notify_callback=self._handle_notification,  # 新增: 连接回调
    )
```

**注意**: `_handle_notification` 保持方法体用于日志输出，不改变调度逻辑。

---

### 3.2 WorkerExecutor 改动

#### 改动: 修复 `shared_rag_hits` 类型处理

**位置**: `WorkerExecutor.__init__`（第 162-170 行）和 `execute`（第 172-253 行）

**问题**: 当前 `shared_rag_hits` 被当作 `dict` 传递，但 `execute` 期望 `list[dict] | None`

**修复**:

```python
def __init__(
    self,
    tools: dict[str, Tool],
    notify_callback: Callable[[TaskNotification], None] | None = None,
    shared_rag_hits: list[dict] | None = None,  # 修复: 改为 list 类型
) -> None:
    self.tools = tools
    self.notify_callback = notify_callback
    self.shared_rag_hits = shared_rag_hits or []  # 默认空列表
```

---

### 3.3 task_notification.py 改动

**无改动**: `TaskNotification` 协议设计正确，保持不变。

---

## 四、风险评估

### 4.1 风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| RAG Lock 导致串行化降低性能 | 低 | 中 | Lock 仅保护 RAG 调用本身，CALC/WEB 仍并行 |
| 引入 asyncio 增加复杂度 | 中 | 中 | 保持 `ThreadPoolExecutor` 方案，仅在回调中使用 asyncio |
| RAG 结果共享逻辑复杂化 | 高 | 中 | 通过 Lock 保证原子性，简化共享判断逻辑 |
| TaskNotification 回调链断裂 | 中 | 高 | 添加单元测试验证回调被调用 |

### 4.2 向后兼容性

- **API 兼容**: `Coordinator.run()` 方法签名不变
- **工具兼容**: `search_knowledge_base` 工具调用方式不变
- **协议兼容**: TaskNotification 数据结构不变

### 4.3 回归风险

**需要验证的场景**:
1. 单 RAG 步骤计划：执行1次 RAG 调用
2. 多 RAG 步骤独立计划：执行1次 RAG 调用（而非多次）
3. RAG + CALC 依赖计划：RAG 先执行，CALC 使用 RAG 结果
4. 无 RAG 计划：CALC/WEB 正常执行

---

## 五、PRD 一致性检查

| PRD 要求 | 当前状态 | 设计改动后 |
|---------|---------|-----------|
| RAG 是工具不是独立服务 | 部分符合（RAG 已工具化） | 符合 |
| RAG 同一 query 只调用1次 | **未实现（并行导致重复）** | 实现（Lock 保证） |
| TaskNotification 协议保留用于可观测性 | 符合（协议完整但未驱动调度） | 符合（协议完整，回调连接用于日志） |
| 静态 DAG 正确工作 | 符合 | 符合（DAG 扫描逻辑保留） |
| RAG Worker 优先执行 | 未明确实现 | 实现（优先级排序 + Lock） |

---

## 六、实施计划

### Phase 1: RAG 互斥（最小改动）
1. 在 `Coordinator` 添加 `threading.Lock`
2. 修改 `_execute_plan` 让 RAG 步骤获取锁后执行
3. RAG 结果写入 `_rag_completed_hits` 后释放锁

### Phase 2: 共享结果传递
1. 修复 `WorkerExecutor.__init__` 的 `shared_rag_hits` 类型
2. 非 RAG 步骤从 `_rag_completed_hits` 读取共享结果

### Phase 3: TaskNotification 回调连接（可选）
1. 在 `Coordinator.__init__` 连接 `notify_callback`
2. 添加可观测性日志

### Phase 4: 测试验证
1. 单元测试：验证 RAG Lock 互斥效果
2. 集成测试：验证多 Worker 场景无重复 RAG 调用
3. 回归测试：确保简单场景不退化
