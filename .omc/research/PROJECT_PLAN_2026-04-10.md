# RAG 与 Agent 协作完整修复方案

> 项目计划 v1.0 | 日期: 2026-04-10 | 状态: 规划完成

---

## 1. Executive Summary

### 问题概述

当前 RAG Agent 协作系统存在 6 个关键问题，分为三类：

| 类别 | 问题 | 优先级 | 风险等级 |
|------|------|--------|---------|
| **Coordinator 执行层** | RAG 重复调用（并行执行导致 shared_rag 空值） | CRITICAL | 资源浪费 + 数据不一致 |
| **Coordinator 执行层** | TaskNotification 回调未连接 | HIGH | 可观测性缺失 |
| **工具层** | MCP event loop 嵌套崩溃 | CRITICAL | 服务无法启动 |
| **工具层** | web_search 无 rate limit | HIGH | 外部服务封禁风险 |
| **工具层** | _safe_eval_math 使用 eval() | MEDIUM | 安全漏洞 |
| **路由层** | build_turn_plan 与 AgentRouter 不一致 | HIGH | 日志与实际执行脱节 |

### 修复目标

1. 保证同一 query 只调用一次 RAG（而非并行多次）
2. 恢复 TaskNotification 回调连接，实现调度驱动可观测性
3. 修复 MCP 嵌套 event loop 崩溃问题
4. 为 web_search 添加并发控制与重试机制
5. 用纯 AST 解释器替代 eval()
6. 让 plan summary 与实际路由决策一致

---

## 2. 详细架构设计

### 2.1 系统架构图

```
用户问题
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│                      AgentRouter                                  │
│  简单问题 ──→ LangGraph ReAct（单轮工具调用）                      │
│  复杂问题 ──→ Coordinator（多步规划）                              │
│  歧义问题 ──→ Clarify（澄清）                                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Planner                                      │
│  build_turn_plan() ←── RouterDecision（新增输入）                 │
│  生成 Plan: [Step1, Step2, Step3]（DAG 结构）                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
         ┌───────────────────┼────────────────────┐
         │                   │                    │
         ▼                   ▼                    ▼
   ┌──────────┐       ┌──────────┐        ┌──────────┐
   │ RAG      │       │ CALC     │        │ WEB      │
   │ Worker   │       │ Worker   │        │ Worker   │
   │ (MUTEX)  │       │          │        │ (Semp)   │
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

### 2.2 核心修复点

#### 修复 1: RAG 互斥执行（coordinator-redesign.md Section 2.2）

```python
class Coordinator:
    def __init__(...):
        self._rag_lock = threading.Lock()  # RAG 互斥锁
        self._rag_completed_hits: list[dict] = []  # 已完成 RAG 结果
        self._rag_executed: bool = False  # RAG 是否已执行标志

    def _execute_rag_with_lock(self, step, context):
        with self._rag_lock:
            if self._rag_executed and self._rag_completed_hits:
                # 复用已有结果，跳过重复调用
                return cached_result
            # 执行 RAG 调用并共享结果
            result, new_hits = self.worker_executor.execute(step, context, rag_hits=None)
            return result, new_hits
```

#### 修复 2: TaskNotification 回调连接（coordinator-redesign.md Section 2.3）

```python
# Coordinator.__init__ 中添加
self.worker_executor = WorkerExecutor(
    tools,
    notify_callback=self._handle_notification,  # 连接回调
)
```

#### 修复 3: MCP EventLoop 修复（tools-fix-design.md Section 1）

```python
# __init__ 中使用 asyncio.run() 作为唯一入口
asyncio.run(self._connect_mcp_servers_async())

# make_sync_wrapper 改为纯线程方案
def make_sync_wrapper(ah):
    def wrapper(input_str: str) -> str:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: asyncio.run(ah(None, input=input_str)))
            return future.result(timeout=30)
    return wrapper
```

#### 修复 4: web_search 限流（tools-fix-design.md Section 2）

```python
_search_semaphore = Semaphore(3)  # 最多 3 并发

def _safe_web_search_with_retry(query, max_retries=2):
    acquired = _search_semaphore.acquire(timeout=15)
    if not acquired:
        return "搜索请求超时：系统繁忙，请稍后重试。"
    try:
        # 指数退避重试逻辑
        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                return process_results(resp)
            except requests.RequestException:
                if attempt < max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                return f"搜索请求失败: {exc}"
    finally:
        _search_semaphore.release()
```

#### 修复 5: _safe_eval_math 移除 eval（tools-fix-design.md Section 3）

```python
class _MathEvaluator(ast.NodeVisitor):
    """纯 AST 数学表达式求值器，无 eval() 调用。"""
    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type is ast.Add: return left + right
        if op_type is ast.Sub: return left - right
        if op_type is ast.Mult: return left * right
        if op_type is ast.Div: return left / right if right != 0 else raise...
        # ...
```

#### 修复 6: build_turn_plan 接受 RouterDecision（router-consistency-design.md Section 2.2）

```python
def build_turn_plan(
    *,
    intent: IntentResult,
    rag_will_run: bool,
    router_decision: RouterDecision | None = None,  # 新增参数
) -> PlanArtifact:
    if router_decision is None:
        # 退化为原有逻辑
        return legacy_build_turn_plan(...)

    if router_decision.route == Route.Clarify:
        return PlanArtifact(..., steps=[PlanStep(..., action=CLARIFY)])
    elif router_decision.route == Route.Coordinator:
        return PlanArtifact(..., steps=[PlanStep(..., action=COORDINATOR)])
    elif router_decision.route == Route.ReAct:
        return PlanArtifact(..., steps=[PlanStep(..., action=REACT)])
```

---

## 3. 修复优先级排序

### CRITICAL（必须立即修复）

| 优先级 | 问题 | 影响 |
|--------|------|------|
| P0 | MCP event loop 嵌套崩溃 | 服务无法启动 |
| P0 | RAG 重复调用 | 资源浪费 + 数据竞争 |

### HIGH（近期修复）

| 优先级 | 问题 | 影响 |
|--------|------|------|
| P1 | build_turn_plan 与路由不一致 | 日志与实际执行脱节 |
| P1 | web_search 无 rate limit | 外部服务封禁风险 |

### MEDIUM（计划内修复）

| 优先级 | 问题 | 影响 |
|--------|------|------|
| P2 | _safe_eval_math 使用 eval() | 安全漏洞（理论风险） |
| P2 | TaskNotification 回调未连接 | 可观测性缺失 |

---

## 4. 依赖关系分析

### 依赖图

```
┌─────────────────────────────────────────────────────────────┐
│                        Phase 1                              │
│  MCP event loop 修复 + tools 修复（registry.py）           │
│  - tools-fix-design.md 问题 1, 2, 3                         │
└─────────────────────────┬───────────────────────────────────┘
                          │ 无依赖
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                        Phase 2                              │
│  Coordinator RAG 去重 + TaskNotification 修复              │
│  - coordinator-redesign.md Section 2.2, 2.3                │
│  依赖: Phase 1 完成后可并行                                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                        Phase 3                              │
│  build_turn_plan 与 Router 一致性修复                       │
│  - router-consistency-design.md Section 2                  │
│  依赖: Phase 2 中的 router 模块改动                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                        Phase 4                              │
│  集成测试 + 回归测试                                         │
└─────────────────────────────────────────────────────────────┘
```

### 并行可能性

- **Phase 1 内部**: MCP 修复与 registry.py 修复可并行（不同文件）
- **Phase 2 内部**: Coordinator 修复与 TaskNotification 修复紧密耦合，串行执行
- **Phase 3**: 依赖 Phase 2 完成的 router 模块，不能提前开始

---

## 5. 分阶段实施计划

### Phase 1: MCP + Tools 修复

**目标**: 修复服务启动崩溃和 web_search 风险

**改动文件**:
- `src/agent/application/agent_service.py`
- `src/agent/tools/registry.py`

**任务分解**:

| 步骤 | 任务 | 文件 | 优先级 |
|------|------|------|--------|
| 1.1 | 重写 `_connect_mcp_servers_async()` 异步方法 | agent_service.py | P0 |
| 1.2 | 修改 `__init__` 使用 `asyncio.run()` | agent_service.py | P0 |
| 1.3 | 简化 `make_sync_wrapper` 为纯线程方案 | agent_service.py | P0 |
| 1.4 | 添加 `Semaphore(3)` 限流 | registry.py | P1 |
| 1.5 | 实现重试 + 指数退避逻辑 | registry.py | P1 |
| 1.6 | 实现纯 AST `_MathEvaluator` 替代 eval | registry.py | P2 |

### Phase 2: Coordinator RAG 去重 + TaskNotification

**目标**: 消除 RAG 重复调用，恢复可观测性

**改动文件**:
- `src/agent/core/planning/coordinator.py`
- `src/agent/core/planning/worker_executor.py`（如需修复类型）

**任务分解**:

| 步骤 | 任务 | 优先级 |
|------|------|--------|
| 2.1 | 添加 `threading.Lock()` 和 `_rag_completed_hits` | P0 |
| 2.2 | 实现 `_execute_rag_with_lock()` 方法 | P0 |
| 2.3 | 修改 `_execute_plan()` 使用 Lock + 共享结果 | P0 |
| 2.4 | 添加 `_get_ready_steps()` 优先级排序 | P1 |
| 2.5 | 连接 `notify_callback` 到 `_handle_notification` | P2 |

### Phase 3: build_turn_plan 与 Router 一致性

**目标**: 让 plan summary 与实际路由一致

**改动文件**:
- `src/agent/core/planning/plan_schema.py`
- `src/agent/core/planning/planner.py`
- `src/agent/application/agent_service.py`

**任务分解**:

| 步骤 | 任务 | 优先级 |
|------|------|--------|
| 3.1 | 扩展 `PlanStepAction` 枚举（添加 REACT, COORDINATOR） | P1 |
| 3.2 | 修改 `build_turn_plan()` 接受 `router_decision` 参数 | P1 |
| 3.3 | 修改 `agent_service.py` 传入 `router_decision` | P1 |
| 3.4 | 统一所有路由分支的 plan 生成逻辑 | P1 |

### Phase 4: 测试验证

**目标**: 确保修复有效，回归测试通过

**任务分解**:

| 步骤 | 任务 | 类型 |
|------|------|------|
| 4.1 | RAG 去重单元测试（验证单次调用） | 单元测试 |
| 4.2 | MCP 连接集成测试 | 集成测试 |
| 4.3 | web_search 限流测试 | 单元测试 |
| 4.4 | _safe_eval_math 安全测试 | 安全测试 |
| 4.5 | build_turn_plan 一致性测试 | 单元测试 |
| 4.6 | 完整对话流程回归测试 | E2E 测试 |

---

## 6. 风险评估

### Phase 1 风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| `asyncio.run()` 在已运行 loop 中调用 | 低 | 高 | 确保 `__init__` 是同步入口点 |
| MCP 服务器连接超时 | 中 | 低 | 异常已打印，不影响主服务启动 |
| Semaphore 阻塞初始化 | 低 | 中 | timeout=15s，超时则跳过 |

### Phase 2 风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| RAG Lock 导致串行化 | 低 | 中 | Lock 仅保护 RAG，CALC/WEB 仍并行 |
| TaskNotification 回调链断裂 | 中 | 中 | 添加单元测试验证回调被调用 |
| shared_rag_hits 类型不匹配 | 高 | 高 | Phase 2 中一并修复 WorkerExecutor 类型 |

### Phase 3 风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 现有测试依赖旧 plan summary 格式 | 中 | 中 | 更新测试断言 |
| 路由边界情况（router_decision=None） | 低 | 低 | 提供向后兼容逻辑 |

### 全局风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 多处改动引入回归 | 中 | 高 | Phase 4 完整回归测试 |
| 改动跨越多个模块 | 中 | 中 | 按 Phase 顺序执行，每 Phase 验证 |

---

## 7. 验收标准

### 7.1 RAG 去重验收

```
验收条件: 当 Planner 生成含多个独立 RAG 步骤的计划时，
实际只执行 1 次 RAG 调用，后续步骤复用结果。

验证方法:
1. 创建含 3 个 RAG 步骤的计划（如 [RAG(s1), RAG(s2), CALC(s3)]）
2. 注入计数器，验证 search_knowledge_base 只被调用 1 次
3. 验证 s2 使用 s1 的结果（非空）
```

### 7.2 MCP EventLoop 验收

```
验收条件: AgentService 可以在任何 event loop 状态下正常初始化。

验证方法:
1. 在已有 loop 环境中创建 AgentService 实例
2. 验证 __init__ 不抛出 RuntimeError
3. 验证 MCP 工具可以正常调用
```

### 7.3 web_search 限流验收

```
验收条件: 最多 3 个并发搜索请求，失败自动重试（最多 3 次）。

验证方法:
1. 并发发起 10 个搜索请求
2. 验证同时执行的请求不超过 3 个
3. 模拟失败场景，验证重试逻辑和指数退避
```

### 7.4 _safe_eval_math 验收

```
验收条件: 数学表达式求值不使用 eval()，危险表达式被拦截。

验证方法:
1. 正常表达式求值结果正确 (2+3)*4 = 20
2. 危险表达式 __import__('os') 被 ValueError 拦截
3. 代码中不包含 eval() 调用
```

### 7.5 build_turn_plan 一致性验收

```
验收条件: meta.last_plan_summary 与实际执行路径一致。

验证方法:
1. ReAct 路由: summary 包含 "react"
2. Coordinator 路由: summary 包含 "coordinator"
3. Clarify 路由: summary 包含 "clarify"
```

### 7.6 TaskNotification 验收

```
验收条件: Worker 完成时触发回调，回调参数包含正确信息。

验证方法:
1. 单元测试 mock notify_callback，验证被调用
2. 验证回调参数包含 sender, type, payload
```

---

## 8. 改动文件汇总

| 文件 | Phase | 改动数 | 优先级 |
|------|-------|--------|--------|
| `src/agent/application/agent_service.py` | 1, 3 | 3 处 | P0 |
| `src/agent/tools/registry.py` | 1 | 3 处 | P0 |
| `src/agent/core/planning/coordinator.py` | 2 | 5 处 | P0 |
| `src/agent/core/planning/worker_executor.py` | 2 | 1 处 | P1 |
| `src/agent/core/planning/plan_schema.py` | 3 | 1 处 | P1 |
| `src/agent/core/planning/planner.py` | 3 | 2 处 | P1 |

**总计**: 6 个文件，约 15 处代码改动

---

## 9. 实施时间线（预估）

| Phase | 内容 | 依赖 | 预估工时 |
|-------|------|------|---------|
| Phase 1 | MCP + Tools 修复 | 无 | 2-3 小时 |
| Phase 2 | Coordinator RAG 去重 | Phase 1 | 3-4 小时 |
| Phase 3 | Router 一致性 | Phase 2 | 1-2 小时 |
| Phase 4 | 测试验证 | Phase 1-3 | 2-3 小时 |
| **总计** | | | **8-12 小时** |

---

## 10. 回滚计划

若 Phase 2 或 Phase 3 引入严重回归：

1. **立即回滚**: `git checkout <phase_commit>` 恢复到 Phase 1 完成状态
2. **问题定位**: 通过 Phase 4 的单元测试精确定位失败点
3. **修复策略**: 将大改动拆分为更小的子步骤，逐个验证

---

## 附录：相关文档索引

- 根因分析: `coordinator-redesign.md`, `tools-fix-design.md`, `router-consistency-design.md`
- 代码位置: `src/agent/core/planning/coordinator.py`, `src/agent/tools/registry.py`, `src/agent/application/agent_service.py`
- 测试目录: `tests/`
