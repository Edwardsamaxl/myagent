# Coordinator 多 Agent 协作设计方案

> 本文档定义 mini-evolving-agent 的多 Agent 协作架构，基于 LangGraph 实现。

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Coordinator                             │
│  (主 Agent：理解问题 → 任务分解 → 调度 Workers → 汇总结果)   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  RAG Worker   │   │ Calc Worker  │   │ Web Worker    │
│   文档检索     │   │   算术计算    │   │   网络搜索    │
└───────────────┘   └───────────────┘   └───────────────┘
```

### 两种执行模式

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **Simple Mode** | 简单问题直接用 LangGraphAgent 回答 | 闲聊、单一工具调用 |
| **Coordinator Mode** | 复杂问题：规划 → 并行执行 → 汇总 | 多步骤、需要多种工具 |

---

## 二、Agent 分工

| Agent | 职责 | 输入 | 输出 |
|-------|------|------|------|
| **Coordinator** | 理解问题、分解任务、调度、汇总 | 用户问题 | 最终回复 |
| **RAG Worker** | 文档检索 | 查询文本 | 检索结果列表 |
| **Calculator Worker** | 算术计算 | 表达式 | 计算结果 |
| **Web Worker** | 网络搜索 | 查询文本 | 搜索结果摘要 |

### 工具到 Worker 的映射

```python
WORKER_TOOLS = {
    "rag": ["search_knowledge_base"],
    "calc": ["calculate"],
    "web": ["web_search"],
    "memory": ["read_memory", "remember"],
}
```

---

## 三、核心数据结构

### 3.1 PlanStepAction（扩展现有枚举）

文件：`src/agent/core/planning/plan_schema.py`

```python
class PlanStepAction(str, Enum):
    """任务步骤动作类型"""
    RAG = "rag"              # 文档检索
    CALC = "calc"            # 算术计算
    WEB = "web"              # 网络搜索
    MEMORY = "memory"        # 记忆操作
    SYNTHESIZE = "synthesize"  # 汇总生成
    AGENT_LOOP = "agent_loop"  # 保留：通用工具循环
    CLARIFY = "clarify"      # 保留：澄清问题
```

### 3.2 PlanStep（扩展依赖字段）

```python
@dataclass
class PlanStep:
    id: str                           # 步骤 ID，如 "s1"
    action: PlanStepAction            # 动作类型
    detail: str                       # 详细描述
    status: PlanStepStatus = PlanStepStatus.PENDING
    depends_on: list[str] = field(default_factory=list)  # 依赖的步骤 ID
    parallel_with: list[str] = field(default_factory=list)  # 可并行的步骤 ID
```

### 3.3 WorkerResult

文件：`src/agent/core/planning/worker_result.py`（新建）

```python
@dataclass
class WorkerResult:
    step_id: str                                    # 对应 PlanStep.id
    action: PlanStepAction                          # 动作类型
    status: Literal["pending", "running", "success", "failed"]
    output: str                                     # 执行结果文本
    error: str | None = None                       # 错误信息
    metadata: dict | None = None                    # 额外信息（如检索的 chunk 数量）

    def is_success(self) -> bool:
        return self.status == "success"
```

### 3.4 CoordinatorResult

```python
@dataclass
class CoordinatorResult:
    answer: str                       # 最终回复
    plan_id: str                      # 计划 ID
    steps: list[PlanStep]            # 计划步骤列表
    worker_results: dict[str, WorkerResult]  # 各 worker 的执行结果
    total_steps: int                 # 总步骤数
    tool_calls: list[str]            # 所有工具调用记录
```

---

## 四、模块详细设计

### 4.1 模块文件结构

```
src/agent/core/planning/
├── __init__.py
├── plan_schema.py      # 【改造】扩展 PlanStepAction 枚举
├── state.py           # 【保留】AgentState
├── nodes.py           # 【保留】工具节点
├── agent_graph.py     # 【保留】build_agent_graph
├── langgraph_agent.py # 【保留】LangGraphAgent
├── planner.py         # 【新建】任务分解逻辑
├── synthesizer.py      # 【新建】结果汇总生成
├── coordinator.py     # 【新建】主调度逻辑
└── worker_result.py   # 【新建】WorkerResult 数据类
```

### 4.2 Planner（任务分解）

文件：`src/agent/core/planning/planner.py`

```python
PLANNER_SYSTEM_PROMPT = """你是一个任务规划专家。
分析用户问题，将其分解为可并行/串行执行的步骤。

可用动作：
- rag: 在知识库中检索相关文档
- calc: 执行算术计算
- web: 在互联网上搜索信息
- memory: 读写长期记忆
- synthesize: 汇总结果生成最终回复

分解原则：
1. 独立任务可以并行执行
2. 有依赖的任务必须串行（后续步骤依赖前序步骤的结果）
3. 每个步骤应该单一职责
4. 简单问题可以直接 synthesize，不需要拆解
5. 检索和计算可以并行，最后 synthesize

输出格式（仅输出 JSON，不要有其他内容）：
{
  "plan_id": "p_时间戳_随机6位",
  "goal": "用户问题的简洁描述",
  "steps": [
    {"id": "s1", "action": "rag", "detail": "...", "depends_on": [], "parallel_with": []},
    {"id": "s2", "action": "calc", "detail": "...", "depends_on": [], "parallel_with": ["s1"]},
    {"id": "s3", "action": "synthesize", "detail": "...", "depends_on": ["s1", "s2"], "parallel_with": []}
  ]
}
"""


class Planner:
    """任务分解器：将用户问题分解为可执行的步骤计划"""

    def __init__(self, model: ModelProvider, config: AgentConfig) -> None:
        self.model = model
        self.config = config

    def create_plan(self, user_input: str, long_context: str = "") -> PlanArtifact:
        """调用 AI 生成任务分解计划"""

        messages: list[Message] = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        ]

        if long_context:
            messages.append({
                "role": "system",
                "content": f"长期上下文（记忆与技能）：\n{long_context}"
            })

        messages.append({
            "role": "user",
            "content": f"请分解这个任务：{user_input}"
        })

        response = self.model.generate(
            messages=messages,
            temperature=0.3,  # 规划时温度不宜过高
            max_tokens=768,
        )

        # 解析 JSON 返回
        try:
            plan_data = json.loads(response)
            return self._parse_plan(plan_data)
        except json.JSONDecodeError:
            # 解析失败，降级为直接 synthesize
            return self._fallback_plan(user_input)

    def _parse_plan(self, plan_data: dict) -> PlanArtifact:
        """解析 AI 返回的 JSON 为 PlanArtifact"""
        steps = []
        for s in plan_data.get("steps", []):
            step = PlanStep(
                id=s["id"],
                action=PlanStepAction(s["action"]),
                detail=s["detail"],
                depends_on=s.get("depends_on", []),
                parallel_with=s.get("parallel_with", []),
            )
            steps.append(step)

        return PlanArtifact(
            plan_id=plan_data.get("plan_id", f"p_{uuid.uuid4().hex[:6]}"),
            goal=plan_data.get("goal", ""),
            steps=steps,
        )

    def _fallback_plan(self, user_input: str) -> PlanArtifact:
        """降级计划：简单问题直接 synthesize"""
        return PlanArtifact(
            plan_id=f"p_{uuid.uuid4().hex[:6]}",
            goal=user_input,
            steps=[
                PlanStep(
                    id="s1",
                    action=PlanStepAction.SYNTHESIZE,
                    detail="直接回答用户问题",
                )
            ],
        )
```

### 4.3 Worker 执行器

文件：`src/agent/core/planning/coordinator.py`

```python
class WorkerExecutor:
    """Worker 执行器：根据 action 类型调用对应的工具"""

    def __init__(self, tools: dict[str, Tool]) -> None:
        self.tools = tools

    def execute(self, step: PlanStep, context: dict[str, WorkerResult]) -> WorkerResult:
        """执行单个步骤，返回结果"""

        action_to_tool = {
            PlanStepAction.RAG: "search_knowledge_base",
            PlanStepAction.CALC: "calculate",
            PlanStepAction.WEB: "web_search",
            PlanStepAction.MEMORY: "read_memory",
        }

        tool_name = action_to_tool.get(step.action)
        if not tool_name or tool_name not in self.tools:
            return WorkerResult(
                step_id=step.id,
                action=step.action,
                status="failed",
                output="",
                error=f"工具不存在: {tool_name}",
            )

        # 对于需要前序结果的步骤，从 context 获取输入
        tool_input = self._build_input(step, context)

        try:
            result = self.tools[tool_name].func(tool_input)
            return WorkerResult(
                step_id=step.id,
                action=step.action,
                status="success",
                output=str(result),
            )
        except Exception as exc:
            return WorkerResult(
                step_id=step.id,
                action=step.action,
                status="failed",
                output="",
                error=str(exc),
            )

    def _build_input(self, step: PlanStep, context: dict[str, WorkerResult]) -> str:
        """根据依赖构建工具输入"""
        if not step.depends_on:
            return step.detail

        # 从依赖步骤中提取结果作为输入
        inputs = []
        for dep_id in step.depends_on:
            if dep_id in context and context[dep_id].is_success():
                inputs.append(context[dep_id].output)

        return "\n".join(inputs) if inputs else step.detail
```

### 4.4 Coordinator（主调度器）

文件：`src/agent/core/planning/coordinator.py`

```python
class Coordinator:
    """Coordinator：主调度器，管理任务分解和执行流程"""

    def __init__(
        self,
        config: AgentConfig,
        model: ModelProvider,
        tools: dict[str, Tool],
    ) -> None:
        self.config = config
        self.model = model
        self.tools = tools
        self.planner = Planner(model, config)
        self.synthesizer = Synthesizer(model, config)
        self.worker_executor = WorkerExecutor(tools)

    def run(
        self,
        user_input: str,
        history_messages: list[dict] | None = None,
        long_context: str = "",
    ) -> CoordinatorResult:
        """执行完整的 Coordinator 流程"""

        history_messages = history_messages or []

        # Step 1: 任务分解
        plan = self.planner.create_plan(user_input, long_context)

        # Step 2: 执行计划
        worker_results = self._execute_plan(plan)

        # Step 3: 汇总生成
        answer = self.synthesizer.synthesize(user_input, plan, worker_results)

        # 收集所有工具调用
        tool_calls = [
            f"{r.action.value}:{r.step_id}"
            for r in worker_results.values()
            if r.is_success()
        ]

        return CoordinatorResult(
            answer=answer,
            plan_id=plan.plan_id,
            steps=plan.steps,
            worker_results=worker_results,
            total_steps=len(plan.steps),
            tool_calls=tool_calls,
        )

    def _execute_plan(self, plan: PlanArtifact) -> dict[str, WorkerResult]:
        """执行计划：处理依赖关系，并行/串行调度"""

        results: dict[str, WorkerResult] = {}
        pending = {s.id: s for s in plan.steps}
        max_iterations = len(plan.steps) * 2  # 防止死循环

        for _ in range(max_iterations):
            if not pending:
                break

            # 找出所有依赖已满足且未执行的步骤
            ready = []
            for step_id, step in pending.items():
                if all(dep in results and results[dep].is_success()
                       for dep in step.depends_on):
                    ready.append(step)

            if not ready:
                # 无法继续执行（可能是循环依赖或全部失败）
                break

            # 并行执行所有就绪的步骤
            for step in ready:
                result = self.worker_executor.execute(step, results)
                results[step.id] = result
                del pending[step.id]

        return results
```

### 4.5 Synthesizer（汇总生成器）

文件：`src/agent/core/planning/synthesizer.py`

```python
SYNTHESIZER_SYSTEM_PROMPT = """你是一个专业的金融知识助手。
根据各步骤的执行结果，汇总生成最终回复。

要求：
1. 优先使用已有数据，避免编造
2. 关键数据需要引用来源
3. 保持回答简洁专业
4. 如果某些步骤失败，在回复中说明
"""


class Synthesizer:
    """结果汇总生成器"""

    def __init__(self, model: ModelProvider, config: AgentConfig) -> None:
        self.model = model
        self.config = config

    def synthesize(
        self,
        user_input: str,
        plan: PlanArtifact,
        worker_results: dict[str, WorkerResult],
    ) -> str:
        """根据计划和各 worker 结果生成最终回复"""

        # 构建汇总上下文
        context_parts = []
        for step in plan.steps:
            if step.id in worker_results:
                result = worker_results[step.id]
                status_icon = "✓" if result.is_success() else "✗"
                context_parts.append(
                    f"{status_icon} [{step.action.value.upper()}] {step.detail}\n"
                    f"结果: {result.output if result.is_success() else result.error}"
                )

        context = "\n\n".join(context_parts)

        # 如果所有步骤都失败，降级为直接生成
        if not any(r.is_success() for r in worker_results.values()):
            return "抱歉，我无法完成您的请求，请尝试重新描述您的问题。"

        messages: list[Message] = [
            {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
            {"role": "user", "content": f"用户问题：{user_input}\n\n执行详情：\n{context}\n\n请生成最终回复。"},
        ]

        return self.model.generate(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
```

---

## 五、与现有代码集成

### 5.1 AgentService 改造

文件：`src/agent/application/agent_service.py`

```python
from ..core.planning.coordinator import Coordinator


class AgentService:
    def __init__(self, config: AgentConfig) -> None:
        # ... 现有初始化 ...

        # 新增：Coordinator（多 Agent 模式）
        self.coordinator = Coordinator(
            config=config,
            model=self.model,
            tools=self.tools,
        )

        # 保留：单一 Agent 模式（用于简单问题）
        self.agent = LangGraphAgent(config=self.config, model=self.model, tools=self.tools)

    def chat(
        self,
        session_id: str,
        user_message: str,
        *,
        use_rag: bool | None = None,
        use_coordinator: bool | None = None,
    ) -> dict[str, Any]:
        """对话入口

        use_coordinator:
            None - 自动选择（根据问题复杂度）
            True - 强制使用 Coordinator
            False - 强制使用简单模式
        """
        rag_on = self.config.rag_enabled if use_rag is None else bool(use_rag)
        use_coord = self._should_use_coordinator(user_message) if use_coordinator is None else use_coordinator

        if use_coord:
            result = self.coordinator.run(
                user_input=user_message,
                history_messages=history,
                long_context=self.build_long_context(),
            )
            return {
                "answer": result.answer,
                "steps_used": result.total_steps,
                "tool_calls": result.tool_calls,
                "session_id": session_id,
                "plan_id": result.plan_id,
            }
        else:
            # 简单模式：直接用 LangGraphAgent
            return self.agent.run(...)

    def _should_use_coordinator(self, user_input: str) -> bool:
        """判断是否使用 Coordinator 模式"""
        # 启发式规则：
        # - 问题较长（> 50 字符）
        # - 包含多个关键词（数据、年份、计算等）
        # - 问及多个实体
        return len(user_input) > 50 or any(
            k in user_input.lower()
            for k in ["多少", "增长", "同比", "年度", "报告", "计算"]
        )
```

### 5.2 配置项

文件：`src/agent/config.py`

```python
@dataclass
class AgentConfig:
    # ... 现有字段 ...

    # Coordinator 配置
    use_coordinator: bool = True          # 是否启用 Coordinator
    coordinator_temperature: float = 0.3   # 规划时温度
    coordinator_max_tokens: int = 768    # 规划时最大 token
```

### 5.3 环境变量

```bash
# .env
USE_COORDINATOR=true
COORDINATOR_TEMPERATURE=0.3
COORDINATOR_MAX_TOKENS=768
```

---

## 六、执行流程图

```
用户：贵州茅台2024年净利润是多少？同比增长多少？

┌─────────────────────────────────────────┐
│           AgentService.chat()              │
│    判断：use_coordinator = True          │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         Coordinator.run()                 │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│        Planner.create_plan()              │
│                                          │
│  分析问题后返回：                         │
│  PlanArtifact:                            │
│    s1: rag (检索财务数据)                 │
│    s2: calc (计算增长率)                  │
│    s3: synthesize (汇总)                  │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│        _execute_plan()                    │
│                                          │
│  第一波（并行，s1 和 s2 互不依赖）：       │
│  ┌─────────────┐    ┌─────────────┐     │
│  │ s1: rag     │ // │ s2: calc    │     │
│  │ (检索年报)   │    │ (并行执行)   │     │
│  └─────────────┘    └─────────────┘     │
│         │                  │             │
│         └────────┬─────────┘             │
│                  ▼                       │
│  第二波（s3 依赖 s1,s2）：                │
│  ┌─────────────────────────┐             │
│  │ s3: synthesize (汇总)   │             │
│  └─────────────────────────┘             │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│       Synthesizer.synthesize()            │
│                                          │
│  输入：用户问题 + s1输出 + s2输出         │
│  输出：最终回复                           │
└─────────────────────────────────────────┘
```

---

## 七、错误处理

### 7.1 步骤执行失败

```python
def _execute_plan(self, plan: PlanArtifact) -> dict[str, WorkerResult]:
    results = {}

    for step in plan.steps:
        result = self.worker_executor.execute(step, results)
        results[step.id] = result

        # 如果关键步骤失败，记录但继续（synthesize 可以尝试降级回复）
        if result.status == "failed" and step.action == PlanStepAction.SYNTHESIZE:
            # synthesize 失败，无法继续
            break

    return results
```

### 7.2 死锁保护

```python
# 如果 pending 中有步骤但没有任何 ready，说明有循环依赖或前置全部失败
if not ready and pending:
    # 将剩余步骤标记为 skipped
    for step_id, step in pending.items():
        results[step_id] = WorkerResult(
            step_id=step_id,
            action=step.action,
            status="failed",
            output="",
            error="依赖步骤执行失败或循环依赖",
        )
    break
```

---

## 八、工具扩展建议

### 8.1 建议增加的工具

| 工具 | 描述 | 优先级 |
|------|------|--------|
| `search_code` | 在工作区搜索代码片段 | 中 |
| `list_files` | 列出工作区文件 | 中 |
| `read_url` | 读取网页内容 | 低 |

### 8.2 MCP 扩展接口

预留 MCP 工具接入能力：

```python
def extend_tools(self, mcp_tools: list[Tool]) -> None:
    """运行时扩展工具列表（支持 MCP）"""
    for tool in mcp_tools:
        self.tools[tool.name] = tool
```

---

## 九、测试计划

### 9.1 单元测试

```python
# tests/unit/test_planner.py
def test_simple_plan():
    planner = Planner(model, config)
    plan = planner.create_plan("计算 2+3", "")
    assert plan.steps[0].action == PlanStepAction.CALC

# tests/unit/test_coordinator.py
def test_parallel_execution():
    coordinator = Coordinator(config, model, tools)
    result = coordinator.run("查询数据", [], "")
    assert result.answer  # 不为空
```

### 9.2 集成测试

```python
# tests/integration/test_coordinator_flow.py
def test_rag_and_calc_flow():
    """测试 RAG + 计算的完整流程"""
    # 1. 摄入测试文档
    # 2. 提问需要计算的问题
    # 3. 验证结果包含检索和计算
```

---

## 十、实现优先级

| 优先级 | 任务 | 工作量 |
|--------|------|--------|
| 1 | 扩展 `plan_schema.py`（PlanStepAction 枚举） | 5 分钟 |
| 2 | 新建 `worker_result.py` | 5 分钟 |
| 3 | 新建 `planner.py`（任务分解） | 1 小时 |
| 4 | 新建 `synthesizer.py`（汇总生成） | 30 分钟 |
| 5 | 新建 `coordinator.py`（主调度） | 2 小时 |
| 6 | 改造 `agent_service.py`（集成） | 30 分钟 |
| 7 | 测试和调优 | 1 小时 |

**总计**：约 5 小时

---

## 附录：相关文件

| 文件 | 说明 |
|------|------|
| `src/agent/core/planning/planner.py` | 任务分解逻辑 |
| `src/agent/core/planning/synthesizer.py` | 结果汇总生成 |
| `src/agent/core/planning/coordinator.py` | 主调度器 |
| `src/agent/core/planning/worker_result.py` | Worker 结果数据类 |
| `src/agent/core/planning/plan_schema.py` | 扩展的 schema |
| `src/agent/application/agent_service.py` | 集成入口 |
